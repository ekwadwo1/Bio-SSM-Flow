#!/usr/bin/env python3
"""
eval_fold0_calib.py — recompute fold 0's calibrated metrics so all 5 folds are
consistent. Fold 0 was trained BEFORE the calibration patch, so its val_log lacks
picp90_cal / temperature. This loads fold-0 best.pt, regenerates on fold-0's
val set, fits temperature T, and writes the calibrated metrics as a new jsonl line
into runs/forecast_fold0/val_log.jsonl so aggregate_results.py picks them up.

No retraining. ~10 min. Run on a free GPU.
Run:  CUDA_VISIBLE_DEVICES=0 python eval_fold0_calib.py
"""
from __future__ import annotations
import os, sys, json, copy
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
import calibration as CAL
import metrics as M

CACHE = "/.../Bio-SSM-Flow/latent_cache_lumiere"
RUN   = "/.../Bio-SSM-Flow/runs/forecast_fold0"
VQGAN = "/.../checkpoints/vqgan_r4/best_vqgan.pt"
DEVICE = "cuda:0"; FOLD, K = 0, 5; R = 32; STEPS = 20; N_IMG = 8


@torch.no_grad()
def main():
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[FOLD]
    st = torch.load(os.path.join(RUN, "best.pt"), map_location=DEVICE, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2,2,4),
                        bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net"))); net.eval(); net.set_week(None)
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False)
    vq.eval().to(DEVICE)
    def v_fn(z,tau,za,dt): return net(z,tau,za,dt)
    print(f"[f0cal] fold 0: {len(val_idx)} val pairs, checkpoint step {st.get('step')}")

    agg = defaultdict(list); per_s=[]; per_y=[]; n_img=0
    for i in val_idx:
        s = ds[i]
        z_a=s["z_a"].unsqueeze(0).to(DEVICE); z_b=s["z_b"].unsqueeze(0).to(DEVICE); dt=s["dt"].view(1).to(DEVICE)
        samp = generate(v_fn, z_a, dt, steps=STEPS, n_samples=R)
        mean = samp.mean(0)
        for k,v in M.latent_fidelity(mean, z_b).items(): agg[k].append(v)
        agg["crps"].append(M.ensemble_crps(samp, z_b))
        agg["picp90_raw"].append(M.picp(samp, z_b, 0.9))
        agg["diversity"].append(M.sample_diversity(samp))
        s2 = samp.var(0).mean(1).reshape(z_b.shape[0],-1).clamp_min(1e-8)
        for k,v in M.uncertainty_block(mean, z_b, s2).items(): agg[k].append(v)
        per_s.append(samp); per_y.append(z_b)
        if n_img < N_IMG:
            xp=vq.decode(mean); xt=vq.decode(z_b)
            for k,v in M.image_fidelity(xp, xt, s["mask_b"].unsqueeze(0).to(DEVICE)).items(): agg[k].append(v)
            del xp,xt; torch.cuda.empty_cache(); n_img+=1
        del samp,mean; torch.cuda.empty_cache()

    # fit temperature on fold-0 val (same protocol as other folds)
    T = CAL.fit_temperature(per_s, per_y, verbose=False)
    picp90_cal = float(np.mean([CAL._picp(CAL.calibrate(s,T), y, 0.9) for s,y in zip(per_s,per_y)]))
    cov_err = CAL.coverage_error(per_s, per_y, T)

    out = {k: float(np.mean(v)) for k,v in agg.items()}
    out.update({"step": int(st.get("step",3000)), "temperature": float(T),
                "picp90_cal": picp90_cal, "cov_err_cal": cov_err})
    print(f"[f0cal] T={T:.2f}  picp90_cal={picp90_cal:.3f}  cov_err={cov_err:.4f}  "
          f"crps={out['crps']:.4f}  err_unc_corr={out.get('err_unc_corr',float('nan')):.3f}")

    # append to fold-0 val_log so aggregate_results picks it up as best-CRPS entry
    logp = os.path.join(RUN, "val_log.jsonl")
    with open(logp, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"[f0cal] appended calibrated fold-0 entry to {logp}")
    print("[f0cal] now re-run: python aggregate_results.py  (all 5 folds consistent)")


if __name__ == "__main__":
    main()
