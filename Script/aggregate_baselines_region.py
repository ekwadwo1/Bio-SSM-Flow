#!/usr/bin/env python3
"""
aggregate_baselines_region.py — 5-fold model-vs-baseline + in-region comparison.

For each fold k in 0..4:
  - load runs/forecast_fold{k}/best.pt (+ its fitted temperature)
  - on fold k's VAL patients, compute:
       MODEL  (calibrated generative mean forecast)
       COPY   (z_b_hat = z_a)                              [sanity floor]
       LINEAR (z_a + (z_a - z_prev)*(dt/dt_prev), else copy) [momentum floor]
    metrics: whole-volume latent MSE, in-region latent MSE, in-region cos,
             + relative improvement of MODEL over COPY (whole & in-region).
  In-region = dilated union of seg-head tumor prediction at baseline & follow-up.

Reports mean +/- std across folds for each, and the KEY paper number:
  in-region improvement of MODEL over COPY (mean +/- std).

CPU-light but uses GPU for generation + decode. ~15-25 min for 5 folds.
Run:  CUDA_VISIBLE_DEVICES=0 python aggregate_baselines_region.py
"""
from __future__ import annotations
import os, sys, glob, re, json
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
import calibration as CAL

CACHE  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
BASE   = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs"
VQGAN  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG    = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
DEVICE = "cuda:0"; K = 5; R = 32; STEPS = 20; N_VAL = 60
TUMOR_CLASSES = (1, 3); DILATE = 3


def build_prev_index():
    byp = defaultdict(list)
    for f in glob.glob(os.path.join(CACHE, "*.pt")):
        m = re.search(r"(Patient-\d+)_week(\d+)\.pt$", os.path.basename(f))
        if m: byp[m.group(1)].append(int(m.group(2)))
    for p in byp: byp[p].sort()
    prev = {}
    for p, ws in byp.items():
        for i in range(1, len(ws)): prev[(p, ws[i])] = ws[i-1]
    return prev

def load_lat(p, w):
    f = os.path.join(CACHE, f"{p}_week{w}.pt")
    return torch.load(f, weights_only=False)["z"].float() if os.path.exists(f) else None

def tumor_roi(seg, vq, z_a, z_b):
    def tum(z):
        lab = seg(vq.decode(z)).argmax(1, keepdim=True)
        return sum((lab == c) for c in TUMOR_CLASSES) > 0
    roi = (tum(z_a) | tum(z_b)).float()
    roi = F.max_pool3d(roi, 2*DILATE+1, 1, DILATE) > 0.5
    return roi

def masked_mse_cos(a, b, m):
    mm = m.expand_as(a)
    if mm.sum() == 0: return None
    av, bv = a[mm], b[mm]
    return F.mse_loss(av, bv).item(), F.cosine_similarity(av.reshape(1,-1), bv.reshape(1,-1)).item()


@torch.no_grad()
def eval_fold(fold, vq, seg, prev_idx):
    ckpt = os.path.join(BASE, f"forecast_fold{fold}", "best.pt")
    st = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    T = float(st.get("temperature", 1.9))
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2,2,4),
                        bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net"))); net.eval(); net.set_week(None)
    def v_fn(z,tau,za,dt): return net(z,tau,za,dt)
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[fold]; val_idx = val_idx[:N_VAL]

    a = defaultdict(list)
    for i in val_idx:
        s = ds[i]
        z_a = s["z_a"].unsqueeze(0).to(DEVICE); z_b = s["z_b"].unsqueeze(0).to(DEVICE); dt = s["dt"].view(1).to(DEVICE)
        z_model = CAL.calibrate(generate(v_fn, z_a, dt, steps=STEPS, n_samples=R), T).mean(0)
        z_copy = z_a
        # linear
        p, wa, wb = s["patient"], s["week_pair"][0], s["week_pair"][1]
        z_lin = z_a
        if (p, wa) in prev_idx:
            zp = load_lat(p, prev_idx[(p,wa)])
            if zp is not None:
                dtp = max(wa - prev_idx[(p,wa)], 1); dtc = max(wb - wa, 1)
                z_lin = z_a + (z_a - zp.unsqueeze(0).to(DEVICE)) * (dtc/dtp)
        roi = tumor_roi(seg, vq, z_a, z_b)
        roi_lat = (F.avg_pool3d(roi.float(), 4) > 0.1)
        # whole
        a["whole_model"].append(F.mse_loss(z_model, z_b).item())
        a["whole_copy"].append(F.mse_loss(z_copy, z_b).item())
        a["whole_lin"].append(min(F.mse_loss(z_lin, z_b).item(), 1e3))  # clip lin blowups
        # in-region
        rm = masked_mse_cos(z_model, z_b, roi_lat); rc = masked_mse_cos(z_copy, z_b, roi_lat)
        if rm and rc:
            a["reg_model"].append(rm[0]); a["reg_copy"].append(rc[0])
            a["reg_cos_model"].append(rm[1]); a["reg_cos_copy"].append(rc[1])
        del z_model; torch.cuda.empty_cache()
    r = {k: float(np.mean(v)) for k,v in a.items() if v}
    r["whole_imp"] = 100*(1 - r["whole_model"]/r["whole_copy"])
    r["reg_imp"]   = 100*(1 - r["reg_model"]/r["reg_copy"])
    return r


def main():
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEVICE)
    seg = ImgSegUNet().to(DEVICE); seg.load_state_dict(torch.load(SEG,map_location=DEVICE,weights_only=False)["net"]); seg.eval()
    for p in list(vq.parameters())+list(seg.parameters()): p.requires_grad_(False)
    prev_idx = build_prev_index()

    per = {}
    for f in range(K):
        print(f"[bl] fold {f} ...", flush=True)
        per[f] = eval_fold(f, vq, seg, prev_idx)
        print(f"  fold {f}: whole_imp={per[f]['whole_imp']:.1f}%  reg_imp={per[f]['reg_imp']:.1f}%")

    def agg(key): 
        v = np.array([per[f][key] for f in per]); return v.mean(), v.std()
    print("\n" + "="*66)
    print("  5-FOLD BASELINE + IN-REGION (mean +/- std)")
    print("="*66)
    for key,label in [("whole_copy","COPY  whole MSE"),("whole_lin","LINEAR whole MSE"),
                      ("whole_model","MODEL whole MSE"),("whole_imp","MODEL vs COPY whole (%)"),
                      ("reg_copy","COPY  in-region MSE"),("reg_model","MODEL in-region MSE"),
                      ("reg_imp","MODEL vs COPY in-region (%)"),
                      ("reg_cos_copy","COPY  in-region cos"),("reg_cos_model","MODEL in-region cos")]:
        m,s = agg(key); print(f"  {label:<28} {m:>9.4f} +/- {s:.4f}")
    print("="*66)
    print("  KEY: MODEL vs COPY IN-REGION improvement is the headline forecasting number.")
    out = {k: {"mean": float(agg(k)[0]), "std": float(agg(k)[1])}
           for k in per[0].keys()}
    out["per_fold"] = per
    json.dump(out, open(os.path.join(BASE,"baselines_region_5fold.json"),"w"), indent=2)
    print(f"\n[saved] baselines_region_5fold.json")


if __name__ == "__main__":
    main()
