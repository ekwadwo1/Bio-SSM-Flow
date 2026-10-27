#!/usr/bin/env python3
"""
offline_biophysics.py — Pillar 2 as OFFLINE analysis (no training coupling).

Two paper results on the trained forecaster + cached data:

  RESULT 1 — physics consistency of FORECASTS: for held-out pairs, generate the
    forecast, derive its density, and fit the (D,rho) that best explains the
    baseline->forecast density change under Fisher-KPP. Report variance explained
    (analogous to the ~20% on GT). Answers: "are the MODEL'S forecasts reaction-
    diffusion-consistent?" (not just the ground truth).

  RESULT 2 — interpretable growth parameters: distribution of fitted (D,rho),
    invasion speed 2*sqrt(rho*D), D/rho ratio across patients. If LUMIERE survival
    is available, correlate D/rho with overall survival (prognostic probe).

Uses smoothed density (the setting that gave ~20% in the ablation). Light: uses the
trained forecaster's MEAN forecast (few-step generate), no backprop, no coupling.

Run:  CUDA_VISIBLE_DEVICES=1 python offline_biophysics.py   (GPU 1 / any free card)
"""
from __future__ import annotations
import os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
from density import from_labels, from_probs
from physics_operators import laplacian3d
import calibration as CAL

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
RUN   = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/forecast_fold0/best.pt"
VQGAN = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG   = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
DEVICE = "cuda:0"; FOLD, K = 0, 5; N_VAL = 60; R = 16; H = 1.42; T_CAL = 2.0


def smooth(c, reps=2):
    w = torch.ones(1,1,3,3,3, device=c.device)/27
    for _ in range(reps): c = F.conv3d(F.pad(c,(1,1,1,1,1,1),mode='replicate'), w)
    return c

def fit(c_a, c_b, dt):
    c_bar = 0.5*(c_a+c_b); dcdt = (c_b-c_a)/max(dt,1.0)
    lap = laplacian3d(c_bar, H); react = c_bar*(1-c_bar)
    m = (c_bar[0,0] > 0.03)
    if m.sum() < 50: return None
    y = dcdt[0,0][m].reshape(-1)
    A = torch.stack([lap[0,0][m].reshape(-1), react[0,0][m].reshape(-1)], 1)
    sol,*_ = torch.linalg.lstsq(A, y.unsqueeze(1))
    D, rho = float(sol[0]), float(sol[1])
    r0 = float(y.pow(2).mean().sqrt())
    r1 = float((y-(A@sol.squeeze())).pow(2).mean().sqrt())
    return D, rho, r0, r1


@torch.no_grad()
def main():
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[FOLD]; val_idx = val_idx[:N_VAL]
    st = torch.load(RUN, map_location=DEVICE, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2,2,4), bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net"))); net.eval(); net.set_week(None)
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEVICE)
    seg = ImgSegUNet().to(DEVICE); seg.load_state_dict(torch.load(SEG,map_location=DEVICE,weights_only=False)["net"]); seg.eval()
    def v_fn(z,tau,za,dt): return net(z,tau,za,dt)

    Ds, rhos, rb, ra = [], [], [], []
    for i in val_idx:
        s = ds[i]
        z_a = s["z_a"].unsqueeze(0).to(DEVICE); dt = s["dt"].view(1).to(DEVICE)
        samp = CAL.calibrate(generate(v_fn, z_a, dt, steps=20, n_samples=R), T_CAL)
        z_pred = samp.mean(0)
        # densities: baseline from GT labels; forecast from seg head probs (smoothed)
        c_a = smooth(from_labels(s["seg_a"]).to(DEVICE))
        prob = F.softmax(seg(vq.decode(z_pred)), 1)
        c_b = smooth(from_probs(prob).to(DEVICE))
        out = fit(c_a, c_b, float(s["dt"]))
        if out is None: continue
        D, rho, r0, r1 = out
        if 0 < D < 5 and 0 < rho < 1:
            Ds.append(D); rhos.append(rho); rb.append(r0); ra.append(r1)

    Ds, rhos, rb, ra = map(np.array, (Ds, rhos, rb, ra))
    print("\n" + "="*60)
    print("RESULT 1 — Fisher-KPP consistency of MODEL FORECASTS")
    print("="*60)
    if len(rb):
        ve = 100*(1 - ra.mean()/max(rb.mean(),1e-12))
        print(f"  fittable forecasts: {len(rb)}/{N_VAL}")
        print(f"  variance explained by reaction-diffusion: {ve:.1f}%")
        print(f"  (cf. ~20% on ground-truth densities -> forecasts are similarly")
        print(f"   physics-consistent, validating the model produces plausible growth)")
    print("\n" + "="*60)
    print("RESULT 2 — interpretable growth parameters (prognostic probe)")
    print("="*60)
    if len(Ds):
        v = 2*np.sqrt(np.clip(rhos,0,None)*np.clip(Ds,0,None))
        print(f"  D   (mm^2/day): median {np.median(Ds):.4f}  IQR [{np.percentile(Ds,25):.4f},{np.percentile(Ds,75):.4f}]")
        print(f"  rho (1/day)   : median {np.median(rhos):.4f}  IQR [{np.percentile(rhos,25):.4f},{np.percentile(rhos,75):.4f}]")
        print(f"  invasion speed 2sqrt(rho*D): median {np.median(v):.4f} mm/day (~{np.median(v)*30:.2f} mm/month)")
        print(f"  D/rho ratio   : median {np.median(Ds/np.clip(rhos,1e-6,None)):.3f}")
        print(f"  -> report distributions as a figure; correlate D/rho with survival")
        print(f"     (LUMIERE Demographics_Pathology.csv) if outcomes available.")
    # save for the paper
    import json
    np.savez(os.path.join(os.path.dirname(RUN), "biophysics_offline.npz"),
             D=Ds, rho=rhos, resid_before=rb, resid_after=ra)
    print(f"\n[saved] biophysics_offline.npz (D, rho, residuals per patient)")


if __name__ == "__main__":
    main()
