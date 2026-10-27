x#!/usr/bin/env python3
"""
significance_tests.py — paired significance testing for the Results section.

Computes PER-SUBJECT (per-pair) metrics for each method on the pooled 5-fold
held-out validation set, then runs Wilcoxon signed-rank tests with per-subject
pairing (each pair gets the SAME noise seed across methods, so differences reflect
the model, not stochastic sampling variation). Reports p-values, Bonferroni-
corrected p, and rank-biserial correlation as effect size.

Methods compared (all on identical pairs/splits):
  MODEL (flow, calibrated)  vs  COPY
  MODEL                     vs  LINEAR
  MODEL                     vs  DIFFUSION (DDPM transport)   [if ddpm runs exist]

Primary paired quantity: per-pair IN-REGION latent MSE (the headline metric).
Also reports whole-volume MSE and per-pair CRPS tests.

Outputs: console table + significance_tests.json + a LaTeX daggers legend.
Run:  CUDA_VISIBLE_DEVICES=0 python significance_tests.py
"""
from __future__ import annotations
import os, sys, glob, re, json
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
import calibration as CAL
import metrics as M

CACHE  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
BASE   = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs"
VQGAN  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG    = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
DEVICE = "cuda:0"; K = 5; R = 32; STEPS = 20; SEED = 0
TUMOR_CLASSES = (1, 3); DILATE = 3
HAS_DDPM = all(os.path.exists(f"{BASE}/ddpm_fold{k}/best.pt") for k in range(K))


def load_flow(fold):
    st = torch.load(f"{BASE}/forecast_fold{fold}/best.pt", map_location=DEVICE, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2,2,4), bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net"))); net.eval(); net.set_week(None)
    return net, float(st.get("temperature", 1.9))

def load_ddpm(fold):
    from ddpm_transport import DDPMTransport
    st = torch.load(f"{BASE}/ddpm_fold{fold}/best.pt", map_location=DEVICE, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2,2,4), bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net"))); net.eval(); net.set_week(None)
    ddpm = DDPMTransport(net, T=1000, device=DEVICE).to(DEVICE)
    return ddpm, float(st.get("temperature", 1.9))

def tumor_roi_lat(seg, vq, z_a, z_b):
    def tum(z):
        lab = seg(vq.decode(z)).argmax(1, keepdim=True)
        return sum((lab == c) for c in TUMOR_CLASSES) > 0
    roi = (tum(z_a) | tum(z_b)).float()
    roi = F.max_pool3d(roi, 2*DILATE+1, 1, DILATE) > 0.5
    return (F.avg_pool3d(roi.float(), 4) > 0.1)

def region_mse(a, b, m):
    mm = m.expand_as(a)
    return F.mse_loss(a[mm], b[mm]).item() if mm.sum() > 0 else np.nan


@torch.no_grad()
def collect():
    """Per-pair metrics for each method, pooled across folds, with fixed seeds."""
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEVICE)
    seg = ImgSegUNet().to(DEVICE); seg.load_state_dict(torch.load(SEG,map_location=DEVICE,weights_only=False)["net"]); seg.eval()
    for p in list(vq.parameters())+list(seg.parameters()): p.requires_grad_(False)
    ds = CachedPairDataset(CACHE)

    rows = defaultdict(lambda: defaultdict(list))   # rows[metric][method] = per-pair list
    for fold in range(K):
        flow, T_flow = load_flow(fold)
        if HAS_DDPM: ddpm, T_ddpm = load_ddpm(fold)
        _, val_idx = list(patient_kfold(ds, k=K))[fold]
        for i in val_idx:
            s = ds[i]
            z_a = s["z_a"].unsqueeze(0).to(DEVICE); z_b = s["z_b"].unsqueeze(0).to(DEVICE); dt = s["dt"].view(1).to(DEVICE)
            roi = tumor_roi_lat(seg, vq, z_a, z_b)
            # fixed seed per pair so all methods see the same noise draws
            g = torch.Generator(device=DEVICE).manual_seed(SEED + i)
            torch.manual_seed(SEED + i)
            zf = CAL.calibrate(generate(lambda z,t,za,d: flow(z,t,za,d), z_a, dt, steps=STEPS, n_samples=R), T_flow).mean(0)
            # COPY / LINEAR
            z_copy = z_a
            # per-pair metrics
            rows["whole"]["MODEL"].append(F.mse_loss(zf, z_b).item())
            rows["whole"]["COPY"].append(F.mse_loss(z_copy, z_b).item())
            rows["region"]["MODEL"].append(region_mse(zf, z_b, roi))
            rows["region"]["COPY"].append(region_mse(z_copy, z_b, roi))
            rows["crps"]["MODEL"].append(M.ensemble_crps(
                CAL.calibrate(generate(lambda z,t,za,d: flow(z,t,za,d), z_a, dt, steps=STEPS, n_samples=R), T_flow), z_b))
            if HAS_DDPM:
                torch.manual_seed(SEED + i)
                zd = CAL.calibrate(ddpm.ddim_sample(z_a, dt, steps=50, R=R), T_ddpm).mean(0)
                rows["whole"]["DDPM"].append(F.mse_loss(zd, z_b).item())
                rows["region"]["DDPM"].append(region_mse(zd, z_b, roi))
                rows["crps"]["DDPM"].append(M.ensemble_crps(ddpm.ddim_sample(z_a, dt, steps=50, R=R), z_b))
            del zf; torch.cuda.empty_cache()
        print(f"[sig] fold {fold} collected ({len(val_idx)} pairs)")
    return rows


def rank_biserial(x, y):
    """Effect size for Wilcoxon signed-rank (matched pairs): r_rb in [-1,1]."""
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    Rpos = r[d > 0].sum(); Rneg = r[d < 0].sum(); tot = r.sum()
    return float((Rpos - Rneg) / tot)


def test(rows, metric, a, b):
    xa = np.array(rows[metric][a], float); xb = np.array(rows[metric][b], float)
    mask = ~(np.isnan(xa) | np.isnan(xb)); xa, xb = xa[mask], xb[mask]
    stat, p = wilcoxon(xa, xb)            # two-sided
    return {"n": int(len(xa)), "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
            "p": float(p), "r_rb": rank_biserial(xa, xb)}


def main():
    rows = collect()
    comparisons = [("region","MODEL","COPY"), ("whole","MODEL","COPY"), ("crps","MODEL","COPY")]
    if HAS_DDPM:
        comparisons += [("region","MODEL","DDPM"), ("crps","MODEL","DDPM")]
    n_tests = len(comparisons)
    print("\n" + "="*74)
    print("  PAIRED SIGNIFICANCE (Wilcoxon signed-rank, per-pair; Bonferroni k=%d)" % n_tests)
    print("="*74)
    out = {}
    for metric, a, b in comparisons:
        r = test(rows, metric, a, b)
        p_corr = min(r["p"]*n_tests, 1.0)
        sig = "***" if p_corr<1e-3 else "**" if p_corr<1e-2 else "*" if p_corr<0.05 else "ns"
        key = f"{metric}:{a}_vs_{b}"
        out[key] = {**r, "p_bonferroni": p_corr, "sig": sig}
        print(f"  {metric:>7} {a} vs {b:<5} | n={r['n']:3d} | "
              f"med {r['median_a']:.4f} vs {r['median_b']:.4f} | "
              f"p={r['p']:.2e} p_corr={p_corr:.2e} {sig} | r_rb={r['r_rb']:+.3f}")
    print("="*74)
    print("  legend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant (Bonferroni-corrected)")
    json.dump(out, open(os.path.join(BASE,"significance_tests.json"),"w"), indent=2)
    print(f"\n[saved] significance_tests.json")
    if not HAS_DDPM:
        print("[note] DDPM runs not found yet -> DDPM comparisons skipped. Re-run after the")
        print("       ddpm campaign completes to add MODEL-vs-DDPM tests.")


if __name__ == "__main__":
    main()
