#!/usr/bin/env python3
"""
baselines.py — CPU-computable sanity-floor baselines (Tier 1) for Bio-SSM-Flow.

Computes the reference bar your generative forecaster MUST beat, on the SAME
validation split (patient-level fold 0) and the SAME metric suite (metrics.py).
No training, no GPU.

  (1) COPY baseline      : z_b_hat = z_a               ("is the model doing anything?")
  (2) LINEAR extrapolate : z_b_hat = z_a + (z_a - z_prev) * (dt / dt_prev)
                           uses the patient's PREVIOUS timepoint to extrapolate the
                           latent trend. Only defined when a prior timepoint exists
                           (LUMIERE's >=3-timepoint trajectories); pairs without a
                           predecessor fall back to copy and are counted separately.

Both are DETERMINISTIC point forecasts, so distributional metrics (CRPS/PICP) are
reported treating the point as a degenerate 1-sample ensemble (CRPS reduces to MAE;
PICP is 0/1) — clearly labelled, since that's the honest comparison for a point
baseline vs a generative model.

Run:  python baselines.py    (matches train.py's fold 0 split)
"""
from __future__ import annotations
import os, sys, re, glob, json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset, patient_kfold
import metrics as M

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cpu"                       # baselines are cheap; keep GPU free for training
FOLD, K = 0, 5


def load_latent(patient, week):
    f = os.path.join(CACHE, f"{patient}_week{week}.pt")
    return torch.load(f, weights_only=False)["z"].float() if os.path.exists(f) else None


def build_prev_index():
    """Map (patient, week) -> previous (week) for linear extrapolation."""
    byp = defaultdict(list)
    for f in glob.glob(os.path.join(CACHE, "*.pt")):
        m = re.search(r"(Patient-\d+)_week(\d+)\.pt$", os.path.basename(f))
        if m: byp[m.group(1)].append(int(m.group(2)))
    for p in byp: byp[p].sort()
    prev = {}
    for p, weeks in byp.items():
        for i in range(1, len(weeks)):
            prev[(p, weeks[i])] = weeks[i-1]
    return prev


def eval_baseline(name, predict_fn, val_idx, ds, prev_idx):
    """predict_fn(sample) -> (z_b_hat [1,c,h,w,d], used_fallback: bool) or None to skip."""
    agg = defaultdict(list); n_fallback = 0; n = 0
    for i in val_idx:
        s = ds[i]
        z_a = s["z_a"].unsqueeze(0); z_b = s["z_b"].unsqueeze(0)
        out = predict_fn(s, z_a)
        if out is None:
            continue
        z_hat, fb = out
        n_fallback += int(fb); n += 1
        # point-forecast fidelity
        for k, v in M.latent_fidelity(z_hat, z_b).items(): agg[k].append(v)
        # distributional metrics with the point as a degenerate 1-sample ensemble
        samp = z_hat.unsqueeze(0)                       # [1,1,c,h,w,d]
        agg["crps_point"].append(M.ensemble_crps(samp, z_b))   # == MAE for 1 sample
        agg["picp90_point"].append(M.picp(samp, z_b, 0.9))
    res = {k: float(np.mean(v)) for k, v in agg.items()}
    res["_n"] = n; res["_n_fallback_to_copy"] = n_fallback
    print(f"\n[{name}]  (n={n}, fell back to copy: {n_fallback})")
    for k in ["latent_mse", "latent_mae", "latent_cos", "crps_point", "picp90_point"]:
        if k in res: print(f"    {k:16s} = {res[k]:.4f}")
    return res


def main():
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[FOLD]
    prev_idx = build_prev_index()
    print(f"baselines on fold {FOLD} val ({len(val_idx)} pairs), same split as train.py")

    # (1) COPY
    def copy_fn(s, z_a): return z_a, False
    r_copy = eval_baseline("COPY  (z_b_hat = z_a)", copy_fn, val_idx, ds, prev_idx)

    # (2) LINEAR extrapolation from previous timepoint
    def linear_fn(s, z_a):
        p, wb = s["patient"], s["week_pair"][1]; wa = s["week_pair"][0]
        key = (p, wa)
        if key in prev_idx:
            w_prev = prev_idx[key]
            z_prev = load_latent(p, w_prev)
            if z_prev is not None:
                dt_prev = max((wa - w_prev), 1); dt_cur = max((wb - wa), 1)
                z_hat = z_a + (z_a - z_prev.unsqueeze(0)) * (dt_cur / dt_prev)
                return z_hat, False
        return z_a, True                     # fallback to copy, flagged
    r_lin = eval_baseline("LINEAR (trend from prev tp)", linear_fn, val_idx, ds, prev_idx)

    out = {"copy": r_copy, "linear": r_lin}
    with open(os.path.join(os.path.dirname(CACHE), "baseline_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "="*60)
    print("REFERENCE BAR (your forecaster must beat these):")
    print(f"  COPY   : latent_mse={r_copy['latent_mse']:.4f}  latent_cos={r_copy['latent_cos']:.4f}  crps(=MAE)={r_copy['crps_point']:.4f}")
    print(f"  LINEAR : latent_mse={r_lin['latent_mse']:.4f}  latent_cos={r_lin['latent_cos']:.4f}  crps(=MAE)={r_lin['crps_point']:.4f}")
    print("="*60)
    print("NOTE: your generative model's step-3000 val was latent_mse=0.260, latent_cos=0.520, CRPS=0.174")
    print("      -> compare mse/cos directly; CRPS is not apples-to-apples (point vs ensemble).")


if __name__ == "__main__":
    main()
