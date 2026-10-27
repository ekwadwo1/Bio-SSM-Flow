#!/usr/bin/env python3
"""
diag_picp.py — is the low PICP genuine OVERCONFIDENCE or a SAMPLING ARTIFACT?

The gate before any loss change. With only 8 samples, the empirical 90% interval is
estimated from the ~1st and ~8th order statistics -> noisy AND biased INWARD, which
systematically underestimates interval width -> artificially LOW PICP even for a
perfectly calibrated model. This test recomputes calibration with increasing sample
counts on a fixed checkpoint.

Reads:  best.pt (or last.pt) from the run dir.
Regenerates on the fold-0 val set with R in {8,16,32,64} and reports, per R:
  - picp90  : coverage of the central 90% interval (target 0.90)
  - picp50  : coverage of the central 50% interval (target 0.50)
  - diversity, crps, latent_cos (accuracy is R-independent-ish, sanity)

Interpretation:
  picp90 RISES toward 0.90 as R grows  -> it was a SAMPLING ARTIFACT; no training fix
      needed, just validate with more samples. Keep pure-CRPS selection.
  picp90 STAYS ~0.44 even at R=64      -> GENUINE overconfidence; a principled training
      fix (beta-NLL or variance floor) is warranted.

Also reports a CALIBRATION CURVE (empirical coverage vs nominal for several nominal
levels) at R=64 — the honest calibration diagnostic for the paper.

Run:  CUDA_VISIBLE_DEVICES=0 python diag_picp.py   (uses whatever GPU is free; small)
"""
from __future__ import annotations
import os, sys, glob, copy
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2

CACHE  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
RUNDIR = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/forecast_fold0"
DEVICE = "cuda:0"
FOLD, K = 0, 5
R_LIST = [8, 16, 32, 64]
GEN_STEPS = 20
N_VAL_PAIRS = 40            # subsample val for speed; raise for final numbers


def picp_at(samples, y, nominal):
    lo = torch.quantile(samples, (1 - nominal) / 2, dim=0)
    hi = torch.quantile(samples, 1 - (1 - nominal) / 2, dim=0)
    return ((y >= lo) & (y <= hi)).float().mean().item()


def load_net():
    for name in ["best.pt", "last.pt"]:
        p = os.path.join(RUNDIR, name)
        if os.path.exists(p):
            print(f"[diag] loading {p}")
            st = torch.load(p, map_location=DEVICE, weights_only=False)
            net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                                bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
            # prefer EMA weights (that's what validate() used)
            sd = st.get("ema", st.get("net"))
            net.load_state_dict(sd, strict=True)
            net.eval(); net.set_week(None)
            print(f"[diag] checkpoint step={st.get('step','?')} best_crps={st.get('best_crps','?')}")
            return net
    raise FileNotFoundError(f"no best.pt/last.pt in {RUNDIR}")


@torch.no_grad()
def main():
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[FOLD]
    val_idx = val_idx[:N_VAL_PAIRS]
    net = load_net()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)

    print(f"\n[diag] PICP vs sample count on {len(val_idx)} val pairs")
    print(f"{'R':>4} | {'picp50':>7} | {'picp90':>7} | {'diversity':>9} | {'lat_cos':>7}")
    print("-" * 48)
    # cache max-R samples per pair once, then subsample R from them (fair + fast)
    Rmax = max(R_LIST)
    per_pair = []
    for i in val_idx:
        s = ds[i]
        z_a = s["z_a"].unsqueeze(0).to(DEVICE); z_b = s["z_b"].unsqueeze(0).to(DEVICE)
        dt = s["dt"].view(1).to(DEVICE)
        samp = generate(v_fn, z_a, dt, steps=GEN_STEPS, n_samples=Rmax)  # [Rmax,1,c,h,w,d]
        per_pair.append((samp.cpu(), z_b.cpu()))

    for R in R_LIST:
        p50s, p90s, divs, coss = [], [], [], []
        for samp, z_b in per_pair:
            sub = samp[:R]                        # first R samples
            p50s.append(picp_at(sub, z_b, 0.50))
            p90s.append(picp_at(sub, z_b, 0.90))
            divs.append(sub.std(0).mean().item())
            coss.append(F.cosine_similarity(sub.mean(0).flatten(1), z_b.flatten(1)).mean().item())
        print(f"{R:>4} | {np.mean(p50s):7.3f} | {np.mean(p90s):7.3f} | "
              f"{np.mean(divs):9.4f} | {np.mean(coss):7.3f}")

    # calibration curve at Rmax: empirical coverage vs nominal
    print(f"\n[diag] calibration curve at R={Rmax} (empirical coverage vs nominal):")
    for nom in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
        cov = np.mean([picp_at(samp, z_b, nom) for samp, z_b in per_pair])
        flag = "" if abs(cov - nom) < 0.1 else "  <-- off"
        print(f"    nominal {nom:.2f} -> empirical {cov:.3f}{flag}")

    print("\nVERDICT GUIDE:")
    print("  picp90 rises toward 0.90 as R grows  -> SAMPLING ARTIFACT (no training fix; validate w/ more samples)")
    print("  picp90 flat & low even at R=64       -> GENUINE overconfidence (apply beta-NLL / variance floor)")


if __name__ == "__main__":
    main()
