#!/usr/bin/env python3
"""
prototype_biophysics.py — validate Fisher-KPP on REAL LUMIERE density pairs.

Runs entirely on cached GROUND-TRUTH segmentations (from_labels), no model, CPU-ok.
Answers three questions that decide how the biophysics enters the paper:

  Q1 CALIBRATION: what is the real magnitude of dc/dt over real intervals? -> sets
     FisherKPPLoss.residual_scale so L_bio is O(1) and lambda_bio is tunable.

  Q2 CONSISTENCY: is real glioma growth even approximately Fisher-KPP-consistent?
     For each consecutive pair, fit the single (D,rho) that MINIMIZES the residual
     (closed-form least squares: R = dcdt - D*lap - rho*react is linear in (D,rho)).
     Report the residual BEFORE (D=rho=0, i.e. "no physics") vs AFTER the best fit.
     A large drop => the physics explains real growth; ~no drop => it doesn't (and we
     must reframe the biophysics claim). This is a genuine scientific check.

  Q3 PARAMETERS: distribution of fitted (D,rho), invasion speed 2*sqrt(rho*D),
     and D/rho ratio across patients -> a results figure + prognostic covariate.

Uses cached seg_a/seg_b via CachedPairDataset (already keyed & paired).
Run:  python prototype_biophysics.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from density import from_labels
from physics_operators import laplacian3d, grad3d

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
H_MM = 1.42          # voxel spacing (mm), LUMIERE 182->128
N_PAIRS = 120        # subsample for speed; raise for the final numbers


def fit_D_rho(c_a, c_b, dt_days, brain=None):
    """
    Least-squares fit of (D,rho) minimizing || dcdt - D*lap(c_bar) - rho*react ||^2.
    Linear in (D,rho): design matrix cols [lap, react], target dcdt. Returns
    (D,rho, resid_before, resid_after) where resid_before uses D=rho=0.
    """
    c_bar = 0.5 * (c_a + c_b)
    dt = max(float(dt_days), 1.0)
    dcdt = (c_b - c_a) / dt
    lap = laplacian3d(c_bar, H_MM)
    react = c_bar * (1.0 - c_bar)
    # restrict to a band around the tumor (where density changes) for a meaningful fit
    m = (c_bar[0, 0] > 0.05) if brain is None else brain.bool()
    if m.sum() < 50:
        return None
    y = dcdt[0, 0][m].reshape(-1)
    A = torch.stack([lap[0, 0][m].reshape(-1), react[0, 0][m].reshape(-1)], 1)  # [n,2]
    sol, *_ = torch.linalg.lstsq(A, y.unsqueeze(1))
    D, rho = float(sol[0]), float(sol[1])
    resid_before = float(y.pow(2).mean().sqrt())                 # no physics
    resid_after = float((y - (A @ sol.squeeze()).squeeze()).pow(2).mean().sqrt())
    return D, rho, resid_before, resid_after


def main():
    ds = CachedPairDataset(CACHE)
    n = min(N_PAIRS, len(ds))
    print(f"[proto] fitting Fisher-KPP on {n} real LUMIERE density pairs\n")
    dcdt_mags, Ds, rhos, rb, ra = [], [], [], [], []
    kept = 0
    for i in range(n):
        s = ds[i]
        c_a = from_labels(s["seg_a"])         # [1,1,128,128,128]
        c_b = from_labels(s["seg_b"])
        dt = float(s["dt"])
        dcdt_mags.append(float(((c_b - c_a) / max(dt, 1.0)).abs().mean()))
        out = fit_D_rho(c_a, c_b, torch.tensor(dt))
        if out is None:
            continue
        D, rho, r0, r1 = out
        # keep physically-plausible fits (positive-ish, bounded)
        if 0 < D < 5 and 0 < rho < 1:
            Ds.append(D); rhos.append(rho); rb.append(r0); ra.append(r1); kept += 1

    dcdt_mags = np.array(dcdt_mags)
    print("=== Q1 CALIBRATION ===")
    print(f"  real |dc/dt| : median {np.median(dcdt_mags):.2e}  "
          f"p90 {np.percentile(dcdt_mags,90):.2e}")
    print(f"  -> set FisherKPPLoss.residual_scale ~ {np.median(dcdt_mags):.1e} "
          f"(makes L_bio O(1))\n")

    print("=== Q2 CONSISTENCY (does real growth obey Fisher-KPP?) ===")
    rb, ra = np.array(rb), np.array(ra)
    if len(rb):
        drop = 100 * (1 - ra.mean() / max(rb.mean(), 1e-12))
        print(f"  residual w/o physics (D=rho=0): {rb.mean():.3e}")
        print(f"  residual w/ best-fit (D,rho)  : {ra.mean():.3e}")
        print(f"  variance explained by physics : {drop:.1f}%  "
              f"(plausible-fit pairs: {kept}/{n})")
        if drop > 30:
            print("  -> Fisher-KPP explains a meaningful share of real growth. Claim holds.")
        elif drop > 10:
            print("  -> modest but real. Frame biophysics as a soft prior, not a hard law.")
        else:
            print("  -> weak. RECONSIDER: biophysics may not fit LUMIERE growth well.")
    print()

    print("=== Q3 PARAMETERS ===")
    if len(Ds):
        Ds, rhos = np.array(Ds), np.array(rhos)
        v = 2 * np.sqrt(np.clip(rhos, 0, None) * np.clip(Ds, 0, None))   # invasion speed
        print(f"  D   (mm^2/day): median {np.median(Ds):.3f}  IQR [{np.percentile(Ds,25):.3f},{np.percentile(Ds,75):.3f}]")
        print(f"  rho (1/day)   : median {np.median(rhos):.3f}  IQR [{np.percentile(rhos,25):.3f},{np.percentile(rhos,75):.3f}]")
        print(f"  invasion speed 2*sqrt(rho*D) (mm/day): median {np.median(v):.3f} "
              f"(~{np.median(v)*30:.2f} mm/month)")
        print(f"  literature: D~0.01-0.1, rho~0.01-0.03, speed~mm/month "
              f"-> {'IN RANGE' if 0.001<np.median(Ds)<1 else 'CHECK UNITS'}")
    print("\n[proto] done.")


if __name__ == "__main__":
    main()
