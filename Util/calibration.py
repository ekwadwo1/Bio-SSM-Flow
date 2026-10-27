#!/usr/bin/env python3
"""
src/calibration.py — post-hoc temperature (spread) calibration for the generative
forecaster. Fit T on VALIDATION, apply at inference. Never fit on test (leak).

Diagnosis established (diag_temp.py): the flow produces well-RANKED but under-
dispersed predictive distributions; a single scalar T that widens the ensemble
spread about its mean restores calibration across all nominal levels (fold 0:
mean coverage error 0.17 -> 0.04 at T=2.0). Widening about the mean leaves the
mean forecast, CRPS-of-mean, PSNR, err_unc_corr UNCHANGED.

    calibrate(samples, T): z_r <- mean + T*(z_r - mean)   # [R,B,...] -> [R,B,...]
    fit_temperature(...) : grid-search T minimizing mean|coverage - nominal| on val
"""
from __future__ import annotations
from typing import List
import numpy as np
import torch

NOMINALS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
DEFAULT_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


def calibrate(samples: torch.Tensor, T: float) -> torch.Tensor:
    """Widen ensemble spread about its own mean by factor T. samples: [R,B,c,h,w,d]."""
    mean = samples.mean(0, keepdim=True)
    return mean + T * (samples - mean)


def _picp(samples: torch.Tensor, y: torch.Tensor, nominal: float) -> float:
    lo = torch.quantile(samples, (1 - nominal) / 2, dim=0)
    hi = torch.quantile(samples, 1 - (1 - nominal) / 2, dim=0)
    return ((y >= lo) & (y <= hi)).float().mean().item()


def coverage_error(per_pair_samples: List, per_pair_y: List, T: float,
                   nominals=NOMINALS) -> float:
    """mean_{nominal} | empirical_coverage(T) - nominal |, averaged over val pairs."""
    errs = []
    for nom in nominals:
        cov = np.mean([_picp(calibrate(s, T), y, nom)
                       for s, y in zip(per_pair_samples, per_pair_y)])
        errs.append(abs(cov - nom))
    return float(np.mean(errs))


def fit_temperature(per_pair_samples: List, per_pair_y: List,
                    grid=DEFAULT_GRID, verbose=True) -> float:
    """
    Grid-search T on VALIDATION ensembles. Returns the T minimizing coverage error.
    per_pair_samples: list of [R,1,c,h,w,d]; per_pair_y: list of [1,c,h,w,d].
    """
    best_T, best_err = 1.0, 1e9
    for T in grid:
        e = coverage_error(per_pair_samples, per_pair_y, T)
        if verbose:
            print(f"    T={T:.2f}  mean|cov-nom|={e:.4f}")
        if e < best_err:
            best_T, best_err = T, e
    if verbose:
        print(f"  -> fitted T = {best_T:.2f} (coverage error {best_err:.4f})")
    return best_T


if __name__ == "__main__":
    # sanity: synthetic under-dispersed gaussian ensemble; the right T should recover ~1/spread
    torch.manual_seed(0)
    R, B, N = 64, 1, 4*8*8*8
    true = torch.zeros(B, 4, 8, 8, 8)
    # ensemble with HALF the needed spread -> optimal T ~ 2.0
    samp = true.unsqueeze(0) + 0.5 * torch.randn(R, B, 4, 8, 8, 8)
    # y drawn from the FULL-spread distribution
    y = true + 1.0 * torch.randn(B, 4, 8, 8, 8)
    T = fit_temperature([samp], [y], grid=[1.0,1.5,2.0,2.5,3.0])
    print(f"recovered T ~ {T} (expected ~2.0 since ensemble spread was half the true)")
    print("calibration.py self-test done.")
