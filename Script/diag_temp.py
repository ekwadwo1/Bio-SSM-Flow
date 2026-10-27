#!/usr/bin/env python3
"""
diag_temp.py — can POST-HOC noise temperature-scaling fix the overconfidence?

Diagnosis (from diag_picp.py): the model's uncertainty is well-RANKED (err_unc_corr
~0.89) but mis-SCALED (calibration curve uniformly below diagonal; even nominal-0.5
covers only 0.32). That "ranking-good, scale-off" pattern is the textbook case where
a ONE-PARAMETER post-hoc calibration fixes it, no retraining.

Mechanism: the conditional flow eps->z_b learned a near-deterministic map (diversity
~0.19). We widen the predictive distribution by scaling the sampling stochasticity.
Two knobs, both post-hoc on the FROZEN best.pt:

  (T_eps)  scale the INITIAL noise:      z0 = T_eps * eps        (eps ~ N(0,I))
  (T_out)  scale the SPREAD about mean:  z_r <- mean + T_out*(z_r - mean)
           (pure post-hoc widening of the ensemble; cannot change the mean, so CRPS
            of the mean & accuracy are UNCHANGED — only calibration changes.)

We sweep T and, per T, report the calibration curve (empirical vs nominal) and pick
the T minimizing mean |empirical - nominal| across levels. If a T straightens the
curve toward the diagonal, overconfidence is SOLVED post-hoc. If no T works (shape
wrong, not just scale), it's a training problem -> beta-NLL / SDE sampling.

Run:  CUDA_VISIBLE_DEVICES=0 python diag_temp.py
"""
from __future__ import annotations
import os, sys
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
R = 64
GEN_STEPS = 20
N_VAL = 40
NOMINALS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
T_OUT_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]     # spread-widening multipliers


def picp_at(samples, y, nominal):
    lo = torch.quantile(samples, (1 - nominal) / 2, dim=0)
    hi = torch.quantile(samples, 1 - (1 - nominal) / 2, dim=0)
    return ((y >= lo) & (y <= hi)).float().mean().item()


def load_net():
    p = os.path.join(RUNDIR, "best.pt")
    st = torch.load(p, map_location=DEVICE, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                        bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net")), strict=True)
    net.eval(); net.set_week(None)
    print(f"[temp] loaded best.pt step={st.get('step','?')}")
    return net


@torch.no_grad()
def main():
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[FOLD]
    val_idx = val_idx[:N_VAL]
    net = load_net()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)

    # generate the ensemble ONCE per pair (T_out is applied post-hoc, no re-gen needed)
    per_pair = []
    for i in val_idx:
        s = ds[i]
        z_a = s["z_a"].unsqueeze(0).to(DEVICE); z_b = s["z_b"].unsqueeze(0).to(DEVICE)
        dt = s["dt"].view(1).to(DEVICE)
        samp = generate(v_fn, z_a, dt, steps=GEN_STEPS, n_samples=R)   # [R,1,c,h,w,d]
        per_pair.append((samp, z_b))

    print(f"\n[temp] T_out sweep — widen ensemble spread about its mean (mean/accuracy unchanged)")
    print(f"{'T_out':>6} | " + " ".join(f"n{int(n*100):02d}" for n in NOMINALS) + " | mean|cov-nom|")
    print("-" * 62)
    best = (None, 1e9, None)
    for T in T_OUT_SWEEP:
        covs = []
        for nom in NOMINALS:
            cvals = []
            for samp, z_b in per_pair:
                mean = samp.mean(0, keepdim=True)
                widened = mean + T * (samp - mean)         # post-hoc widening
                cvals.append(picp_at(widened, z_b, nom))
            covs.append(np.mean(cvals))
        err = np.mean([abs(c - n) for c, n in zip(covs, NOMINALS)])
        row = " ".join(f"{c:.2f}" for c in covs)
        flag = ""
        if err < best[1]:
            best = (T, err, covs); flag = ""
        print(f"{T:>6.1f} | {row} | {err:.4f}")

    Tb, errb, covb = best
    print("\n" + "=" * 62)
    print(f"BEST T_out = {Tb:.1f}  (mean |coverage - nominal| = {errb:.4f})")
    print("calibration at best T_out:")
    for nom, c in zip(NOMINALS, covb):
        print(f"    nominal {nom:.2f} -> empirical {c:.3f}  {'OK' if abs(c-nom)<0.1 else 'off'}")
    print("=" * 62)
    print("READ:")
    print("  best T_out straightens curve toward diagonal (mean err < ~0.08) -> POST-HOC")
    print("     CALIBRATION SOLVES IT. Widen ensemble by T_out at inference; no retraining.")
    print("     (mean forecast & CRPS-of-mean unchanged; only interval widths change.)")
    print("  no T_out works (curve stays bent / err stays high) -> SHAPE is wrong, not scale")
    print("     -> training fix needed (beta-NLL or SDE sampling).")


if __name__ == "__main__":
    main()
