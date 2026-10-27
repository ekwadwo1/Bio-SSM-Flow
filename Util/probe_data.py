#!/usr/bin/env python3
"""
probe_data.py — NO NETWORK. Is the flow-matching target even learnable from inputs?

The convnet converged to predicting ZERO (loss frozen at exactly 1.0 = Var(target)).
That means zero is the loss-minimizing output, which only happens if the mapping
(z_s, z_a, s) -> u is not consistently learnable. This probe checks WHY, with pure
tensor math and no model:

  (1) confirm predicting-zero loss == target variance (~1.0)  [the trap]
  (2) analytic velocity u = (z_s - z_a)/s : is it STABLE across s, or does it
      explode near s=0?  (naive FM param blows up as s->0)
  (3) does the TRUE constant target u=z_b-z_a have consistent scale, and is
      cos(z_s-z_a, u) high (i.e. is the direction recoverable)?
  (4) closed-form least-squares: fit u from features [ (z_s-z_a), s, 1 ] per voxel
      -> if a LINEAR map nails it, the target is trivially learnable and the bug
      is purely optimization/parameterization; if not, the pairing is the issue.

Run:  python probe_data.py   (CPU fine, instant)
"""
from __future__ import annotations
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from flow import interpolate

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"

ds = CachedPairDataset(CACHE)
b = [ds[i] for i in range(2)]
z_a = torch.stack([x["z_a"] for x in b])          # [2,4,32,32,32]
z_b = torch.stack([x["z_b"] for x in b])
u_true = z_b - z_a
vel_scale = float(u_true.std().clamp_min(1e-6))
tgt = u_true / vel_scale

print(f"VEL_SCALE={vel_scale:.4f}")
print(f"(1) Var(target) = {float(tgt.var()):.4f}  |  E[target^2] = {float((tgt**2).mean()):.4f}")
print(f"    -> predicting ZERO gives MSE = E[target^2] = {float((tgt**2).mean()):.4f}")
print(f"    (convnet's frozen loss was 1.0 -> it IS predicting zero)\n")

# (2) analytic velocity (z_s - z_a)/s across a range of s
print("(2) analytic u_hat = (z_s - z_a)/s   vs   true u = z_b - z_a:")
for s_val in [0.01, 0.1, 0.3, 0.5, 0.9, 0.99]:
    s = torch.full((2,), s_val)
    z_s, u = interpolate(z_a, z_b, s)
    u_hat = (z_s - z_a) / s.view(-1,1,1,1,1)
    err = (u_hat - u_true).abs().mean().item()
    mag = u_hat.abs().mean().item()
    print(f"    s={s_val:4.2f} | |u_hat|~{mag:.4f} | mean|u_hat - u_true|={err:.4e} "
          f"(should be ~0 — u_hat exactly recovers u)")

# (3) is the CONSTANT target consistent, and is direction recoverable at mid-s?
s = torch.full((2,), 0.5)
z_s, u = interpolate(z_a, z_b, s)
d = z_s - z_a
cos_dir = F.cosine_similarity(d.flatten(1), u_true.flatten(1)).mean().item()
print(f"\n(3) at s=0.5: cos(z_s - z_a, u_true) = {cos_dir:.4f} "
      f"(should be 1.0 — (z_s-z_a) is exactly 0.5*u)")

# (4) closed-form: can a per-voxel LINEAR map from (z_s - z_a) predict u at fixed s?
#     if yes, target is trivially learnable; parameterization/optimization is the bug.
print("\n(4) least-squares recoverability at s=0.5:")
X = (z_s - z_a).reshape(-1, 1)            # feature: (z_s - z_a)
Y = u_true.reshape(-1, 1)                 # target
# u = 2*(z_s - z_a) exactly at s=0.5, so slope should be 2.0
sol = torch.linalg.lstsq(X, Y).solution
pred = X @ sol
r2 = 1 - ((pred - Y)**2).sum() / ((Y - Y.mean())**2).sum()
print(f"    fitted slope = {float(sol.item()):.4f} (should be 2.0)  R^2 = {float(r2):.4f}")
print("\nINTERPRETATION:")
print("  If (2) errors are ~0 and (3) cos~1 and (4) R^2~1 and slope~2:")
print("    -> the target IS trivially recoverable; the bug is that the network")
print("       never learns to use s. Likely: s conditioning too weak, OR the")
print("       standard fix -> parameterize the net to predict from (z_s - z_a)")
print("       directly, or predict x1 (=z_b) instead of velocity.")
