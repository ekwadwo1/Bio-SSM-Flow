#!/usr/bin/env python3
"""
probe_echo.py — is the velocity net IGNORING the interpolant z_s and echoing z_a?

Hypothesis: the denoiser injects z_a (as z_obs) at every block, so the output is
driven by z_a and barely depends on z_s. Then v is ~constant in z_s and the
"forecast" collapses to ~z_a (cosine to z_b stuck at cos(z_a,z_b)).

This probe needs NO training. It measures, on an untrained (and a lightly-trained)
net:
  A) sensitivity of v to z_s      (vary z_s, hold z_a)   -> should be LARGE
  B) sensitivity of v to z_a      (vary z_a, hold z_s)   -> if this dominates A, that's the bug
  C) cos(z_a, z_b) for the 4 pairs -> compare to the stuck forecast cosine (~0.66)

Run:  CUDA_VISIBLE_DEVICES=0 python probe_echo.py
"""
from __future__ import annotations
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from velocity_net import VelocityNet

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"

ds = CachedPairDataset(CACHE)
b = [ds[i] for i in range(4)]
z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
dt  = torch.stack([x["dt"] for x in b]).to(DEVICE)

# (C) what is cos(z_a, z_b)? If ~0.66, the stuck forecast is just echoing z_a.
cos_ab = F.cosine_similarity(z_a.flatten(1), z_b.flatten(1)).mean().item()
print(f"[probe] cos(z_a, z_b) = {cos_ab:.4f}   "
      f"(if the stuck forecast cosine ~0.66 matches this -> model echoes z_a)")

net = VelocityNet(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                  bottleneck_depth=4, use_checkpoint=False).to(DEVICE).eval()
net.set_week(None)

with torch.no_grad():
    s_mid = torch.full((4,), 0.5, device=DEVICE)
    # baseline output at the true interpolant midpoint
    z_s0, _ = (0.5 * z_a + 0.5 * z_b), None
    v_ref, _ = net(z_s0, s_mid, z_a, dt)

    # (A) perturb z_s only (hold z_a): how much does v change?
    z_s_pert = z_s0 + 0.5 * torch.randn_like(z_s0)
    v_dzs, _ = net(z_s_pert, s_mid, z_a, dt)
    sens_zs = (v_dzs - v_ref).abs().mean().item()

    # (B) perturb z_a only (hold z_s): how much does v change?
    z_a_pert = z_a + 0.5 * torch.randn_like(z_a)
    v_dza, _ = net(z_s0, s_mid, z_a_pert, dt)
    sens_za = (v_dza - v_ref).abs().mean().item()

print(f"[probe] sensitivity of v to z_s (main input) = {sens_zs:.5f}")
print(f"[probe] sensitivity of v to z_a (condition)  = {sens_za:.5f}")
ratio = sens_za / max(sens_zs, 1e-8)
print(f"[probe] z_a/z_s sensitivity ratio = {ratio:.2f}")
print("[probe] interpretation:")
print("   ratio >> 1  -> output dominated by z_a conditioning (ECHO bug confirmed)")
print("   ratio ~ 1   -> balanced; stall is elsewhere")
print("   sens_zs ~ 0 -> net ignores its main input entirely (severe echo)")
