#!/usr/bin/env python3
"""
probe_fix.py — confirm the CONTROLLED velocity-head fix (fast, 2 pairs, ~90s).

Diagnosis established:
  - out_norm's GroupNorm+SiLU strangles the velocity head (arm A frozen).
  - Removing it entirely lets the head express scale but explodes then recovers
    (arm B: |v| hit 38, settled to 0.30). We need a CONTROLLED version.

This probe replaces the head path with the canonical flow recipe:
    out_norm(GroupNorm, NO SiLU) -> eps_head(zero-init) -> * learnable out_scale
and trains with grad-clip + no weight decay. Expect: loss -> near 0, |v| -> ~tgt,
NO explosion. If so, this is the exact patch for the real denoiser.

Run:  CUDA_VISIBLE_DEVICES=0 python probe_fix.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s
from models.denoiser import URSSMDenoiser

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"; N = 2; STEPS = 300; LR = 1e-3


class VelHead(nn.Module):
    """GroupNorm (scale-preserving, no SiLU) -> zero-init conv -> learnable scale."""
    def __init__(self, d_h: int, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, d_h), d_h)     # stabilize, but no SiLU rectification
        self.conv = nn.Conv3d(d_h, c, 1)
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.conv(self.norm(x)) * self.scale


def main():
    ds = CachedPairDataset(CACHE)
    b = [ds[i] for i in range(N)]
    z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
    z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
    vel_scale = float((z_b - z_a).std().clamp_min(1e-6))
    print(f"VEL_SCALE={vel_scale:.4f}")

    torch.manual_seed(0)
    net = URSSMDenoiser(latent_dim=4, d_h=128, stage_depths=(1, 1, 1),
                        bottleneck_depth=1, use_checkpoint=False).to(DEVICE)
    # swap the strangling out_norm+eps_head for the controlled velocity head
    net.out_norm = nn.Identity().to(DEVICE)
    net.eps_head = VelHead(128, 4).to(DEVICE)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    net.train(); t0 = time.time()
    for step in range(1, STEPS + 1):
        s = sample_s(N, DEVICE)
        z_s, u = interpolate(z_a, z_b, s)
        target = u / vel_scale
        v, _ = net(z_s, s * 1000.0, z_a)
        loss = F.mse_loss(v.float(), target.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step == 1 or step % 30 == 0:
            print(f"  step {step:3d} | loss {float(loss):.5f} | "
                  f"|v|~{float(v.detach().abs().mean()):.4f} (tgt {float(target.abs().mean()):.3f}) "
                  f"| out_scale {float(net.eps_head.scale):.3f}")

    net.eval()
    with torch.no_grad():
        z = z_a.clone(); M = 8
        for m in range(M):
            v, _ = net(z, torch.full((N,), m/M*1000., device=DEVICE), z_a)
            z_mid = z + (0.5/M)*(v*vel_scale)
            v2, _ = net(z_mid, torch.full((N,), (m+0.5)/M*1000., device=DEVICE), z_a)
            z = z + (1.0/M)*(v2*vel_scale)
    cos = F.cosine_similarity(z.flatten(1), z_b.flatten(1)).mean().item()
    base = F.mse_loss(z_a, z_b).item(); mse = F.mse_loss(z, z_b).item()
    print(f"\n{time.time()-t0:.0f}s | forecast MSE {mse:.5f} (base {base:.5f}) | cosine {cos:.4f}")
    print(f"VERDICT: {'PASS -> this is the patch' if (cos>0.9 and mse<0.5*base) else 'still not clean'}")


if __name__ == "__main__":
    main()
