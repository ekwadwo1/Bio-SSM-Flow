#!/usr/bin/env python3
"""
probe_outnorm.py — is the final out_norm GroupNorm strangling the velocity head?

FAST (~40s): 2 pairs, a SMALL denoiser (stage_depths=(1,1,1), bottleneck=1) so we
iterate quickly instead of waiting ~280s.

We monkeypatch the denoiser's output path two ways and overfit each:
  (A) BASELINE: out_norm(GroupNorm+SiLU) -> eps_head          (Paper-1 default)
  (B) BYPASS  : eps_head on RAW decoder features * learnable scale (skip GroupNorm)

If (B) overfits (|v|->tgt, loss->~0) and (A) stalls, the GroupNorm-before-head
is the culprit and we patch the real model the same way.

Run:  CUDA_VISIBLE_DEVICES=0 python probe_outnorm.py
"""
from __future__ import annotations
import os, sys, time, types
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s
from models.denoiser import URSSMDenoiser

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"; N = 2; STEPS = 200; LR = 2e-3

ds = CachedPairDataset(CACHE)
b = [ds[i] for i in range(N)]
z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
vel_scale = float((z_b - z_a).std().clamp_min(1e-6))


class ScaledHead(nn.Module):
    """eps_head followed by a learnable scalar — a valid nn.Module child."""
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.head(x) * self.scale


def build(bypass: bool):
    net = URSSMDenoiser(latent_dim=4, d_h=128, stage_depths=(1, 1, 1),
                        bottleneck_depth=1, use_checkpoint=False).to(DEVICE)
    torch.nn.init.zeros_(net.eps_head.weight)
    if net.eps_head.bias is not None: torch.nn.init.zeros_(net.eps_head.bias)
    if bypass:
        # replace out_norm with identity; wrap head in a learnable-scale module
        net.out_norm = nn.Identity().to(DEVICE)
        net.eps_head = ScaledHead(net.eps_head).to(DEVICE)
    return net


def run(bypass: bool, tag: str):
    torch.manual_seed(0)
    net = build(bypass)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    params = list(net.parameters())
    net.train(); t0 = time.time()
    last = None
    for step in range(1, STEPS + 1):
        s = sample_s(N, DEVICE)
        z_s, u = interpolate(z_a, z_b, s)
        target = u / vel_scale
        v, _ = net(z_s, s * 1000.0, z_a)
        loss = F.mse_loss(v.float(), target.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        last = (float(loss), float(v.detach().abs().mean()))
        if step % 50 == 0 or step == 1:
            print(f"  [{tag}] step {step:3d} | loss {last[0]:.5f} | |v|~{last[1]:.4f} (tgt {float(target.abs().mean()):.3f})")
    print(f"  [{tag}] {time.time()-t0:.0f}s  final loss {last[0]:.5f}  |v|~{last[1]:.4f}\n")
    return last[0]


print(f"VEL_SCALE={vel_scale:.4f}  (small denoiser, 2 pairs, {STEPS} steps)\n")
la = run(False, "A: out_norm")
lb = run(True,  "B: bypass  ")
print(f"VERDICT: A(out_norm)={la:.4f}  B(bypass)={lb:.4f}")
print("  B << A  -> GroupNorm-before-head was strangling the velocity. Patch the real model.")
print("  B ~ A   -> not out_norm; deeper architectural issue, try next probe.")
