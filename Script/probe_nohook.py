#!/usr/bin/env python3
"""
probe_nohook.py — does the forward-HOOK in VelocityNet freeze learning?

We bypass VelocityNet entirely and drive the raw URSSMDenoiser directly:
  - s in [0,1] rescaled to s*1000 (the fix VelocityNet also does)
  - z_a passed as z_obs conditioning
  - NO forward hook, NO interval embedding, NO cond_scale gate
  - zero-init eps_head, data-calibrated target scaling, fp32, weight_decay=0

If THIS overfits 4 pairs (|v|->tgt, cosine->~1), the bug is the hook/interval
plumbing in VelocityNet, and we rewrite interval injection cleanly.
If this ALSO stalls, the denoiser body itself resists this task -> architectural.

Run:  CUDA_VISIBLE_DEVICES=0 python probe_nohook.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s
from models.denoiser import URSSMDenoiser

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"; N = 4; STEPS = 250; LR = 1e-3

ds = CachedPairDataset(CACHE)
b = [ds[i] for i in range(N)]
z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
vel_scale = float((z_b - z_a).std().clamp_min(1e-6))
print(f"[nohook] VEL_SCALE={vel_scale:.4f}")

net = URSSMDenoiser(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                    bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
torch.nn.init.zeros_(net.eps_head.weight)
if net.eps_head.bias is not None: torch.nn.init.zeros_(net.eps_head.bias)

opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)  # NO decay
net.train(); t0 = time.time()
for step in range(1, STEPS + 1):
    s = sample_s(N, DEVICE)
    z_s, u = interpolate(z_a, z_b, s)
    target = u / vel_scale
    v, _ = net(z_s, s * 1000.0, z_a)          # DIRECT call, s->s*1000, z_a as z_obs
    loss = F.mse_loss(v.float(), target.float())
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if step == 1 or step % 25 == 0:
        print(f"  step {step:4d} | loss {float(loss):.5f} | "
              f"|v|~{float(v.detach().abs().mean()):.4f} (tgt {float(target.abs().mean()):.4f})")

net.eval()
with torch.no_grad():
    z = z_a.clone(); M = 8
    for m in range(M):
        v, _ = net(z, torch.full((N,), m/M*1000., device=DEVICE), z_a)
        z_mid = z + (0.5/M) * (v * vel_scale)
        v2, _ = net(z_mid, torch.full((N,), (m+0.5)/M*1000., device=DEVICE), z_a)
        z = z + (1.0/M) * (v2 * vel_scale)
cos = F.cosine_similarity(z.flatten(1), z_b.flatten(1)).mean().item()
base = F.mse_loss(z_a, z_b).item(); mse = F.mse_loss(z, z_b).item()
print(f"\n[nohook] {time.time()-t0:.1f}s | forecast MSE {mse:.5f} (base {base:.5f}) | cosine {cos:.4f}")
print(f"[nohook] VERDICT: {'PASS -> the HOOK was the bug' if (cos>0.9 and mse<0.5*base) else 'STILL STUCK -> denoiser body resists this task'}")
