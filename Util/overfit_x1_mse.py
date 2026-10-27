#!/usr/bin/env python3
"""
overfit_x1_mse.py — x1-prediction with PURE MSE (no variance head).

Last run proved the heteroscedastic loss cheats: loss->-1.5 while x1_mse stayed
0.408 (model shrinks log-sigma^2 instead of predicting z_b). Remove that escape
hatch entirely: train the endpoint prediction with plain MSE. The variance head
is trained LATER, after x1 is good (warmup) — not now.

Also bumps LR and lets it run longer, since the MSE-only runs learned but slowly.
Fast rig: 2 pairs, small denoiser.
Run:  CUDA_VISIBLE_DEVICES=0 python overfit_x1_mse.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s, forecast
from velocity_net import VelocityNet

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"; N = 2; STEPS = 600; LR = 3e-3

def main():
    torch.manual_seed(0)
    ds = CachedPairDataset(CACHE)
    b = [ds[i] for i in range(N)]
    z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
    z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
    dt  = torch.stack([x["dt"] for x in b]).to(DEVICE)
    print(f"[x1-mse] {N} pairs | z_b std {z_b.std():.3f} | mean|z_b-z_a| {(z_b-z_a).abs().mean():.3f}")

    net = VelocityNet(latent_dim=4, d_h=128, stage_depths=(1,1,1),
                      bottleneck_depth=1, use_checkpoint=False).to(DEVICE)
    net.set_week(None)
    # NOTE: do NOT zero-init here — x1 target is z_b (~0.6 std), a nonzero endpoint.
    # Let the head init normally so it can express a nonzero endpoint from step 1.
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)

    def x1_fn(z_s, s, z_a_, dt_):
        return net(z_s, s, z_a_, dt_)

    net.train(); t0 = time.time()
    for step in range(1, STEPS+1):
        s = sample_s(N, DEVICE)
        z_s, _ = interpolate(z_a, z_b, s)
        x1, _ = net(z_s, s, z_a, dt)                 # head output = x1_hat (predict z_b)
        loss = F.mse_loss(x1.float(), z_b.float())    # PURE MSE on endpoint
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d} | mse {float(loss):.5f} | "
                  f"|x1|~{float(x1.detach().abs().mean()):.3f} (z_b~{float(z_b.abs().mean()):.3f})")

    net.eval()
    zb_hat, _ = forecast(x1_fn, z_a, dt, steps=8)
    mse = F.mse_loss(zb_hat.float(), z_b).item(); base = F.mse_loss(z_a, z_b).item()
    cos = F.cosine_similarity(zb_hat.flatten(1), z_b.flatten(1)).mean().item()
    print(f"\n[x1-mse] {time.time()-t0:.0f}s | forecast MSE {mse:.5f} (base {base:.5f}) | cosine {cos:.4f}")
    ok = cos > 0.9 and mse < 0.5*base
    print(f"[x1-mse] VERDICT: {'PASS — endpoint prediction learns (variance head was the saboteur)' if ok else 'STILL STUCK — deeper'}")

if __name__ == "__main__":
    main()
