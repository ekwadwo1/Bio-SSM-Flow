#!/usr/bin/env python3
"""
overfit_test2.py — SELF-CONTAINED overfit sanity test (no edits to flow.py needed).

Fixes baked in vs v1:
  (1) velocity-target scaling: u is standardized by its own std (computed from the
      4 pairs) so the regression target is ~unit magnitude. The net was previously
      stuck predicting ~0 because the target (mean|u|~0.23) is tiny and the final
      GroupNorm flattens output scale -> "regress to zero" plateau at vel_mse~E[u^2].
  (2) zero-init eps_head: the velocity head starts at exactly 0 and builds up
      (standard flow/diffusion practice), avoiding the overshoot->collapse dynamic.

Only imports the STABLE pieces (interpolate, sample_s) from flow.py. Everything
else is local, so there is no cross-file drift.

Run:  CUDA_VISIBLE_DEVICES=0 python overfit_test2.py
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s          # stable, already self-tested
from velocity_net import VelocityNet

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"
N_PAIRS = 4
STEPS = 250
LR = 1e-3


def main():
    torch.manual_seed(0)
    ds = CachedPairDataset(CACHE)
    batch = [ds[i] for i in range(N_PAIRS)]
    z_a = torch.stack([b["z_a"] for b in batch]).to(DEVICE)
    z_b = torch.stack([b["z_b"] for b in batch]).to(DEVICE)
    dt  = torch.stack([b["dt"] for b in batch]).to(DEVICE)

    # (1) calibrate target scale from the data: std of the true velocity
    u_all = (z_b - z_a)
    vel_scale = float(u_all.std().clamp_min(1e-6))
    print(f"[overfit2] {N_PAIRS} pairs | dt(days) {[round(float(x)) for x in dt]}")
    print(f"[overfit2] VEL_SCALE (std of z_b - z_a) = {vel_scale:.4f}")

    net = VelocityNet(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                      bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    # (2) zero-init the velocity head
    torch.nn.init.zeros_(net.denoiser.eps_head.weight)
    if net.denoiser.eps_head.bias is not None:
        torch.nn.init.zeros_(net.denoiser.eps_head.bias)
    net.set_week(None)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)

    net.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        s = sample_s(N_PAIRS, DEVICE)
        z_s, u = interpolate(z_a, z_b, s)
        target = u / vel_scale                              # ~unit-magnitude
        v, _ = net(z_s, s, z_a, dt)                          # fp32
        loss = F.mse_loss(v.float(), target.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gh = net.denoiser.eps_head.weight.grad
        ghn = float(gh.norm()) if gh is not None else -1.0
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            print(f"  step {step:4d} | loss {float(loss):8.5f} | "
                  f"|v|~{float(v.detach().abs().mean()):.4f} "
                  f"(tgt {float(target.abs().mean()):.4f}) | head.grad {ghn:.2e}")

    # ── forecast: integrate learned velocity (rescaled) from z_a ──
    net.eval()
    with torch.no_grad():
        z = z_a.clone()
        M = 8
        for m in range(M):
            s_m = torch.full((N_PAIRS,), m / M, device=DEVICE)
            v, _ = net(z, s_m, z_a, dt)
            s_mid = torch.full((N_PAIRS,), (m + 0.5) / M, device=DEVICE)
            z_mid = z + (0.5 / M) * (v * vel_scale)
            v_mid, _ = net(z_mid, s_mid, z_a, dt)
            z = z + (1.0 / M) * (v_mid * vel_scale)
        zb_hat = z
    mse = F.mse_loss(zb_hat, z_b).item()
    base = F.mse_loss(z_a, z_b).item()
    cos = F.cosine_similarity(zb_hat.flatten(1), z_b.flatten(1)).mean().item()
    print(f"\n[overfit2] {time.time()-t0:.1f}s")
    print(f"[overfit2] forecast MSE {mse:.5f}  (baseline z_a->z_b {base:.5f})")
    print(f"[overfit2] forecast cosine {cos:.4f}  (target ~1.0)")
    ok = cos > 0.9 and mse < 0.5 * base
    print(f"[overfit2] VERDICT: {'PASS — engine learns the transport' if ok else 'STILL STUCK'}")


if __name__ == "__main__":
    main()
