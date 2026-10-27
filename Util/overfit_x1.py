#!/usr/bin/env python3
"""
overfit_x1.py — overfit test under X1-PREDICTION (predict endpoint z_b).

The decisive rerun. v-prediction collapsed to zero (proven: loss==Var(target)).
x1-prediction predicts z_b directly, which is a fixed per-pair target and should
be trivially learnable. If THIS overfits (x1_mse->~0, forecast cosine->~1), the
parameterization was the whole bug and the engine works.

Fast config: 2 pairs, small denoiser, ~90s.
Run:  CUDA_VISIBLE_DEVICES=0 python overfit_x1.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s, HeteroscedasticX1Loss, forecast
from velocity_net import VelocityNet

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"; N = 2; STEPS = 300; LR = 1e-3


def main():
    torch.manual_seed(0)
    ds = CachedPairDataset(CACHE)
    b = [ds[i] for i in range(N)]
    z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
    z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
    dt  = torch.stack([x["dt"] for x in b]).to(DEVICE)
    print(f"[x1] {N} pairs | mean|z_b-z_a|={(z_b-z_a).abs().mean():.3f}")

    net = VelocityNet(latent_dim=4, d_h=128, stage_depths=(1, 1, 1),
                      bottleneck_depth=1, use_checkpoint=False).to(DEVICE)
    net.set_week(None)
    # the head now predicts z_b (an O(0.6)-std target). zero-init so it starts at 0
    # and builds toward z_b; keep it (helps but is not load-bearing anymore).
    torch.nn.init.zeros_(net.denoiser.eps_head.weight)
    if net.denoiser.eps_head.bias is not None:
        torch.nn.init.zeros_(net.denoiser.eps_head.bias)

    loss_fn = HeteroscedasticX1Loss()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)

    # x1_fn adapter: VelocityNet's head output is now interpreted as x1_hat (=z_b).
    def x1_fn(z_s, s, z_a_, dt_):
        x1, logvar = net(z_s, s, z_a_, dt_)
        return x1, logvar

    net.train(); t0 = time.time()
    for step in range(1, STEPS + 1):
        s = sample_s(N, DEVICE)
        z_s, _ = interpolate(z_a, z_b, s)
        x1, logvar = net(z_s, s, z_a, dt)
        loss, _, logs = loss_fn(z_b, x1, logvar)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step == 1 or step % 30 == 0:
            print(f"  step {step:3d} | loss {float(loss):8.4f} | "
                  f"x1_mse {logs['x1_mse']:.5f} | |x1|~{float(x1.detach().abs().mean()):.3f} "
                  f"(z_b~{float(z_b.abs().mean()):.3f})")

    net.eval()
    zb_hat, _ = forecast(x1_fn, z_a, dt, steps=8)
    zb_hat = zb_hat.float()
    mse = F.mse_loss(zb_hat, z_b).item(); base = F.mse_loss(z_a, z_b).item()
    cos = F.cosine_similarity(zb_hat.flatten(1), z_b.flatten(1)).mean().item()
    print(f"\n[x1] {time.time()-t0:.0f}s | forecast MSE {mse:.5f} (base {base:.5f}) | cosine {cos:.4f}")
    ok = cos > 0.9 and mse < 0.5 * base
    print(f"[x1] VERDICT: {'PASS — engine learns (x1-prediction fixed it)' if ok else 'STILL STUCK'}")


if __name__ == "__main__":
    main()
