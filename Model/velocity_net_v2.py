#!/usr/bin/env python3
"""
src/models/velocity_net_v2.py — VelocityNet with a LOCAL RESIDUAL path (Option A).

WHY (proven by localize_stage.py):
  A per-voxel MLP on cat[z_s, z_a] learns endpoint prediction in ~5 steps on the
  SAME real latents where the SSM-U-Net denoiser DID NOT REACH mse<0.05 in 400.
  The SSM's global 1D scans + U-Net downsampling destroy the local, high-frequency
  mapping this task needs. Fix: give the network the easy local map for FREE via a
  per-voxel residual branch, and let the SSM denoiser add global refinement on top.

Design:
    x1_hat = local_head( cat[z_s, z_a] )            # per-voxel 1x1x1 MLP (learns fast)
             + refine_scale * denoiser(z_s, s; z_a) # SSM global correction (starts ~0)

  - local_head is a small 1x1x1 conv MLP: exactly the structure that learned in 5 steps.
  - refine_scale is a learnable scalar init at 0, so at start the net == the local MLP
    (guaranteed fast learning), and the SSM's contribution grows only if it helps.
  - keeps the SSM in the loop -> preserves the state-space thesis identity, but the
    SSM is now a RESIDUAL refiner, not the sole predictor.
  - dt / week_a interval conditioning still flow into the SSM branch via the hook.

Head predicts x1 (=z_b), matching flow.py's x1-prediction. Returns (x1_hat, logvar).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import sys, os
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
from models.denoiser import URSSMDenoiser
from models.ssm_block import SinusoidalTimestepEmb

T_EMB = 1000.0
DT_MEAN, DT_STD = 90.0, 90.0
WK_MEAN, WK_STD = 60.0, 60.0


class LocalHead(nn.Module):
    """Per-voxel 1x1x1 MLP: cat[z_s, z_a] (2c) -> x1 (c). The fast learner."""
    def __init__(self, c: int = 4, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2 * c, d, 1), nn.SiLU(),
            nn.Conv3d(d, d, 1), nn.SiLU(),
            nn.Conv3d(d, c, 1),
        )
    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_s, z_a], dim=1))


class VelocityNetV2(nn.Module):
    """
    x1_hat = LocalHead(z_s, z_a) + refine_scale * SSM_denoiser(z_s, s; z_a)
    Also returns the SSM's heteroscedastic log-variance head as forecast uncertainty.
    """
    def __init__(self, latent_dim: int = 4, d_h: int = 128,
                 use_week: bool = True, local_width: int = 256,
                 **denoiser_kwargs) -> None:
        super().__init__()
        self.local = LocalHead(latent_dim, local_width)
        self.denoiser = URSSMDenoiser(latent_dim=latent_dim, d_h=d_h, **denoiser_kwargs)
        # SSM refinement starts OFF (scale 0) -> net begins as the fast local MLP
        self.refine_scale = nn.Parameter(torch.zeros(1))

        # interval conditioning into the SSM branch (same hook mechanism as v1)
        self.use_week = use_week
        base_dim = 4 * d_h
        self.dt_embed = SinusoidalTimestepEmb(base_dim)
        self.wk_embed = SinusoidalTimestepEmb(base_dim) if use_week else None
        self.cond_scale = nn.Parameter(torch.zeros(1))
        self._pending_cond: Optional[torch.Tensor] = None
        self._week_a: Optional[torch.Tensor] = None
        self.denoiser.time_embed.base.register_forward_hook(self._add_interval)

    def _add_interval(self, module, inp, out):
        if self._pending_cond is None:
            return out
        return out + torch.tanh(self.cond_scale) * self._pending_cond

    def set_week(self, week_a: Optional[torch.Tensor]):
        self._week_a = week_a

    def forward(self, z_s: torch.Tensor, s: torch.Tensor,
                z_a: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # local per-voxel prediction (the fast path)
        x1_local = self.local(z_s, z_a)

        # interval embedding -> hook -> SSM branch
        dtn = (dt - DT_MEAN) / DT_STD
        cond = self.dt_embed(dtn * T_EMB)
        if self.use_week and self._week_a is not None:
            wkn = (self._week_a - WK_MEAN) / WK_STD
            cond = cond + self.wk_embed(wkn * T_EMB)
        self._pending_cond = cond
        refine, logvar = self.denoiser(z_s, s * T_EMB, z_a)   # s->s*1000 rescale
        self._pending_cond = None

        x1_hat = x1_local + self.refine_scale * refine
        return x1_hat, logvar


if __name__ == "__main__":
    import time
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cached_pair_dataset import CachedPairDataset
    from flow import interpolate, sample_s, forecast

    DEV = "cuda:0"; N = 2; STEPS = 300; LR = 1e-3
    ds = CachedPairDataset("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere")
    b = [ds[i] for i in range(N)]
    z_a = torch.stack([x["z_a"] for x in b]).to(DEV)
    z_b = torch.stack([x["z_b"] for x in b]).to(DEV)
    dt  = torch.stack([x["dt"] for x in b]).to(DEV)

    torch.manual_seed(0)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(1, 1, 1),
                        bottleneck_depth=1, use_checkpoint=False).to(DEV)
    net.set_week(None)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)

    def x1_fn(z, s, za, dtt):
        return net(z, s, za, dtt)

    net.train(); t0 = time.time(); hit = None
    for step in range(1, STEPS + 1):
        s = sample_s(N, DEV)
        z_s, _ = interpolate(z_a, z_b, s)
        x1, _ = net(z_s, s, z_a, dt)
        loss = F.mse_loss(x1.float(), z_b.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if hit is None and float(loss) < 0.05: hit = step
        if step == 1 or step % 25 == 0:
            print(f"  step {step:3d} | mse {float(loss):.5f} | "
                  f"|x1|~{float(x1.detach().abs().mean()):.3f} (z_b~{float(z_b.abs().mean()):.3f}) | "
                  f"refine_scale {float(net.refine_scale):.3f}")

    net.eval()
    zb_hat, _ = forecast(x1_fn, z_a, dt, steps=8)
    mse = F.mse_loss(zb_hat.float(), z_b).item(); base = F.mse_loss(z_a, z_b).item()
    cos = F.cosine_similarity(zb_hat.flatten(1), z_b.flatten(1)).mean().item()
    print(f"\n{time.time()-t0:.0f}s | reached mse<0.05 at step: {hit if hit else 'NO'}")
    print(f"forecast MSE {mse:.5f} (base {base:.5f}) | cosine {cos:.4f}")
    ok = cos > 0.9 and mse < 0.5 * base
    print(f"VERDICT: {'PASS — engine learns (local residual fixed it)' if ok else 'STILL STUCK'}")
