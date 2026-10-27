#!/usr/bin/env python3
"""
velocity_net_v3.py — VelocityNetV2 with a FUNCTIONAL interval pathway.

Why: in V2, Δt enters only through the SSM branch, gated by tanh(cond_scale) with
cond_scale = 0 at init, and the SSM branch itself is multiplied by refine_scale = 0 at
init. The product of two zero-initialised gates gives the interval pathway ~no gradient.
Measured result on all 5 folds: true vs shuffled vs constant Δt -> identical CRPS.

Changes vs V2 (architecture only; V2 checkpoints still load with V2):
  (1) LocalHeadDt: the per-voxel head receives standardised Δt as an extra input
      channel (cat[z_s, z_a, Δt_map]) -> Δt has a direct, ungated path to the output.
  (2) interval gate on the SSM branch is a plain scalar initialised at `cond_init`
      (no tanh, not zero).
  (3) refine_scale initialised at `refine_init` (not zero).
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from velocity_net_v2 import VelocityNetV2, DT_MEAN, DT_STD, WK_MEAN, WK_STD, T_EMB


class LocalHeadDt(nn.Module):
    """Per-voxel 1x1x1 MLP on cat[z_s, z_a, Δt_map] (2c+1 channels) -> c."""
    def __init__(self, c: int = 4, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2 * c + 1, d, 1), nn.SiLU(),
            nn.Conv3d(d, d, 1), nn.SiLU(),
            nn.Conv3d(d, c, 1),
        )

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor, dtn: torch.Tensor) -> torch.Tensor:
        dmap = dtn.view(-1, 1, 1, 1, 1).expand(-1, 1, *z_s.shape[2:]).to(z_s.dtype)
        return self.net(torch.cat([z_s, z_a, dmap], dim=1))


class VelocityNetV3(VelocityNetV2):
    def __init__(self, latent_dim: int = 4, d_h: int = 128, use_week: bool = True,
                 local_width: int = 256, cond_init: float = 0.5, refine_init: float = 0.1,
                 **denoiser_kwargs) -> None:
        super().__init__(latent_dim=latent_dim, d_h=d_h, use_week=use_week,
                         local_width=local_width, **denoiser_kwargs)
        self.local = LocalHeadDt(latent_dim, local_width)      # replaces V2's LocalHead
        with torch.no_grad():
            self.cond_scale.fill_(cond_init)
            self.refine_scale.fill_(refine_init)

    # V2's __init__ registers self._add_interval as a forward hook; because this method
    # is overridden here, the registered hook is this version (plain scalar, no tanh).
    def _add_interval(self, module, inp, out):
        if self._pending_cond is None:
            return out
        return out + self.cond_scale * self._pending_cond

    def forward(self, z_s, s, z_a, dt):
        dtn = (dt.float() - DT_MEAN) / DT_STD
        v_local = self.local(z_s, z_a, dtn)

        cond = self.dt_embed(dtn * T_EMB)
        if self.use_week and self._week_a is not None:
            cond = cond + self.wk_embed(((self._week_a - WK_MEAN) / WK_STD) * T_EMB)
        self._pending_cond = cond
        refine, logvar = self.denoiser(z_s, s * T_EMB, z_a)
        self._pending_cond = None
        return v_local + self.refine_scale * refine, logvar


if __name__ == "__main__":
    dev = "cuda:0"
    net = VelocityNetV3(latent_dim=4, d_h=128, stage_depths=(1, 1, 1),
                        bottleneck_depth=1, use_checkpoint=False).to(dev).eval()
    net.set_week(None)
    z = torch.randn(2, 4, 32, 32, 32, device=dev)
    za = torch.randn_like(z)
    t = torch.full((2,), 0.5, device=dev)
    with torch.no_grad():
        v1, lv = net(z, t, za, torch.tensor([30., 30.], device=dev))
        v2, _ = net(z, t, za, torch.tensor([300., 300.], device=dev))
    rel = ((v1 - v2).abs().mean() / v1.abs().mean()).item()
    print(f"output {tuple(v1.shape)}, logvar {tuple(lv.shape)}")
    print(f"Δt sensitivity at INIT (30d vs 300d): {rel:.4f}  (must be clearly > 0)")
    assert rel > 1e-3, "Δt has no effect at init — pathway still broken"
    print("velocity_net_v3.py self-test passed.")
