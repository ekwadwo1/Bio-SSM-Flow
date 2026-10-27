#!/usr/bin/env python3
"""
src/models/velocity_net.py — flow-matching velocity field over the Paper-1 SSM denoiser.

WRAP, DON'T EDIT. This module CONTAINS an unmodified URSSMDenoiser and adapts it
into a velocity field v_theta(z_s, s; z_a, dt) for conditional flow matching:

  1. z_a (baseline) plays the role of the denoiser's z_obs conditioning.
  2. flow pseudo-time s in [0,1] is rescaled to s*T_EMB (=1000) BEFORE the
     denoiser's SinusoidalTimestepEmb, whose frequencies are calibrated for
     the [0,1000] integer range. Feeding raw s in [0,1] makes it time-blind.
  3. Δt (and optionally week_a) are embedded the SAME way and ADDED to the
     denoiser's timestep embedding — riding the existing per-token additive
     injection (AxisCycledSSMBlock Step 1: H + t_emb + z_obs_proj) without
     touching any SSM block or the URA gate.
  4. the denoiser's eps head output is reinterpreted as the VELOCITY.
     v_theta (log-variance head) carries over unchanged as forecast uncertainty.

Interval normalization: Δt (days) and week_a (weeks) are standardized to O(1)
before embedding, so their sinusoidal arguments are well-scaled. Stats are
config, not learned, so they're identical train/val/test.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

# make the Paper-1 package importable (models/denoiser.py etc.)
import sys, os
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
from models.denoiser import URSSMDenoiser
from models.ssm_block import SinusoidalTimestepEmb

T_EMB = 1000.0          # the integer range the denoiser's sinusoidal embedding expects

# Interval standardization (from LUMIERE: dt median ~84d, range 7..343; weeks 0..~500)
DT_MEAN, DT_STD = 90.0, 90.0          # days  -> ~O(1)
WK_MEAN, WK_STD = 60.0, 60.0          # weeks -> ~O(1)


class VelocityNet(nn.Module):
    """
    v_theta(z_s, s; z_a, dt) -> (velocity[B,c,h,w,d], logvar[B,N]).

    Contains an unmodified URSSMDenoiser. Adds an interval-embedding MLP that is
    added to the denoiser's internal timestep embedding via a forward hook on its
    `time_embed` module, so we never edit Paper-1 code.
    """
    def __init__(self, latent_dim: int = 4, d_h: int = 128,
                 use_week: bool = True, **denoiser_kwargs) -> None:
        super().__init__()
        self.denoiser = URSSMDenoiser(latent_dim=latent_dim, d_h=d_h,
                                      **denoiser_kwargs)
        self.use_week = use_week
        base_dim = 4 * d_h                       # matches TimestepEmbedder.base output dim

        # interval embedders -> produce a vector in the SAME base space, which we
        # ADD to the denoiser's base timestep embedding before its per-stage projs.
        self.dt_embed = SinusoidalTimestepEmb(base_dim)
        self.wk_embed = SinusoidalTimestepEmb(base_dim) if use_week else None
        # small gate so interval conditioning starts near-zero and grows (stability)
        self.cond_scale = nn.Parameter(torch.zeros(1))

        self._pending_cond: Optional[torch.Tensor] = None
        # hook the denoiser's base sinusoidal embedding to add our interval vector
        self.denoiser.time_embed.base.register_forward_hook(self._add_interval)

    def _add_interval(self, module, inp, out):
        # out: [B, base_dim] = base timestep embedding inside TimestepEmbedder
        if self._pending_cond is None:
            return out
        return out + torch.tanh(self.cond_scale) * self._pending_cond

    def forward(self, z_s: torch.Tensor, s: torch.Tensor,
                z_a: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_s : [B,c,h,w,d]  interpolated latent
        s   : [B]          flow pseudo-time in [0,1]
        z_a : [B,c,h,w,d]  baseline latent (conditioning; == denoiser z_obs)
        dt  : [B]          interval in DAYS
        """
        B = z_s.shape[0]
        # standardize + embed interval covariates into the base space
        dtn = (dt - DT_MEAN) / DT_STD
        cond = self.dt_embed(dtn * T_EMB)            # reuse sinusoidal scale
        # (week_a is optional; passed via attribute to avoid changing the signature)
        if self.use_week and getattr(self, "_week_a", None) is not None:
            wkn = (self._week_a - WK_MEAN) / WK_STD
            cond = cond + self.wk_embed(wkn * T_EMB)
        self._pending_cond = cond                    # consumed by the hook

        # rescale flow-time into the [0,1000] range the embedding expects
        t_scaled = (s * T_EMB)

        # z_a is the conditioning latent (denoiser's z_obs slot); eps-head -> velocity
        velocity, logvar = self.denoiser(z_s, t_scaled, z_a)
        self._pending_cond = None
        return velocity, logvar

    def set_week(self, week_a: Optional[torch.Tensor]):
        """Optionally supply absolute treatment-time (weeks) for conditioning."""
        self._week_a = week_a


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("needs CUDA for mamba — skipping"); raise SystemExit
    torch.manual_seed(0)
    B, c, h = 2, 4, 32
    net = VelocityNet(latent_dim=c, d_h=128, stage_depths=(2, 2, 4),
                      bottleneck_depth=4, use_checkpoint=False).to(dev)
    z_s = torch.randn(B, c, h, h, h, device=dev)
    z_a = torch.randn(B, c, h, h, h, device=dev)

    # KEY TEST: velocity must DEPEND on s (proves the s->s*1000 fix works;
    # a time-blind net would give identical output for different s).
    with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.bfloat16):
        net.set_week(None)
        v0, _ = net(z_s, torch.zeros(B, device=dev), z_a, torch.full((B,), 84., device=dev))
        v1, _ = net(z_s, torch.ones(B, device=dev),  z_a, torch.full((B,), 84., device=dev))
        d_s = (v0 - v1).abs().mean().item()
        # and must DEPEND on dt (proves interval conditioning is wired)
        va, _ = net(z_s, torch.full((B,), .5, device=dev), z_a, torch.full((B,), 30.,  device=dev))
        vb, _ = net(z_s, torch.full((B,), .5, device=dev), z_a, torch.full((B,), 300., device=dev))
        d_dt = (va - vb).abs().mean().item()
    print(f"velocity shape {tuple(v0.shape)}, logvar OK")
    print(f"sensitivity to s : mean|v(s=0)-v(s=1)| = {d_s:.4e}  (must be > 0 -> not time-blind)")
    print(f"sensitivity to dt: mean|v(dt=30)-v(dt=300)| = {d_dt:.4e}  (>0 once cond_scale trains; ~0 at init is OK)")
    assert d_s > 1e-5, "TIME-BLIND: s has no effect — the s->s*1000 rescale failed."
    print("velocity_net.py forward + s-sensitivity checks passed.")
