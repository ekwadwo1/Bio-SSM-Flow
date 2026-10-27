#!/usr/bin/env python3
"""
src/physics/fisher_kpp.py — Fisher-KPP reaction-diffusion residual (headline novelty).

Constrains the forecast so the density change baseline->follow-up is consistent with
proliferation-invasion physics:

    dc/dt = div(D grad c) + rho * c * (1 - c)                (Fisher-KPP)

Discretized between two timepoints over interval dt (days), midpoint (Crank-Nicolson):
    c_bar = 0.5 (c_a + c_b)
    R = (c_b - c_a)/dt  -  div(D grad c_bar)  -  rho * c_bar * (1 - c_bar)
    L_bio = mean_{brain} R^2

Design (validated choices):
  - IMAGE space (128^3), operators from physics_operators (analytic-tested).
  - isotropic scalar D to start (anisotropy = later extension via a D-field).
  - (D, rho) can be fixed constants (GT-prototyping) or patient-specific (param head).
  - UNIT SCALING is explicit and logged: dc/dt is O(1/dt)~1e-2/day (tiny), so we
    report the three terms separately and expose a residual normalizer so L_bio is
    O(1) and commensurate with the flow loss (else lambda_bio is untunable).

Literature scales (Swanson/Tracqui): D_w ~ 0.01-0.1 mm^2/day, rho ~ 0.01-0.03 /day,
invasion speed v = 2 sqrt(rho D) ~ mm/month. Voxel h ~ 1.42 mm (LUMIERE 182->128).
"""
from __future__ import annotations
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from physics_operators import div_D_grad
except ImportError:
    from .physics_operators import div_D_grad

Number = Union[float, torch.Tensor]


def fisher_kpp_residual(c_a: torch.Tensor, c_b: torch.Tensor, dt_days: torch.Tensor,
                        D: Number, rho: Number, h: float = 1.42
                        ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    c_a, c_b : [B,1,H,W,D] density at baseline / follow-up
    dt_days  : [B] interval (days)
    D        : scalar or [B,1,1,1,1] diffusivity (mm^2/day)
    rho      : scalar or [B,1,1,1,1] proliferation (1/day)
    returns per-voxel residual [B,1,H,W,D] and a term-magnitude log.
    """
    dt = dt_days.view(-1, 1, 1, 1, 1).clamp_min(1.0)      # days, guard dt>=1
    c_bar = 0.5 * (c_a + c_b)
    dcdt = (c_b - c_a) / dt                                # [B,1,H,W,D]
    diff = div_D_grad(c_bar, D, h)                         # div(D grad c_bar)
    reac = rho * c_bar * (1.0 - c_bar)                     # logistic growth
    R = dcdt - diff - reac
    log = {"dcdt_mag": float(dcdt.abs().mean()),
           "diff_mag": float(diff.abs().mean()),
           "reac_mag": float(reac.abs().mean()),
           "resid_mag": float(R.abs().mean())}
    return R, log


class FisherKPPLoss(nn.Module):
    """
    L_bio = mean_{brain} (R / scale)^2 , with `scale` normalizing the residual to O(1)
    so lambda_bio is tunable. brain_mask restricts to intracranial voxels.
    """
    def __init__(self, h: float = 1.42, residual_scale: float = 1e-2) -> None:
        super().__init__()
        self.h = h
        self.scale = residual_scale        # ~ magnitude of dc/dt; tune from the logs

    def forward(self, c_a, c_b, dt_days, D, rho, brain_mask=None):
        R, log = fisher_kpp_residual(c_a, c_b, dt_days, D, rho, self.h)
        Rn = R / self.scale
        if brain_mask is not None:
            m = brain_mask.view(R.shape).float()
            L = (Rn.pow(2) * m).sum() / m.sum().clamp_min(1.0)
        else:
            L = Rn.pow(2).mean()
        log["L_bio"] = float(L.detach())
        return L, log


class GrowthParamHead(nn.Module):
    """
    Patient-specific (D, rho) from pooled baseline+follow-up latents.
    Positivity via softplus; ranges biased to literature scales at init.
    Use later (couples to VelocityNetV2). For GT-prototyping, use fixed constants.
    """
    def __init__(self, latent_dim: int = 4, hidden: int = 64,
                 D_init: float = 0.02, rho_init: float = 0.02) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))
        # bias init so softplus(output) ~ literature scale
        import math
        self.net[-1].bias.data = torch.tensor(
            [math.log(math.exp(D_init) - 1), math.log(math.exp(rho_init) - 1)])
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        f = torch.cat([z_a.mean((2, 3, 4)), z_b.mean((2, 3, 4))], dim=1)  # [B,2c]
        out = F.softplus(self.net(f))                                     # [B,2] >0
        D = out[:, 0].view(-1, 1, 1, 1, 1)
        rho = out[:, 1].view(-1, 1, 1, 1, 1)
        return D, rho


if __name__ == "__main__":
    # Prototype on SYNTHETIC densities that DO obey Fisher-KPP: evolve c_a by one
    # explicit step, then the residual for that (D,rho) must be ~0; wrong (D,rho) large.
    from physics_operators import div_D_grad
    torch.manual_seed(0)
    H = 32; h = 1.42
    c_a = torch.zeros(1, 1, H, H, H)
    cc = H // 2
    c_a[0, 0, cc-3:cc+3, cc-3:cc+3, cc-3:cc+3] = 0.8         # a tumor seed
    D_true, rho_true, dt = 0.02, 0.02, 30.0
    # one explicit Fisher-KPP step to make c_b consistent with (D_true,rho_true)
    dc = div_D_grad(c_a, D_true, h) + rho_true * c_a * (1 - c_a)
    c_b = (c_a + dt * dc).clamp(0, 1)
    dt_t = torch.tensor([dt])

    R_true, log_t = fisher_kpp_residual(c_a, c_b, dt_t, D_true, rho_true, h)
    R_wrong, log_w = fisher_kpp_residual(c_a, c_b, dt_t, D_true*5, rho_true*5, h)
    print("term magnitudes (correct D,rho):", {k: round(v,5) for k,v in log_t.items()})
    print(f"residual |R| correct (D,rho) = {log_t['resid_mag']:.2e}")
    print(f"residual |R| WRONG   (D,rho) = {log_w['resid_mag']:.2e}  (should be >> correct)")
    assert log_t['resid_mag'] < 0.3 * log_w['resid_mag'], "residual not sensitive to (D,rho)!"

    loss_fn = FisherKPPLoss(h=h)
    L, log = loss_fn(c_a, c_b, dt_t, D_true, rho_true)
    print(f"L_bio (correct) = {L.item():.4f}  terms: {[(k,round(v,4)) for k,v in log.items()]}")

    # differentiability: gradient flows to density
    c_b2 = c_b.clone().requires_grad_(True)
    L2, _ = loss_fn(c_a, c_b2, dt_t, D_true, rho_true)
    L2.backward()
    assert c_b2.grad is not None and c_b2.grad.abs().sum() > 0
    print("differentiable: gradient flows to c_b  OK")
    print("fisher_kpp.py self-test passed.")
