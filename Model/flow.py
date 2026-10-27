#!/usr/bin/env python3
"""
src/flow/flow.py — CONDITIONAL flow matching: noise -> z_b, conditioned on (z_a, dt).

THE LEAK-PROOF FORMULATION (this is the whole point):
  Previous versions interpolated z_a<->z_b, so z_b leaked into the network input and
  it learned to algebraically un-mix instead of learning progression. Here the
  transport is from NOISE to z_b:

      z_tau = (1-tau)*eps + tau*z_b ,   eps ~ N(0,I) ,  tau in [0,1]
      u     = dz_tau/dtau = z_b - eps          (target velocity)
      v_theta(z_tau, tau ; z_a, dt) regresses u

  z_a enters ONLY as conditioning, never inside the interpolant -> no algebraic path
  from (z_tau, z_a, tau) to z_b -> the net must learn p(z_b | z_a, dt). This is
  standard conditional flow matching (as in class/image-conditioned generation),
  conditioned on the baseline latent + interval instead of a label.

INFERENCE = GENERATION:
  sample eps ~ N(0,I); integrate dz/dtau = v_theta(z, tau; z_a, dt) from eps (tau=0)
  to z_b_hat (tau=1). Draw R samples -> a DISTRIBUTION of plausible futures.
  mean = point forecast; spread = forecast uncertainty (-> pseudoprogression signal).

Denoiser-agnostic: takes v_fn(z_tau, tau, z_a, dt) -> (v, logvar).
"""
from __future__ import annotations
from typing import Callable, Tuple
import torch
import torch.nn as nn

VFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]


def sample_tau(batch: int, device, eps: float = 1e-3) -> torch.Tensor:
    return torch.rand(batch, device=device).clamp(eps, 1.0 - eps)


def noise_interpolate(z_b: torch.Tensor, tau: torch.Tensor, eps: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Noise->data interpolant.  z_tau = (1-tau)*eps + tau*z_b ;  u = z_b - eps.
    eps is passed in (so callers can fix it for reconstruction tests).
    """
    t = tau.view(-1, 1, 1, 1, 1)
    z_tau = (1.0 - t) * eps + t * z_b
    u = z_b - eps
    return z_tau, u


class HeteroscedasticFMLoss(nn.Module):
    """
    Heteroscedastic NLL on the VELOCITY (noise->z_b transport):
        L = mean[ 0.5 ( ||u - v||^2 / sigma^2 + log sigma^2 ) ] + beta ||v_tilde||^2
    No leak now, so v-prediction is well-posed (standard CFM). logvar per-token [B,N];
    velocity error reduced over channels to [B,N].
    """
    def __init__(self, log_var_range: Tuple[float, float] = (-10.0, 10.0),
                 beta: float = 1e-4) -> None:
        super().__init__()
        self.lo, self.hi = log_var_range
        self.beta = beta

    def forward(self, u: torch.Tensor, v: torch.Tensor, logvar: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        B, c = u.shape[:2]
        v_tilde = logvar.clamp(self.lo, self.hi)
        inv_var = torch.exp(-v_tilde)
        err = (u - v).pow(2).reshape(B, c, -1).mean(dim=1)
        nll = 0.5 * (err * inv_var + v_tilde)
        loss = nll.mean() + self.beta * v_tilde.pow(2).mean()
        logs = {"fm_nll": float(nll.mean().detach()),
                "vel_mse": float(err.mean().detach()),
                "sigma2_mean": float(torch.exp(v_tilde).mean().detach())}
        return loss, v_tilde.detach(), logs


@torch.no_grad()
def generate(v_fn: VFn, z_a: torch.Tensor, dt: torch.Tensor,
             steps: int = 20, n_samples: int = 1, eps0: torch.Tensor = None
             ) -> torch.Tensor:
    """
    Generate futures by integrating dz/dtau = v_fn(z, tau; z_a, dt) from eps (tau=0)
    to tau=1, with a midpoint step. Returns [n_samples, B, c, h, w, d].
    Pass eps0 to fix the noise (deterministic reconstruction test).
    """
    B = z_a.shape[0]; device = z_a.device
    outs = []
    for r in range(n_samples):
        z = eps0.clone() if eps0 is not None else torch.randn_like(z_a)
        for m in range(steps):
            tau = m / steps
            v, _ = v_fn(z, torch.full((B,), tau, device=device), z_a, dt)
            tau_mid = (m + 0.5) / steps
            z_mid = z + (0.5 / steps) * v
            v_mid, _ = v_fn(z_mid, torch.full((B,), tau_mid, device=device), z_a, dt)
            z = z + (1.0 / steps) * v_mid
        outs.append(z)
    return torch.stack(outs, 0)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, c, h = 2, 4, 8
    z_b = torch.randn(B, c, h, h, h)
    z_a = torch.randn(B, c, h, h, h)
    N = h * h * h

    tau0 = torch.zeros(B); tau1 = torch.ones(B)
    eps = torch.randn_like(z_b)
    z0, _ = noise_interpolate(z_b, tau0, eps)
    z1, _ = noise_interpolate(z_b, tau1, eps)
    assert torch.allclose(z0, eps, atol=1e-5), "tau=0 must give eps"
    assert torch.allclose(z1, z_b, atol=1e-5), "tau=1 must give z_b"
    print("noise interpolant endpoints OK (tau=0->eps, tau=1->z_b)")

    # oracle velocity: constant u = z_b - eps recovers z_b via the ODE from that eps
    def oracle(z, tau, za, dt):
        return (z_b - eps), torch.zeros(z.shape[0], N)
    zb_hat = generate(oracle, z_a, torch.zeros(B), steps=10, n_samples=1, eps0=eps)[0]
    err = (zb_hat - z_b).abs().max().item()
    print(f"ODE with oracle velocity + fixed eps reconstructs z_b: max|err|={err:.2e}")
    assert err < 1e-3
    print("flow.py (conditional noise->z_b CFM) self-test passed.")
