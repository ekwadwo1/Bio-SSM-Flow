#!/usr/bin/env python3
"""
src/ddpm_transport.py — DIFFUSION transport, as a controlled ablation of the flow
transport. Everything else (VelocityNetV2 backbone, VQGAN latent space, data,
splits, metrics, calibration) is held IDENTICAL; only the generative transport
changes: DDPM (noise-prediction) instead of flow matching.

This isolates the flow-matching-vs-diffusion choice for a fair SOTA-transport
comparison (TaDiff is a DDPM; its public release lacks training code and its
treatment mechanism is inapplicable to single-protocol data, so we compare the
transport on our own identical pipeline).

Conditional DDPM (predict-noise), conditioned on baseline z_a and interval dt:
  forward:  z_s = sqrt(abar_s) z_b + sqrt(1-abar_s) eps
  net:      the SAME VelocityNetV2 head, reinterpreted as an EPS predictor
  loss:     heteroscedastic NLL on eps (mirrors the flow loss structure)
  sample:   DDIM reverse from noise, conditioned on (z_a, dt)

The point forecast is the DDIM mean; an R-sample ensemble gives the predictive
distribution, calibrated post-hoc by the same temperature protocol. Metrics are
computed with the identical metrics.py so numbers sit in the same table.
"""
from __future__ import annotations
from typing import Tuple
import math, torch
import torch.nn as nn
import torch.nn.functional as F


# ---- cosine noise schedule (Nichol & Dhariwal), T steps ----
def cosine_abar(T: int, s: float = 0.008, device="cpu"):
    t = torch.arange(T + 1, device=device, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(1e-6, 0.9999)
    return abar[1:].float()          # [T], abar_s for s=1..T


class DDPMTransport(nn.Module):
    """
    Conditional DDPM transport reusing VelocityNetV2 as an eps-predictor.
    The net's forward(z_s, s_norm, z_a, dt) is unchanged; we interpret its first
    output as eps_hat and its second as logvar (same dual head as the flow model).
    """
    def __init__(self, net, T: int = 1000, device="cuda:0"):
        super().__init__()
        self.net = net
        self.T = T
        self.register_buffer("abar", cosine_abar(T, device=device))

    def _gather(self, s_idx):
        a = self.abar[s_idx]                          # [B]
        return a.view(-1, 1, 1, 1, 1)

    def q_sample(self, z_b, s_idx, eps):
        a = self._gather(s_idx)
        return a.sqrt() * z_b + (1 - a).sqrt() * eps

    def training_loss(self, z_b, z_a, dt, loss_fn, warmup=False):
        B = z_b.shape[0]; device = z_b.device
        s_idx = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(z_b)
        z_s = self.q_sample(z_b, s_idx, eps)
        s_norm = (s_idx.float() / self.T)
        eps_hat, logvar = self.net(z_s, s_norm, z_a, dt)
        if warmup:
            # variance-warmup: pure eps-MSE, MATCHING the flow recipe's first phase
            loss = torch.nn.functional.mse_loss(eps_hat.float(), eps.float())
            logs = {"vel_mse": float(loss.detach()), "sigma2_mean": float("nan")}
        else:
            loss, v_tilde, logs = loss_fn(eps, eps_hat, logvar)
        return loss, logs

    @torch.no_grad()
    def ddim_sample(self, z_a, dt, steps=50, R=1, eta=0.0):
        """DDIM reverse from noise, conditioned on (z_a, dt). Returns [R,B,c,h,w,d]."""
        B = z_a.shape[0]; device = z_a.device
        seq = torch.linspace(self.T - 1, 0, steps + 1, device=device).long()
        outs = []
        for _ in range(R):
            z = torch.randn_like(z_a)
            for i in range(steps):
                s, s_next = seq[i], seq[i + 1]
                a_s = self.abar[s].view(1, 1, 1, 1, 1)
                a_n = self.abar[s_next].view(1, 1, 1, 1, 1) if s_next >= 0 else torch.ones_like(a_s)
                s_norm = torch.full((B,), float(s) / self.T, device=device)
                eps_hat, _ = self.net(z, s_norm, z_a, dt)
                z0 = (z - (1 - a_s).sqrt() * eps_hat) / a_s.sqrt()
                z0 = z0.clamp(-5, 5)
                sigma = eta * ((1 - a_n) / (1 - a_s)).sqrt() * (1 - a_s / a_n).sqrt()
                z = a_n.sqrt() * z0 + (1 - a_n - sigma ** 2).clamp(min=0).sqrt() * eps_hat
                if eta > 0 and s_next > 0:
                    z = z + sigma * torch.randn_like(z)
            outs.append(z)
        return torch.stack(outs, 0)


if __name__ == "__main__":
    # smoke test with a stub net that returns (eps_hat, logvar) of correct shapes
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    B, c, h = 2, 4, 8; N = h*h*h
    class Stub(nn.Module):
        def set_week(self, *_): pass
        def forward(self, z, s, za, dt):
            return torch.zeros_like(z), torch.zeros(z.shape[0], N, device=z.device)
    ddpm = DDPMTransport(Stub(), T=1000, device=dev).to(dev)
    z_b = torch.randn(B, c, h, h, h, device=dev)
    z_a = torch.randn(B, c, h, h, h, device=dev)
    dt = torch.tensor([84., 56.], device=dev)
    s_idx = torch.randint(0, 1000, (B,), device=dev); eps = torch.randn_like(z_b)
    zs = ddpm.q_sample(z_b, s_idx, eps)
    assert zs.shape == z_b.shape
    samp = ddpm.ddim_sample(z_a, dt, steps=5, R=3)
    assert samp.shape == (3, B, c, h, h, h)
    print("q_sample OK, ddim_sample OK ->", tuple(samp.shape))
    print("ddpm_transport.py smoke test passed.")
