#!/usr/bin/env python3
"""
src/physics/operators.py — differentiable finite-difference operators for the
Fisher-KPP reaction-diffusion residual. IMAGE space (voxel grid), NOT latent.

WHY image space: div(D grad c) requires the anatomical grid; spatial derivatives
in the compressed latent are physically meaningless. So the biophysics residual
is evaluated on the decoded density field.

Operators (all as fixed-kernel conv3d, so gradients flow to the density surrogate):
  grad3d(c)            -> (gx,gy,gz)      central differences
  div3d(fx,fy,fz)      -> scalar          central-difference divergence
  laplacian3d(c)       -> scalar          7-point stencil
  div_D_grad(c, D)     -> div(D * grad c) with scalar or field D (isotropic here)

Boundary: Neumann (zero-flux) via replicate padding — no tumor flux out of the brain.
Spacing: voxel size h (mm); LUMIERE atlas resized 182->128 so h ~= 182/128*1mm ~= 1.42mm.

THE CRITICAL TEST (bottom): numeric laplacian of a Gaussian vs the closed form.
If that fails, the physics is silently wrong. Everything else depends on it.
"""
from __future__ import annotations
from typing import Tuple, Union
import torch
import torch.nn.functional as F

Number = Union[float, torch.Tensor]


def _pad_replicate(c: torch.Tensor) -> torch.Tensor:
    # c: [B,1,H,W,D] -> pad 1 each side, replicate (Neumann / zero-flux)
    return F.pad(c, (1, 1, 1, 1, 1, 1), mode="replicate")


def grad3d(c: torch.Tensor, h: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Central-difference gradient. c:[B,1,H,W,D] -> gx,gy,gz each [B,1,H,W,D]."""
    cp = _pad_replicate(c)
    gx = (cp[:, :, 2:, 1:-1, 1:-1] - cp[:, :, :-2, 1:-1, 1:-1]) / (2 * h)
    gy = (cp[:, :, 1:-1, 2:, 1:-1] - cp[:, :, 1:-1, :-2, 1:-1]) / (2 * h)
    gz = (cp[:, :, 1:-1, 1:-1, 2:] - cp[:, :, 1:-1, 1:-1, :-2]) / (2 * h)
    return gx, gy, gz


def div3d(fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """Central-difference divergence of a vector field -> scalar [B,1,H,W,D]."""
    fxp = _pad_replicate(fx); fyp = _pad_replicate(fy); fzp = _pad_replicate(fz)
    dfx = (fxp[:, :, 2:, 1:-1, 1:-1] - fxp[:, :, :-2, 1:-1, 1:-1]) / (2 * h)
    dfy = (fyp[:, :, 1:-1, 2:, 1:-1] - fyp[:, :, 1:-1, :-2, 1:-1]) / (2 * h)
    dfz = (fzp[:, :, 1:-1, 1:-1, 2:] - fzp[:, :, 1:-1, 1:-1, :-2]) / (2 * h)
    return dfx + dfy + dfz


def laplacian3d(c: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """7-point Laplacian stencil. c:[B,1,H,W,D] -> [B,1,H,W,D]."""
    cp = _pad_replicate(c)
    center = cp[:, :, 1:-1, 1:-1, 1:-1]
    lap = (cp[:, :, 2:, 1:-1, 1:-1] + cp[:, :, :-2, 1:-1, 1:-1] +
           cp[:, :, 1:-1, 2:, 1:-1] + cp[:, :, 1:-1, :-2, 1:-1] +
           cp[:, :, 1:-1, 1:-1, 2:] + cp[:, :, 1:-1, 1:-1, :-2] -
           6.0 * center) / (h * h)
    return lap


def div_D_grad(c: torch.Tensor, D: Number, h: float = 1.0) -> torch.Tensor:
    """
    div(D grad c). For scalar/isotropic D this is D * laplacian(c); we implement the
    general div(D*grad c) so a spatial D-field (anisotropy extension) drops in later.
    D: scalar float/tensor, or a field [B,1,H,W,D].
    """
    gx, gy, gz = grad3d(c, h)
    if torch.is_tensor(D) and D.dim() == c.dim():
        gx, gy, gz = D * gx, D * gy, D * gz
    else:
        gx, gy, gz = D * gx, D * gy, D * gz   # scalar broadcast
    return div3d(gx, gy, gz, h)


# ============================ CRITICAL UNIT TEST ============================
def _test_against_analytic():
    """
    Numeric laplacian of a 3D Gaussian vs the closed form.
    For g(r)=exp(-|r|^2/(2s^2)), lap g = g * ((|r|^2 - 3 s^2)/s^4).
    Relative error must be tiny (<1e-2) away from boundaries. If not, STOP.
    """
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    Hs = 48; h = 1.0; s = 6.0
    ax = torch.arange(Hs, dtype=torch.float64, device=dev) - Hs / 2
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    r2 = X**2 + Y**2 + Z**2
    g = torch.exp(-r2 / (2 * s**2))[None, None]            # [1,1,H,H,H]
    lap_num = laplacian3d(g, h)
    lap_ana = g * ((r2[None, None] - 3 * s**2) / s**4)
    # compare in the interior (avoid replicate-pad boundary)
    sl = slice(4, Hs - 4)
    ln = lap_num[..., sl, sl, sl]; la = lap_ana[..., sl, sl, sl]
    rel = (ln - la).abs().mean() / (la.abs().mean() + 1e-12)
    print(f"[operators] laplacian vs analytic: rel_err = {float(rel):.2e}  "
          f"(must be < 1e-2)")
    assert rel < 1e-2, "FD Laplacian is WRONG — do not build physics on this."

    # div(D grad c) with scalar D must equal D * laplacian(c)
    D = 0.013
    lhs = div_D_grad(g.float(), D, h)
    rhs = D * laplacian3d(g.float(), h)
    d = (lhs - rhs).abs().max().item()
    print(f"[operators] div(D grad c) == D*lap(c) for scalar D: max|diff| = {d:.2e}")
    assert d < 1e-4

    # gradients flow (differentiability)
    c = torch.randn(1, 1, 16, 16, 16, device=dev, requires_grad=True)
    div_D_grad(c, 0.01, 1.0).sum().backward()
    assert c.grad is not None and c.grad.abs().sum() > 0
    print("[operators] differentiable: gradient flows to c  OK")
    print("[operators] ALL OPERATOR TESTS PASSED.")


if __name__ == "__main__":
    _test_against_analytic()
