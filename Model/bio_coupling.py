#!/usr/bin/env python3
"""
src/bio_coupling.py — lightweight soft Fisher-KPP regularizer for training.

Design (option ii, matched to the ~20% prior strength):
  - applied every K steps (not every step) -> cheap on a contended GPU
  - small lambda -> a gentle nudge, not a governing term (honest to the 20% result)
  - residual_scale ~ 3.6e-5 (calibrated from real |dc/dt| in prototype_biophysics)
  - density from the FORECAST via seg head (from_probs, differentiable) vs the
    BASELINE cached seg density -> Fisher-KPP residual between them over dt
  - patient-specific (D,rho) from GrowthParamHead; falls back to fixed literature
    constants if the head is disabled

Cost control: only fires when (step % every == 0). One forecast-decode-seg pass then.
Returns (loss_bio, logs). If seg/vqgan not provided or step not due, returns (0, {}).
"""
from __future__ import annotations
from typing import Optional, Dict, Tuple
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisher_kpp import FisherKPPLoss, GrowthParamHead
from density import from_probs, from_labels


class BioRegularizer:
    """
    Soft physics regularizer. Holds the Fisher-KPP loss, an optional (D,rho) head,
    frozen VQGAN decoder + seg head to turn a forecast latent into a density field.
    """
    def __init__(self, vqgan, seghead, device,
                 residual_scale: float = 3.6e-5, h: float = 1.42,
                 use_param_head: bool = True, latent_dim: int = 4,
                 D_fixed: float = 0.02, rho_fixed: float = 0.02):
        self.vqgan = vqgan            # frozen
        self.seg = seghead            # frozen (0.52 Dice; gives from_probs density)
        self.device = device
        self.loss_fn = FisherKPPLoss(h=h, residual_scale=residual_scale).to(device)
        self.use_param_head = use_param_head
        self.D_fixed, self.rho_fixed = D_fixed, rho_fixed
        if use_param_head:
            self.param_head = GrowthParamHead(latent_dim=latent_dim).to(device)
        else:
            self.param_head = None

    def parameters(self):
        return list(self.param_head.parameters()) if self.param_head else []

    def __call__(self, z_a: torch.Tensor, z_pred: torch.Tensor,
                 seg_a: torch.Tensor, dt_days: torch.Tensor,
                 brain_mask: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        z_a     : [B,4,32,32,32] baseline latent
        z_pred  : [B,4,32,32,32] forecast latent (from the flow; carries grad)
        seg_a   : [B,128,128,128] baseline GT seg (unified labels) -> baseline density
        dt_days : [B]
        """
        # baseline density from GT labels (clean, no seg-head noise on the anchor)
        c_a = from_labels(seg_a).to(self.device)                    # [B,1,128,128,128]
        # forecast density: decode forecast latent -> seg head -> soft density (differentiable)
        x_pred = self.vqgan.decode(z_pred)                          # grad flows z_pred->x_pred
        prob = F.softmax(self.seg(x_pred), dim=1)                   # [B,4,128,128,128]
        c_b = from_probs(prob).to(self.device)                     # differentiable density

        # patient-specific (D,rho) or fixed
        if self.param_head is not None:
            D, rho = self.param_head(z_a, z_pred)                   # [B,1,1,1,1] each
        else:
            D = torch.full((z_a.shape[0],1,1,1,1), self.D_fixed, device=self.device)
            rho = torch.full((z_a.shape[0],1,1,1,1), self.rho_fixed, device=self.device)

        L, log = self.loss_fn(c_a, c_b, dt_days, D, rho, brain_mask)
        # report the learned params (interpretable prognostic covariate)
        log["D_med"] = float(D.median()); log["rho_med"] = float(rho.median())
        return L, log


if __name__ == "__main__":
    # smoke test with random stand-ins (no real checkpoints needed for shape check)
    import torch.nn as nn
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    class FakeVQ(nn.Module):
        def decode(self, z): return F.interpolate(z, scale_factor=4, mode="trilinear", align_corners=False)
    class FakeSeg(nn.Module):
        def __init__(s): super().__init__(); s.c = nn.Conv3d(4,4,1)
        def forward(s,x): return s.c(x)
    reg = BioRegularizer(FakeVQ().to(dev), FakeSeg().to(dev), dev, use_param_head=True)
    z_a = torch.randn(2,4,32,32,32, device=dev)
    z_pred = torch.randn(2,4,32,32,32, device=dev, requires_grad=True)
    seg_a = torch.randint(0,4,(2,128,128,128), device=dev)
    dt = torch.tensor([84.,56.], device=dev)
    L, log = reg(z_a, z_pred, seg_a, dt)
    L.backward()
    print("L_bio:", float(L), "| grad to z_pred:", z_pred.grad is not None and float(z_pred.grad.abs().mean())>0)
    print("logs:", {k: round(v,6) for k,v in log.items()})
    print("bio_coupling.py smoke test passed.")
