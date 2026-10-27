#!/usr/bin/env python3
"""
src/physics/density.py — tumor-cell-density surrogate c in [0,1] from segmentation.

The Fisher-KPP field is CELL DENSITY, not image intensity or class labels. We map
the UNIFIED subregion labels (0=bg, 1=NETC, 2=SNFH, 3=ET) to a soft density via the
established imaging-threshold convention (Swanson/Konukoglu): enhancing tumor ~ high
density, non-enhancing core ~ mid, edema/FLAIR-hyperintensity ~ low (invasive rim).

  ET   (3) -> 0.90    (dense proliferating tumor)
  NETC (1) -> 0.50    (non-enhancing / necrotic core)
  SNFH (2) -> 0.16    (edema / infiltrative low-density margin)
  bg   (0) -> 0.0

Two modes:
  from_labels(seg)          : hard labels -> density (for GT-seg prototyping)
  from_probs(prob[B,K,...]) : soft class probabilities -> density (differentiable,
                              for when a seg head predicts on the forecast latent)

Weights are configurable and become a small ablation in the paper.
"""
from __future__ import annotations
from typing import Dict
import torch

# unified-label -> density weight
DEFAULT_W: Dict[int, float] = {0: 0.0, 1: 0.50, 2: 0.16, 3: 0.90}


def from_labels(seg: torch.Tensor, w: Dict[int, float] = None) -> torch.Tensor:
    """
    seg: [B,H,W,D] or [H,W,D] integer unified labels {0,1,2,3}.
    returns density c in [0,1], same spatial shape with a channel dim: [B,1,H,W,D].
    """
    w = w or DEFAULT_W
    if seg.dim() == 3:
        seg = seg.unsqueeze(0)
    c = torch.zeros_like(seg, dtype=torch.float32)
    for lab, val in w.items():
        c = c + (seg == lab).float() * val
    return c.unsqueeze(1).clamp(0, 1)          # [B,1,H,W,D]


def from_probs(prob: torch.Tensor, w: Dict[int, float] = None) -> torch.Tensor:
    """
    prob: [B,K,H,W,D] softmax class probabilities (K=4 unified classes).
    Differentiable soft density = sum_k w_k * prob_k. -> [B,1,H,W,D].
    Use this when coupling to a learned seg head (gradients flow to the seg logits).
    """
    w = w or DEFAULT_W
    B, K = prob.shape[:2]
    wv = torch.tensor([w.get(k, 0.0) for k in range(K)], device=prob.device,
                      dtype=prob.dtype).view(1, K, 1, 1, 1)
    c = (prob * wv).sum(1, keepdim=True)
    return c.clamp(0, 1)                        # [B,1,H,W,D]


if __name__ == "__main__":
    # sanity: labels -> density, and soft==hard when probs are one-hot
    import torch.nn.functional as F
    seg = torch.tensor([[[0,1],[2,3]]]).reshape(1,1,2,2,1).squeeze(1)  # [1,2,2,1]
    seg = torch.zeros(1,4,4,4, dtype=torch.long)
    seg[0,0,:,:] = 3; seg[0,1,:,:] = 1; seg[0,2,:,:] = 2; seg[0,3,:,:] = 0
    c = from_labels(seg)
    print("from_labels density range:", float(c.min()), float(c.max()),
          "| ET slice mean:", float(c[0,0,0].mean()), "(expect 0.90)")
    prob = F.one_hot(seg, 4).permute(0,4,1,2,3).float()   # one-hot -> [B,4,...]
    c2 = from_probs(prob)
    print("from_probs == from_labels on one-hot:", torch.allclose(c, c2))
    assert torch.allclose(c, c2)
    print("density.py self-test passed.")
