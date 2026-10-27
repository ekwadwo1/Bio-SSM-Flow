#!/usr/bin/env python3
"""
src/models/seg_head.py — latent -> segmentation head for Bio-SSM-Flow.

Maps a [B,4,32,32,32] latent to a [B,4,128,128,128] subregion segmentation in the
UNIFIED label space (0=bg, 1=NETC, 2=SNFH, 3=ET). Trained on all cached (latent,seg)
pairs (real encoded latents + harmonized DeepBraTumIA labels), then applied to the
FORECAST latent at inference to predict the future mask and to feed the biophysics
density (from_probs).

Design: 3 conv blocks (GN+GELU) at latent res, trilinear upsample x4 (32->128), 1x1x1
classifier. ~small; trains fast on 560 cached timepoints. Same channel order as VQGAN.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class SegHead(nn.Module):
    def __init__(self, latent_dim=4, n_classes=4, width=64, upsample=4):
        super().__init__()
        self.up = upsample
        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1),
                                 nn.GroupNorm(min(32, co), co), nn.GELU())
        self.body = nn.Sequential(blk(latent_dim, width), blk(width, width), blk(width, width))
        self.head = nn.Conv3d(width, n_classes, 1)

    def forward(self, z):                                    # z: [B,4,32,32,32]
        h = self.body(z)
        h = F.interpolate(h, scale_factor=self.up, mode="trilinear", align_corners=False)
        return self.head(h)                                  # [B,n_classes,128,128,128]


class DiceCELoss(nn.Module):
    """Soft Dice + class-weighted CE. Down-weights via sample_weight (silver labels)."""
    def __init__(self, n_classes=4, class_w=(0.1, 1.0, 1.0, 2.0), eps=1.0):
        super().__init__()
        self.n = n_classes; self.eps = eps
        self.register_buffer("cw", torch.tensor(class_w))

    def forward(self, logits, target, sample_weight=1.0):
        # logits [B,K,H,W,D]; target [B,H,W,D] int
        ce = F.cross_entropy(logits, target, weight=self.cw.to(logits.device), reduction="none")
        ce = (ce.mean((1, 2, 3)) * sample_weight).mean()
        p = F.softmax(logits, 1)
        t = F.one_hot(target.clamp(0, self.n - 1), self.n).permute(0, 4, 1, 2, 3).float()
        inter = (p * t).sum((2, 3, 4)); denom = p.sum((2, 3, 4)) + t.sum((2, 3, 4))
        dice = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean(1)   # [B]
        dice = (dice * sample_weight).mean()
        return ce + dice, {"ce": float(ce.detach()), "dice_loss": float(dice.detach())}


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = SegHead().to(dev)
    z = torch.randn(2, 4, 32, 32, 32, device=dev)
    out = net(z)
    print("seg logits:", tuple(out.shape), "(expect [2,4,128,128,128])")
    assert out.shape == (2, 4, 128, 128, 128)
    seg = torch.randint(0, 4, (2, 128, 128, 128), device=dev)
    L, logs = DiceCELoss().to(dev)(out, seg)
    print("loss:", float(L), logs, "| params:", sum(p.numel() for p in net.parameters()))
    print("seg_head.py self-test passed.")
