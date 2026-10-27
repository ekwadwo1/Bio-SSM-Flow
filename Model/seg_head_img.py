#!/usr/bin/env python3
"""
src/models/seg_head_img.py — IMAGE-space segmentation head (decode-then-segment).

Diagnosis (latent-space head hit meanDice 0.46; SNFH ok but thin ET/NETC washed out):
the VQGAN latent, optimized for reconstruction, loses fine tumor detail. Fix: segment
the DECODED 128^3 volume, where thin structures are restored.

Pipeline: latent [B,4,32,32,32] --(frozen VQGAN.decode)--> image [B,4,128,128,128]
          --> UNet-lite 3D segmenter --> [B,4,128,128,128] logits (unified 4 classes).

This is differentiable end-to-end through the decoder (needed later for biophysics
density from the forecast). At train time we decode cached latents; at inference we
decode the FORECAST latent and segment it.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


def gn(c):
    # largest group count in {32,16,8,4,2,1} that divides c
    g = next(g for g in (32,16,8,4,2,1) if c % g == 0)
    return nn.GroupNorm(g, c)

class DBlk(nn.Module):
    def __init__(s, ci, co):
        super().__init__()
        s.c = nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), gn(co), nn.GELU(),
                            nn.Conv3d(co, co, 3, padding=1), gn(co), nn.GELU())
    def forward(s, x): return s.c(x)

class ImgSegUNet(nn.Module):
    """Compact 3D U-Net on 128^3 x4 -> 4-class seg. ~3 levels, memory-conscious."""
    def __init__(self, in_ch=4, n_classes=4, w=(24, 48, 96, 192)):
        super().__init__()
        self.e0 = DBlk(in_ch, w[0]); self.e1 = DBlk(w[0], w[1])
        self.e2 = DBlk(w[1], w[2]);  self.bott = DBlk(w[2], w[3])
        self.pool = nn.MaxPool3d(2)
        self.u2 = nn.ConvTranspose3d(w[3], w[2], 2, 2); self.d2 = DBlk(w[2]*2, w[2])
        self.u1 = nn.ConvTranspose3d(w[2], w[1], 2, 2); self.d1 = DBlk(w[1]*2, w[1])
        self.u0 = nn.ConvTranspose3d(w[1], w[0], 2, 2); self.d0 = DBlk(w[0]*2, w[0])
        self.out = nn.Conv3d(w[0], n_classes, 1)
    def forward(s, x):                          # x:[B,4,128,128,128]
        e0 = s.e0(x); e1 = s.e1(s.pool(e0))
        e2 = s.e2(s.pool(e1)); b = s.bott(s.pool(e2))
        d2 = s.d2(torch.cat([s.u2(b), e2], 1))
        d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        d0 = s.d0(torch.cat([s.u0(d1), e0], 1))
        return s.out(d0)                         # [B,4,128,128,128]


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = ImgSegUNet().to(dev)
    x = torch.randn(1, 4, 128, 128, 128, device=dev)
    with torch.no_grad(): y = net(x)
    print("img-seg logits:", tuple(y.shape), "| params:", sum(p.numel() for p in net.parameters()))
    assert y.shape == (1, 4, 128, 128, 128)
    print("seg_head_img.py self-test passed.")
