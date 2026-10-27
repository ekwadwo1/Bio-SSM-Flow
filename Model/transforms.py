#!/usr/bin/env python3
"""
src/data/transforms.py — preprocessing that feeds the (frozen) VQGAN encoder.

CRITICAL PREMISE: the VQGAN is a function fitted to ONE input distribution.
Whatever normalization trained it MUST be reproduced here, or its latents are
garbage. We implement the BraTS/MONAI-standard normalization (per-volume,
per-channel z-score within the brain/nonzero region) and then PROVE it matches
via an encode->decode PSNR gate in cache_latents.py. Do not skip that gate.

What this does, per timepoint:
  image [4,H,W,D] : resize -> 128^3 (trilinear), z-score within nonzero per channel
  seg   [H,W,D]   : resize -> 128^3 (NEAREST — never blend labels)
  mask  [H,W,D]   : resize -> 128^3 (NEAREST), boolean

Resize (not crop): LUMIERE atlas is 182x218x182; cropping to 128^3 would clip
brain. We resize the whole brain into the cube (voxel size becomes ~1.4mm).
That is the right trade for whole-brain forecasting; note it in the paper.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
from monai.transforms import Resize, NormalizeIntensity

INPUT_SIZE = (128, 128, 128)      # what the Paper-1 VQGAN expects


def _resize_image(x: torch.Tensor, size=INPUT_SIZE) -> torch.Tensor:
    """x: [C,H,W,D] float -> [C,128,128,128] trilinear."""
    r = Resize(spatial_size=size, mode="trilinear", align_corners=False)
    return r(x.float())


def _resize_label(y: torch.Tensor, size=INPUT_SIZE) -> torch.Tensor:
    """y: [H,W,D] -> [128,128,128] nearest (labels/masks must not blend)."""
    r = Resize(spatial_size=size, mode="nearest")
    out = r(y.unsqueeze(0).float())          # [1,128,128,128]
    return out.squeeze(0)


def preprocess(image: torch.Tensor, seg: torch.Tensor,
               mask: torch.Tensor, size=INPUT_SIZE) -> Dict[str, torch.Tensor]:
    """
    image : [4,H,W,D]  (T1pre,T1post,FLAIR,T2)
    seg   : [H,W,D]    unified labels {0,1,2,3}
    mask  : [H,W,D]    brain mask (bool-ish)
    returns dict of 128^3 tensors, image z-scored within-brain per channel.
    """
    img = _resize_image(image, size)                       # [4,128,128,128]
    # z-score within nonzero (brain) region, per channel — BraTS/MONAI standard.
    norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    img = norm(img)
    seg128 = _resize_label(seg, size).to(torch.uint8)      # [128,128,128]
    mask128 = (_resize_label(mask.float(), size) > 0.5)    # [128,128,128] bool
    return {"image": img, "seg": seg128, "mask": mask128}


if __name__ == "__main__":
    # synthetic smoke test (no data dependency)
    img = torch.rand(4, 182, 218, 182) * 500.0 + 100.0     # raw-ish intensities
    seg = torch.randint(0, 4, (182, 218, 182))
    mask = (img[1] > 120).to(torch.uint8)
    out = preprocess(img, seg, mask)
    print("image", tuple(out["image"].shape),
          "mean~0?", round(float(out["image"][out["image"] != 0].mean()), 3),
          "std~1?", round(float(out["image"][out["image"] != 0].std()), 3))
    print("seg", tuple(out["seg"].shape), "labels", torch.unique(out["seg"]).tolist())
    print("mask", tuple(out["mask"].shape), "vox", int(out["mask"].sum()))
