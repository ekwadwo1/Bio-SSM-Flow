#!/usr/bin/env python3
"""
scripts/cache_latents.py — encode every LUMIERE timepoint ONCE, cache to disk.

WHY: 469 pairs re-encoding the frozen VQGAN every epoch wastes days. We encode
each UNIQUE timepoint once (~508 encodes, not ~938), store the latent, and let
training read small latent tensors. Per-timepoint (not per-pair) halves the work.

THE GATE (do not bypass): before caching anything, we encode->decode a handful of
real LUMIERE volumes and measure PSNR. This simultaneously validates (a) that our
normalization matches what trained the VQGAN and (b) that the encoder transfers
from BraTS to LUMIERE. If mean PSNR is high -> reuse the checkpoint, cache, move on.
If it is low -> STOP; the frozen Paper-1 VQGAN does not transfer and must be
retrained/fine-tuned on the Paper-2 cohort before latents mean anything.

Edit VQGAN_CKPT and LUMIERE_ROOT, then run. Writes to CACHE_DIR/{patient}_{week}.pt
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumiere_dataset import discover, LTimepoint, CHANNEL_ORDER
from harmonize_labels import harmonize
from transforms import preprocess
from vqgan3d import VQGAN3D, compute_psnr           # Paper-1 code, reused as-is

# ── EDIT ─────────────────────────────────────────────────────────────
LUMIERE_ROOT = "/.../LUMIERE_Imaging/Imaging"
VQGAN_CKPT   = "/.../checkpoints/vqgan_r4/best_vqgan.pt"     # <-- give me this path
CACHE_DIR    = "/.../Bio-SSM-Flow/latent_cache_lumiere"
DEVICE       = "cuda:0"
PSNR_GATE    = 25.0                                   # below this -> VQGAN does not transfer; retrain
N_GATE       = 6                                      # cases for the reconstruction gate
# ─────────────────────────────────────────────────────────────────────


def load_vqgan(ckpt_path: str) -> VQGAN3D:
    vq = VQGAN3D(in_channels=4, latent_dim=4, base_channels=64,
                 channel_mult=(1, 2, 4, 8), n_res_blocks=2, downsample_factor=4)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"VQGAN checkpoint not found: {ckpt_path}\n"
            f"Set VQGAN_CKPT to your Paper-1 trained weights.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("gen", ckpt.get("state_dict", ckpt))
    missing, unexpected = vq.load_state_dict(state, strict=False)
    print(f"[vqgan] loaded {ckpt_path}")
    print(f"[vqgan] missing={len(missing)} unexpected={len(unexpected)} keys "
          f"(some are expected: discriminator/quantizer not needed for encode/decode)")
    vq.eval().to(DEVICE)
    for p in vq.parameters():
        p.requires_grad_(False)
    return vq


def _timepoint_tensors(tp: LTimepoint):
    import nibabel as nib
    img = np.stack([np.asanyarray(nib.load(tp.contrast(c)).dataobj).astype(np.float32)
                    for c in CHANNEL_ORDER], 0)
    seg = np.asanyarray(nib.load(tp.seg()).dataobj)
    seg = harmonize(seg, "lumiere")
    bm = tp.brain_mask()
    mask = (np.asanyarray(nib.load(bm).dataobj) if os.path.exists(bm) else (img[1] > 0))
    pp = preprocess(torch.from_numpy(img), torch.from_numpy(seg).long(),
                    torch.from_numpy(np.asarray(mask)))
    return pp


@torch.no_grad()
def reconstruction_gate(vq: VQGAN3D, tps, n=N_GATE) -> float:
    print(f"\n{'='*64}\n  RECONSTRUCTION GATE — does the VQGAN transfer to LUMIERE?\n{'='*64}")
    psnrs = []
    for tp in tps[:n]:
        pp = _timepoint_tensors(tp)
        x = pp["image"].unsqueeze(0).to(DEVICE)           # [1,4,128,128,128], z-scored
        z = vq.encode(x)
        xr = vq.decode(z)
        p = compute_psnr(x, xr)
        psnrs.append(p)
        print(f"  {tp.patient} week-{tp.week:<4d} PSNR={p:5.2f} dB  "
              f"z:{tuple(z.shape[1:])}")
    m = float(np.mean(psnrs))
    print(f"\n  mean encode->decode PSNR = {m:.2f} dB   (gate = {PSNR_GATE} dB)")
    if m < PSNR_GATE:
        print(f"  !! FAIL: VQGAN does NOT transfer to LUMIERE (or normalization mismatch).")
        print(f"     DO NOT cache these latents. Options:")
        print(f"       (a) fine-tune the VQGAN decoder+encoder on LUMIERE, or")
        print(f"       (b) retrain the VQGAN from scratch on the Paper-2 glioma cohort.")
        print(f"     Tell me and I'll write the VQGAN (re)training script.")
    else:
        print(f"  OK: reconstruction good -> normalization matches & domain transfers. Caching.")
    return m


@torch.no_grad()
def cache_all(vq: VQGAN3D):
    os.makedirs(CACHE_DIR, exist_ok=True)
    pats = discover(LUMIERE_ROOT)
    # unique complete timepoints across all patients
    tps = [t for tps in pats.values() for t in tps if t.complete()]
    print(f"\n[cache] {len(tps)} unique complete timepoints -> {CACHE_DIR}")

    gate = reconstruction_gate(vq, tps)
    if gate < PSNR_GATE:
        print("[cache] aborting: fix the VQGAN first (see options above).")
        return

    done = 0
    for tp in tps:
        out = os.path.join(CACHE_DIR, f"{tp.patient}_week{tp.week}.pt")
        if os.path.exists(out):
            done += 1
            continue
        pp = _timepoint_tensors(tp)
        x = pp["image"].unsqueeze(0).to(DEVICE)
        z = vq.encode(x).squeeze(0).to(torch.float16).cpu()   # [4,32,32,32]
        torch.save({
            "z": z,                                            # latent (fp16)
            "seg": pp["seg"].to(torch.uint8),                  # [128,128,128] labels
            "mask": pp["mask"],                                # [128,128,128] bool
            "patient": tp.patient, "week": tp.week,
        }, out)
        done += 1
        if done % 50 == 0:
            print(f"  cached {done}/{len(tps)}")
    print(f"[cache] done: {done} timepoints in {CACHE_DIR}")
    print(f"[cache] NEXT: the training dataset reads these by (patient,week) and forms pairs.")


def _run():
    vq = load_vqgan(VQGAN_CKPT)
    cache_all(vq)


if __name__ == "__main__":
    _run()
