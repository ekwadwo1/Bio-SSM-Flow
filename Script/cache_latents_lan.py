#!/usr/bin/env python3
"""
cache_latents_lan.py — re-encode every LUMIERE timepoint with LESION-AWARE normalisation.

Why: per-scan z-scoring over ALL nonzero voxels (transforms.preprocess) shifts every
normal-tissue intensity whenever the lesion changes. The validity diagnostic showed
the forecaster's entire advantage over Copy equals that global shift. Here each
contrast is z-scored with the mean/std of NORMAL-APPEARING brain only
(brain mask minus the dilated union of all lesion labels of that scan), and the
result is applied to all nonzero voxels (background stays 0), exactly mirroring
MONAI NormalizeIntensity(nonzero=True, channel_wise=True) except for which voxels
define the statistics.

Writes a NEW cache (latent_cache_lumiere_lan/); the original cache is untouched.

Before caching, two checks run and are printed:
  1. DRIFT CHECK (no VQGAN): on consecutive post-operative pairs, the share of the
     baseline->follow-up normal-tissue difference explained by a global per-contrast
     affine intensity map, under OLD vs NEW normalisation. NEW should be far lower.
  2. VQGAN GATE: encode->decode PSNR (whole volume and brain voxels) under OLD vs NEW
     normalisation on the same cases. Aborts if NEW whole-volume PSNR < 25 dB.

Run (CPU for the drift check, GPU for encoding):
  CUDA_VISIBLE_DEVICES=0 python cache_latents_lan.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_latents as CL                                   # LUMIERE_ROOT, VQGAN_CKPT, load_vqgan, DEVICE
from lumiere_dataset import discover, CHANNEL_ORDER
from harmonize_labels import harmonize
from transforms import preprocess, _resize_image, _resize_label
from vqgan3d import compute_psnr

CACHE_LAN = os.path.join(HERE, "latent_cache_lumiere_lan")
DILATE = 3          # voxels (128^3 grid) of lesion dilation excluded from the statistics
MIN_REF = 1000      # minimum normal-tissue voxels per contrast; else fall back to nonzero stats
N_GATE = 6
N_DRIFT = 20
PSNR_GATE = 25.0


def _plain(t):
    return t.as_tensor() if hasattr(t, "as_tensor") else t


def load_raw(tp):
    img = np.stack([np.asanyarray(nib.load(tp.contrast(c)).dataobj).astype(np.float32)
                    for c in CHANNEL_ORDER], 0)
    seg = harmonize(np.asanyarray(nib.load(tp.seg()).dataobj), "lumiere")
    bm = tp.brain_mask()
    mask = np.asanyarray(nib.load(bm).dataobj) if os.path.exists(bm) else (img[1] > 0)
    return torch.from_numpy(img), torch.from_numpy(seg).long(), torch.from_numpy(np.asarray(mask))


def preprocess_lan(image, seg, mask):
    """Same geometry as transforms.preprocess; statistics from normal-appearing brain only."""
    img = _plain(_resize_image(image)).float()                       # [4,128,128,128]
    seg128 = _plain(_resize_label(seg)).to(torch.uint8)
    mask128 = _plain(_resize_label(mask.float())) > 0.5
    lesion = F.max_pool3d((seg128 > 0).float()[None, None], 2 * DILATE + 1, 1, DILATE)[0, 0] > 0.5
    out = torch.zeros_like(img)
    n_fallback = 0
    for c in range(img.shape[0]):
        nz = img[c] != 0
        ref = nz & mask128 & ~lesion
        if int(ref.sum()) < MIN_REF:
            ref, n_fallback = nz, n_fallback + 1
        mu, sd = img[c][ref].mean(), img[c][ref].std().clamp_min(1e-6)
        out[c][nz] = (img[c][nz] - mu) / sd
    return {"image": out, "seg": seg128, "mask": mask128, "lesion": lesion, "fallback": n_fallback}


def preprocess_old(image, seg, mask):
    pp = preprocess(image, seg, mask)
    return {k: _plain(v) for k, v in pp.items()}


def explained_by_global_affine(xa, xb, m):
    """Share of mean-squared baseline->follow-up difference on m removed by per-contrast a*xa+b."""
    raw, res = 0.0, 0.0
    for c in range(xa.shape[0]):
        p, y = xa[c][m], xb[c][m]
        pm, ym = p.mean(), y.mean()
        a = ((p - pm) * (y - ym)).mean() / ((p - pm) ** 2).mean().clamp_min(1e-8)
        raw += float(((p - y) ** 2).mean())
        res += float(((a * p + (ym - a * pm) - y) ** 2).mean())
    return 100 * (1 - res / max(raw, 1e-12))


def drift_check(pats):
    print(f"\n{'=' * 70}\n  DRIFT CHECK — normal-tissue difference explained by a global intensity map\n{'=' * 70}")
    old_v, new_v, n = [], [], 0
    for pid, tps in pats.items():
        good = [t for t in tps if t.complete() and t.week > 0]
        cand = [(a, b) for a, b in zip(good[:-1], good[1:]) if (b.week - a.week) * 7 >= 7]
        if not cand:
            continue
        a, b = cand[0]
        ra, rb = load_raw(a), load_raw(b)
        na, nb = preprocess_lan(*ra), preprocess_lan(*rb)
        oa, ob = preprocess_old(*ra), preprocess_old(*rb)
        normal = na["mask"] & nb["mask"] & ~na["lesion"] & ~nb["lesion"]
        old_v.append(explained_by_global_affine(oa["image"], ob["image"], normal))
        new_v.append(explained_by_global_affine(na["image"], nb["image"], normal))
        print(f"  {pid} week {a.week}->{b.week}:  OLD {old_v[-1]:5.1f}%   NEW {new_v[-1]:5.1f}%")
        n += 1
        if n >= N_DRIFT:
            break
    print(f"\n  mean explained share: OLD {np.mean(old_v):.1f}%   NEW {np.mean(new_v):.1f}%   (n={n})")
    print("  EXPECT NEW << OLD. If not, the normalisation fix is not removing the drift.")
    return float(np.mean(old_v)), float(np.mean(new_v))


@torch.no_grad()
def vqgan_gate(vq, tps):
    print(f"\n{'=' * 70}\n  VQGAN GATE — encode->decode PSNR, OLD vs NEW normalisation\n{'=' * 70}")
    rows = []
    for tp in tps[:N_GATE]:
        raw = load_raw(tp)
        r = {}
        for tag, fn in (("old", preprocess_old), ("new", preprocess_lan)):
            pp = fn(*raw)
            x = pp["image"].unsqueeze(0).to(CL.DEVICE)
            xr = vq.decode(vq.encode(x))
            m = pp["mask"].to(CL.DEVICE)
            L = (x[0][:, m].max() - x[0][:, m].min()).clamp_min(1e-6)
            mse_b = ((xr[0] - x[0]) ** 2)[:, m].mean().clamp_min(1e-12)
            r[f"{tag}_whole"] = float(compute_psnr(x, xr))
            r[f"{tag}_brain"] = float(10 * torch.log10(L * L / mse_b))
        rows.append(r)
        print(f"  {tp.patient} week {tp.week:<4d}  OLD whole {r['old_whole']:5.2f} brain {r['old_brain']:5.2f} | "
              f"NEW whole {r['new_whole']:5.2f} brain {r['new_brain']:5.2f} dB")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"\n  mean: OLD whole {mean['old_whole']:.2f} brain {mean['old_brain']:.2f} | "
          f"NEW whole {mean['new_whole']:.2f} brain {mean['new_brain']:.2f} dB  (gate {PSNR_GATE} dB on NEW whole)")
    return mean


@torch.no_grad()
def main():
    pats = discover(CL.LUMIERE_ROOT)
    tps = [t for ts in pats.values() for t in ts if t.complete()]
    print(f"[lan] {len(tps)} complete timepoints -> {CACHE_LAN}")
    old_share, new_share = drift_check(pats)

    vq = CL.load_vqgan(CL.VQGAN_CKPT)
    gate = vqgan_gate(vq, tps)
    if gate["new_whole"] < PSNR_GATE:
        print("[lan] ABORT: VQGAN reconstruction under lesion-aware normalisation is below the gate.")
        return

    os.makedirs(CACHE_LAN, exist_ok=True)
    done, fallbacks, t0 = 0, 0, time.time()
    for tp in tps:
        out = os.path.join(CACHE_LAN, f"{tp.patient}_week{tp.week}.pt")
        if os.path.exists(out):
            done += 1
            continue
        pp = preprocess_lan(*load_raw(tp))
        fallbacks += int(pp["fallback"] > 0)
        z = vq.encode(pp["image"].unsqueeze(0).to(CL.DEVICE)).squeeze(0).to(torch.float16).cpu()
        torch.save({"z": z, "seg": pp["seg"], "mask": pp["mask"], "patient": tp.patient,
                    "week": tp.week, "norm": "lesion_aware", "dilate": DILATE}, out)
        done += 1
        if done % 50 == 0:
            print(f"  cached {done}/{len(tps)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[lan] done: {done} timepoints in {CACHE_LAN}; timepoints using fallback stats: {fallbacks}")
    print(f"[lan] summary: drift explained OLD {old_share:.1f}% -> NEW {new_share:.1f}% | "
          f"VQGAN brain PSNR OLD {gate['old_brain']:.2f} -> NEW {gate['new_brain']:.2f} dB")


if __name__ == "__main__":
    main()
