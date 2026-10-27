#!/usr/bin/env python3
"""
src/data/cached_pair_dataset.py — training dataset over CACHED latents.

Reads latent_cache_lumiere/{patient}_week{W}.pt (560 files) and forms CONSECUTIVE
forecasting pairs per patient. This is what training iterates: it never touches
NIfTI, never re-encodes — just loads fp16 [4,32,32,32] latents from disk.

Each cache file: {'z':[4,32,32,32] fp16, 'seg':[128^3] uint8, 'mask':[128^3] bool,
                  'patient':str, 'week':int}

A training item is an ordered pair (a -> b) of the same patient:
  z_a, z_b        : baseline / follow-up latents (float32)
  seg_b, mask_b   : follow-up segmentation + brain mask (image space, for L_seg / biophysics domain)
  seg_a           : baseline segmentation (for baseline density c_a in the physics residual)
  dt              : interval in days  (b.week - a.week) * 7   [> 0, guaranteed by min filter]
  week_a          : absolute treatment-time of baseline (weeks) — conditioning covariate

Patient-level grouping is exposed (`.patients`, `.patient_of(i)`) so k-fold CV can
split BY PATIENT and never leak a patient's timepoints across folds.
"""
from __future__ import annotations
import os, glob, re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

_NAME_RE = re.compile(r"(Patient-\d+)_week(\d+)\.pt$")


@dataclass
class CachedPairDataset(Dataset):
    cache_dir: str
    min_dt_days: float = 7.0
    pairs: List[Tuple[str, str]] = field(default_factory=list, init=False)   # (fileA, fileB)
    patient_of_pair: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        # group cache files by patient, sorted by week
        byp: Dict[str, List[Tuple[int, str]]] = {}
        for f in glob.glob(os.path.join(self.cache_dir, "*.pt")):
            m = _NAME_RE.search(os.path.basename(f))
            if not m:
                continue
            byp.setdefault(m.group(1), []).append((int(m.group(2)), f))
        for pid in byp:
            byp[pid].sort()
        n_pat = len(byp)
        for pid, tps in byp.items():
            for i in range(len(tps) - 1):
                (wa, fa), (wb, fb) = tps[i], tps[i + 1]
                if (wb - wa) * 7.0 < self.min_dt_days:
                    continue
                self.pairs.append((fa, fb))
                self.patient_of_pair.append(pid)
        self.patients = sorted(byp.keys())
        print(f"[cache-pairs] patients={n_pat}  consecutive_pairs={len(self.pairs)}  "
              f"cache_dir={self.cache_dir}")

    def patient_of(self, i: int) -> str:
        return self.patient_of_pair[i]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Dict:
        fa, fb = self.pairs[i]
        a = torch.load(fa, weights_only=False)
        b = torch.load(fb, weights_only=False)
        dt_days = float((b["week"] - a["week"]) * 7)
        return {
            "z_a": a["z"].float(), "z_b": b["z"].float(),         # [4,32,32,32]
            "seg_a": a["seg"].long(), "seg_b": b["seg"].long(),   # [128,128,128]
            "mask_b": b["mask"], "mask_a": a["mask"],             # [128,128,128] bool
            "dt": torch.tensor(dt_days, dtype=torch.float32),
            "week_a": torch.tensor(float(a["week"]), dtype=torch.float32),
            "patient": a["patient"], "week_pair": (a["week"], b["week"]),
        }


def patient_kfold(ds: CachedPairDataset, k: int = 5, seed: int = 42):
    """Yield (train_idx, val_idx) with NO patient shared across the split."""
    import numpy as np
    rng = np.random.default_rng(seed)
    pats = list(ds.patients)
    rng.shuffle(pats)
    folds = [pats[f::k] for f in range(k)]           # round-robin patient assignment
    for f in range(k):
        val_pat = set(folds[f])
        tr = [i for i in range(len(ds)) if ds.patient_of(i) not in val_pat]
        va = [i for i in range(len(ds)) if ds.patient_of(i) in val_pat]
        yield tr, va


if __name__ == "__main__":
    CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
    ds = CachedPairDataset(CACHE)
    s = ds[0]
    print("sample:", s["patient"], s["week_pair"],
          "| z_a", tuple(s["z_a"].shape), s["z_a"].dtype,
          "| seg_b labels", torch.unique(s["seg_b"]).tolist(),
          "| dt", float(s["dt"]), "| week_a", float(s["week_a"]))
    tr, va = next(patient_kfold(ds, k=5))
    # verify no patient leaks across the split
    ptr = {ds.patient_of(i) for i in tr}; pva = {ds.patient_of(i) for i in va}
    print(f"fold0: train_pairs={len(tr)} val_pairs={len(va)} "
          f"patient_overlap={len(ptr & pva)} (must be 0)")
