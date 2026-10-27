#!/usr/bin/env python3
"""
src/data/lumiere_dataset.py — LUMIERE longitudinal glioma loader (PRIMARY glioma source).

VERIFIED FACTS this is built on (measured from disk 2026-08-30):
  - root: .../LUMIERE_Imaging/Imaging/Patient-###/week-###/
  - atlas space (co-registered across timepoints, skull-stripped):
      DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz
      DeepBraTumIA-segmentation/atlas/skull_strip/{ct1,t1,t2,flair}_skull_strip.nii.gz
      DeepBraTumIA-segmentation/atlas/skull_strip/brain_mask.nii.gz
  - 91 patients; median 6 timepoints (max 19); 73 patients have >=3 timepoints.
  - seg labels {0,1,2,3}; VERIFIED order 1=ET, 2=NETC, 3=SNFH (NOT BraTS order).
  - week-### encodes weeks since reference (≈ weeks since treatment) -> real intervals.
  - some weeks are partially processed (missing seg or a contrast) -> dropped w/ log.

Design decisions (confirmed with the researcher):
  - CONSECUTIVE pairs (week_k -> week_{k+1}) for the biophysics/forecasting core.
    (Finite-difference physics residual is only valid between adjacent timepoints.)
    all_pairs=True optionally emits every earlier->later combination (augmentation).
  - Expose BOTH dt_days (pair gap) AND week_a (absolute treatment-time of baseline)
    as conditioning covariates. week_a is a free, clinically meaningful signal.
  - Contrast channel order [T1pre=t1, T1post=ct1, FLAIR=flair, T2=t2] — identical to
    Paper-1 and the (mets) ALPTDG loader, so channels are consistent across datasets.
  - Atlas space => NO registration needed, but we ASSERT same-grid per patient rather
    than assume it (the discipline that caught the ALPTDG misalignment).

Raw tensors + metadata out; MONAI resample/normalize/crop live in a separate pipeline.
"""
from __future__ import annotations
import os, glob, re, warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

try:
    from harmonize_labels import harmonize, ET
except ImportError:
    from .harmonize_labels import harmonize, ET

# LUMIERE atlas contrast filename -> unified channel name (fixed 4-channel order)
CONTRAST_FILE = {"T1pre": "t1", "T1post": "ct1", "FLAIR": "flair", "T2": "t2"}
CHANNEL_ORDER = ["T1pre", "T1post", "FLAIR", "T2"]     # -> [4,H,W,D]
WEEK_RE = re.compile(r"week-(\d+)")                     # week-044 -> 44 ; week-000-1 -> 0 (see parse)
PATIENT_RE = re.compile(r"(Patient-\d+)")


def _junk(p: str) -> bool:
    return os.path.basename(p).startswith("._")


def _load(path: str) -> np.ndarray:
    if _junk(path):
        raise ValueError(f"AppleDouble junk passed to loader: {path}")
    return np.asanyarray(nib.load(path).dataobj)


def _parse_week(week_dir: str) -> Optional[int]:
    """'week-044' -> 44. 'week-000-1'/'week-000-2' are same-week repeats -> 0 (kept, ordered by suffix)."""
    m = WEEK_RE.search(os.path.basename(week_dir))
    return int(m.group(1)) if m else None


@dataclass
class LTimepoint:
    patient: str
    week: int
    order_key: Tuple[int, str]     # (week, raw folder name) to keep week-000-1 < week-000-2
    atlas: str                     # .../DeepBraTumIA-segmentation/atlas
    def seg(self) -> str:
        return os.path.join(self.atlas, "segmentation", "seg_mask.nii.gz")
    def contrast(self, name: str) -> str:
        return os.path.join(self.atlas, "skull_strip", f"{CONTRAST_FILE[name]}_skull_strip.nii.gz")
    def brain_mask(self) -> str:
        return os.path.join(self.atlas, "skull_strip", "brain_mask.nii.gz")
    def complete(self) -> bool:
        if not os.path.exists(self.seg()):
            return False
        return all(os.path.exists(self.contrast(c)) for c in CHANNEL_ORDER)


def discover(root: str) -> Dict[str, List[LTimepoint]]:
    """root = .../LUMIERE_Imaging/Imaging"""
    pats: Dict[str, List[LTimepoint]] = {}
    for wk_dir in sorted(glob.glob(os.path.join(root, "*", "week-*"))):
        if _junk(wk_dir) or not os.path.isdir(wk_dir):
            continue
        pm = PATIENT_RE.search(wk_dir)
        wk = _parse_week(wk_dir)
        if pm is None or wk is None:
            continue
        atlas = os.path.join(wk_dir, "DeepBraTumIA-segmentation", "atlas")
        if not os.path.isdir(atlas):
            continue          # timepoint not DeepBraTumIA-processed -> skip
        tp = LTimepoint(pm.group(1), wk, (wk, os.path.basename(wk_dir)), atlas)
        pats.setdefault(pm.group(1), []).append(tp)
    for pid in pats:
        pats[pid].sort(key=lambda t: t.order_key)
    return pats


@dataclass
class LumiereDataset(Dataset):
    root: str                                   # .../LUMIERE_Imaging/Imaging
    all_pairs: bool = False                     # False=consecutive only (default)
    convention: str = "lumiere"
    assert_semantics: bool = True               # verify label 1 is T1c-brightest on first item
    min_dt_days: float = 7.0                     # drop pairs with interval < this (kills dt=0 same-week repeats)
    pairs: List[Tuple[LTimepoint, LTimepoint]] = field(default_factory=list, init=False)
    _grid_warned: set = field(default_factory=set, init=False)
    _sem_checked: bool = field(default=False, init=False)

    def __post_init__(self):
        pats = discover(self.root)
        n_pat = len(pats)
        n_ge3 = 0
        n_drop = 0
        n_degenerate = 0
        for pid, tps in pats.items():
            good = [t for t in tps if t.complete()]
            n_drop += len(tps) - len(good)
            if len(good) >= 3:
                n_ge3 += 1
            if len(good) < 2:
                continue
            # same-grid assertion per patient (atlas space SHOULD guarantee this)
            self._assert_same_grid(pid, good)
            if self.all_pairs:
                cand = [(good[i], good[j])
                        for i in range(len(good)) for j in range(i + 1, len(good))]
            else:
                cand = [(good[i], good[i + 1]) for i in range(len(good) - 1)]
            for a, b in cand:
                if (b.week - a.week) * 7.0 < self.min_dt_days:
                    n_degenerate += 1          # same-week / zero-interval repeat -> not a forecasting pair
                    continue
                self.pairs.append((a, b))
        print(f"[LUMIERE] patients={n_pat}  >=3tp(biophysics)={n_ge3}  "
              f"{'all' if self.all_pairs else 'consecutive'}_pairs={len(self.pairs)}  "
              f"partial_timepoints_dropped={n_drop}  "
              f"degenerate_dt<{self.min_dt_days:.0f}d_dropped={n_degenerate}")
        if not self.pairs:
            warnings.warn("[LUMIERE] 0 pairs — check root path.")

    def _assert_same_grid(self, pid: str, tps: List[LTimepoint]):
        try:
            shapes = {nib.load(t.contrast("T1post")).shape for t in tps}
        except Exception:
            return
        if len(shapes) > 1 and pid not in self._grid_warned:
            self._grid_warned.add(pid)
            warnings.warn(f"[LUMIERE] {pid}: timepoints span multiple grids {shapes} — "
                          f"atlas space was assumed co-registered but ISN'T for this patient. "
                          f"Exclude or register before use.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _stack(self, tp: LTimepoint) -> np.ndarray:
        return np.stack([_load(tp.contrast(c)).astype(np.float32) for c in CHANNEL_ORDER], 0)

    def _maybe_check_semantics(self, tp: LTimepoint, x: np.ndarray, s: np.ndarray):
        """Assert label mapped to ET is the brightest on T1post — once, loudly, at load time."""
        if self._sem_checked or not self.assert_semantics:
            return
        self._sem_checked = True
        t1c = x[CHANNEL_ORDER.index("T1post")]
        uni = s  # already harmonized
        means = {k: float(t1c[uni == k].mean()) for k in np.unique(uni) if k != 0 and (uni == k).any()}
        if means and max(means, key=means.get) != ET:
            raise AssertionError(
                f"[LUMIERE] label-semantics check FAILED: unified-ET ({ET}) is not the "
                f"brightest T1c class (means={means}). The crosswalk order is wrong — STOP.")

    def __getitem__(self, i: int) -> Dict:
        a, b = self.pairs[i]
        x_a, x_b = self._stack(a), self._stack(b)
        s_a = harmonize(_load(a.seg()), self.convention)
        s_b = harmonize(_load(b.seg()), self.convention)
        self._maybe_check_semantics(a, x_a, s_a)
        bm = _load(a.brain_mask()).astype(bool) if os.path.exists(a.brain_mask()) else (x_a[1] > 0)
        dt_days = float((b.week - a.week) * 7)
        return {
            "x_a": torch.from_numpy(x_a), "x_b": torch.from_numpy(x_b),   # [4,H,W,D]
            "s_a": torch.from_numpy(s_a).long(), "s_b": torch.from_numpy(s_b).long(),
            "brain_mask": torch.from_numpy(bm),                           # [H,W,D] biophysics domain
            "dt": torch.tensor(dt_days, dtype=torch.float32),            # pair interval (days)
            "week_a": torch.tensor(float(a.week), dtype=torch.float32),  # abs. treatment-time (weeks)
            "patient": a.patient, "week_pair": (a.week, b.week),
            "cohort": "lumiere", "label_quality": "silver",             # DeepBraTumIA = auto/silver
        }


if __name__ == "__main__":
    ROOT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/LUMIERE_Imaging/Imaging"
    ds = LumiereDataset(ROOT, all_pairs=False)
    if len(ds):
        s = ds[0]
        print("sample:", s["patient"], s["week_pair"],
              "| x_a", tuple(s["x_a"].shape),
              "| s_b labels", torch.unique(s["s_b"]).tolist(),
              "| dt(d)", float(s["dt"]), "| week_a", float(s["week_a"]),
              "| mask vox", int(s["brain_mask"].sum()))
