#!/usr/bin/env python3
"""
src/data/alptdg_dataset.py — UCSF-ALPTDG forecasting-pair loader.

Confirmed facts this is built on (measured from disk 2026-08-30):
  - root: .../UCSF-ALPTDG/UCSF_BrainMetastases_TRAIN/   (folder mislabeled 'Metastases'; it IS glioma)
  - 324 case folders named  #####A , #####B , ...  (patient id = 5 digits, timepoint = letter)
  - per case: {id}_T1pre, {id}_T1post, {id}_FLAIR, {id}_T2Synth, {id}_BraTS-seg (.nii.gz)
  - BraTS-seg labels {0,1,2,3}; raw 3 = enhancing (verified via T1c intensity)
  - atlas/BraTS-processed => timepoints of a patient share a grid (co-registered)

Design decisions (challenge them if you disagree):
  - A "sample" is an ORDERED pair (timepoint k -> timepoint k+1) of the SAME patient.
    Baseline = earlier letter, follow-up = later letter.
  - Patients with only one timepoint are DROPPED for forecasting (logged, not silent).
  - Contrasts stacked as [T1pre, T1post, FLAIR, T2Synth] = 4 channels (matches Paper-1 C=4).
    (T2Synth is synthesized where native T2 was missing — flagged per case.)
  - Interval dt: ALPTDG does not ship acquisition dates in the filename. We default
    dt=None and REQUIRE the caller to supply intervals from the clinical table, OR
    fall back to a nominal unit interval with a loud warning. Do NOT silently fake dt.

This module returns raw tensors + metadata. MONAI transforms (resample, normalize,
crop) are applied by a separate transform pipeline, NOT here, so the loader stays
inspectable and the science stays decoupled from augmentation.
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
    from harmonize_labels import harmonize
except ImportError:
    from .harmonize_labels import harmonize

CONTRASTS = ["T1pre", "T1post", "FLAIR", "T2Synth"]   # -> 4 channels, fixed order
CASE_RE = re.compile(r"^(\d+)([A-Z])$")               # 100104B -> ('100104','B'); id length varies


def _is_junk(p: str) -> bool:
    return os.path.basename(p).startswith("._")


def _load(path: str) -> np.ndarray:
    if _is_junk(path):
        raise ValueError(f"AppleDouble junk passed to loader: {path}")
    return np.asanyarray(nib.load(path).dataobj)


@dataclass
class Timepoint:
    patient: str
    tp: str                    # 'A','B',...
    folder: str
    contrasts: Dict[str, str]  # name -> path
    seg: Optional[str]
    def complete(self) -> bool:
        return all(c in self.contrasts for c in CONTRASTS) and self.seg is not None


def discover(root: str) -> Dict[str, List[Timepoint]]:
    """Group case folders by patient id, sorted by timepoint letter."""
    pats: Dict[str, List[Timepoint]] = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or _is_junk(d):
            continue
        m = CASE_RE.match(os.path.basename(d))
        if not m:
            continue
        pid, tp = m.group(1), m.group(2)
        files = [f for f in glob.glob(os.path.join(d, "*.nii.gz")) if not _is_junk(f)]
        contrasts, seg = {}, None
        for f in files:
            b = os.path.basename(f)
            if "BraTS-seg" in b:
                seg = f
            else:
                for c in CONTRASTS:
                    if b.endswith(f"_{c}.nii.gz"):
                        contrasts[c] = f
        pats.setdefault(pid, []).append(
            Timepoint(pid, tp, d, contrasts, seg))
    for pid in pats:
        pats[pid].sort(key=lambda t: t.tp)
    return pats


@dataclass
class ALPTDGDataset(Dataset):
    """
    Forecasting-pair dataset. Each item = (baseline, follow-up) of one patient.

    intervals : optional {patient_id: {('A','B'): days, ...}} from clinical table.
                If None, dt is emitted as NaN and a warning is raised ONCE.
    keep_incomplete : if False, drop timepoints missing any contrast or the seg.
    """
    root: str
    intervals: Optional[Dict[str, Dict[Tuple[str, str], float]]] = None
    keep_incomplete: bool = False
    convention: str = "alptdg"
    pairs: List[Tuple[Timepoint, Timepoint]] = field(default_factory=list, init=False)
    _warned_dt: bool = field(default=False, init=False)

    def __post_init__(self):
        pats = discover(self.root)
        n_pat = len(pats)
        n_single, n_incomplete = 0, 0
        for pid, tps in pats.items():
            good = [t for t in tps if t.complete() or self.keep_incomplete]
            n_incomplete += len(tps) - len([t for t in tps if t.complete()])
            if len(good) < 2:
                n_single += 1
                continue
            for k in range(len(good) - 1):
                self.pairs.append((good[k], good[k + 1]))
        print(f"[ALPTDG] patients={n_pat}  forecasting_pairs={len(self.pairs)}  "
              f"single_timepoint_dropped={n_single}  incomplete_timepoints={n_incomplete}")
        if not self.pairs:
            warnings.warn("[ALPTDG] produced 0 pairs — check root path / naming.")

    def __len__(self) -> int:
        return len(self.pairs)

    def _stack(self, tp: Timepoint) -> np.ndarray:
        vols = [_load(tp.contrasts[c]).astype(np.float32) for c in CONTRASTS]
        return np.stack(vols, axis=0)   # [4,H,W,D]

    def _interval(self, a: Timepoint, b: Timepoint) -> float:
        if self.intervals and a.patient in self.intervals:
            dt = self.intervals[a.patient].get((a.tp, b.tp))
            if dt is not None:
                return float(dt)
        if not self._warned_dt:
            warnings.warn("[ALPTDG] no acquisition intervals supplied — dt=NaN. "
                          "Biophysics/interval-conditioning REQUIRE real dt; "
                          "load them from the clinical table before training.")
            self._warned_dt = True
        return float("nan")

    def __getitem__(self, i: int) -> Dict:
        a, b = self.pairs[i]
        x_a, x_b = self._stack(a), self._stack(b)
        s_a = harmonize(_load(a.seg), self.convention)
        s_b = harmonize(_load(b.seg), self.convention)
        return {
            "x_a": torch.from_numpy(x_a),                 # [4,H,W,D] baseline
            "x_b": torch.from_numpy(x_b),                 # [4,H,W,D] follow-up
            "s_a": torch.from_numpy(s_a).long(),          # [H,W,D] unified labels
            "s_b": torch.from_numpy(s_b).long(),
            "dt":  torch.tensor(self._interval(a, b), dtype=torch.float32),
            "patient": a.patient, "tp_a": a.tp, "tp_b": b.tp,
            "cohort": "alptdg", "label_quality": "gold",
        }


if __name__ == "__main__":
    # Point at your machine's ALPTDG root and smoke-test discovery + one item.
    ROOT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UCSF-ALPTDG/UCSF_BrainMetastases_TRAIN"
    ds = ALPTDGDataset(ROOT)
    if len(ds):
        s = ds[0]
        print("sample:", s["patient"], s["tp_a"], "->", s["tp_b"],
              "| x_a", tuple(s["x_a"].shape), "| s_b labels",
              torch.unique(s["s_b"]).tolist(), "| dt", float(s["dt"]))
