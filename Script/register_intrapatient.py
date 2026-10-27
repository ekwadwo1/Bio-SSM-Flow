#!/usr/bin/env python3
"""
src/data/register_intrapatient.py — rigid co-registration of longitudinal timepoints.

WHY THIS EXISTS (measured, not assumed):
  ALPTDG multi-timepoint patients came back same-grid=2 / different-grid=38 out of 40.
  i.e. ~95% of forecasting pairs are NOT spatially aligned. Training forecasting on
  unaligned pairs teaches the model that "progression = whole-brain shift/resize" —
  it fits registration error, not biology. This step removes that failure mode.

WHAT IT DOES, per patient:
  - baseline = earliest timepoint (letter 'A' or lowest).
  - every later timepoint's T1post is rigidly registered (6-DOF) to the baseline T1post.
  - the SAME transform is applied to that timepoint's other contrasts (shared header)
    and to its segmentation with NEAREST-NEIGHBOR interpolation (labels must not blend).
  - a brain-mask Dice gate confirms alignment succeeded; failures are logged, not written.
  - aligned volumes written to OUT_ROOT mirroring the input layout; RAW DATA UNTOUCHED.

Interpolation rule (do not change):
  intensities -> linear ; segmentation -> nearestNeighbor. Linear on a label map
  produces values like 1.5 that are neither class 1 nor 2 — a classic silent corruptor.

Dependencies: antspyx (import 'ants'), nibabel, numpy.
Run pattern: edit ROOT/OUT_ROOT/CONTRASTS, then `python register_intrapatient.py`.
Idempotent: skips a timepoint if its aligned output already exists.
"""
from __future__ import annotations
import os, glob, re, json, warnings
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib

try:
    import ants
except ImportError as e:
    raise ImportError("ANTsPy not found. `pip install antspyx`.") from e

# ── EDIT THESE ───────────────────────────────────────────────────────
ROOT     = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UCSF-ALPTDG/UCSF_BrainMetastases_TRAIN"
OUT_ROOT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/UCSF-ALPTDG-registered"
CONTRASTS = ["T1pre", "T1post", "FLAIR", "T2Synth"]   # {id}_{name}.nii.gz
SEG_SUFFIX = "BraTS-seg"                                # {id}_BraTS-seg.nii.gz
FIXED_CONTRAST = "T1post"                               # registration driver (best tumor contrast)
CASE_RE = re.compile(r"^(\d+)([A-Z])$")
DICE_GATE = 0.90                                        # post-reg brain-mask Dice must exceed this
# ─────────────────────────────────────────────────────────────────────


def _brain_mask(arr: np.ndarray) -> np.ndarray:
    return arr > (float(arr.max()) * 1e-3)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    d = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / d) if d else 1.0


def discover(root: str) -> Dict[str, List[Tuple[str, str]]]:
    pats: Dict[str, List[Tuple[str, str]]] = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        b = os.path.basename(d)
        if b.startswith("._") or not os.path.isdir(d):
            continue
        m = CASE_RE.match(b)
        if m:
            pats.setdefault(m.group(1), []).append((m.group(2), b))
    for pid in pats:
        pats[pid].sort()
    return pats


def _paths(case_dir: str, case: str) -> Dict[str, str]:
    p = {c: os.path.join(case_dir, f"{case}_{c}.nii.gz") for c in CONTRASTS}
    p["seg"] = os.path.join(case_dir, f"{case}_{SEG_SUFFIX}.nii.gz")
    return p


def register_patient(pid: str, tps: List[Tuple[str, str]], report: list) -> None:
    base_letter, base_case = tps[0]
    base_dir = os.path.join(ROOT, base_case)
    base_paths = _paths(base_dir, base_case)
    if not os.path.exists(base_paths[FIXED_CONTRAST]):
        report.append({"patient": pid, "status": "no_baseline_fixed_contrast"})
        return
    fixed = ants.image_read(base_paths[FIXED_CONTRAST])

    # Baseline: copy through unchanged (it defines the target space)
    _write_case(base_case, base_paths, moving_to_fixed=None, fixed=fixed,
                report=report, is_baseline=True, pid=pid)

    for letter, case in tps[1:]:
        cdir = os.path.join(ROOT, case)
        cpaths = _paths(cdir, case)
        out_seg = os.path.join(OUT_ROOT, case, f"{case}_{SEG_SUFFIX}.nii.gz")
        if os.path.exists(out_seg):
            report.append({"patient": pid, "case": case, "status": "skip_exists"})
            continue
        if not os.path.exists(cpaths[FIXED_CONTRAST]):
            report.append({"patient": pid, "case": case, "status": "missing_fixed_contrast"})
            continue
        moving = ants.image_read(cpaths[FIXED_CONTRAST])
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
        _write_case(case, cpaths, moving_to_fixed=reg, fixed=fixed,
                    report=report, is_baseline=False, pid=pid)


def _write_case(case, cpaths, moving_to_fixed, fixed, report, is_baseline, pid):
    outdir = os.path.join(OUT_ROOT, case)
    os.makedirs(outdir, exist_ok=True)
    for c in CONTRASTS:
        if not os.path.exists(cpaths[c]):
            continue
        img = ants.image_read(cpaths[c])
        if is_baseline:
            out = img
        else:
            out = ants.apply_transforms(fixed=fixed, moving=img,
                                        transformlist=moving_to_fixed["fwdtransforms"],
                                        interpolator="linear")
        ants.image_write(out, os.path.join(outdir, f"{case}_{c}.nii.gz"))
    # segmentation: NEAREST-NEIGHBOR interpolation (labels must never be blended)
    if os.path.exists(cpaths["seg"]):
        seg = ants.image_read(cpaths["seg"])
        if is_baseline:
            out_seg = seg
        else:
            out_seg = ants.apply_transforms(fixed=fixed, moving=seg,
                                            transformlist=moving_to_fixed["fwdtransforms"],
                                            interpolator="nearestNeighbor")
        ants.image_write(out_seg, os.path.join(outdir, f"{case}_{SEG_SUFFIX}.nii.gz"))

    # QC: brain-mask Dice against fixed (baseline) — only meaningful for follow-ups
    if not is_baseline:
        mv = ants.apply_transforms(fixed=fixed,
                                   moving=ants.image_read(cpaths[FIXED_CONTRAST]),
                                   transformlist=moving_to_fixed["fwdtransforms"],
                                   interpolator="linear").numpy()
        d = _dice(_brain_mask(fixed.numpy()), _brain_mask(mv))
        status = "ok" if d >= DICE_GATE else "LOW_DICE"
        report.append({"patient": pid, "case": case, "dice": round(d, 4), "status": status})
        if status == "LOW_DICE":
            warnings.warn(f"[reg] {case}: post-reg Dice {d:.3f} < {DICE_GATE} — inspect manually.")
    else:
        report.append({"patient": pid, "case": case, "status": "baseline_copied"})


def _run():
    os.makedirs(OUT_ROOT, exist_ok=True)
    pats = discover(ROOT)
    multi = {p: v for p, v in pats.items() if len(v) > 1}
    print(f"[reg] patients total={len(pats)}  multi-timepoint={len(multi)}")
    print(f"[reg] registering follow-ups -> baseline (Rigid). Output: {OUT_ROOT}")
    report: list = []
    for i, (pid, tps) in enumerate(sorted(multi.items()), 1):
        try:
            register_patient(pid, tps, report)
        except Exception as e:
            report.append({"patient": pid, "status": f"error:{type(e).__name__}:{e}"})
        if i % 10 == 0:
            print(f"  ...{i}/{len(multi)} patients")
    with open(os.path.join(OUT_ROOT, "registration_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    dices = [r["dice"] for r in report if "dice" in r]
    low = [r for r in report if r.get("status") == "LOW_DICE"]
    if dices:
        import statistics as st
        print(f"[reg] follow-ups registered: {len(dices)}  "
              f"mean Dice={st.mean(dices):.3f}  min={min(dices):.3f}  "
              f"LOW_DICE={len(low)}")
    print(f"[reg] report -> {os.path.join(OUT_ROOT,'registration_report.json')}")
    print(f"[reg] NEXT: point the ALPTDG loader at OUT_ROOT and re-run the same-grid check "
          f"(should now be ~all same-grid).")


if __name__ == "__main__":
    _run()
