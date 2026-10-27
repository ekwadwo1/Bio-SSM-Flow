#!/usr/bin/env python3
"""
01_inspect_data.py — Establish ground truth about the data BEFORE modeling.

Answers the questions we must NOT guess:
  (A) What segmentation label integers does each dataset actually use?
      (BraTS-2021, BraTS-2024-post, LUMIERE, ALPTDG all differ — we read, not assume.)
  (B) How many usable patients/timepoints per dataset?  (feasibility counts)
  (C) Are longitudinal timepoints already co-registered?  (brain-mask Dice)
      -> if Dice > 0.95, SKIP registration entirely.

Dependencies: numpy, nibabel  (SimpleITK optional, only for spacing niceties).
Jupyter-safe: edit ROOTS below, then run.  No argparse.

NOTE ON LAYOUTS: folder structures vary. This script is deliberately robust —
it recursively finds every '*seg*.nii.gz' under a root and groups by a patient
regex. Adjust the regex per dataset if grouping looks wrong in the output.
"""
from __future__ import annotations
import glob, os, re, collections
import numpy as np
import nibabel as nib

# ======================================================================
# EDIT THESE to your machine's paths. Point at the EXTRACTED dirs
# (unzip BraTS-2024 first:  unzip BraTS2024-BraTS-GLI-TrainingData.zip -d <dir>)
# ======================================================================
ROOTS = {
    "BraTS-GLI-post":  {"root": "/path/to/BraTS2024-GLI-Training-extracted",
                        "pid_re": r"(BraTS-GLI-\d{5})-(\d{3})"},
    "UCSF-ALPTDG":     {"root": "/path/to/UCSF-ALPTDG",
                        "pid_re": r"(\d{6})([A-Z])"},          # 100101A -> pid 100101, tp A
    "LUMIERE":         {"root": "/path/to/LUMIERE/Imaging",
                        "pid_re": r"(Patient-\d+)"},           # adjust to real LUMIERE naming
    "UCSD-PTGBM":      {"root": "/path/to/UCSD-PTGBM",
                        "pid_re": r"(sub-\w+?)_?(ses-\w+)?"},  # adjust to real naming
}
MODALITY_HINTS = ["t1c", "t1n", "t1ce", "t1", "t2w", "t2", "t2f", "flair"]
MAX_CASES_PER_DATASET = 400   # cap for a fast first pass; raise later


def find_segs(root):
    pats = ["*seg*.nii.gz", "*Seg*.nii.gz", "*-seg.nii*"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(root, "**", p), recursive=True)
    return sorted(set(out))


def group_by_patient(segs, pid_re):
    rx = re.compile(pid_re)
    groups = collections.defaultdict(list)
    for s in segs:
        m = rx.search(os.path.basename(s)) or rx.search(s)
        if not m:
            groups["__UNMATCHED__"].append(s); continue
        pid = m.group(1)
        tp = m.group(2) if m.lastindex and m.lastindex >= 2 else "0"
        groups[pid].append((tp, s))
    for pid in groups:
        if pid != "__UNMATCHED__":
            groups[pid] = sorted(groups[pid])
    return groups


def label_histogram(seg_path, max_voxels=None):
    arr = np.asanyarray(nib.load(seg_path).dataobj)
    vals, counts = np.unique(arr, return_counts=True)
    return dict(zip(vals.astype(int).tolist(), counts.astype(int).tolist()))


def modality_for(seg_path):
    """Find a co-located intensity volume to build a brain mask."""
    d = os.path.dirname(seg_path)
    cands = glob.glob(os.path.join(d, "*.nii.gz"))
    for hint in MODALITY_HINTS:
        for c in cands:
            b = os.path.basename(c).lower()
            if hint in b and "seg" not in b:
                return c
    return None


def brain_mask(vol_path):
    """BraTS-style data is skull-stripped -> nonzero = brain. Good enough for a Dice check."""
    arr = np.asanyarray(nib.load(vol_path).dataobj).astype(np.float32)
    return arr > (arr.max() * 1e-3)


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    if a.shape != b.shape:
        return None  # different grids -> NOT co-registered, flag it
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom else 1.0


def inspect(name, cfg):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    root = cfg["root"]
    if not os.path.isdir(root):
        print(f"  [skip] root not found: {root}  (edit ROOTS)")
        return
    segs = find_segs(root)
    print(f"  seg files found: {len(segs)}")
    if not segs:
        print("  [skip] no '*seg*.nii.gz' under root — check path/layout.")
        return
    groups = group_by_patient(segs[:MAX_CASES_PER_DATASET], cfg["pid_re"])
    unmatched = groups.pop("__UNMATCHED__", [])
    n_pat = len(groups)
    tps = [len(v) for v in groups.values()]
    print(f"  patients (grouped): {n_pat}")
    if tps:
        print(f"  timepoints/patient: min {min(tps)}  median {int(np.median(tps))}  max {max(tps)}")
    print(f"  patients w/ >=3 timepoints (biophysics-eligible): {sum(t>=3 for t in tps)}")
    print(f"  patients w/ >=2 timepoints (forecasting pairs):   {sum(t>=2 for t in tps)}")
    if unmatched:
        print(f"  !! {len(unmatched)} seg files did NOT match pid_re — fix regex. e.g.:")
        print(f"     {os.path.basename(unmatched[0])}")

    # (A) LABEL INTEGERS — aggregated across up to 30 segs
    print(f"\n  --- (A) actual label integers (aggregated) ---")
    agg = collections.Counter()
    sample = [s for grp in groups.values() for _, s in grp][:30] if tps else segs[:30]
    sample = [s if isinstance(s, str) else s for s in sample]
    for s in sample:
        try:
            for k, v in label_histogram(s).items():
                agg[k] += v
        except Exception as e:
            print(f"     (read error {os.path.basename(s)}: {e})")
    total = sum(agg.values()) or 1
    for k in sorted(agg):
        print(f"     label {k:>2d}:  {100*agg[k]/total:6.3f}%  of voxels")
    print(f"     >> RECORD THESE. They define this dataset's convention for the crosswalk.")

    # (B/C) spacing/shape + intra-patient registration check
    print(f"\n  --- (B/C) geometry + intra-patient co-registration (mask Dice) ---")
    dices, shape_set = [], set()
    checked = 0
    for pid, tp_list in groups.items():
        if len(tp_list) < 2:
            continue
        vols = [modality_for(s) for _, s in tp_list]
        vols = [v for v in vols if v]
        if len(vols) < 2:
            continue
        try:
            m0 = brain_mask(vols[0])
            shape_set.add(m0.shape)
            for v in vols[1:]:
                mk = brain_mask(v)
                shape_set.add(mk.shape)
                d = dice(m0, mk)
                if d is None:
                    print(f"     {pid}: timepoints on DIFFERENT grids -> NOT co-registered")
                else:
                    dices.append(d)
            checked += 1
        except Exception as e:
            print(f"     ({pid} check error: {e})")
        if checked >= 25:
            break
    if dices:
        dd = np.array(dices)
        print(f"     brain-mask Dice across timepoints:  mean {dd.mean():.3f}  min {dd.min():.3f}  (n={len(dd)})")
        if dd.mean() > 0.95:
            print(f"     >> ALREADY CO-REGISTERED. Skip registration for {name}.")
        else:
            print(f"     >> NOT reliably aligned. {name} needs rigid intra-patient registration (ANTsPy).")
    else:
        print(f"     (no multi-timepoint pairs with intensity volumes found to test)")
    if len(shape_set) > 1:
        print(f"     !! multiple volume shapes seen {sorted(shape_set)} -> resample to common grid.")
    elif shape_set:
        print(f"     common shape: {shape_set.pop()}")


def _run():
    print("BIO-SSM-FLOW :: data ground-truth inspection")
    print("Edit ROOTS paths at top, then re-run. Report the label-integer blocks back.")
    for name, cfg in ROOTS.items():
        try:
            inspect(name, cfg)
        except Exception as e:
            print(f"  [error] {name}: {type(e).__name__}: {e}")
    print(f"\n{'='*72}\n  DONE. Next: paste the (A) label blocks + (C) Dice lines back to me.\n{'='*72}")


if __name__ == "__main__":
    _run()

