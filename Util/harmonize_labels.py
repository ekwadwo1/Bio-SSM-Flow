#!/usr/bin/env python3
"""
src/data/harmonize_labels.py — unify 4 label conventions into ONE space.

Confirmed-from-disk integer conventions (do NOT edit from memory — these were
read off the actual .nii.gz files on 2026-08-30):

  BraTS-2021 (Paper 1)         : {0,1,2,4}  = bg, NCR,  ED,   ET          (no 3, no RC)
  BraTS-2024 post-treatment    : {0,1,2,3,4}= bg, NETC, SNFH, ET,  RC     (RC only here)
  UCSF-ALPTDG                  : {0,1,2,3}  = bg, ?,    ?,    ?           (NO label 4)
  LUMIERE / DeepBraTumIA       : {0,1,2,3}  = bg, ?,    ?,    ?           (VERIFY meaning)

UNIFIED TARGET SPACE (4 classes — RC dropped because only 1 of 4 datasets has it):
  0 = background
  1 = non-enhancing tumor core   (NETC / NCR / necrotic)
  2 = non-enhancing FLAIR hyperintensity / edema (SNFH / ED)
  3 = enhancing tumor            (ET)
  (RC handled separately, ONLY for BraTS-GLI, via KEEP_RC flag.)

IMPORTANT UNRESOLVED ITEM: the *semantic* order of ALPTDG's {1,2,3} and
LUMIERE's {1,2,3} is ASSUMED to follow the BraTS ET/ED/NCR ordering below.
This assumption is flagged loudly and must be visually confirmed on a few
cases (overlay label 3 on T1c — it should sit on the enhancing rim) before
you trust cross-dataset training. See verify_semantic_order().
"""
from __future__ import annotations
import os, glob
from typing import Dict, List
import numpy as np
import nibabel as nib

# ── Unified class ids ────────────────────────────────────────────────
BG, NETC, SNFH, ET, RC = 0, 1, 2, 3, 4   # RC only used when KEEP_RC=True

# ── Per-dataset raw->unified integer maps (from confirmed conventions) ──
# NOTE: ALPTDG/LUMIERE orderings are the ASSUMED BraTS-style mapping.
#       Confirm with verify_semantic_order() before trusting.
CROSSWALK: Dict[str, Dict[int, int]] = {
    "brats2021":  {0: BG, 1: NETC, 2: SNFH, 4: ET},                 # NCR->NETC, ED->SNFH, ET->ET
    "brats2024":  {0: BG, 1: NETC, 2: SNFH, 3: ET, 4: RC},          # native target scheme
    "alptdg":     {0: BG, 1: NETC, 2: SNFH, 3: ET},                 # NOTE: this folder turned out to be BraTS-MET (metastases), not glioma — not used in the glioma pipeline
    "lumiere":    {0: BG, 1: ET, 2: NETC, 3: SNFH},                 # VERIFIED 2026-08-30: DeepBraTumIA order is 1=enhancing (T1c-brightest), 3=edema (largest, FLAIR-bright), 2=necrotic core. NOT the BraTS order.
}

# Expected raw label set per dataset — asserts we never see an unmapped id.
EXPECTED_RAW: Dict[str, set] = {
    "brats2021": {0, 1, 2, 4},
    "brats2024": {0, 1, 2, 3, 4},
    "alptdg":    {0, 1, 2, 3},
    "lumiere":   {0, 1, 2, 3},
}


def harmonize(seg: np.ndarray, convention: str, keep_rc: bool = False) -> np.ndarray:
    """
    Map a raw segmentation array to the unified label space.

    Parameters
    ----------
    seg        : integer label volume (any shape)
    convention : one of CROSSWALK keys
    keep_rc    : if False, any RC (unified id 4) is folded to background.
                 RC is only meaningful for brats2024; for all training
                 datasets keep_rc=False so the space is a clean 4 classes.

    Raises if an unexpected raw label id is encountered (guards silent
    convention drift, e.g. a stray label from a bad file).
    """
    if convention not in CROSSWALK:
        raise KeyError(f"unknown convention {convention!r}; "
                       f"expected one of {list(CROSSWALK)}")
    seg = np.asarray(seg)
    raw = set(np.unique(seg).astype(int).tolist())
    unexpected = raw - EXPECTED_RAW[convention]
    if unexpected:
        raise ValueError(
            f"[{convention}] unexpected raw labels {sorted(unexpected)} "
            f"(expected subset of {sorted(EXPECTED_RAW[convention])}). "
            f"A corrupt file or wrong convention tag is likely.")

    out = np.zeros_like(seg, dtype=np.int64)
    for raw_id, uni_id in CROSSWALK[convention].items():
        if uni_id == RC and not keep_rc:
            uni_id = BG          # fold resection cavity into background
        out[seg == raw_id] = uni_id
    return out


def _read_seg(path: str) -> np.ndarray:
    if os.path.basename(path).startswith("._"):
        raise ValueError(f"AppleDouble junk file passed to reader: {path}")
    return np.asanyarray(nib.load(path).dataobj).astype(np.int64)


def audit_convention(seg_paths: List[str], convention: str,
                     n: int = 60) -> None:
    """
    Sanity report: aggregate raw label frequencies across n files and
    confirm they fall within EXPECTED_RAW. Run once per dataset.
    """
    paths = [p for p in seg_paths if not os.path.basename(p).startswith("._")]
    from collections import Counter
    agg = Counter()
    for p in paths[:n]:
        for k, v in zip(*np.unique(_read_seg(p), return_counts=True)):
            agg[int(k)] += int(v)
    tot = sum(agg.values()) or 1
    print(f"[{convention}] audited {min(n,len(paths))} files "
          f"({len(paths)} available)")
    for k in sorted(agg):
        tag = "" if k in EXPECTED_RAW[convention] else "  <-- UNEXPECTED!"
        print(f"    raw {k}: {100*agg[k]/tot:6.3f}%{tag}")
    stray = set(agg) - EXPECTED_RAW[convention]
    if stray:
        raise ValueError(f"[{convention}] stray labels {sorted(stray)} — fix before training.")
    print(f"    -> OK, matches expected {sorted(EXPECTED_RAW[convention])}")


def verify_semantic_order(t1c_path: str, seg_path: str, convention: str) -> Dict:
    """
    THE assumption-buster. For a case, report the mean T1c intensity under
    each raw label. Enhancing tumor (mapped to ET=3) MUST have the highest
    mean T1c intensity. If a different raw id is brightest, the assumed
    order in CROSSWALK is WRONG for that dataset and must be corrected.
    """
    t1c = np.asanyarray(nib.load(t1c_path).dataobj).astype(np.float32)
    seg = _read_seg(seg_path)
    report = {}
    for raw_id in sorted(set(np.unique(seg).astype(int)) - {0}):
        m = seg == raw_id
        report[raw_id] = float(t1c[m].mean()) if m.any() else float("nan")
    brightest = max(report, key=report.get) if report else None
    et_raw = [k for k, v in CROSSWALK[convention].items() if v == ET]
    et_raw = et_raw[0] if et_raw else None
    ok = (brightest == et_raw)
    print(f"[{convention}] mean T1c by raw label: "
          + ", ".join(f"{k}:{v:.1f}" for k, v in report.items()))
    print(f"    brightest raw label = {brightest}; "
          f"CROSSWALK maps raw {et_raw} -> ET(3)")
    print(f"    {'OK — order consistent' if ok else '!! MISMATCH — fix CROSSWALK order for this dataset'}")
    return {"means": report, "brightest": brightest, "assumed_ET_raw": et_raw, "ok": ok}


if __name__ == "__main__":
    # Smoke test on synthetic labels (no data dependency)
    for conv, raws in EXPECTED_RAW.items():
        fake = np.array(sorted(raws), dtype=np.int64)
        out = harmonize(fake, conv, keep_rc=(conv == "brats2024"))
        print(f"{conv:10s} {fake.tolist()} -> {out.tolist()}")
    print("\nharmonize() smoke test passed.")
    print("NEXT: run audit_convention() and verify_semantic_order() on real files.")
