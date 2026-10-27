#!/usr/bin/env python3
"""
diagnose_ext.py - inspect an external dataset (and LUMIERE for comparison)
before building the external-validation pipeline. Read-only: changes nothing.

Usage:
  python diagnose_ext.py --ext /REAL/path/to/ALPTDG --lum /REAL/path/to/LUMIERE 2>&1 | tee /tmp/diag_ext.txt
"""
import argparse, os, sys, json
import numpy as np

NIFTI = (".nii", ".nii.gz")
META = (".csv", ".tsv", ".xlsx", ".xls", ".json")


def hdr(t):
    print(f"\n{'=' * 22} {t} {'=' * 22}")


def walk_files(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(exts):
                out.append(os.path.join(dp, f))
    return sorted(out)


def show_tree(path, max_depth=3, max_lines=40):
    base = path.rstrip("/").count(os.sep)
    n = 0
    for dp, dns, fns in os.walk(path):
        depth = dp.rstrip("/").count(os.sep) - base
        if depth > max_depth:
            dns[:] = []
            continue
        print("  " * depth + os.path.basename(dp) + "/")
        n += 1
        for f in sorted(fns)[:15]:
            print("  " * (depth + 1) + f)
            n += 1
            if n >= max_lines:
                print("  ... (truncated)")
                return


def inspect_root(root, tag):
    hdr(f"{tag}: {root}")
    if not os.path.isdir(root):
        print(f"!! Path does not exist: {root}")
        return
    entries = sorted(os.listdir(root))
    print(f"top-level entries: {len(entries)} (first 20)")
    for e in entries[:20]:
        print("  ", e + ("/" if os.path.isdir(os.path.join(root, e)) else ""))

    subdirs = [os.path.join(root, e) for e in entries if os.path.isdir(os.path.join(root, e))]
    if subdirs:
        hdr(f"{tag}: tree of first subject/folder")
        show_tree(subdirs[0])

    files = walk_files(root, NIFTI)
    hdr(f"{tag}: NIfTI files = {len(files)}")
    n_met = sum("met" in os.path.basename(f).lower() for f in files)
    if n_met:
        print(f"!! WARNING: {n_met} filenames contain 'MET' - check this is not BraTS-Metastases")
    if not files:
        return

    import nibabel as nib
    first_dir = os.path.dirname(files[0])
    print(f"sample folder: {first_dir}")
    for f in sorted(p for p in files if os.path.dirname(p) == first_dir):
        try:
            im = nib.load(f)
            a = np.asarray(im.dataobj, dtype=np.float32)
            nz = a[a != 0]
            p99 = float(np.percentile(nz, 99)) if nz.size else 0.0
            print(f"  {os.path.basename(f):45s} shape={im.shape} "
                  f"spacing={tuple(round(float(z), 2) for z in im.header.get_zooms()[:3])} "
                  f"orient={''.join(nib.aff2axcodes(im.affine))} dtype={im.get_data_dtype()} "
                  f"min={a.min():.2f} max={a.max():.2f} p99(nonzero)={p99:.2f}")
        except Exception as e:
            print(f"  {os.path.basename(f)}: ERROR {e}")

    segs = [f for f in files if any(k in os.path.basename(f).lower() for k in ("seg", "label", "mask"))]
    hdr(f"{tag}: label files = {len(segs)} (showing up to 5)")
    for f in segs[:5]:
        try:
            v, c = np.unique(np.asarray(nib.load(f).dataobj), return_counts=True)
            print(f"  {os.path.basename(f)}: " + ", ".join(f"{int(x)}:{int(n)}" for x, n in zip(v, c)))
        except Exception as e:
            print(f"  {os.path.basename(f)}: ERROR {e}")

    metas = walk_files(root, META)
    hdr(f"{tag}: metadata files = {len(metas)} (showing up to 5)")
    for f in metas[:5]:
        print(f"\n### {f}")
        try:
            if f.lower().endswith(".json"):
                d = json.load(open(f))
                print("  keys:", list(d)[:30] if isinstance(d, dict) else f"list of {len(d)}")
            else:
                import pandas as pd
                if f.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(f)
                else:
                    df = pd.read_csv(f, sep="\t" if f.lower().endswith(".tsv") else ",")
                print("  shape:", df.shape)
                print("  columns:", df.columns.tolist())
                print(df.head(3).to_string())
        except Exception as e:
            print(f"  ERROR reading: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True, help="external dataset root (e.g. UCSF-ALPTDG)")
    ap.add_argument("--lum", required=True, help="LUMIERE images that were fed to the encoder")
    a = ap.parse_args()
    for p in (a.ext, a.lum):
        if p.startswith("/path/to"):
            sys.exit(f"!! '{p}' is still a placeholder - replace it with the real path.")
    inspect_root(a.ext, "EXT")
    inspect_root(a.lum, "LUM")
