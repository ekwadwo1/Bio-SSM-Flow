#!/usr/bin/env python3
"""
Phase-0 shape audit for polar / harmonic OD-OC segmentation.

Runs two experiments on GROUND-TRUTH MASKS ONLY (no fundus images, no GPU):

  P0-1  Representation ceiling
        Extract the boundary radius function r(theta) about a polar anchor,
        truncate its Fourier series to K harmonics, re-render the mask, and
        measure Dice against the original. This upper-bounds what any
        harmonic-parameterised model can achieve.

  P0-2  Star-convexity audit
        For each ray from the anchor, check whether the occupancy profile is
        radially monotone (a run of 1s then 0s). Reports the fraction of rays
        and the fraction of pixels that violate star-convexity, plus the
        "star-hull ceiling" -- the Dice you lose to non-star-convexity alone,
        before any band-limiting.

  Bonus  Nesting audit: fraction of images where the cup is fully inside the
         disc in the ground truth, and the mean radial violation.

Usage
-----
    python phase0_shape_audit.py --masks /path/to/Drishti-GS --convention auto
    python phase0_shape_audit.py --masks /path/REFUGE --convention refuge --name REFUGE
    python phase0_shape_audit.py --masks A --masks B --outdir results

Outputs
-------
    <outdir>/<name>_per_image.csv    one row per image
    <outdir>/<name>_summary.csv      mean/median per K
    <outdir>/<name>_ceiling.png      Dice vs K curve
    printed summary table

Dependencies: numpy, scipy, pillow, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

IMG_EXTS = {".png", ".tif", ".tiff", ".bmp", ".gif", ".jpg", ".jpeg"}

DEFAULT_K_VALUES = [2, 4, 6, 8, 12, 16, 24, 32, 48]

N_THETA = 720   # angular samples of r(theta); 0.5 deg resolution
N_RHO = 512     # radial samples per ray when probing occupancy

# Filename patterns used to decide whether a mask file is cup or disc.
# Checked in order; first match wins.
CUP_PATTERNS = [r"cup", r"(?<![a-z])oc(?![a-z])", r"opticcup"]
DISC_PATTERNS = [r"disc", r"disk", r"(?<![a-z])od(?![a-z])",
                 r"opticdisc", r"(?<![a-z])onh(?![a-z])"]


# --------------------------------------------------------------------------
# Mask loading
# --------------------------------------------------------------------------

#: Words that commonly trail a role token ("ODseg", "cupSoftmap", "OD_mask").
#: Replacing them with a separator turns "odseg" into "od_", so that plain
#: word-boundary matching on "od" / "oc" works without false positives on
#: words like "model" or "product".
SUFFIX_WORDS = r"(seg|softmap|soft|map|mask|annotation|annot|label|gt|ann|contour|boundary)"


def _classify_by_name(name: str) -> str | None:
    """Return 'cup', 'disc', or None based on the filename."""
    low = Path(name).stem.lower()
    low = re.sub(SUFFIX_WORDS, "_", low)
    low = re.sub(r"[^a-z0-9]+", "_", low)
    for pat in CUP_PATTERNS:
        if re.search(pat, low):
            return "cup"
    for pat in DISC_PATTERNS:
        if re.search(pat, low):
            return "disc"
    return None


def _stem_key(path: Path) -> str:
    """Strip common cup/disc suffixes so cup and disc files of the same case pair up."""
    s = path.stem.lower()
    for pat in [
        r"[_\-]?cupsegsoftmap", r"[_\-]?odsegsoftmap",
        r"[_\-]?cupseg", r"[_\-]?odseg", r"[_\-]?discseg",
        r"[_\-]?cup", r"[_\-]?disc", r"[_\-]?disk",
        r"[_\-]?oc", r"[_\-]?od",
        r"[_\-]?mask", r"[_\-]?gt", r"[_\-]?seg",
        r"[_\-]?expert\d*", r"[_\-]?rater\d*", r"[_\-]?avg",
    ]:
        s = re.sub(pat + r"$", "", s)
    return s


def _read_gray(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr


def _decode_multilabel(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Decode a single-file multi-label mask.

    REFUGE convention: 0 = cup, 128 = neuroretinal rim, 255 = background.
    So disc = (v <= 128), cup = (v == 0). Some exports invert this.
    Returns (disc, cup) as bool arrays, or None if the file does not look
    like a 3-level mask.
    """
    vals = np.unique(arr)
    if len(vals) != 3:
        return None
    lo, mid, hi = sorted(vals.tolist())
    # Background is whichever level occupies the most pixels (it always does).
    counts = {v: int((arr == v).sum()) for v in (lo, mid, hi)}
    bg = max(counts, key=counts.get)
    if bg == hi:                       # dark = foreground (REFUGE)
        disc = arr <= mid
        cup = arr <= lo
    elif bg == lo:                     # bright = foreground (inverted)
        disc = arr >= mid
        cup = arr >= hi
    else:
        return None
    return disc, cup


def _binarise(arr: np.ndarray) -> np.ndarray:
    """Threshold a binary or soft-map mask at the midpoint of its range."""
    a = arr.astype(np.float32)
    if a.max() <= 1.0 + 1e-6:
        return a >= 0.5
    return a >= (a.max() / 2.0)


def _clean(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill interior holes."""
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(mask)


@dataclass
class Case:
    key: str
    disc: np.ndarray | None
    cup: np.ndarray | None
    n_components_disc: int = 1
    n_components_cup: int = 1


def load_cases(root: Path, convention: str) -> list[Case]:
    """Walk a directory and assemble (disc, cup) mask pairs."""
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No image files found under {root}")

    buckets: dict[str, dict[str, np.ndarray]] = {}
    raw_components: dict[str, dict[str, int]] = {}

    for f in files:
        try:
            arr = _read_gray(f)
        except Exception as e:                      # noqa: BLE001
            print(f"  ! skipping unreadable {f.name}: {e}", file=sys.stderr)
            continue

        key = _stem_key(f)
        role = _classify_by_name(f.name) if convention != "refuge" else None

        if role is None:
            dec = _decode_multilabel(arr)
            if dec is not None:
                disc, cup = dec
                b = buckets.setdefault(key, {})
                rc = raw_components.setdefault(key, {})
                for nm, m in (("disc", disc), ("cup", cup)):
                    rc[nm] = int(ndimage.label(m)[1])
                    b[nm] = _clean(m)
                continue
            if convention == "auto":
                # Unlabelled binary file with no role hint: cannot place it.
                continue

        if role is None:
            continue

        m = _binarise(arr)
        b = buckets.setdefault(key, {})
        rc = raw_components.setdefault(key, {})
        rc[role] = int(ndimage.label(m)[1])
        b[role] = _clean(m)

    cases = []
    for key, b in sorted(buckets.items()):
        rc = raw_components.get(key, {})
        cases.append(
            Case(
                key=key,
                disc=b.get("disc"),
                cup=b.get("cup"),
                n_components_disc=rc.get("disc", 0),
                n_components_cup=rc.get("cup", 0),
            )
        )
    return [c for c in cases if c.disc is not None or c.cup is not None]


# --------------------------------------------------------------------------
# Polar geometry
# --------------------------------------------------------------------------

def centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def occupancy_profiles(mask: np.ndarray, cx: float, cy: float,
                       n_theta: int = N_THETA, n_rho: int = N_RHO
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Sample the mask along n_theta rays from (cx, cy).

    Returns
      prof  (n_theta, n_rho) bool occupancy, sampled at rho in (0, rho_max]
      rho   (n_rho,) the radii sampled, in pixels
      rho_max
    """
    h, w = mask.shape
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    # Max distance from the anchor to any image corner, so rays always exit.
    rho_max = float(np.hypot(max(cx, w - 1 - cx), max(cy, h - 1 - cy))) + 1.0
    rho = np.linspace(rho_max / n_rho, rho_max, n_rho)

    xs = cx + rho[None, :] * np.cos(theta)[:, None]
    ys = cy + rho[None, :] * np.sin(theta)[:, None]

    xi = np.rint(xs).astype(np.int64)
    yi = np.rint(ys).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)

    prof = mask[yi, xi] & inside
    return prof, rho, rho_max


def radius_from_profile(prof: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Outermost occupied radius per ray -- i.e. the star-hull boundary r(theta)."""
    any_hit = prof.any(axis=1)
    # index of last True along each row
    last = prof.shape[1] - 1 - np.argmax(prof[:, ::-1], axis=1)
    r = np.where(any_hit, rho[last], 0.0)
    return r


def star_convexity_stats(prof: np.ndarray, d_rho: float,
                         min_gap_px: float = 2.0,
                         min_run_px: float = 2.0) -> float:
    """
    Fraction of rays that are genuinely NOT radially monotone.

    A ray is monotone iff its occupancy profile is 1...1 0...0 with no
    re-entry. Nearest-neighbour sampling of a jagged rasterised boundary
    produces spurious one-sample gaps, so a re-entry only counts when the
    gap and the re-entered run each exceed a physical pixel tolerance.
    Set min_gap_px = min_run_px = 0 for the raw, intolerant count.
    """
    n_rays, n_rho = prof.shape
    if n_rays == 0:
        return 0.0
    min_gap = max(1, int(np.ceil(min_gap_px / d_rho)))
    min_run = max(1, int(np.ceil(min_run_px / d_rho)))

    violating = 0
    for i in range(n_rays):
        p = prof[i]
        if not p.any():
            continue
        # Run-length encode the ray.
        edges = np.flatnonzero(np.diff(p.astype(np.int8))) + 1
        bounds = np.concatenate(([0], edges, [n_rho]))
        runs = [(bounds[j], bounds[j + 1], bool(p[bounds[j]]))
                for j in range(len(bounds) - 1)]
        # Find the first exit, i.e. the first zero-run that follows a one-run.
        seen_one = False
        gap_len = 0
        bad = False
        for start, stop, val in runs:
            length = stop - start
            if val:
                if seen_one and gap_len >= min_gap and length >= min_run:
                    bad = True
                    break
                seen_one = True
                gap_len = 0
            else:
                if seen_one:
                    gap_len += length
        if bad:
            violating += 1
    return float(violating / n_rays)


def render_from_radius(r_theta: np.ndarray, shape: tuple[int, int],
                       cx: float, cy: float) -> np.ndarray:
    """Rasterise {(rho, theta) : rho <= r(theta)} back into image space."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    rho = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)                       # (-pi, pi]
    n = len(r_theta)
    # r_theta is sampled on linspace(-pi, pi, n, endpoint=False); interpolate periodically.
    idx = (th + np.pi) / (2 * np.pi) * n
    i0 = np.floor(idx).astype(np.int64) % n
    i1 = (i0 + 1) % n
    frac = idx - np.floor(idx)
    r_at = r_theta[i0] * (1 - frac) + r_theta[i1] * frac
    return rho <= r_at


def truncate_harmonics(r_theta: np.ndarray, K: int) -> np.ndarray:
    """Keep the DC term plus the first K harmonics of r(theta)."""
    F = np.fft.rfft(r_theta)
    F[K + 1:] = 0
    out = np.fft.irfft(F, n=len(r_theta))
    return np.maximum(out, 0.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / s)


# --------------------------------------------------------------------------
# Per-case analysis
# --------------------------------------------------------------------------

def analyse_structure(mask: np.ndarray, cx: float, cy: float,
                      k_values: list[int],
                      min_gap_px: float = 2.0,
                      min_run_px: float = 2.0) -> dict:
    """Run P0-1 and P0-2 for one structure about a given anchor."""
    prof, rho, _ = occupancy_profiles(mask, cx, cy)
    r = radius_from_profile(prof, rho)
    d_rho = float(rho[1] - rho[0])

    ray_viol = star_convexity_stats(prof, d_rho, min_gap_px, min_run_px)

    # Star-hull ceiling: what you lose to non-star-convexity alone (K = infinity).
    # The hull always contains the mask, so the violation is the extra area
    # the hull is forced to include.
    hull = render_from_radius(r, mask.shape, cx, cy)
    extra = float(np.logical_and(hull, ~mask).sum() / max(mask.sum(), 1))
    out = {
        "star_ray_violation": ray_viol,
        "star_area_violation": extra,
        "star_convex": bool(ray_viol == 0.0),
        "dice_starhull": dice(hull, mask),
    }
    for K in k_values:
        rk = truncate_harmonics(r, K)
        out[f"dice_K{K}"] = dice(render_from_radius(rk, mask.shape, cx, cy), mask)
    return out


def nesting_stats(disc: np.ndarray, cup: np.ndarray) -> dict:
    if disc is None or cup is None or cup.sum() == 0:
        return {}
    escaped = np.logical_and(cup, ~disc).sum()
    return {
        "cup_nested": bool(escaped == 0),
        "cup_escape_frac": float(escaped / cup.sum()),
    }


def run(root: Path, name: str, convention: str, k_values: list[int],
        limit: int | None, min_gap_px: float = 2.0, min_run_px: float = 2.0,
        jitter: list[float] | None = None, seed: int = 0) -> pd.DataFrame:
    print(f"\n=== {name} ===")
    cases = load_cases(root, convention)
    if limit:
        cases = cases[:limit]
    print(f"  {len(cases)} case(s) assembled from {root}")
    if not cases:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for i, c in enumerate(cases, 1):
        if i % 25 == 0 or i == len(cases):
            print(f"  ... {i}/{len(cases)}")
        row: dict = {"dataset": name, "case": c.key}

        # Polar anchor: the DISC centroid, used for BOTH structures.
        # This mirrors the single-anchor framework; a cup-centroid variant is
        # also recorded so you can see how much the shared anchor costs.
        anchor = None
        if c.disc is not None and c.disc.sum() > 0:
            anchor = centroid(c.disc)
        elif c.cup is not None and c.cup.sum() > 0:
            anchor = centroid(c.cup)
        if anchor is None:
            continue
        cx, cy = anchor

        tol = dict(min_gap_px=min_gap_px, min_run_px=min_run_px)

        if c.disc is not None and c.disc.sum() > 0:
            for k, v in analyse_structure(c.disc, cx, cy, k_values, **tol).items():
                row[f"disc_{k}"] = v
            row["disc_area"] = int(c.disc.sum())
            row["disc_raw_components"] = c.n_components_disc

        if c.cup is not None and c.cup.sum() > 0:
            for k, v in analyse_structure(c.cup, cx, cy, k_values, **tol).items():
                row[f"cup_{k}"] = v
            # Same structure, but anchored on its own centroid.
            ccx, ccy = centroid(c.cup)
            own = analyse_structure(c.cup, ccx, ccy, k_values, **tol)
            row["cupown_dice_starhull"] = own["dice_starhull"]
            for K in k_values:
                row[f"cupown_dice_K{K}"] = own[f"dice_K{K}"]
            row["cup_area"] = int(c.cup.sum())
            row["cup_raw_components"] = c.n_components_cup

        # Anchor sensitivity: how much does a mislocalised ONH centre cost?
        for j in (jitter or []):
            ang = rng.uniform(0, 2 * np.pi)
            jx, jy = cx + j * np.cos(ang), cy + j * np.sin(ang)
            for struct, m in (("disc", c.disc), ("cup", c.cup)):
                if m is None or m.sum() == 0:
                    continue
                res = analyse_structure(m, jx, jy, [16], **tol)
                row[f"{struct}_jit{int(j)}_dice_K16"] = res["dice_K16"]
                row[f"{struct}_jit{int(j)}_star_convex"] = res["star_convex"]

        row.update(nesting_stats(c.disc, c.cup))
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarise(df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    recs = []
    for struct in ("disc", "cup", "cupown"):
        col_hull = f"{struct}_dice_starhull"
        if col_hull not in df.columns:
            continue
        for K in k_values:
            col = f"{struct}_dice_K{K}"
            if col not in df.columns:
                continue
            s = df[col].dropna()
            recs.append({
                "structure": struct,
                "K": K,
                "n": len(s),
                "dice_mean": s.mean(),
                "dice_median": s.median(),
                "dice_p5": s.quantile(0.05),
                "dice_min": s.min(),
            })
        s = df[col_hull].dropna()
        recs.append({
            "structure": struct, "K": "inf (star hull)", "n": len(s),
            "dice_mean": s.mean(), "dice_median": s.median(),
            "dice_p5": s.quantile(0.05), "dice_min": s.min(),
        })
    return pd.DataFrame(recs)


def print_report(df: pd.DataFrame, k_values: list[int], name: str) -> None:
    print(f"\n--- {name}: P0-2 star-convexity audit ---")
    for struct in ("disc", "cup"):
        col = f"{struct}_star_convex"
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(bool)
        ray = df[f"{struct}_star_ray_violation"].dropna()
        area = df[f"{struct}_star_area_violation"].dropna()
        print(f"  {struct.upper():5s}  star-convex about disc centroid: "
              f"{100 * s.mean():6.2f}% of {len(s)} masks")
        print(f"         mean violating rays: {100 * ray.mean():.3f}%   "
              f"mean extra area if hulled: {100 * area.mean():.4f}% "
              f"(worst {100 * area.max():.3f}%)")

    if "cup_nested" in df.columns:
        s = df["cup_nested"].dropna().astype(bool)
        esc = df["cup_escape_frac"].dropna()
        print(f"  NEST   cup fully inside disc in GT: {100 * s.mean():6.2f}% "
              f"of {len(s)} pairs (mean escape {100 * esc.mean():.4f}% of cup)")

    print(f"\n--- {name}: P0-1 harmonic reconstruction ceiling (mean Dice) ---")
    header = "  struct  " + "".join(f"K={K:<3d} " for K in k_values) + " hull"
    print(header)
    for struct in ("disc", "cup", "cupown"):
        if f"{struct}_dice_starhull" not in df.columns:
            continue
        cells = []
        for K in k_values:
            c = f"{struct}_dice_K{K}"
            cells.append(f"{df[c].mean():.4f}" if c in df.columns else "  -  ")
        hull = df[f"{struct}_dice_starhull"].mean()
        label = "cup*" if struct == "cupown" else struct
        print(f"  {label:7s} " + " ".join(cells) + f"  {hull:.4f}")
    if "cupown_dice_starhull" in df.columns:
        print("  (cup* = cup anchored on its own centroid, for comparison)")

    jit_cols = sorted({int(m.group(1)) for c in df.columns
                       if (m := re.search(r"_jit(\d+)_dice_K16$", c))})
    if jit_cols:
        print(f"\n--- {name}: anchor sensitivity (Dice at K=16, star-convex rate) ---")
        for struct in ("disc", "cup"):
            base = f"{struct}_dice_K16"
            if base not in df.columns:
                continue
            parts = [f"0px {df[base].mean():.4f}"]
            for j in jit_cols:
                c = f"{struct}_jit{j}_dice_K16"
                sc = f"{struct}_jit{j}_star_convex"
                if c in df.columns:
                    rate = 100 * df[sc].dropna().astype(bool).mean()
                    parts.append(f"{j}px {df[c].mean():.4f} ({rate:.0f}% SC)")
            print(f"  {struct.upper():5s} " + "   ".join(parts))

    # Verdict against the thresholds in the proposal.
    print(f"\n--- {name}: verdict at K=16 ---")
    for struct, thr in (("disc", 0.98), ("cup", 0.96)):
        c = f"{struct}_dice_K16"
        if c not in df.columns:
            continue
        m = df[c].mean()
        p5 = df[c].quantile(0.05)
        verdict = "PASS" if m >= thr else "BELOW TARGET"
        print(f"  {struct.upper():5s} mean {m:.4f} (target >= {thr})  "
              f"5th pct {p5:.4f}   -> {verdict}")


def plot_ceiling(df: pd.DataFrame, k_values: list[int], out: Path, name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception as e:                          # noqa: BLE001
        print(f"  (skipping plot: {e})")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for struct, style in (("disc", "-o"), ("cup", "-s")):
        cols = [f"{struct}_dice_K{K}" for K in k_values]
        if cols[0] not in df.columns:
            continue
        means = [df[c].mean() for c in cols]
        p5 = [df[c].quantile(0.05) for c in cols]
        ax.plot(k_values, means, style, label=f"{struct.upper()} mean")
        ax.fill_between(k_values, p5, means, alpha=0.15)
        hull = df[f"{struct}_dice_starhull"].mean()
        ax.axhline(hull, ls=":", lw=1,
                   color=ax.lines[-1].get_color())
    ax.axhline(0.98, ls="--", lw=0.8, color="grey")
    ax.axhline(0.96, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Number of harmonics K")
    ax.set_ylabel("Reconstruction Dice vs ground truth")
    ax.set_title(f"P0-1 representation ceiling — {name}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", action="append", required=True, type=Path,
                    help="Directory of ground-truth masks. Repeatable.")
    ap.add_argument("--name", action="append", default=None,
                    help="Label for each --masks dir (defaults to folder name).")
    ap.add_argument("--convention", default="auto",
                    choices=["auto", "separate", "refuge"],
                    help="auto: infer from filenames and pixel values; "
                         "separate: cup/disc in distinct files; "
                         "refuge: single 3-level mask per case.")
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES,
                    help=f"Harmonic counts to sweep (default {DEFAULT_K_VALUES})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N cases (for a quick check).")
    ap.add_argument("--min-gap-px", type=float, default=2.0,
                    help="Radial gap (px) a re-entry must exceed to count as a "
                         "genuine star-convexity violation. 0 = intolerant.")
    ap.add_argument("--min-run-px", type=float, default=2.0,
                    help="Length (px) a re-entered run must exceed to count.")
    ap.add_argument("--jitter", type=float, nargs="*", default=None,
                    help="Anchor displacement magnitudes in px, e.g. --jitter 5 10 20. "
                         "Adds columns showing the cost of ONH mislocalisation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("phase0_results"))
    args = ap.parse_args()

    names = args.name or []
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_df = []
    for i, root in enumerate(args.masks):
        name = names[i] if i < len(names) else root.name
        df = run(root, name, args.convention, args.k, args.limit,
                 args.min_gap_px, args.min_run_px, args.jitter, args.seed)
        if df.empty:
            print(f"  no usable cases in {root}")
            continue
        df.to_csv(args.outdir / f"{name}_per_image.csv", index=False)
        summarise(df, args.k).to_csv(args.outdir / f"{name}_summary.csv", index=False)
        print_report(df, args.k, name)
        plot_ceiling(df, args.k, args.outdir / f"{name}_ceiling.png", name)
        all_df.append(df)

    if len(all_df) > 1:
        combined = pd.concat(all_df, ignore_index=True)
        combined.to_csv(args.outdir / "ALL_per_image.csv", index=False)
        print_report(combined, args.k, "ALL DATASETS")

    print(f"\nResults in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
