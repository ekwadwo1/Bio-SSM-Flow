#!/usr/bin/env python3
"""
aggregate_results.py — turn the finished 5-fold campaign into the headline
mean +/- std results table. CPU-only, reads existing logs. No GPU, no retraining.

For each fold it reads runs/ddpm_fold{k}/val_log.jsonl, selects the validation
entry with the BEST CRPS (the checkpoint best.pt corresponds to), and collects the
full metric suite. Then reports mean +/- std across folds for every metric, plus
the per-fold calibration temperature T.

Outputs:
  - console table (mean +/- std)
  - results_5fold.json  (machine-readable, for the paper)
  - a LaTeX-ready booktabs table fragment (results_5fold_table.tex)

Run:  python aggregate_results.py
"""
from __future__ import annotations
import os, json, glob
import numpy as np

BASE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs"
FOLDS = [0, 1, 2, 3, 4]

# metrics to aggregate (present in val_log.jsonl); label -> pretty name + higher_better
METRICS = [
    ("crps",         "CRPS",              False),
    ("picp90_cal",   "PICP@90 (cal.)",    None),   # target 0.90, report as-is
    ("picp90_raw",   "PICP@90 (raw)",     None),
    ("cov_err_cal",  "Calib. error",      False),
    ("latent_cos",   "Latent cos",        True),
    ("latent_mse",   "Latent MSE",        False),
    ("psnr",         "PSNR (dB)",         True),
    ("ssim",         "SSIM",              True),
    ("err_unc_corr", "Err-Unc corr",      True),
    ("nll",          "NLL",               False),
    ("reg_ece",      "ECE",               False),
    ("diversity",    "Sample spread",     None),
    ("temperature",  "Temperature T",     None),
]


def best_entry(fold):
    """Return the val_log entry with the lowest CRPS for a fold (matches best.pt selection)."""
    p = os.path.join(BASE, f"ddpm_fold{fold}", "val_log.jsonl")
    if not os.path.exists(p):
        print(f"  [warn] fold {fold}: no val_log.jsonl"); return None
    entries = [json.loads(l) for l in open(p) if l.strip()]
    if not entries:
        print(f"  [warn] fold {fold}: empty val_log"); return None
    # pick the entry with min crps (best.pt tracks best CRPS)
    cand = [e for e in entries if "crps" in e]
    return min(cand, key=lambda e: e["crps"]) if cand else entries[-1]


def main():
    print(f"[agg] reading {len(FOLDS)} folds from {BASE}\n")
    per_fold = {}
    for f in FOLDS:
        e = best_entry(f)
        if e is not None:
            per_fold[f] = e
            print(f"  fold {f}: best-CRPS entry @ step {e.get('step','?')}  "
                  f"CRPS={e.get('crps',float('nan')):.4f}  T={e.get('temperature','-')}")
    print()

    # aggregate
    rows = []
    for key, name, hib in METRICS:
        vals = [per_fold[f][key] for f in per_fold if key in per_fold[f]]
        if not vals:
            continue
        vals = np.array(vals, float)
        rows.append((name, key, vals.mean(), vals.std(), len(vals), vals.tolist()))

    # console
    print("="*72)
    print(f"  5-FOLD RESULTS (mean +/- std over {len(per_fold)} folds)")
    print("="*72)
    print(f"  {'Metric':<18} {'mean':>10} {'std':>9}   per-fold")
    print("  " + "-"*70)
    for name, key, m, s, n, vv in rows:
        pf = " ".join(f"{x:.3f}" for x in vv)
        print(f"  {name:<18} {m:>10.4f} {s:>9.4f}   [{pf}]")
    print("="*72)

    # json
    out = {name: {"mean": m, "std": s, "n": n, "per_fold": vv}
           for name, key, m, s, n, vv in rows}
    with open(os.path.join(BASE, "results_5fold.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # LaTeX fragment (headline metrics only, formatted mean$\pm$std)
    head = ["CRPS", "PICP@90 (cal.)", "Calib. error", "PSNR (dB)", "SSIM",
            "Err-Unc corr", "Latent cos"]
    latex = ["% headline 5-fold results (paste into Results)",
             "\\begin{tabular}{@{}lc@{}}", "\\toprule",
             "Metric & Bio-SSM-Flow (mean$\\pm$std) \\\\", "\\midrule"]
    d = {name: (m, s) for name, key, m, s, n, vv in rows}
    for h in head:
        if h in d:
            m, s = d[h]
            latex.append(f"{h} & ${m:.3f}\\pm{s:.3f}$ \\\\")
    latex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(BASE, "results_5fold_table.tex"), "w") as fh:
        fh.write("\n".join(latex))
    print(f"\n[saved] results_5fold.json  and  results_5fold_table.tex  in {BASE}")
    print("[note] fill the methodology \\TODO PSNR with the PSNR mean above.")


if __name__ == "__main__":
    main()
