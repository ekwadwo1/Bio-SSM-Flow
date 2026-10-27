#!/usr/bin/env python3
"""
aggregate_unified.py — every Results number, computed ONLY from eval_unified.py outputs.

Protocol:
  * Temperature is CROSS-FITTED: within each fold's validation set, patients are split
    into two halves; T is fitted on one half and applied to the other (and vice versa).
    Every calibrated number is therefore out-of-sample. T is fitted on the --cal_region
    (default: brain) to the FINITE-R IDEAL coverage (Monte-Carlo, same quantile rule),
    so it does not over-widen to reach the nominal level with an R-member ensemble.
  * Every per-pair metric is first averaged per PATIENT; fold values are means over the
    fold's patients; tables report mean ± std over folds.
  * Hypothesis tests are paired Wilcoxon signed-rank tests over PATIENTS (not pairs),
    with rank-biserial effect size, patient-bootstrap 95% CI of the relative improvement,
    and Holm correction over the whole pre-specified family.
  * Segmentation Dice is reported for fold 0 only as primary (the seg head was trained
    on the fold-0 split, so folds 1-4 are not held-out for it).

Run:
  python aggregate_unified.py --runs runs/eval_unified/flow runs/eval_unified/ddpm \
                              --names Bio-SSM-Flow DDPM
  (the first run is the primary model; Copy/Linear rows are taken from it)

Outputs (--out, default runs/eval_unified/summary):
  results_unified.json   everything
  table_main.tex         main table (mean ± std over folds)
  fig2_data.json         calibration curves, finite-R ideal, rank histograms, err-unc per fold
"""
from __future__ import annotations
import os, re, glob, json, argparse
import numpy as np
import torch
from scipy.stats import wilcoxon


# ───────────────────────────── loading ─────────────────────────────
def load_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    files = sorted(glob.glob(os.path.join(d, "per_pair_fold*.json")),
                   key=lambda f: int(re.search(r"fold(\d+)", f).group(1)))
    if not files:
        raise FileNotFoundError(f"no per_pair_fold*.json in {d}")
    recs, arrs = [], {}
    for f in files:
        k = int(re.search(r"fold(\d+)", f).group(1))
        R = json.load(open(f))
        A = np.load(os.path.join(d, f"arrays_fold{k}.npz"))
        assert len(R) == A["crps_whole"].shape[0], f"record/array mismatch in fold {k}"
        for r in R:
            r["_row"] = len(recs)
            recs.append(r)
        for key in A.files:
            arrs.setdefault(key, []).append(A[key])
    return meta, recs, {k: np.concatenate(v, 0) for k, v in arrs.items()}


def ideal_coverage(R, nominal, n=200000, seed=0):
    """Expected coverage of a PERFECTLY calibrated R-member ensemble under torch.quantile."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(R, n, generator=g)
    y = torch.randn(n, generator=g)
    q = torch.tensor(np.concatenate([(1 - nominal) / 2, 1 - (1 - nominal) / 2]), dtype=torch.float32)
    Q = torch.quantile(x, q, dim=0)
    lo, hi = Q[:len(nominal)], Q[len(nominal):]
    return ((y >= lo) & (y <= hi)).float().mean(1).numpy()


def crossfit(recs, arrs, ideal, region, seed=0):
    cov = arrs[f"cov_{region}"]                      # [P, nT, nQ]
    Tidx = np.full(len(recs), -1, dtype=int)
    Tfit = {}
    for k in sorted({r["fold"] for r in recs}):
        rows = np.array([r["_row"] for r in recs if r["fold"] == k])
        pats = sorted({recs[i]["patient"] for i in rows})
        rng = np.random.RandomState(seed + k)
        rng.shuffle(pats)
        half = set(pats[:len(pats) // 2])
        inA = np.array([recs[i]["patient"] in half for i in rows])

        def fit(sel):
            if not sel.any():                        # tiny smoke tests: fall back to the other half
                sel = ~sel
            c = np.nanmean(cov[rows[sel]], 0)       # [nT, nQ]
            return int(np.nanargmin(np.abs(c - ideal[None]).mean(1)))
        tA, tB = fit(inA), fit(~inA)
        Tidx[rows[inA]] = tB                         # half A evaluated with T fitted on half B
        Tidx[rows[~inA]] = tA
        Tfit[int(k)] = (tA, tB)
    return Tidx, Tfit


def prepare(name, d, cal_region):
    meta, recs, arrs = load_run(d)
    nominal = np.array(meta["NOMINAL"])
    Tg = np.array(meta["T_GRID"])
    R = int(meta["protocol"]["R"])
    ideal = ideal_coverage(R, nominal)
    t1 = int(np.argmin(np.abs(Tg - 1.0)))
    Tidx, Tfit = crossfit(recs, arrs, ideal, cal_region)
    for r in recs:
        j, ti = r["_row"], Tidx[r["_row"]]
        for reg in ("whole", "brain", "roi"):
            r[f"crps_raw_{reg}"] = float(arrs[f"crps_{reg}"][j, t1])
            r[f"crps_cal_{reg}"] = float(arrs[f"crps_{reg}"][j, ti])
        r["T_applied"] = float(Tg[ti])
    return dict(name=name, dir=d, meta=meta, recs=recs, arrs=arrs, nominal=nominal, Tg=Tg,
                R=R, ideal=ideal, t1=t1, Tidx=Tidx,
                Tfit={k: (float(Tg[a]), float(Tg[b])) for k, (a, b) in Tfit.items()})


# ───────────────────────────── statistics ─────────────────────────────
def by_patient(recs, key):
    acc = {}
    for r in recs:
        v = r.get(key, np.nan)
        v = np.nan if v is None else float(v)
        acc.setdefault(r["patient"], {"fold": r["fold"], "v": []})["v"].append(v)
    out = {}
    for p, d in acc.items():
        v = np.array(d["v"], float)
        out[p] = (d["fold"], float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
    return out


def fold_summary(recs, key):
    pm = by_patient(recs, key)
    folds = sorted({f for f, _ in pm.values()})
    per = []
    for k in folds:
        v = np.array([x for f, x in pm.values() if f == k], float)
        per.append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
    per = np.array(per)
    return dict(mean=float(np.nanmean(per)), std=float(np.nanstd(per)), per_fold=per.tolist())


def fold_rel_improvement(recs, key_model, key_ref):
    pa, pb = by_patient(recs, key_model), by_patient(recs, key_ref)
    per = []
    for k in sorted({f for f, _ in pa.values()}):
        ps = [p for p in pa if pa[p][0] == k and np.isfinite(pa[p][1]) and np.isfinite(pb[p][1])]
        per.append(100 * (1 - np.mean([pa[p][1] for p in ps]) / np.mean([pb[p][1] for p in ps])))
    per = np.array(per)
    return dict(mean=float(per.mean()), std=float(per.std()), per_fold=per.tolist())


def paired_test(a, b, n_boot=5000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    _, p = wilcoxon(a, b)
    d = a - b
    nz = d[d != 0]
    rk = np.argsort(np.argsort(np.abs(nz))) + 1
    rrb = float((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()) if len(nz) else 0.0
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), (n_boot, len(a)))
    rel = 100 * (1 - a[idx].mean(1) / b[idx].mean(1))
    return dict(n_patients=int(len(a)), mean_a=float(a.mean()), mean_b=float(b.mean()),
                rel_improvement_pct=float(100 * (1 - a.mean() / b.mean())),
                ci95_pct=[float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                p=float(p), r_rb=rrb)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def aligned_within(recs, ka, kb):
    pa, pb = by_patient(recs, ka), by_patient(recs, kb)
    ps = sorted(pa)
    return [pa[p][1] for p in ps], [pb[p][1] for p in ps]


def aligned_across(recs_a, recs_b, key):
    common = {(r["fold"], r["idx"]) for r in recs_a} & {(r["fold"], r["idx"]) for r in recs_b}
    ra = [r for r in recs_a if (r["fold"], r["idx"]) in common]
    rb = [r for r in recs_b if (r["fold"], r["idx"]) in common]
    pa, pb = by_patient(ra, key), by_patient(rb, key)
    ps = sorted(set(pa) & set(pb))
    return [pa[p][1] for p in ps], [pb[p][1] for p in ps], len(common)


# ───────────────────────────── calibration ─────────────────────────────
def calib_summary(run, reg):
    recs, arrs, Tidx, t1 = run["recs"], run["arrs"], run["Tidx"], run["t1"]
    ideal, nominal = run["ideal"], run["nominal"]
    cov = arrs[f"cov_{reg}"]
    i90 = int(np.argmin(np.abs(nominal - 0.9)))
    curves = {"raw": [], "cal": []}
    for k in sorted({r["fold"] for r in recs}):
        rows = np.array([r["_row"] for r in recs if r["fold"] == k])
        curves["raw"].append(np.nanmean(cov[rows, t1], 0))
        curves["cal"].append(np.nanmean(cov[rows, Tidx[rows]], 0))
    out = {}
    for which, cs in curves.items():
        cs = np.array(cs)
        e_i = np.mean(np.abs(cs - ideal[None]), 1)
        e_n = np.mean(np.abs(cs - nominal[None]), 1)
        out[which] = dict(curves_per_fold=cs.tolist(),
                          calerr_vs_ideal=dict(mean=float(e_i.mean()), std=float(e_i.std()), per_fold=e_i.tolist()),
                          calerr_vs_nominal=dict(mean=float(e_n.mean()), std=float(e_n.std()), per_fold=e_n.tolist()),
                          picp90=dict(mean=float(cs[:, i90].mean()), std=float(cs[:, i90].std()),
                                      per_fold=cs[:, i90].tolist()))
    rk = arrs[f"rank_{reg}"].astype(np.float64)
    rows = np.arange(len(recs))
    h_raw, h_cal = rk[rows, t1].sum(0), rk[rows, Tidx].sum(0)
    out["rank_hist_raw"] = (h_raw / h_raw.sum()).tolist()
    out["rank_hist_cal"] = (h_cal / h_cal.sum()).tolist()
    return out


# ───────────────────────────── segmentation ─────────────────────────────
def direction(v0, v1, rel=0.10, abs_min=50):
    d = v1 - v0
    return 0 if abs(d) <= max(rel * v0, abs_min) else (1 if d > 0 else -1)


def seg_summary(recs, folds=(0,)):
    sub = [r for r in recs if r["fold"] in folds and "dice_forecast_TC" in r]
    if not sub:
        return None
    out = {"folds": list(folds), "n_pairs": len(sub)}
    for reg in ("ET", "NETC", "SNFH", "TC", "WT"):
        out[reg] = {src: float(np.nanmean(list(zip(*by_patient(sub, f"dice_{src}_{reg}").values()))[1]))
                    for src in ("forecast", "copy", "ceiling")}
    for r in sub:
        r["_dir_true"] = direction(r["vol_TC_gt_a"], r["vol_TC_gt_b"])
        r["_dir_pred"] = direction(r["vol_TC_seg_za"], r["vol_TC_seg_fc"])
        r["_dir_ok_model"] = float(r["_dir_true"] == r["_dir_pred"])
        r["_dir_ok_copy"] = float(r["_dir_true"] == 0)
    out["TC_direction_accuracy"] = {
        "model": float(np.mean(list(zip(*by_patient(sub, "_dir_ok_model").values()))[1])),
        "copy_predicts_stable": float(np.mean(list(zip(*by_patient(sub, "_dir_ok_copy").values()))[1])),
        "true_distribution": {str(v): int(sum(r["_dir_true"] == v for r in sub)) for v in (-1, 0, 1)}}
    return out


# ───────────────────────────── main ─────────────────────────────
def fmt(d, prec=3):
    return f"${d['mean']:.{prec}f}\\pm{d['std']:.{prec}f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--cal_region", default="brain", choices=["whole", "brain", "roi"])
    ap.add_argument("--out", default="runs/eval_unified/summary")
    args = ap.parse_args()
    names = args.names or [os.path.basename(r.rstrip("/")) for r in args.runs]
    os.makedirs(args.out, exist_ok=True)
    runs = [prepare(n, d, args.cal_region) for n, d in zip(names, args.runs)]
    prim = runs[0]
    res = {"protocol": dict(cal_region=args.cal_region, runs=dict(zip(names, args.runs)),
                            note="patient-level aggregation; cross-fitted T; Holm-corrected Wilcoxon"),
           "runs": {}}

    point_keys = ["mse_whole", "mse_brain", "mse_roi", "mse_roiwt", "cos_whole", "cos_roi"]
    for run in runs:
        recs = run["recs"]
        s = {"n_pairs": len(recs), "n_patients": len({r["patient"] for r in recs}),
             "folds": sorted({r["fold"] for r in recs}),
             "T_crossfit_per_fold(halfA,halfB)": run["Tfit"],
             "finite_R_ideal_coverage": dict(zip([str(q) for q in run["nominal"]], run["ideal"].tolist()))}
        for key in point_keys:
            s[f"{key}_model"] = fold_summary(recs, f"{key}_model")
        for reg in ("whole", "brain", "roi"):
            s[f"crps_raw_{reg}"] = fold_summary(recs, f"crps_raw_{reg}")
            s[f"crps_cal_{reg}"] = fold_summary(recs, f"crps_cal_{reg}")
            s[f"errunc_{reg}"] = fold_summary(recs, f"errunc_{reg}")
            s[f"calibration_{reg}"] = calib_summary(run, reg)
        s["roi_improvement_vs_copy_pct"] = fold_rel_improvement(recs, "mse_roi_model", "mse_roi_copy")
        s["whole_improvement_vs_copy_pct"] = fold_rel_improvement(recs, "mse_whole_model", "mse_whole_copy")
        s["diversity"] = fold_summary(recs, "diversity")
        if "psnr_model" in recs[0]:
            for key in ("psnr_model", "ssim_model"):
                s[key] = fold_summary(recs, key)
        s["segmentation_fold0"] = seg_summary(recs, folds=(0,))
        s["segmentation_all_folds_NOT_heldout"] = seg_summary(recs, folds=tuple(s["folds"]))
        res["runs"][run["name"]] = s

    base = {}
    for key in point_keys:
        base[f"{key}_copy"] = fold_summary(prim["recs"], f"{key}_copy")
        base[f"{key}_linear"] = fold_summary(prim["recs"], f"{key}_linear")
    base["linear_pairs_with_previous_timepoint"] = int(sum(r["has_prev"] for r in prim["recs"]))
    if "psnr_copy" in prim["recs"][0]:
        base["psnr_copy"] = fold_summary(prim["recs"], "psnr_copy")
        base["ssim_copy"] = fold_summary(prim["recs"], "ssim_copy")
    res["baselines"] = base

    # pre-specified test family (patient level), Holm-corrected together
    tests = []
    for ka, kb, label in [("mse_roi_model", "mse_roi_copy", "ROI MSE: model vs Copy"),
                          ("mse_whole_model", "mse_whole_copy", "whole MSE: model vs Copy"),
                          ("mse_roi_model", "mse_roi_linear", "ROI MSE: model vs Linear"),
                          ("mse_whole_model", "mse_whole_linear", "whole MSE: model vs Linear")]:
        a, b = aligned_within(prim["recs"], ka, kb)
        tests.append((f"{prim['name']} | {label}", paired_test(a, b)))
    for other in runs[1:]:
        for key, label in [("crps_cal_brain", "calibrated CRPS (brain)"), ("mse_roi_model", "ROI MSE")]:
            a, b, n_common = aligned_across(prim["recs"], other["recs"], key)
            t = paired_test(a, b)
            t["n_common_pairs"] = n_common
            tests.append((f"{prim['name']} vs {other['name']} | {label}", t))
    adj = holm([t["p"] for _, t in tests])
    res["tests"] = {}
    for (label, t), pa in zip(tests, adj):
        t["p_holm"] = float(pa)
        t["sig"] = "***" if pa < 1e-3 else "**" if pa < 1e-2 else "*" if pa < 0.05 else "ns"
        res["tests"][label] = t

    with open(os.path.join(args.out, "results_unified.json"), "w") as f:
        json.dump(res, f, indent=2)

    # main table
    lines = ["% generated by aggregate_unified.py — do not edit numbers by hand",
             "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
             "Method & Whole MSE$\\downarrow$ & ROI MSE$\\downarrow$ & ROI improv.\\ (\\%)$\\uparrow$ & "
             "CRPS$\\downarrow$ & PSNR (dB)$\\uparrow$ & SSIM$\\uparrow$ & Calib.\\ err$\\downarrow$ \\\\",
             "\\midrule"]
    ps = lambda k: fmt(base[k], 2) if k in base else "--"
    ss = lambda k: fmt(base[k], 3) if k in base else "--"
    lines.append(f"\\textsc{{Copy}} & {fmt(base['mse_whole_copy'])} & {fmt(base['mse_roi_copy'])} & -- & -- & "
                 f"{ps('psnr_copy')} & {ss('ssim_copy')} & -- \\\\")
    lines.append(f"\\textsc{{Linear}} (capped) & {fmt(base['mse_whole_linear'])} & {fmt(base['mse_roi_linear'])} "
                 f"& -- & -- & -- & -- & -- \\\\")
    for run in runs[::-1]:
        s = res["runs"][run["name"]]
        p_ = fmt(s["psnr_model"], 2) if "psnr_model" in s else "--"
        q_ = fmt(s["ssim_model"], 3) if "ssim_model" in s else "--"
        lines.append(f"{run['name']} & {fmt(s['mse_whole_model'])} & {fmt(s['mse_roi_model'])} & "
                     f"{fmt(s['roi_improvement_vs_copy_pct'], 1)} & {fmt(s[f'crps_cal_{args.cal_region}'])} & "
                     f"{p_} & {q_} & {fmt(s[f'calibration_{args.cal_region}']['cal']['calerr_vs_ideal'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(args.out, "table_main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # figure-2 data
    fig = {"nominal": prim["nominal"].tolist(), "finite_R_ideal": prim["ideal"].tolist(), "R": prim["R"],
           "cal_region": args.cal_region, "runs": {}}
    for run in runs:
        s = res["runs"][run["name"]]
        c = s[f"calibration_{args.cal_region}"]
        fig["runs"][run["name"]] = dict(
            raw_curves_per_fold=c["raw"]["curves_per_fold"], cal_curves_per_fold=c["cal"]["curves_per_fold"],
            rank_hist_raw=c["rank_hist_raw"], rank_hist_cal=c["rank_hist_cal"],
            errunc_per_fold={reg: s[f"errunc_{reg}"]["per_fold"] for reg in ("whole", "brain", "roi")},
            T_crossfit=s["T_crossfit_per_fold(halfA,halfB)"])
    with open(os.path.join(args.out, "fig2_data.json"), "w") as f:
        json.dump(fig, f, indent=2)

    # console summary
    print("=" * 78)
    print(f"  UNIFIED RESULTS  (patient-level, mean ± std over folds; T cross-fitted on {args.cal_region})")
    print("=" * 78)
    print(f"  Copy   : whole MSE {base['mse_whole_copy']['mean']:.4f}  ROI MSE {base['mse_roi_copy']['mean']:.4f}")
    print(f"  Linear : whole MSE {base['mse_whole_linear']['mean']:.4f}  ROI MSE {base['mse_roi_linear']['mean']:.4f}"
          f"  ({base['linear_pairs_with_previous_timepoint']} pairs have a previous timepoint)")
    for run in runs:
        s = res["runs"][run["name"]]
        cb = s[f"calibration_{args.cal_region}"]
        print(f"  {run['name']:<14}: whole MSE {s['mse_whole_model']['mean']:.4f}  ROI MSE {s['mse_roi_model']['mean']:.4f}"
              f"  ROI improv {s['roi_improvement_vs_copy_pct']['mean']:.1f}±{s['roi_improvement_vs_copy_pct']['std']:.1f}%")
        print(f"  {'':<14}  CRPS raw {s[f'crps_raw_{args.cal_region}']['mean']:.4f} -> cal "
              f"{s[f'crps_cal_{args.cal_region}']['mean']:.4f} | PICP@90 raw {cb['raw']['picp90']['mean']:.3f} -> cal "
              f"{cb['cal']['picp90']['mean']:.3f} (finite-R ideal {prim['ideal'][int(np.argmin(np.abs(prim['nominal'] - 0.9)))]:.3f})")
        print(f"  {'':<14}  calib err vs ideal: raw {cb['raw']['calerr_vs_ideal']['mean']:.4f} -> cal "
              f"{cb['cal']['calerr_vs_ideal']['mean']:.4f} | err-unc (brain/roi) "
              f"{s['errunc_brain']['mean']:.3f}/{s['errunc_roi']['mean']:.3f} | T per fold {s['T_crossfit_per_fold(halfA,halfB)']}")
        if "psnr_model" in s:
            print(f"  {'':<14}  PSNR {s['psnr_model']['mean']:.2f} dB  SSIM {s['ssim_model']['mean']:.3f}  (brain voxels, vs decoded follow-up)")
    print("-" * 78)
    for label, t in res["tests"].items():
        print(f"  {label:<48} n={t['n_patients']:>3} rel {t['rel_improvement_pct']:+6.1f}% "
              f"[{t['ci95_pct'][0]:+.1f},{t['ci95_pct'][1]:+.1f}]  p_holm={t['p_holm']:.1e} {t['sig']}  r_rb={t['r_rb']:+.2f}")
    print("=" * 78)
    print(f"[saved] {args.out}/results_unified.json, table_main.tex, fig2_data.json")


if __name__ == "__main__":
    main()
