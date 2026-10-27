#!/usr/bin/env python3
"""
plot_fig3_skill.py — Fig. 3: forecasting skill relative to reference forecasts.

  (a) Tumor-region MSE skill relative to persistence, 100 x (1 - MSE / MSE_persistence), for
      damped persistence, voxel-wise AR(1), and Bio-SSM-Flow in LUMIERE and the two external
      cohorts; patient-level paired estimates with 95% patient-bootstrap intervals.
  (b) Relative improvement of Bio-SSM-Flow over AR(1) by region and metric, per cohort
      (positive values favor Bio-SSM-Flow; for cosine similarity the sign is aligned).

All estimates come from aggregate_unified.paired_test on patient-level means, i.e. the same
computation as the statistics in the Results, so numbers and intervals match the text.
Sources: runs/eval_unified/{flow,ar1,damped} (internal; AR(1) and damped persistence with
their cross-fitted temperature, ~1) and runs/external_eval/{alptdg,mu}/per_pair.jsonl
(external; baselines at T = 1 as pre-registered).
Outputs: figures/fig_skill.pdf, .png, _data.json.

Usage:  python plot_fig3_skill.py            (real data)
        python plot_fig3_skill.py --synthetic (layout test)
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COHORTS = [("lumiere", "LUMIERE"), ("alptdg", "UCSF-ALPTDG"), ("mu", "MU-Glioma-Post")]
# forecasters in (a): colour + marker; Bio-SSM-Flow keeps its blue from Fig. 2
FORE = [("damped", "Damped persistence", "#8C8C8C", "D"),
        ("ar1", "Voxel-wise AR(1)", "#4D4D4D", "^"),
        ("flow", "Bio-SSM-Flow", "#0072B2", "o")]
# cohorts in (b): Okabe-Ito colours + distinct markers
CSTYLE = {"lumiere": ("#000000", "o"), "alptdg": ("#E69F00", "s"), "mu": ("#009E73", "D")}
# rows of (b): label, internal keys (flow, ar1), external keys (flow, ar1), higher_is_better
ROWS = [("Tumor-region MSE", ("mse_roi_model",) * 2, ("mse_roi_flow", "mse_roi_ar1"), False),
        ("Tumor-region CRPS", ("crps_cal_roi",) * 2, ("crps_cal_roi_flow", "crps_roi_ar1"), False),
        ("Tumor-region direction", ("cos_roi_model",) * 2, ("cos_roi_flow", "cos_roi_ar1"), True),
        ("Brain CRPS", ("crps_cal_brain",) * 2, ("crps_cal_brain_flow", "crps_brain_ar1"), False),
        ("Brain MSE", ("mse_brain_model",) * 2, ("mse_brain_flow", "mse_brain_ar1"), False),
        ("Whole-volume MSE", ("mse_whole_model",) * 2, ("mse_whole_flow", "mse_whole_ar1"), False)]


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.markersize": 4.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ───────────────────────────── statistics ─────────────────────────────
def compute_stats():
    import aggregate_unified as AG

    def by_p(recs, key):
        return {p: v for p, (_, v) in AG.by_patient(recs, key).items()}

    def test(da, db, higher=False):
        common = sorted(set(da) & set(db))
        a = np.array([da[p] for p in common], float); b = np.array([db[p] for p in common], float)
        ok = np.isfinite(a) & np.isfinite(b)
        t = AG.paired_test(a[ok], b[ok])
        rel, lo, hi = t["rel_improvement_pct"], t["ci95_pct"][0], t["ci95_pct"][1]
        if higher:                                   # paired_test assumes lower is better
            rel, lo, hi = -rel, -hi, -lo
        return dict(rel=float(rel), lo=float(lo), hi=float(hi), p=float(t["p"]), n=int(ok.sum()))

    recs = {}
    run = {n: AG.prepare(n, os.path.join(HERE, "runs", "eval_unified", n), "brain")["recs"] for n in ("flow", "ar1", "damped")}
    recs["lumiere"] = dict(flow=run["flow"], ar1=run["ar1"], damped=run["damped"], copy=run["flow"], ext=False)
    for key in ("alptdg", "mu"):
        p = os.path.join(HERE, "runs", "external_eval", key, "per_pair.jsonl")
        R = [json.loads(l) for l in open(p)]
        recs[key] = dict(flow=R, ar1=R, damped=R, copy=R, ext=True)

    stats = {"a": {}, "b": {}}
    for ck, _ in COHORTS:
        r = recs[ck]
        suf = (lambda m: f"mse_roi_{m}") if r["ext"] else (lambda m: "mse_roi_copy" if m == "copy" else "mse_roi_model")
        copy = by_p(r["copy"], suf("copy"))
        stats["a"][ck] = {m: test(by_p(r[m], suf(m)), copy) for m, _, _, _ in FORE}
        stats["b"][ck] = {}
        for label, ik, ek, higher in ROWS:
            kf, ka = ek if r["ext"] else ik
            stats["b"][ck][label] = test(by_p(r["flow"], kf), by_p(r["ar1"], ka), higher)
    return stats


def synthetic_stats():
    a = {"lumiere": dict(damped=(33.4, 31.5, 35.2), ar1=(32.5, 30.6, 34.4), flow=(35.4, 33.6, 37.2)),
         "alptdg": dict(damped=(-10.1, -18.0, -3.0), ar1=(-14.6, -23.0, -7.0), flow=(-4.7, -12.0, 1.3)),
         "mu": dict(damped=(33.8, 31.9, 35.6), ar1=(33.4, 31.5, 35.2), flow=(34.6, 32.7, 36.4))}
    b = {"lumiere": [(4.3, 2.9, 5.8), (4.4, 3.4, 5.5), (9.0, 7.4, 10.8), (1.1, 0.2, 1.9), (-1.4, -2.6, -0.3), (-1.8, -3.1, -0.5)],
         "alptdg": [(8.6, 6.1, 11.1), (7.6, 6.2, 9.0), (2.5, 1.2, 3.7), (9.0, 7.5, 10.4), (11.7, 9.9, 13.5), (13.3, 11.7, 15.0)],
         "mu": [(1.8, 0.6, 3.0), (2.7, 1.8, 3.6), (4.7, 3.1, 6.4), (0.6, -0.2, 1.4), (-0.6, -1.5, 0.4), (-1.1, -2.3, 0.0)]}
    n = {"lumiere": 81, "alptdg": 187, "mu": 143}
    S = {"a": {}, "b": {}}
    for ck, _ in COHORTS:
        S["a"][ck] = {m: dict(rel=v[0], lo=v[1], hi=v[2], p=0.0, n=n[ck]) for m, v in a[ck].items()}
        S["b"][ck] = {ROWS[i][0]: dict(rel=v[0], lo=v[1], hi=v[2], p=0.0, n=n[ck]) for i, v in enumerate(b[ck])}
    return S


# ───────────────────────────── figure ─────────────────────────────
def make_figure(S, out_stem):
    apply_style()
    fig = plt.figure(figsize=(7.16, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.18], wspace=0.62, left=0.075, right=0.985, bottom=0.17, top=0.83)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (a) skill relative to persistence
    ax_a.axhline(0, color="black", lw=0.8, zorder=1)
    ax_a.text(2.55, 0.8, "Persistence", ha="right", va="bottom", fontsize=6.5)
    offs = {"damped": -0.24, "ar1": 0.0, "flow": 0.24}
    for i, (ck, cname) in enumerate(COHORTS):
        if i > 0:
            ax_a.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)
        for m, _, col, mk in FORE:
            s = S["a"][ck][m]
            ax_a.errorbar(i + offs[m], s["rel"], yerr=[[s["rel"] - s["lo"]], [s["hi"] - s["rel"]]], fmt=mk,
                          color=col, mfc=col, mec=col, ms=4.4 if m == "flow" else 3.8, lw=0.9, capsize=1.6, zorder=3)
    lo_all = min(S["a"][ck][m]["lo"] for ck, _ in COHORTS for m, *_ in FORE)
    hi_all = max(S["a"][ck][m]["hi"] for ck, _ in COHORTS for m, *_ in FORE)
    ax_a.set_ylim(min(-5, lo_all - 4), hi_all + 6)
    ax_a.set_xlim(-0.5, len(COHORTS) - 0.5)
    ax_a.set_xticks(range(len(COHORTS)))
    nlab = [S["a"][ck]["flow"]["n"] for ck, _ in COHORTS]
    ax_a.set_xticklabels([f"{c}\n$n$ = {n}" for (_, c), n in zip(COHORTS, nlab)], fontsize=6.8)
    ax_a.tick_params(axis="x", length=0)
    ax_a.set_ylabel("Tumor-region MSE skill\nrelative to persistence (%)")
    ax_a.set_title("Skill relative to persistence", loc="left", pad=3)
    ax_a.grid(axis="y", color="0.92", lw=0.5, zorder=0); ax_a.set_axisbelow(True)
    leg_a = [Line2D([], [], ls="none", marker=mk, color=col, ms=4.0, label=lab) for _, lab, col, mk in FORE]
    ax_a.legend(handles=leg_a, loc="upper center", bbox_to_anchor=(0.5, 1.21), ncol=3, handletextpad=0.2, columnspacing=0.9)

    # (b) forest plot, Bio-SSM-Flow vs AR(1)
    n_rows = len(ROWS)
    yoff = {"lumiere": 0.22, "alptdg": 0.0, "mu": -0.22}
    ax_b.axvspan(-100, 0, color="0.965", lw=0, zorder=0)
    ax_b.axvline(0, color="black", lw=0.8, zorder=1)
    ax_b.axhline(n_rows - 3.5, color="0.75", lw=0.5, ls=(0, (2, 2)), zorder=1)
    for j, (label, *_rest) in enumerate(ROWS):
        y = n_rows - 1 - j
        for ck, _ in COHORTS:
            s = S["b"][ck][label]; col, mk = CSTYLE[ck]
            filled = ck != "alptdg"
            ax_b.errorbar(s["rel"], y + yoff[ck], xerr=[[s["rel"] - s["lo"]], [s["hi"] - s["rel"]]], fmt=mk,
                          color=col, mfc=col if filled else "white", mec=col, ms=3.8, lw=0.9, capsize=1.4, zorder=3)
    xs = [S["b"][ck][r[0]][k] for ck, _ in COHORTS for r in ROWS for k in ("lo", "hi")]
    xmin, xmax = min(xs), max(xs)
    pad = 0.08 * (xmax - xmin)
    ax_b.set_xlim(min(xmin - pad, -pad), xmax + pad)
    ax_b.set_ylim(-1.15, n_rows - 0.4)
    ax_b.set_yticks(range(n_rows))
    ax_b.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=7)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Improvement of Bio-SSM-Flow over AR(1) (%)")
    ax_b.set_title("Bio-SSM-Flow vs. voxel-wise AR(1)", loc="left", pad=3)
    ax_b.text(0.99, -0.92, "favors Bio-SSM-Flow \u2192", transform=ax_b.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color="0.35")
    x_left = ax_b.get_xlim()[0]
    ax_b.text(x_left / 2, -0.85, "\u2190 favors\nAR(1)", ha="center", va="center", fontsize=6.5, color="0.35", linespacing=1.0)
    ax_b.grid(axis="x", color="0.92", lw=0.5, zorder=0); ax_b.set_axisbelow(True)
    leg_b = [Line2D([], [], ls="none", marker=CSTYLE[ck][1], color=CSTYLE[ck][0],
                    mfc=CSTYLE[ck][0] if ck != "alptdg" else "white", ms=4.0, label=c) for ck, c in COHORTS]
    ax_b.legend(handles=leg_b, loc="upper center", bbox_to_anchor=(0.5, 1.21), ncol=3, handletextpad=0.2, columnspacing=1.0)

    for ax, letter, dx in ((ax_a, "a", -0.3), (ax_b, "b", -0.42)):
        ax.text(dx, 1.012, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")

    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    with open(out_stem + "_data.json", "w") as f:
        json.dump(S, f, indent=1)

    print("% (a) MSE skill relative to persistence (%), tumor region")
    for ck, cname in COHORTS:
        print(f"   {cname:15s} " + " | ".join(f"{lab} {S['a'][ck][m]['rel']:+.1f} [{S['a'][ck][m]['lo']:+.1f},{S['a'][ck][m]['hi']:+.1f}]"
                                         for m, lab, _, _ in FORE) + f" | n={S['a'][ck]['flow']['n']}")
    print("% (b) improvement of Bio-SSM-Flow over AR(1) (%)")
    for label, *_ in ROWS:
        print(f"   {label:24s} " + " | ".join(f"{c} {S['b'][ck][label]['rel']:+.1f} [{S['b'][ck][label]['lo']:+.1f},{S['b'][ck][label]['hi']:+.1f}]"
                                            for ck, c in COHORTS))
    print(f"[fig3] saved {out_stem}.pdf, .png, _data.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig_skill"))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.synthetic:
        make_figure(synthetic_stats(), args.out + "_SYNTHETIC")
    else:
        make_figure(compute_stats(), args.out)


if __name__ == "__main__":
    main()
