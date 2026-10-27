#!/usr/bin/env python3
"""
plot_fig2_calibration.py — Fig. 2: calibration of the predictive distribution.

  (a) Reliability diagram, LUMIERE, brain tokens: empirical vs. nominal coverage of central
      intervals for the raw and the cross-fitted calibrated Bio-SSM-Flow ensemble and for AR(1),
      with the coverage expected of a calibrated 32-member ensemble; 95% patient-bootstrap bands.
  (b) Rank histograms, LUMIERE, brain tokens: raw vs. calibrated ensemble (relative to uniform).
  (c) Calibration error without and with temperature scaling for LUMIERE (cross-fitted
      temperature) and the two external cohorts (transferred temperature), AR(1) for reference;
      95% patient-bootstrap intervals.

Everything is computed from saved outputs; nothing is typed in by hand:
  internal  runs/eval_unified/flow and runs/eval_unified/ar1 (via aggregate_unified.prepare,
            which also performs the cross-fitting used in the paper)
  external  runs/external_eval/{alptdg,mu}/per_pair.jsonl
Outputs: fig_calibration.pdf (vector, fonts embedded), fig_calibration.png (preview),
         fig_calibration_data.json (every plotted value).

Usage (from the Bio-SSM-Flow folder):
  python plot_fig2_calibration.py                 # real data
  python plot_fig2_calibration.py --synthetic     # layout test without data
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Okabe-Ito colour-blind-safe palette; meaning is also carried by marker and line style.
COL = dict(raw="#D55E00", cal="#0072B2", ar1="#6E6E6E", ideal="#000000")
COHORTS = [("lumiere", "LUMIERE", "cross-fitted"),
           ("alptdg", "UCSF-ALPTDG", "transferred"),
           ("mu", "MU-Glioma-Post", "transferred")]
R_MEMBERS = 32


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ───────────────────────────── statistics helpers ─────────────────────────────
def patient_matrix(curves_by_patient):
    pats = sorted(curves_by_patient)
    return np.stack([np.nanmean(np.stack(curves_by_patient[p]), 0) for p in pats])


def mean_band(M, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(M), (n_boot, len(M)))
    boots = np.nanmean(M[idx], axis=1)
    return np.nanmean(M, 0), np.nanpercentile(boots, 2.5, 0), np.nanpercentile(boots, 97.5, 0)


def calerr_ci(M, ideal, n_boot=2000, seed=0):
    ce = lambda X: float(np.mean(np.abs(np.nanmean(X, 0) - ideal)))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(M), (n_boot, len(M)))
    b = np.array([ce(M[i]) for i in idx])
    return ce(M), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ───────────────────────────── data loading ─────────────────────────────
def _cal_index(rec, crps_row, Tg):
    for key in ("T_cal", "T_cross", "T_crossfit", "T"):
        if key in rec and rec[key] is not None:
            return int(np.argmin(np.abs(Tg - float(rec[key])))), 0.0
    target = rec["crps_cal_brain"]
    i = int(np.nanargmin(np.abs(crps_row - target)))
    return i, abs(float(crps_row[i]) - float(target))


def load_internal(flow_dir, ar1_dir):
    import aggregate_unified as AG
    out = {}
    for name, d in (("flow", flow_dir), ("ar1", ar1_dir)):
        run = AG.prepare(name, d, "brain")
        Tg, ideal = np.asarray(run["Tg"], float), np.asarray(run["ideal"], float)
        cov, crps, rank = (np.asarray(run["arrs"][k]) for k in ("cov_brain", "crps_brain", "rank_brain"))
        i1 = int(np.argmin(np.abs(Tg - 1.0)))
        raw, cal = {}, {}
        rk_raw, rk_cal = np.zeros(rank.shape[-1]), np.zeros(rank.shape[-1])
        worst = 0.0
        for r in run["recs"]:
            row = r["_row"]
            ic, err = _cal_index(r, crps[row], Tg)
            worst = max(worst, err)
            raw.setdefault(r["patient"], []).append(cov[row, i1])
            cal.setdefault(r["patient"], []).append(cov[row, ic])
            rk_raw += rank[row, i1]; rk_cal += rank[row, ic]
        if worst > 1e-5:
            print(f"[fig2] WARNING: {name}: calibrated temperature recovered with CRPS mismatch up to {worst:.2e}")
        out[name] = dict(raw=patient_matrix(raw), cal=patient_matrix(cal), rank_raw=rk_raw, rank_cal=rk_cal, ideal=ideal)
    meta = json.load(open(os.path.join(flow_dir, "meta.json")))
    out["nominal"] = np.asarray(meta["NOMINAL"], float)
    return out


def load_external(ext_dir):
    out = {}
    for key, _, _ in COHORTS[1:]:
        p = os.path.join(ext_dir, key, "per_pair.jsonl")
        if not os.path.exists(p):
            continue
        raw, cal, ar1 = {}, {}, {}
        for line in open(p):
            r = json.loads(line)
            if "cov_raw_brain_flow" in r:
                raw.setdefault(r["patient"], []).append(np.asarray(r["cov_raw_brain_flow"], float))
                cal.setdefault(r["patient"], []).append(np.asarray(r["cov_cal_brain_flow"], float))
                ar1.setdefault(r["patient"], []).append(np.asarray(r["cov_brain_ar1"], float))
        out[key] = dict(raw=patient_matrix(raw), cal=patient_matrix(cal), ar1=patient_matrix(ar1))
    return out


def synthetic():
    rng = np.random.RandomState(1)
    nom = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    ideal = nom * R_MEMBERS / (R_MEMBERS + 1) - 0.03 * nom ** 3
    def mat(curve, n, sd=0.03):
        return np.clip(curve[None] + rng.normal(0, sd, (n, len(nom))), 0, 1)
    raw = ideal * 0.84
    flow = dict(raw=mat(raw, 81), cal=mat(ideal - 0.004, 81), ideal=ideal,
                rank_raw=1 + 1.6 * np.abs(np.linspace(-1, 1, R_MEMBERS + 1)) ** 3, rank_cal=1 + 0.08 * rng.randn(R_MEMBERS + 1))
    ar1 = dict(raw=mat(ideal - 0.005, 81), cal=mat(ideal - 0.005, 81), ideal=ideal,
               rank_raw=np.ones(R_MEMBERS + 1), rank_cal=np.ones(R_MEMBERS + 1))
    internal = dict(flow=flow, ar1=ar1, nominal=nom)
    external = dict(alptdg=dict(raw=mat(ideal + 0.01, 292), cal=mat(np.minimum(ideal + 0.088, 0.995), 292), ar1=mat(ideal + 0.059, 292)),
                    mu=dict(raw=mat(ideal - 0.103, 149), cal=mat(ideal - 0.008, 149), ar1=mat(ideal - 0.005, 149)))
    return internal, external


# ───────────────────────────── figure ─────────────────────────────
def make_figure(internal, external, out_stem):
    apply_style()
    nom = internal["nominal"]
    F, A = internal["flow"], internal["ar1"]
    ideal = F["ideal"]
    data = {"nominal": nom.tolist(), "ideal_coverage": ideal.tolist()}

    fig = plt.figure(figsize=(7.16, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 0.95], wspace=0.55, left=0.07, right=0.995, bottom=0.2, top=0.8)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) reliability diagram
    ax_a.plot(nom, ideal, ls=(0, (4, 2)), lw=0.9, color=COL["ideal"], zorder=4)
    series = [("raw", F["raw"], COL["raw"], dict(ls="--", marker="o", mfc="white")),
              ("cal", F["cal"], COL["cal"], dict(ls="-", marker="o", mfc=COL["cal"])),
              ("ar1", A["raw"], COL["ar1"], dict(ls="-", lw=0.9, marker="^", mfc=COL["ar1"], ms=3.2))]
    data["panel_a"] = {}
    for key, M, c, kw in series:
        m, lo, hi = mean_band(M)
        ax_a.fill_between(nom, lo, hi, color=c, alpha=0.16, lw=0, zorder=1)
        ax_a.plot(nom, m, color=c, mec=c, zorder=3, **kw)
        data["panel_a"][key] = dict(mean=m.tolist(), lo=lo.tolist(), hi=hi.tolist(), n_patients=int(len(M)))
    ax_a.set_xlim(0.05, 1.0); ax_a.set_ylim(0.0, 1.0)
    ax_a.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9]); ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlabel("Nominal coverage"); ax_a.set_ylabel("Empirical coverage")
    ax_a.set_title("Reliability (LUMIERE)", loc="left", pad=3)
    ax_a.set_aspect("equal", adjustable="box")

    # (b) rank histograms relative to uniform
    ranks = np.arange(R_MEMBERS + 1)
    rr = F["rank_raw"] / F["rank_raw"].sum() * (R_MEMBERS + 1)
    rc = F["rank_cal"] / F["rank_cal"].sum() * (R_MEMBERS + 1)
    ax_b.fill_between(ranks, rc, step="mid", color=COL["cal"], alpha=0.22, lw=0)
    ax_b.step(ranks, rc, where="mid", color=COL["cal"], lw=1.0)
    ax_b.step(ranks, rr, where="mid", color=COL["raw"], lw=1.0, ls="--")
    ax_b.axhline(1.0, color=COL["ideal"], lw=0.8, ls=(0, (1, 1.5)))
    ax_b.set_xlim(-0.5, R_MEMBERS + 0.5); ax_b.set_ylim(0, max(rr.max(), rc.max()) * 1.08)
    ax_b.set_xticks([0, 8, 16, 24, 32])
    ax_b.set_xlabel("Rank of observation among 32 members"); ax_b.set_ylabel("Frequency relative to uniform")
    ax_b.set_title("Rank histogram (LUMIERE)", loc="left", pad=3)
    data["panel_b"] = dict(raw=rr.tolist(), calibrated=rc.tolist())

    # (c) calibration error by cohort (horizontal: full cohort names stay readable)
    data["panel_c"] = {}
    ypos = {key: len(COHORTS) - 1 - i for i, (key, _, _) in enumerate(COHORTS)}
    for key, name, how in COHORTS:
        y = ypos[key]
        if key == "lumiere":
            Mr, Mc, Ma = F["raw"], F["cal"], A["raw"]
        elif key in external:
            Mr, Mc, Ma = external[key]["raw"], external[key]["cal"], external[key]["ar1"]
        else:
            ax_c.text(0.002, y, "pending", ha="left", va="center", fontsize=6.5, color="0.5")
            continue
        r_, c_, a_ = (calerr_ci(M, ideal) for M in (Mr, Mc, Ma))
        yf, ya = y + 0.1, y - 0.24
        ax_c.annotate("", xy=(c_[0], yf), xytext=(r_[0], yf),
                      arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.8, mutation_scale=7, shrinkA=4.5, shrinkB=7.5))
        ax_c.errorbar(r_[0], yf, xerr=[[r_[0] - r_[1]], [r_[2] - r_[0]]], fmt="o", color=COL["raw"],
                      mfc="white", mec=COL["raw"], ms=4.2, lw=0.8, capsize=1.5, zorder=3)
        ax_c.errorbar(c_[0], yf, xerr=[[c_[0] - c_[1]], [c_[2] - c_[0]]], fmt="o", color=COL["cal"],
                      mfc=COL["cal"], mec=COL["cal"], ms=4.2, lw=0.8, capsize=1.5, zorder=3)
        ax_c.errorbar(a_[0], ya, xerr=[[a_[0] - a_[1]], [a_[2] - a_[0]]], fmt="^", color=COL["ar1"],
                      mfc=COL["ar1"], mec=COL["ar1"], ms=3.6, lw=0.7, capsize=1.2, zorder=2)
        data["panel_c"][key] = dict(raw=r_, calibrated=c_, ar1=a_, n_patients=int(len(Mr)))
        print(f"[fig2] {name:15s} calibration error raw {r_[0]:.4f} -> calibrated {c_[0]:.4f} | AR(1) {a_[0]:.4f} | n={len(Mr)}")
    ax_c.set_yticks([ypos[k] for k, _, _ in COHORTS])
    ax_c.set_yticklabels([f"{n}\n{h} T" for _, n, h in COHORTS], fontsize=6.8)
    ax_c.set_ylim(-0.6, len(COHORTS) - 0.4)
    ax_c.set_xlim(0, None)
    ax_c.set_xlabel("Calibration error")
    ax_c.set_title("Calibration error by cohort", loc="left", pad=3)
    ax_c.tick_params(axis="y", length=0)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax.text(-0.2, 1.1, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
        ax.grid(axis="x" if ax is ax_c else "y", color="0.92", lw=0.5, zorder=0); ax.set_axisbelow(True)

    handles = [Line2D([], [], color=COL["ideal"], ls=(0, (4, 2)), lw=0.9),
               Line2D([], [], color=COL["raw"], ls="--", marker="o", mfc="white"),
               Line2D([], [], color=COL["cal"], ls="-", marker="o"),
               Line2D([], [], color=COL["ar1"], ls="-", lw=0.9, marker="^", ms=3.2),
               Line2D([], [], color=COL["ideal"], ls=(0, (1, 1.5)), lw=0.8)]
    labels = ["Ideal (calibrated 32-member ensemble)", "Bio-SSM-Flow, raw", "Bio-SSM-Flow, calibrated", "AR(1)", "Uniform ranks"]
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0), handlelength=2.2, columnspacing=1.4)

    fig.savefig(out_stem + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    with open(out_stem + "_data.json", "w") as f:
        json.dump(data, f, indent=1)
    print(f"[fig2] saved {out_stem}.pdf, {out_stem}.png, {out_stem}_data.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_dir", default=os.path.join(HERE, "runs", "eval_unified", "flow"))
    ap.add_argument("--ar1_dir", default=os.path.join(HERE, "runs", "eval_unified", "ar1"))
    ap.add_argument("--external_dir", default=os.path.join(HERE, "runs", "external_eval"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig_calibration"))
    ap.add_argument("--synthetic", action="store_true", help="layout test with synthetic data")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.synthetic:
        internal, external = synthetic()
        args.out += "_SYNTHETIC"
    else:
        internal, external = load_internal(args.flow_dir, args.ar1_dir), load_external(args.external_dir)
    make_figure(internal, external, args.out)


if __name__ == "__main__":
    main()
