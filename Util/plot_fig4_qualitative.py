#!/usr/bin/env python3
"""
plot_fig4_qualitative.py — Fig. 4: representative forecasts, one row per cohort.

Case selection (fixed rule, no hand-picking): in each cohort, among pairs with an evaluable
tumor region, the pair whose tumor-region improvement over AR(1), 1 - MSE_flow / MSE_AR(1),
is closest to the cohort median. Improvements come from the saved evaluation records.

Columns (contrast-enhanced T1, axial slice with the largest tumor-region area):
  baseline (decoded) | observed follow-up (decoded) | Bio-SSM-Flow forecast (mean of 8 decoded
  members) | AR(1) forecast (decoded) | |forecast - observed| | calibrated uncertainty
  (voxel-wise SD of the 8 decoded members x T). Error and uncertainty share one colour scale.

Uses the frozen fold model assigned to the pair, its full-fold temperature, the frozen AR(1) of
the same fold, and the same per-pair random seed as the evaluation.

Usage:  CUDA_VISIBLE_DEVICES=0 python plot_fig4_qualitative.py         (real data, GPU)
        python plot_fig4_qualitative.py --synthetic                     (layout test, CPU)
Outputs: figures/fig_qualitative.pdf, .png, _data.json
"""
from __future__ import annotations
import os, sys, json, glob, argparse
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COHORTS = [("lumiere", "LUMIERE"), ("alptdg", "UCSF-ALPTDG"), ("mu", "MU-Glioma-Post")]
COLS = ["Baseline", "Follow-up\n(observed)", "Bio-SSM-Flow\nforecast", "AR(1)\nforecast",
        "|Forecast \u2212\nobserved|", "Uncertainty\n(calibrated)"]
K_DECODE, T1C = 8, 1                      # decoded members; channel index of contrast-enhanced T1
ASPECT = (218 / 128) / (182 / 128)       # display aspect: 128^3 grid resampled from 182 x 218 x 182 (1 mm)
CONTOUR = "#00E5FF"


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ───────────────────────────── case selection ─────────────────────────────
def select_cases():
    cases = {}
    flow = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "eval_unified", "flow", "per_pair_fold*.json"))):
        for r in json.load(open(f)):
            flow[(r["fold"], r["idx"])] = r
    ar1 = {}
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "eval_unified", "ar1", "per_pair_fold*.json"))):
        for r in json.load(open(f)):
            ar1[(r["fold"], r["idx"])] = r
    imp = []
    for key, r in flow.items():
        a = ar1.get(key)
        if a and r.get("roi_tokens", 0) > 0 and np.isfinite(r["mse_roi_model"]) and a["mse_roi_model"] > 0:
            imp.append((1 - r["mse_roi_model"] / a["mse_roi_model"], r))
    cases["lumiere"] = _median_case(imp)
    for ck in ("alptdg", "mu"):
        R = [json.loads(l) for l in open(os.path.join(HERE, "runs", "external_eval", ck, "per_pair.jsonl"))]
        imp = [(1 - r["mse_roi_flow"] / r["mse_roi_ar1"], r) for r in R
               if r.get("roi_tokens", 0) > 0 and r["mse_roi_ar1"] > 0 and np.isfinite(r["mse_roi_flow"])]
        cases[ck] = _median_case(imp)
    return cases


def _median_case(imp):
    vals = np.array([v for v, _ in imp])
    med = float(np.median(vals))
    i = int(np.argmin(np.abs(vals - med)))
    return dict(rec=imp[i][1], improvement=float(vals[i]), median=med, n=len(vals))


# ───────────────────────────── forecasting one pair ─────────────────────────────
def load_pair(ck, rec):
    import torch
    import eval_unified as EU
    if ck == "lumiere":
        from cached_pair_dataset import CachedPairDataset
        s = CachedPairDataset(EU.CACHE)[rec["idx"]]
        return s["z_a"], s["z_b"], s["seg_a"], s["seg_b"], s["mask_b"], float(s["dt"])
    if ck == "alptdg":
        fa = os.path.join(HERE, "latent_cache_alptdg", f"{rec['patient']}_t1.pt")
        fb = os.path.join(HERE, "latent_cache_alptdg", f"{rec['patient']}_t2.pt")
    else:
        fa = os.path.join(HERE, "latent_cache_mu", f"{rec['patient']}_tp{rec['tp_a']}.pt")
        fb = os.path.join(HERE, "latent_cache_mu", f"{rec['patient']}_tp{rec['tp_b']}.pt")
    a, b = torch.load(fa, weights_only=False), torch.load(fb, weights_only=False)
    return a["z"], b["z"], a["seg"], b["seg"], b["mask"], float(rec.get("dt_used", 84.0))


def forecast_case(ck, case, vq, nets, base, Tk):
    import torch
    import torch.nn.functional as F
    import eval_unified as EU
    rec = case["rec"]
    k = int(rec["fold"])
    z_a, z_b, seg_a, seg_b, mask_b, dt = load_pair(ck, rec)
    dev = EU.DEV
    z_a = z_a.float()[None].to(dev); z_b = z_b.float()[None].to(dev)
    seg_a = seg_a.to(dev).long().reshape(*seg_a.shape[-3:]); seg_b = seg_b.to(dev).long().reshape(*seg_b.shape[-3:])
    brain = mask_b.to(dev).bool().reshape(*mask_b.shape[-3:])
    if k not in nets:
        nets[k] = EU.load_model(SimpleNamespace(transport="flow", ckpt_pattern=""), k)[0]
    ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=32, chunk=16, steps=20, ddim_steps=50)
    if ck == "lumiere":
        seed = 0 * 100003 + int(rec["idx"])
    else:
        import eval_external as EE
        seed = EE.pair_seed(rec, 0)
    with torch.no_grad():
        torch.manual_seed(seed)
        samp = EU.sample_ensemble(ns, nets[k], None, z_a, torch.tensor([dt], device=dev))
        members = torch.stack([vq.decode(samp[r])[0] for r in range(K_DECODE)])        # [K,4,D,H,W]
        b_ = base[k]
        x_ar1 = vq.decode((b_["alpha"] + b_["beta"] * z_a[0])[None])[0]
        x_a, x_b = vq.decode(z_a)[0], vq.decode(z_b)[0]
        fc = members.mean(0)
        unc = Tk[k] * members[:, T1C].std(0)
        roi = F.max_pool3d((EU.label_mask(seg_a, EU.CORE) | EU.label_mask(seg_b, EU.CORE)).float()[None, None], 7, 1, 3)[0, 0] > 0.5
        roi = roi & brain
    zsl = int(torch.argmax(roi.sum((0, 1))).item())
    sl = lambda v: v[..., zsl].detach().float().cpu().numpy()
    return dict(base=sl(x_a[T1C]), obs=sl(x_b[T1C]), fc=sl(fc[T1C]), ar1=sl(x_ar1[T1C]),
                err=np.abs(sl(fc[T1C]) - sl(x_b[T1C])), unc=sl(unc), roi=sl(roi.float()) > 0.5,
                brain=sl(brain.float()) > 0.5, slice=zsl, fold=k)


def compute_real(cases):
    import torch
    import eval_unified as EU
    from vqgan3d import VQGAN3D
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck_ = torch.load(EU.VQGAN, map_location="cpu", weights_only=False)
    vq.load_state_dict(ck_.get("gen", ck_), strict=False)
    vq = vq.eval().to(EU.DEV)
    Tk = {int(k): float(v) for k, v in json.load(open(os.path.join(HERE, "runs", "FROZEN_T_full_fold.json"))).items()}
    B = torch.load(os.path.join(HERE, "runs", "FROZEN_BASELINES.pt"), map_location="cpu", weights_only=False)
    base = {k: {n: B[k]["ar1"][n].to(EU.DEV).float() for n in ("alpha", "beta")} for k in B}
    nets, panels = {}, {}
    for ck, name in COHORTS:
        print(f"[fig4] {name}: pair {case_id(ck, cases[ck]['rec'])} (fold {cases[ck]['rec']['fold']}), "
              f"improvement {100 * cases[ck]['improvement']:+.1f}% (cohort median {100 * cases[ck]['median']:+.1f}%, n={cases[ck]['n']})", flush=True)
        panels[ck] = forecast_case(ck, cases[ck], vq, nets, base, Tk)
    return panels


def case_id(ck, rec):
    if ck == "lumiere":
        return f"{rec['patient']} weeks {rec.get('week_a')}->{rec.get('week_b')}"
    return f"{rec['patient']} tp {rec.get('tp_a')}->{rec.get('tp_b')}"


def synthetic():
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:128, 0:128]
    brain = ((xx - 64) / 44) ** 2 + ((yy - 66) / 54) ** 2 < 1
    panels, cases = {}, {}
    for i, (ck, _) in enumerate(COHORTS):
        tum = ((xx - 50 - 8 * i) / 10) ** 2 + ((yy - 55) / 12) ** 2 < 1
        base = brain * (0.2 + 0.3 * rng.rand(128, 128)) + tum * 1.5
        obs = base + tum * 0.4 + 0.1 * rng.randn(128, 128) * brain
        fc = base + tum * 0.25
        ar1 = 0.6 * base
        unc = brain * (0.08 + 0.25 * tum + 0.03 * rng.rand(128, 128))
        panels[ck] = dict(base=base, obs=obs, fc=fc, ar1=ar1, err=np.abs(fc - obs) * brain, unc=unc,
                          roi=tum, brain=brain, slice=64, fold=i)
        cases[ck] = dict(improvement=[0.043, 0.081, 0.017][i], median=[0.043, 0.081, 0.017][i], n=[81, 187, 143][i],
                         rec=dict(patient=f"SYN-{i}", fold=i, idx=0, tp_a=1, tp_b=2))
    return panels, cases


# ───────────────────────────── figure ─────────────────────────────
def make_figure(panels, cases, out_stem):
    apply_style()
    errs = np.concatenate([p["err"][p["brain"]] for p in panels.values()] + [p["unc"][p["brain"]] for p in panels.values()])
    vmax = float(np.percentile(errs, 99))
    boxes = {}
    for ck, _ in COHORTS:
        ys, xs = np.where(panels[ck]["brain"])
        boxes[ck] = (max(ys.min() - 3, 0), ys.max() + 4, max(xs.min() - 3, 0), xs.max() + 4)
    # height/width of a displayed panel (after rot90 rows <- second axis) at the physical aspect
    ratio = max((b[3] - b[2]) * ASPECT / (b[1] - b[0]) for b in boxes.values())
    W, left, right, top, bottom, wsp, hsp = 7.16, 0.085, 0.955, 0.9, 0.02, 0.04, 0.05
    panel_w = (right - left) * W / (6 + 0.06 + 6 * wsp)
    H = 3 * panel_w * ratio * (1 + hsp) / (top - bottom)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 7, width_ratios=[1] * 6 + [0.06], wspace=wsp, hspace=hsp,
                          left=left, right=right, top=top, bottom=bottom)
    hm = None
    for i, (ck, name) in enumerate(COHORTS):
        p = panels[ck]
        y0, y1, x0, x1 = boxes[ck]
        crop = lambda a: np.rot90(a[y0:y1, x0:x1])
        lo, hi = np.percentile(p["base"][p["brain"]], [1, 99.5])
        imgs = [p["base"], p["obs"], p["fc"], p["ar1"], p["err"], p["unc"]]
        for j, im in enumerate(imgs):
            ax = fig.add_subplot(gs[i, j])
            if j < 4:
                ax.imshow(crop(np.where(p["brain"], im, lo)), cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos", aspect=ASPECT)
            else:
                m = np.ma.masked_where(~crop(p["brain"]), crop(im))
                cmap = plt.get_cmap("magma").copy(); cmap.set_bad("black")
                hm = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="lanczos", aspect=ASPECT)
            ax.contour(crop(p["roi"].astype(float)), levels=[0.5], colors=CONTOUR, linewidths=0.55)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_facecolor("black")
            if i == 0:
                ax.set_title(COLS[j], fontsize=7, pad=2.5, linespacing=1.05)
            if j == 0:
                imp = 100 * cases[ck]["improvement"]
                ax.set_ylabel(f"{name}\n{imp:+.1f}% vs. AR(1)", fontsize=7, labelpad=3, linespacing=1.15)
    cax = fig.add_subplot(gs[:, 6])
    cb = fig.colorbar(hm, cax=cax)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.set_label("Intensity (z-score units)", fontsize=6.5)
    fig.savefig(out_stem + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    meta = {ck: dict(pair=case_id(ck, cases[ck]["rec"]) if "patient" in cases[ck]["rec"] else None,
                     fold=int(cases[ck]["rec"]["fold"]), improvement_pct=100 * cases[ck]["improvement"],
                     cohort_median_pct=100 * cases[ck]["median"], n_pairs=cases[ck]["n"], slice=int(panels[ck]["slice"]))
            for ck, _ in COHORTS}
    meta["shared_colour_scale_max"] = vmax
    with open(out_stem + "_data.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[fig4] saved {out_stem}.pdf, .png, _data.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "fig_qualitative"))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.synthetic:
        panels, cases = synthetic()
        make_figure(panels, cases, args.out + "_SYNTHETIC")
    else:
        cases = select_cases()
        make_figure(compute_real(cases), cases, args.out)


if __name__ == "__main__":
    main()
