#!/usr/bin/env python3
"""
diag_validity.py — three validity checks on the FROZEN flow model (default: fold 0).

A. INTENSITY DRIFT. Is the model forecasting tumor evolution, or mostly predicting the
   global intensity shift caused by per-timepoint z-score normalisation?
   In image space, each prediction P gets a per-contrast affine intensity map a*P+b fitted
   to the true follow-up on NORMAL brain (brain minus dilated tumor+edema of both
   timepoints). This is an ORACLE correction (it uses the truth) and is diagnostic only.
   Compared on the tumor ROI (GT ET∪NETC at baseline∪follow-up, dilated 3 voxels):
     model(raw)     vs copy(raw)
     model(raw)     vs copy(matched)   — copy is given the oracle intensity correction
     model(matched) vs copy(matched)   — both corrected: the fair "tumor-only" comparison
   Also: % of Copy's normal-tissue error explained by a global affine intensity shift.

B. SEGMENTATION WITH A FAIR BASELINE. "Pipeline Copy" = seg-head(decode(z_a)) vs GT
   follow-up, i.e. Copy passed through the SAME decode->segment pipeline as the forecast.
   Also: forecast (decoded ensemble mean), single decoded sample, ceiling (decoded true
   follow-up), and GT Copy (GT baseline labels). TC growth direction vs trivial baselines.

C. IMAGE FIDELITY OF SAMPLES VS MEAN. PSNR/SSIM (brain voxels) for Copy, the decoded
   ensemble-mean latent, individual decoded samples, and the image-space mean of K
   decoded samples — tests whether the blur of the latent mean explains the SSIM gap.

Reuses eval_unified.py (hash-verified frozen models, sampling, masks, metrics) and
aggregate_unified.py (patient-level statistics). Seg metrics are held-out only for fold 0.

Run:  CUDA_VISIBLE_DEVICES=0 python diag_validity.py                 # fold 0, ~1 h
      CUDA_VISIBLE_DEVICES=0 python diag_validity.py --max_pairs 3 --tag smoke
Out:  runs/diag_validity/<tag>/per_pair_fold{k}.json, summary.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_unified as EU            # sets up paths; frozen-model loading, sampling, metrics
import aggregate_unified as AG       # by_patient, paired_test, holm, direction
from cached_pair_dataset import CachedPairDataset, patient_kfold
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet

DEV = EU.DEV
PRED_NAMES = ("copy", "model", "sample", "imgmean")
SEG_SOURCES = ("gt_copy", "pipeline_copy", "forecast", "sample", "ceiling")


def dilate(m, r):
    return F.max_pool3d(m.float()[None, None], 2 * r + 1, 1, r)[0, 0] > 0.5


def region_mse(P, Y, m):
    if int(m.sum()) == 0:
        return float("nan")
    return float(((P - Y) ** 2)[:, m].mean())


def affine_match(P, Y, fit_mask):
    """Per-contrast least-squares a,b so that a*P+b ≈ Y on fit_mask (oracle; diagnostic)."""
    out = torch.empty_like(P)
    for c in range(P.shape[0]):
        p, y = P[c][fit_mask], Y[c][fit_mask]
        pm, ym = p.mean(), y.mean()
        a = ((p - pm) * (y - ym)).mean() / ((p - pm) ** 2).mean().clamp_min(1e-8)
        out[c] = a * P[c] + (ym - a * pm)
    return out


def seg_labels(seg, x):
    return seg(x[None]).argmax(1)[0]


@torch.no_grad()
def run_fold(args, k, ds, folds, vq, seg, outdir):
    out = os.path.join(outdir, f"per_pair_fold{k}.json")
    if os.path.exists(out) and not args.overwrite:
        print(f"[diag] fold {k}: exists -> reusing {out}")
        return json.load(open(out))
    ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=args.R, chunk=args.chunk,
                         steps=20, ddim_steps=50)
    net, ddpm, info = EU.load_model(ns, k)
    val = list(folds[k][1])
    if args.max_pairs:
        val = val[:args.max_pairs]
    print(f"[diag] fold {k}: {len(val)} pairs | R={args.R} K={args.K} | {info['path']} "
          f"({'verified' if info['verified'] else 'UNVERIFIED'})", flush=True)

    recs, t0 = [], time.time()
    for n_done, i in enumerate(val):
        s = ds[i]
        z_a = s["z_a"][None].to(DEV).float()
        z_b = s["z_b"][None].to(DEV).float()
        dt = s["dt"].view(1).to(DEV).float()
        seg_a = s["seg_a"].to(DEV).long().reshape(*s["seg_a"].shape[-3:])
        seg_b = s["seg_b"].to(DEV).long().reshape(*s["seg_b"].shape[-3:])
        brain = s["mask_b"].to(DEV).bool().reshape(*s["mask_b"].shape[-3:])
        roi = dilate(EU.label_mask(seg_a, EU.CORE) | EU.label_mask(seg_b, EU.CORE), 3) & brain
        tumor_any = dilate(EU.label_mask(seg_a, EU.WHOLE_TUMOR) | EU.label_mask(seg_b, EU.WHOLE_TUMOR), 5)
        normal = brain & ~tumor_any

        torch.manual_seed(args.seed * 100003 + int(i))
        samp = EU.sample_ensemble(ns, net, ddpm, z_a, dt)             # [R,1,c,h,w,d]
        mean = samp.mean(0)
        x_b, x_a, x_fc = vq.decode(z_b)[0], vq.decode(z_a)[0], vq.decode(mean)[0]
        xs = [vq.decode(samp[r])[0] for r in range(min(args.K, args.R))]
        x_im = torch.stack(xs).mean(0)

        rec = dict(fold=int(k), idx=int(i), patient=str(s["patient"]),
                   roi_vox=int(roi.sum()), normal_vox=int(normal.sum()))
        fit_ok = int(normal.sum()) >= 1000
        rec["matching_done"] = bool(fit_ok)

        # A + C: image-space errors (raw and intensity-matched) and fidelity
        for name, plist in (("copy", [x_a]), ("model", [x_fc]), ("sample", xs), ("imgmean", [x_im])):
            acc = {}
            for P in plist:
                Pm = affine_match(P, x_b, normal) if fit_ok else P
                for reg, m in (("roi", roi), ("brain", brain), ("normal", normal)):
                    acc.setdefault(f"mse_{reg}_raw", []).append(region_mse(P, x_b, m))
                    acc.setdefault(f"mse_{reg}_matched", []).append(region_mse(Pm, x_b, m))
                acc.setdefault("psnr", []).append(EU.psnr_masked(P, x_b, brain))
                acc.setdefault("ssim", []).append(EU.ssim_masked(P, x_b, brain))
            for key, v in acc.items():
                rec[f"{name}_{key}"] = float(np.nanmean(v))

        # B: segmentation with a pipeline-matched Copy baseline
        labs = {"gt_copy": seg_a,
                "pipeline_copy": seg_labels(seg, x_a),
                "forecast": seg_labels(seg, x_fc),
                "sample": seg_labels(seg, xs[0]),
                "ceiling": seg_labels(seg, x_b)}
        for src, lab in labs.items():
            for reg, cls in EU.SEG_REGIONS.items():
                rec[f"dice_{src}_{reg}"] = EU.dice(EU.label_mask(lab, cls), EU.label_mask(seg_b, cls))
            rec[f"vol_TC_{src}"] = int(EU.label_mask(lab, EU.SEG_REGIONS["TC"]).sum())
        rec["vol_TC_gt_b"] = int(EU.label_mask(seg_b, EU.SEG_REGIONS["TC"]).sum())

        recs.append(rec)
        del samp, mean, xs, x_im, x_a, x_b, x_fc
        if (n_done + 1) % 10 == 0 or n_done == 0:
            el = time.time() - t0
            print(f"    fold {k}: {n_done + 1}/{len(val)} | {el / (n_done + 1):.1f}s/pair | "
                  f"ETA {el / (n_done + 1) * (len(val) - n_done - 1) / 60:.0f} min", flush=True)
            torch.cuda.empty_cache()

    with open(out, "w") as f:
        json.dump(recs, f)
    return recs


def summarize(recs):
    def pm(key):
        d = AG.by_patient(recs, key)
        return np.array([d[p][1] for p in sorted(d)], float)

    S = {"n_pairs": len(recs), "n_patients": len({r["patient"] for r in recs}),
         "folds": sorted({r["fold"] for r in recs}),
         "pairs_without_intensity_matching": int(sum(not r["matching_done"] for r in recs))}

    S["A_image_errors"] = {n: {k: float(np.nanmean(pm(f"{n}_{k}"))) for k in
                               ("mse_roi_raw", "mse_roi_matched", "mse_brain_raw", "mse_brain_matched",
                                "mse_normal_raw", "mse_normal_matched", "psnr", "ssim")}
                           for n in PRED_NAMES}
    c = S["A_image_errors"]["copy"]
    S["A_copy_normal_tissue_error_explained_by_global_intensity_pct"] = \
        100 * (1 - c["mse_normal_matched"] / c["mse_normal_raw"])

    tests = {
        "A1 ROI image MSE: model(raw) vs copy(raw)": AG.paired_test(pm("model_mse_roi_raw"), pm("copy_mse_roi_raw")),
        "A2 ROI image MSE: model(raw) vs copy(matched)": AG.paired_test(pm("model_mse_roi_raw"), pm("copy_mse_roi_matched")),
        "A3 ROI image MSE: model(matched) vs copy(matched)": AG.paired_test(pm("model_mse_roi_matched"), pm("copy_mse_roi_matched")),
        "C1 ROI image MSE: sample(raw) vs copy(raw)": AG.paired_test(pm("sample_mse_roi_raw"), pm("copy_mse_roi_raw")),
        "B1 Dice TC: forecast vs pipeline copy": AG.paired_test(pm("dice_forecast_TC"), pm("dice_pipeline_copy_TC")),
        "B2 Dice WT: forecast vs pipeline copy": AG.paired_test(pm("dice_forecast_WT"), pm("dice_pipeline_copy_WT")),
    }
    adj = AG.holm([t["p"] for t in tests.values()])
    for (label, t), pa in zip(tests.items(), adj):
        t["p_holm"] = float(pa)
    S["tests"] = tests

    S["B_dice"] = {src: {reg: float(np.nanmean(pm(f"dice_{src}_{reg}"))) for reg in EU.SEG_REGIONS}
                   for src in SEG_SOURCES}
    for r in recs:
        t = AG.direction(r["vol_TC_gt_copy"], r["vol_TC_gt_b"])
        r["_ok_model"] = float(AG.direction(r["vol_TC_pipeline_copy"], r["vol_TC_forecast"]) == t)
        r["_ok_ceiling"] = float(AG.direction(r["vol_TC_pipeline_copy"], r["vol_TC_ceiling"]) == t)
        r["_ok_grow"] = float(t == 1)
        r["_ok_stable"] = float(t == 0)
        r["_relerr_forecast"] = abs(r["vol_TC_forecast"] - r["vol_TC_gt_b"]) / max(r["vol_TC_gt_b"], 50)
        r["_relerr_pipecopy"] = abs(r["vol_TC_pipeline_copy"] - r["vol_TC_gt_b"]) / max(r["vol_TC_gt_b"], 50)
    S["B_TC_direction_accuracy"] = {k: float(np.nanmean(pm(f"_ok_{k}")))
                                    for k in ("model", "ceiling", "grow", "stable")}
    S["B_TC_volume_relerr"] = {"forecast": float(np.nanmean(pm("_relerr_forecast"))),
                               "pipeline_copy": float(np.nanmean(pm("_relerr_pipecopy")))}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--K", type=int, default=4, help="decoded samples for per-sample / image-mean metrics")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    outdir = os.path.join(HERE, "runs", "diag_validity", args.tag)
    os.makedirs(outdir, exist_ok=True)

    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck = torch.load(EU.VQGAN, map_location="cpu", weights_only=False)
    vq.load_state_dict(ck.get("gen", ck), strict=False)
    vq.eval().to(DEV)
    seg = ImgSegUNet().to(DEV)
    seg.load_state_dict(torch.load(EU.SEG, map_location=DEV, weights_only=False)["net"])
    seg.eval()
    for p in list(vq.parameters()) + list(seg.parameters()):
        p.requires_grad_(False)

    ds = CachedPairDataset(EU.CACHE)
    folds = list(patient_kfold(ds, k=5))
    recs = []
    for k in args.folds:
        recs += run_fold(args, k, ds, folds, vq, seg, outdir)

    S = summarize(recs)
    S["protocol"] = vars(args)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(S, f, indent=2)

    A = S["A_image_errors"]
    print("=" * 80)
    print(f"  VALIDITY DIAGNOSTICS  (folds {S['folds']}, {S['n_patients']} patients, {S['n_pairs']} pairs; "
          f"patient-level means)")
    print("=" * 80)
    print("  A. Image-space tumor-ROI MSE (vs decoded follow-up)      raw      intensity-matched")
    for n in PRED_NAMES:
        print(f"     {n:<10}                                        {A[n]['mse_roi_raw']:.4f}      {A[n]['mse_roi_matched']:.4f}")
    print(f"     Copy's normal-tissue error explained by a global intensity shift: "
          f"{S['A_copy_normal_tissue_error_explained_by_global_intensity_pct']:.1f}%")
    print("  C. Image fidelity on brain voxels                         PSNR     SSIM")
    for n in PRED_NAMES:
        print(f"     {n:<10}                                        {A[n]['psnr']:.2f}    {A[n]['ssim']:.3f}")
    print("  B. Dice vs GT follow-up (fold-0 seg head: held-out only for fold 0)")
    print("     " + " ".join(f"{reg:>6}" for reg in EU.SEG_REGIONS))
    for src in SEG_SOURCES:
        print(f"     " + " ".join(f"{S['B_dice'][src][reg]:6.3f}" for reg in EU.SEG_REGIONS) + f"   {src}")
    d = S["B_TC_direction_accuracy"]
    print(f"     TC growth direction: model {d['model']:.3f} | ceiling {d['ceiling']:.3f} | "
          f"always-grow {d['grow']:.3f} | always-stable {d['stable']:.3f}")
    v = S["B_TC_volume_relerr"]
    print(f"     TC volume rel. error: forecast {v['forecast']:.3f} | pipeline copy {v['pipeline_copy']:.3f}")
    print("-" * 80)
    for label, t in S["tests"].items():
        print(f"  {label:<48} n={t['n_patients']:>3}  mean {t['mean_a']:.4f} vs {t['mean_b']:.4f}  "
              f"p_holm={t['p_holm']:.1e}  r_rb={t['r_rb']:+.2f}")
    print("=" * 80)
    print("  READ: A3 (both intensity-matched) is the key line. If the model still clearly beats")
    print("        Copy there, its advantage is tumor-level forecasting, not global intensity drift.")
    print(f"[saved] {outdir}/summary.json")


if __name__ == "__main__":
    main()
