#!/usr/bin/env python3
"""
baseline_ar1.py — classical PROBABILISTIC forecasting baselines in latent space.

The decisive question: does the generative model beat what a simple statistical
forecaster achieves, both as a point forecast and as a calibrated distribution?

Fitted on each fold's TRAINING pairs only (same pairs the flow model was trained on),
evaluated on that fold's validation pairs with EXACTLY the eval_unified.py metrics:

  clim    Climatology:  z_b ~ N(mu_b, sigma^2)              (ignores the baseline)
  damped  Damped persistence: z_b ~ N(mu_b + beta_c (z_a - mu_a), sigma^2),
          one beta per latent channel (shrinks the baseline toward the voxel mean)
  ar1     Voxelwise AR(1): z_b ~ N(alpha_v + beta_v z_a, sigma_v^2), per latent voxel

mu, beta, alpha, sigma are per latent voxel/channel except where stated. The point
forecast is the ANALYTIC predictive mean (a slightly stronger baseline than a sample
mean); distributional metrics use R Gaussian samples, like the generative model.

Writes runs/eval_unified/<mode>/{per_pair_fold*.json, arrays_fold*.npz, meta.json},
so aggregate_unified.py compares it with the flow model directly (patient-level
Wilcoxon, cross-fitted temperature, Holm).

Run:   CUDA_VISIBLE_DEVICES=0 python baseline_ar1.py            # all modes, all folds, minutes
Then:  python aggregate_unified.py --runs runs/eval_unified/flow runs/eval_unified/ar1 \
           runs/eval_unified/damped runs/eval_unified/clim \
           --names Bio-SSM-Flow AR1 Damped Climatology --out runs/eval_unified/summary_baselines
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_unified as EU
from cached_pair_dataset import CachedPairDataset, patient_kfold

DEV = EU.DEV


@torch.no_grad()
def fit(ds, idx, mode):
    A = torch.stack([ds[i]["z_a"] for i in idx]).to(DEV).float()     # [N,c,h,w,d]
    B = torch.stack([ds[i]["z_b"] for i in idx]).to(DEV).float()
    ma, mb = A.mean(0), B.mean(0)
    if mode == "clim":
        beta = torch.zeros_like(ma)
    else:
        da, db = A - ma, B - mb
        if mode == "ar1":
            beta = (da * db).mean(0) / (da * da).mean(0).clamp_min(1e-6)
        else:   # damped: one coefficient per channel
            bc = (da * db).sum((0, 2, 3, 4)) / (da * da).sum((0, 2, 3, 4)).clamp_min(1e-6)
            beta = bc.view(-1, 1, 1, 1).expand_as(ma).clone()
    alpha = mb - beta * ma
    sigma = (B - (alpha + beta * A)).std(0).clamp_min(1e-4)
    info = dict(n_train_pairs=len(idx), beta_mean=float(beta.mean()), beta_median=float(beta.median()),
                sigma_mean=float(sigma.mean()))
    if mode == "damped":
        info["beta_per_channel"] = [float(b) for b in beta[:, 0, 0, 0]]
    return alpha, beta, sigma, info


@torch.no_grad()
def eval_fold(mode, k, ds, folds, prev, args, outdir, meta):
    out_json = os.path.join(outdir, f"per_pair_fold{k}.json")
    out_npz = os.path.join(outdir, f"arrays_fold{k}.npz")
    if os.path.exists(out_json) and os.path.exists(out_npz) and not args.overwrite:
        print(f"[{mode}] fold {k}: exists -> skip"); return
    tr, va = folds[k]
    alpha, beta, sigma, info = fit(ds, tr, mode)
    meta["fits"][f"fold{k}"] = info
    print(f"[{mode}] fold {k}: fitted on {len(tr)} train pairs | beta mean {info['beta_mean']:.3f} "
          f"| sigma mean {info['sigma_mean']:.3f} | evaluating {len(va)} val pairs", flush=True)
    recs = []
    arr = {f"{q}_{r}": [] for q in ("crps", "cov", "rank") for r in EU.DIST_REGIONS}
    for i in va:
        s = ds[i]
        z_a = s["z_a"][None].to(DEV).float()
        z_b = s["z_b"][None].to(DEV).float()
        C = z_b.shape[1]
        seg_a = s["seg_a"].to(DEV).long().reshape(*s["seg_a"].shape[-3:])
        seg_b = s["seg_b"].to(DEV).long().reshape(*s["seg_b"].shape[-3:])
        brain = s["mask_b"].to(DEV).bool().reshape(*s["mask_b"].shape[-3:])
        brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
        roi_tok = EU.roi_latent(seg_a, seg_b, EU.CORE)
        roiwt_tok = EU.roi_latent(seg_a, seg_b, EU.WHOLE_TUMOR)
        regions = {"whole": None, "brain": brain_tok, "roi": roi_tok, "roiwt": roiwt_tok}

        mean = (alpha + beta * z_a[0])[None]                                   # analytic predictive mean
        torch.manual_seed(args.seed * 100003 + int(i))
        samp = (mean[0] + sigma * torch.randn(args.R, *mean.shape[1:], device=DEV)).unsqueeze(1)
        z_lin, has_prev = EU.linear_forecast(s, z_a, prev, args.linear_cap)

        rec = dict(fold=int(k), idx=int(i), patient=str(s["patient"]),
                   week_a=int(s["week_pair"][0]), week_b=int(s["week_pair"][1]),
                   dt=float(s["dt"]), has_prev=bool(has_prev),
                   brain_tokens=int(brain_tok.sum()), roi_tokens=int(roi_tok.sum()),
                   roiwt_tokens=int(roiwt_tok.sum()), diversity=float(sigma.mean()))
        for name, pred in (("model", mean), ("copy", z_a), ("linear", z_lin)):
            for reg, tm in regions.items():
                mse, cos = EU.masked_mse_cos(pred, z_b, tm)
                rec[f"mse_{reg}_{name}"], rec[f"cos_{reg}_{name}"] = mse, cos

        S_all, y_all = samp[:, 0].reshape(args.R, -1), z_b[0].reshape(-1)
        g = torch.Generator(device=DEV).manual_seed(args.seed * 100003 + int(i))
        for reg in EU.DIST_REGIONS:
            tm = regions[reg]
            if tm is None:
                S, y = S_all, y_all
            else:
                em = tm[None].expand(C, *tm.shape).reshape(-1)
                if int(em.sum()) == 0:
                    arr[f"crps_{reg}"].append(np.full(len(EU.T_GRID), np.nan))
                    arr[f"cov_{reg}"].append(np.full((len(EU.T_GRID), len(EU.NOMINAL)), np.nan))
                    arr[f"rank_{reg}"].append(np.zeros((len(EU.T_GRID), args.R + 1)))
                    continue
                S, y = S_all[:, em], y_all[em]
            c_, v_, r_ = EU.dist_metrics(S, y, g)
            arr[f"crps_{reg}"].append(c_); arr[f"cov_{reg}"].append(v_); arr[f"rank_{reg}"].append(r_)

        err_tok = (mean - z_b).pow(2).mean(1)[0].sqrt()
        unc_tok = sigma.pow(2).mean(0).sqrt()
        for reg in EU.DIST_REGIONS:
            tm = regions[reg]
            if tm is None:
                rec[f"errunc_{reg}"] = EU.spearman(err_tok.reshape(-1), unc_tok.reshape(-1))
            elif int(tm.sum()) < 10:
                rec[f"errunc_{reg}"] = float("nan")
            else:
                rec[f"errunc_{reg}"] = EU.spearman(err_tok[tm], unc_tok[tm])
        recs.append(rec)

    np.savez_compressed(out_npz, **{key: np.stack(v).astype(np.int32 if key.startswith("rank") else np.float32)
                                    for key, v in arr.items()})
    with open(out_json, "w") as f:
        json.dump(recs, f)
    print(f"[{mode}] fold {k} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["ar1", "damped", "clim"], choices=["ar1", "damped", "clim"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--linear_cap", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = CachedPairDataset(EU.CACHE)
    folds = list(patient_kfold(ds, k=5))
    prev = EU.build_prev_index()
    for mode in args.modes:
        outdir = os.path.join(HERE, "runs", "eval_unified", mode)
        os.makedirs(outdir, exist_ok=True)
        meta = dict(protocol=dict(transport=mode, R=args.R, linear_cap=args.linear_cap, seed=args.seed,
                                  point_forecast="analytic predictive mean"),
                    T_GRID=EU.T_GRID.tolist(), NOMINAL=EU.NOMINAL.tolist(),
                    created=datetime.datetime.now().isoformat(timespec="seconds"), fits={})
        t0 = time.time()
        for k in args.folds:
            eval_fold(mode, k, ds, folds, prev, args, outdir, meta)
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{mode}] all folds done in {(time.time() - t0) / 60:.1f} min -> {outdir}")


if __name__ == "__main__":
    main()
