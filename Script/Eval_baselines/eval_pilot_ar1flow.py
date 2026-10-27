#!/usr/bin/env python3
"""
eval_pilot_ar1flow.py — evaluate the AR(1)-anchored flow pilot and apply the
PRE-SPECIFIED success criteria.

Writes runs/eval_unified/ar1flow/{per_pair_fold*.json, arrays_fold*.npz, meta.json}
in exactly the eval_unified.py format (latent metrics; no image decoding), then compares
it on the SAME patients with the current flow model (runs/eval_unified/flow) and the
voxelwise AR(1) baseline (runs/eval_unified/ar1), using patient-level paired tests.

PRE-SPECIFIED SUCCESS (all three):
  C1  brain MSE not worse than AR(1): lower 95% CI of relative improvement > -1%
  C2  tumor-ROI MSE AND calibrated tumor-ROI CRPS better than AR(1), Holm p < 0.05
  C3  tumor-ROI MSE not worse than the current flow: lower 95% CI > -1%

Run (one fold per GPU, then summarise):
  CUDA_VISIBLE_DEVICES=0 python eval_pilot_ar1flow.py --folds 0
  CUDA_VISIBLE_DEVICES=1 python eval_pilot_ar1flow.py --folds 1
  python eval_pilot_ar1flow.py --summarize_only
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_unified as EU
import aggregate_unified as AG
from cached_pair_dataset import CachedPairDataset, patient_kfold
from velocity_net_v2 import VelocityNetV2

DEV = EU.DEV
OUTDIR = os.path.join(HERE, "runs", "eval_unified", "ar1flow")


@torch.no_grad()
def eval_fold(k, args, ds, folds, prev):
    out_json = os.path.join(OUTDIR, f"per_pair_fold{k}.json")
    out_npz = os.path.join(OUTDIR, f"arrays_fold{k}.npz")
    if os.path.exists(out_json) and os.path.exists(out_npz) and not args.overwrite:
        print(f"[ar1flow-eval] fold {k}: exists -> skip"); return
    ck_path = os.path.join(HERE, "runs", "pilot_ar1flow", f"fold{k}", "final.pt")
    st = torch.load(ck_path, map_location=DEV, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                        use_checkpoint=False).to(DEV)
    net.load_state_dict(st["ema"]); net.eval(); net.set_week(None)
    alpha, beta, s = st["alpha"].to(DEV), st["beta"].to(DEV), st["sigma"].to(DEV)
    ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=args.R, chunk=args.chunk, steps=20, ddim_steps=50)
    va = folds[k][1]
    print(f"[ar1flow-eval] fold {k}: {len(va)} val pairs | {ck_path} (sha {EU.sha256(ck_path)[:12]})", flush=True)

    recs, t0 = [], time.time()
    arr = {f"{q}_{r}": [] for q in ("crps", "cov", "rank") for r in EU.DIST_REGIONS}
    for n_done, i in enumerate(va):
        s_ = ds[i]
        z_a = s_["z_a"][None].to(DEV).float(); z_b = s_["z_b"][None].to(DEV).float()
        dt = s_["dt"].view(1).to(DEV).float()
        C = z_b.shape[1]
        seg_a = s_["seg_a"].to(DEV).long().reshape(*s_["seg_a"].shape[-3:])
        seg_b = s_["seg_b"].to(DEV).long().reshape(*s_["seg_b"].shape[-3:])
        brain = s_["mask_b"].to(DEV).bool().reshape(*s_["mask_b"].shape[-3:])
        brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
        roi_tok = EU.roi_latent(seg_a, seg_b, EU.CORE)
        roiwt_tok = EU.roi_latent(seg_a, seg_b, EU.WHOLE_TUMOR)
        regions = {"whole": None, "brain": brain_tok, "roi": roi_tok, "roiwt": roiwt_tok}

        torch.manual_seed(args.seed * 100003 + int(i))
        r = EU.sample_ensemble(ns, net, None, z_a, dt)                   # [R,1,c,h,w,d] residuals
        samp = (alpha + beta * z_a[0]) + s * r                           # anchored forecasts
        mean = samp.mean(0)
        z_lin, has_prev = EU.linear_forecast(s_, z_a, prev, 1.0)

        rec = dict(fold=int(k), idx=int(i), patient=str(s_["patient"]),
                   week_a=int(s_["week_pair"][0]), week_b=int(s_["week_pair"][1]),
                   dt=float(s_["dt"]), has_prev=bool(has_prev),
                   brain_tokens=int(brain_tok.sum()), roi_tokens=int(roi_tok.sum()),
                   roiwt_tokens=int(roiwt_tok.sum()), diversity=float(samp.std(0).mean()))
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
        unc_tok = samp.var(0).mean(1)[0].sqrt()
        for reg in EU.DIST_REGIONS:
            tm = regions[reg]
            if tm is None:
                rec[f"errunc_{reg}"] = EU.spearman(err_tok.reshape(-1), unc_tok.reshape(-1))
            elif int(tm.sum()) < 10:
                rec[f"errunc_{reg}"] = float("nan")
            else:
                rec[f"errunc_{reg}"] = EU.spearman(err_tok[tm], unc_tok[tm])
        recs.append(rec)
        if (n_done + 1) % 10 == 0 or n_done == 0:
            el = time.time() - t0
            print(f"    fold {k}: {n_done + 1}/{len(va)} | {el / (n_done + 1):.1f}s/pair | "
                  f"ETA {el / (n_done + 1) * (len(va) - n_done - 1) / 60:.0f} min", flush=True)
            torch.cuda.empty_cache()

    np.savez_compressed(out_npz, **{key: np.stack(v).astype(np.int32 if key.startswith("rank") else np.float32)
                                    for key, v in arr.items()})
    with open(out_json, "w") as f:
        json.dump(recs, f)
    print(f"[ar1flow-eval] fold {k} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def criteria():
    runs = {n: AG.prepare(n, os.path.join(HERE, "runs", "eval_unified", n), "brain")
            for n in ("ar1flow", "flow", "ar1")}
    A = runs["ar1flow"]["recs"]

    def cmp(other, key):
        a, b, n_common = AG.aligned_across(A, runs[other]["recs"], key)
        t = AG.paired_test(a, b); t["n_common_pairs"] = n_common
        return t

    t_brain = cmp("ar1", "mse_brain_model")
    t_roi = cmp("ar1", "mse_roi_model")
    t_crps = cmp("ar1", "crps_cal_roi")
    for t, pa in zip((t_roi, t_crps), AG.holm([t_roi["p"], t_crps["p"]])):
        t["p_holm"] = float(pa)
    t_vsflow = cmp("flow", "mse_roi_model")
    extra = {f"{o} | {k}": cmp(o, k) for o in ("ar1", "flow")
             for k in ("mse_whole_model", "cos_roi_model", "crps_cal_brain", "crps_raw_brain")}

    c1 = t_brain["ci95_pct"][0] > -1.0
    c2 = all(t["p_holm"] < 0.05 and t["mean_a"] < t["mean_b"] for t in (t_roi, t_crps))
    c3 = t_vsflow["ci95_pct"][0] > -1.0
    line = lambda lab, t: (f"  {lab:<44} {t['mean_a']:.4f} vs {t['mean_b']:.4f}  rel {t['rel_improvement_pct']:+5.1f}% "
                           f"[{t['ci95_pct'][0]:+.1f},{t['ci95_pct'][1]:+.1f}]  "
                           + (f"p_holm={t['p_holm']:.1e}" if "p_holm" in t else f"p={t['p']:.1e}")
                           + f"  n={t['n_patients']}")
    print("=" * 100)
    print("  AR(1)-ANCHORED FLOW PILOT — pre-specified criteria (patient-level, same patients across models)")
    print("=" * 100)
    print(line("C1 brain MSE vs AR(1)  [need lower CI > -1%]", t_brain), "->", "PASS" if c1 else "FAIL")
    print(line("C2a ROI MSE vs AR(1)   [need Holm p<0.05, better]", t_roi))
    print(line("C2b ROI calibrated CRPS vs AR(1)", t_crps), "->", "PASS" if c2 else "FAIL")
    print(line("C3 ROI MSE vs current flow [need lower CI > -1%]", t_vsflow), "->", "PASS" if c3 else "FAIL")
    print("-" * 100)
    print("  Secondary (unadjusted; for cos_roi a NEGATIVE rel means the anchored flow is better):")
    for lab, t in extra.items():
        print(line(lab, t))
    print("=" * 100)
    verdict = c1 and c2 and c3
    print(f"  VERDICT: {'SUCCESS -> freeze the AR(1)-anchored flow for external validation' if verdict else 'NOT MET -> freeze the current flow model'}")
    res = dict(C1=dict(test=t_brain, pass_=c1), C2=dict(roi=t_roi, crps=t_crps, pass_=c2),
               C3=dict(test=t_vsflow, pass_=c3), secondary=extra, success=verdict)
    with open(os.path.join(OUTDIR, "pilot_criteria.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {OUTDIR}/pilot_criteria.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    meta_path = os.path.join(OUTDIR, "meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump(dict(protocol=dict(transport="ar1flow", R=args.R, steps=20, seed=args.seed),
                           T_GRID=EU.T_GRID.tolist(), NOMINAL=EU.NOMINAL.tolist(),
                           created=datetime.datetime.now().isoformat(timespec="seconds")), f, indent=2)
    if not args.summarize_only:
        ds = CachedPairDataset(EU.CACHE)
        folds = list(patient_kfold(ds, k=5))
        prev = EU.build_prev_index()
        for k in args.folds:
            eval_fold(k, args, ds, folds, prev)
    done = [k for k in range(5) if os.path.exists(os.path.join(OUTDIR, f"per_pair_fold{k}.json"))]
    if len(done) >= 2 or args.summarize_only:
        criteria()
    else:
        print(f"[ar1flow-eval] folds finished so far: {done}; run --summarize_only once all pilot folds are done.")


if __name__ == "__main__":
    main()
