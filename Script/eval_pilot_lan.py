#!/usr/bin/env python3
"""
eval_pilot_lan.py — evaluate the lesion-aware pilot against the PRE-SPECIFIED criterion.

Runs (post-operative validation pairs of the requested folds, patient-level statistics):
  lan      : pilot, lesion-aware data, target z_b
  lan_res  : pilot, lesion-aware data, residual target (z_a + s*Δ)
  ref_v2   : the FROZEN reported V2 model on the ORIGINAL cache, same post-op pairs

Point forecast (primary, fixed in advance): image-space mean of K decoded samples.
Secondary: the decoded ensemble-mean latent.

Per run it reports:
  * Copy's normal-tissue error explained by a global intensity map (built-in check:
    should be near 0 for lan / lan_res if the normalisation fix works)
  * A1: tumor-ROI image MSE, forecast vs Copy (raw)
  * A3: the same after per-contrast intensity matching of BOTH on normal tissue (oracle)
  * latent tumor-ROI MSE vs Copy; raw CRPS and PICP@90 (brain tokens); PSNR/SSIM

PRE-SPECIFIED SUCCESS (per pilot variant): A1 AND A3 favour the forecast with
Holm-corrected p < 0.05 (Holm over the two primary tests).

Run (parallel, then summarise):
  CUDA_VISIBLE_DEVICES=0 python eval_pilot_lan.py --runs lan ref_v2
  CUDA_VISIBLE_DEVICES=1 python eval_pilot_lan.py --runs lan_res
  python eval_pilot_lan.py --runs lan lan_res ref_v2 --summarize_only
"""
from __future__ import annotations
import os, sys, json, time, argparse
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_unified as EU
import aggregate_unified as AG
from diag_validity import dilate, region_mse, affine_match
from cached_pair_dataset import CachedPairDataset, patient_kfold
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D

DEV = EU.DEV
CACHE_LAN = os.path.join(HERE, "latent_cache_lumiere_lan")
OUT = os.path.join(HERE, "runs", "pilot_lan", "eval")
T1 = int(np.argmin(np.abs(EU.T_GRID - 1.0)))
I90 = int(np.argmin(np.abs(EU.NOMINAL - 0.9)))


def postop(ds, idx):
    return [i for i in idx if not os.path.basename(ds.pairs[i][0]).endswith("_week0.pt")]


def load_run_model(run, k):
    if run == "ref_v2":
        net, _, info = EU.load_model(SimpleNamespace(transport="flow", ckpt_pattern=""), k)
        return net, None, EU.CACHE, info
    path = os.path.join(HERE, "runs", "pilot_lan", f"{run}_fold{k}", "final.pt")
    st = torch.load(path, map_location=DEV, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                        use_checkpoint=False).to(DEV)
    net.load_state_dict(st["ema"]); net.eval(); net.set_week(None)
    return net, st.get("res_scale"), CACHE_LAN, dict(path=path, sha256=EU.sha256(path), step=st.get("step"))


@torch.no_grad()
def eval_run(run, k, args, vq):
    out = os.path.join(OUT, f"{run}_fold{k}.json")
    if os.path.exists(out) and not args.overwrite:
        print(f"[pilot-eval] {run} fold {k}: exists -> reuse"); return
    net, res_scale, cache, info = load_run_model(run, k)
    ds = CachedPairDataset(cache)
    val = postop(ds, list(patient_kfold(ds, k=5))[k][1])
    if args.max_pairs:
        val = val[:args.max_pairs]
    ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=args.R, chunk=args.chunk, steps=20, ddim_steps=50)
    print(f"[pilot-eval] {run} fold {k}: {len(val)} post-op pairs | {info['path']}", flush=True)
    recs, t0 = [], time.time()
    for n_done, i in enumerate(val):
        s = ds[i]
        z_a = s["z_a"][None].to(DEV).float(); z_b = s["z_b"][None].to(DEV).float()
        dt = s["dt"].view(1).to(DEV).float()
        seg_a = s["seg_a"].to(DEV).long().reshape(*s["seg_a"].shape[-3:])
        seg_b = s["seg_b"].to(DEV).long().reshape(*s["seg_b"].shape[-3:])
        brain = s["mask_b"].to(DEV).bool().reshape(*s["mask_b"].shape[-3:])
        C = z_b.shape[1]

        torch.manual_seed(args.seed * 100003 + int(i))
        samp = EU.sample_ensemble(ns, net, None, z_a, dt)
        if res_scale:
            samp = z_a.unsqueeze(0) + res_scale * samp
        mean = samp.mean(0)
        rec = dict(run=run, fold=int(k), patient=str(s["patient"]),
                   week_a=int(s["week_pair"][0]), week_b=int(s["week_pair"][1]))

        # latent metrics
        roi_tok = EU.roi_latent(seg_a, seg_b, EU.CORE)
        brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
        rec["lat_mse_roi_model"], _ = EU.masked_mse_cos(mean, z_b, roi_tok)
        rec["lat_mse_roi_copy"], _ = EU.masked_mse_cos(z_a, z_b, roi_tok)
        em = brain_tok[None].expand(C, *brain_tok.shape).reshape(-1)
        g = torch.Generator(device=DEV).manual_seed(args.seed * 100003 + int(i))
        crps, cov, _ = EU.dist_metrics(samp[:, 0].reshape(args.R, -1)[:, em], z_b[0].reshape(-1)[em], g)
        rec["crps_brain_raw"], rec["picp90_brain_raw"] = float(crps[T1]), float(cov[T1, I90])

        # image metrics
        x_b, x_a, x_fc = vq.decode(z_b)[0], vq.decode(z_a)[0], vq.decode(mean)[0]
        x_im = torch.stack([vq.decode(samp[r])[0] for r in range(min(args.K, args.R))]).mean(0)
        roi = dilate(EU.label_mask(seg_a, EU.CORE) | EU.label_mask(seg_b, EU.CORE), 3) & brain
        tumor_any = dilate(EU.label_mask(seg_a, EU.WHOLE_TUMOR) | EU.label_mask(seg_b, EU.WHOLE_TUMOR), 5)
        normal = brain & ~tumor_any
        fit_ok = int(normal.sum()) >= 1000
        for name, P in (("copy", x_a), ("imgmean", x_im), ("latmean", x_fc)):
            Pm = affine_match(P, x_b, normal) if fit_ok else P
            for reg, m in (("roi", roi), ("normal", normal)):
                rec[f"{name}_mse_{reg}_raw"] = region_mse(P, x_b, m)
                rec[f"{name}_mse_{reg}_matched"] = region_mse(Pm, x_b, m)
        for name, P in (("copy", x_a), ("imgmean", x_im)):
            rec[f"{name}_psnr"] = EU.psnr_masked(P, x_b, brain)
            rec[f"{name}_ssim"] = EU.ssim_masked(P, x_b, brain)
        recs.append(rec)
        del samp, mean, x_b, x_a, x_fc, x_im
        if (n_done + 1) % 10 == 0 or n_done == 0:
            el = time.time() - t0
            print(f"    {run} fold {k}: {n_done + 1}/{len(val)} | {el / (n_done + 1):.1f}s/pair | "
                  f"ETA {el / (n_done + 1) * (len(val) - n_done - 1) / 60:.0f} min", flush=True)
            torch.cuda.empty_cache()
    with open(out, "w") as f:
        json.dump(dict(info=info, res_scale=res_scale, records=recs), f)


def summarize(run, folds):
    recs = []
    for k in folds:
        p = os.path.join(OUT, f"{run}_fold{k}.json")
        if os.path.exists(p):
            recs += json.load(open(p))["records"]
    if not recs:
        print(f"  {run}: no results yet"); return None

    def pm(key):
        d = AG.by_patient(recs, key)
        return np.array([d[p][1] for p in sorted(d)], float)

    mean = lambda key: float(np.nanmean(pm(key)))
    expl = 100 * (1 - mean("copy_mse_normal_matched") / mean("copy_mse_normal_raw"))
    tests = {
        "A1 imgmean vs copy (raw)": AG.paired_test(pm("imgmean_mse_roi_raw"), pm("copy_mse_roi_raw")),
        "A3 imgmean vs copy (both matched)": AG.paired_test(pm("imgmean_mse_roi_matched"), pm("copy_mse_roi_matched")),
    }
    adj = AG.holm([t["p"] for t in tests.values()])
    for t, pa in zip(tests.values(), adj):
        t["p_holm"] = float(pa)
    secondary = {
        "A1' latent-mean vs copy (raw)": AG.paired_test(pm("latmean_mse_roi_raw"), pm("copy_mse_roi_raw")),
        "A3' latent-mean vs copy (matched)": AG.paired_test(pm("latmean_mse_roi_matched"), pm("copy_mse_roi_matched")),
        "L latent ROI MSE vs copy": AG.paired_test(pm("lat_mse_roi_model"), pm("lat_mse_roi_copy")),
    }
    success = all(t["p_holm"] < 0.05 and t["mean_a"] < t["mean_b"] for t in tests.values())
    S = dict(run=run, n_patients=len({r["patient"] for r in recs}), n_pairs=len(recs),
             copy_normal_error_explained_pct=expl, primary=tests, secondary=secondary,
             crps_brain_raw=mean("crps_brain_raw"), picp90_brain_raw=mean("picp90_brain_raw"),
             psnr=dict(forecast=mean("imgmean_psnr"), copy=mean("copy_psnr")),
             ssim=dict(forecast=mean("imgmean_ssim"), copy=mean("copy_ssim")),
             success=bool(success) if run != "ref_v2" else None)
    print(f"\n  ── {run}  ({S['n_patients']} patients, {S['n_pairs']} post-op pairs)")
    print(f"     Copy normal-tissue error explained by global intensity: {expl:.1f}%")
    for label, t in {**tests, **secondary}.items():
        ph = f"p_holm={t['p_holm']:.1e}" if "p_holm" in t else f"p={t['p']:.1e}"
        print(f"     {label:<36} {t['mean_a']:.4f} vs {t['mean_b']:.4f}  rel {t['rel_improvement_pct']:+6.1f}% "
              f"[{t['ci95_pct'][0]:+.1f},{t['ci95_pct'][1]:+.1f}]  {ph}  r_rb={t['r_rb']:+.2f}")
    print(f"     CRPS(brain, raw) {S['crps_brain_raw']:.4f} | PICP@90 raw {S['picp90_brain_raw']:.3f} | "
          f"PSNR {S['psnr']['forecast']:.2f} vs {S['psnr']['copy']:.2f} | SSIM {S['ssim']['forecast']:.3f} vs {S['ssim']['copy']:.3f}")
    if run != "ref_v2":
        print(f"     PRE-SPECIFIED CRITERION (A1 and A3 favour forecast, Holm p<0.05): "
              f"{'MET' if success else 'NOT MET'}")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["lan", "lan_res", "ref_v2"],
                    choices=["lan", "lan_res", "ref_v2"])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--K", type=int, default=8, help="decoded samples averaged in image space (primary forecast)")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if not args.summarize_only:
        vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
        ck = torch.load(EU.VQGAN, map_location="cpu", weights_only=False)
        vq.load_state_dict(ck.get("gen", ck), strict=False)
        vq.eval().to(DEV)
        for p in vq.parameters():
            p.requires_grad_(False)
        for run in args.runs:
            for k in args.folds:
                eval_run(run, k, args, vq)

    print("=" * 96)
    print(f"  LESION-AWARE PILOT — folds {args.folds}, post-operative pairs, patient-level statistics")
    print("=" * 96)
    summ = {r: summarize(r, args.folds) for r in args.runs}
    with open(os.path.join(OUT, "pilot_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("=" * 96)
    print(f"[saved] {OUT}/pilot_summary.json")


if __name__ == "__main__":
    main()
