#!/usr/bin/env python3
"""
image_metrics_internal.py — image-space fidelity on LUMIERE with the paper's settings, for the
Results sentence "Brain PSNR and SSIM of the forecast ... were [RERUN-image] in LUMIERE".

Per validation pair (5-fold CV, frozen fold models): R = 32 ensemble members, 20 midpoint steps,
the same per-pair seed as eval_unified.py; forecast = average of the first 8 decoded members;
persistence = decoded baseline; reference = decoded observed follow-up. PSNR and SSIM over brain
voxels with eval_unified's own functions (identical to the external evaluation). Patient-level
means are reported (pairs averaged per patient first).

Run (split the folds over both GPUs; resumable):
  CUDA_VISIBLE_DEVICES=0 python image_metrics_internal.py --folds 0 1 2
  CUDA_VISIBLE_DEVICES=1 python image_metrics_internal.py --folds 3 4
  python image_metrics_internal.py --summarize
Output: runs/image_metrics_internal/per_pair.jsonl, summary.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "runs", "image_metrics_internal")
K_DECODE, R, SEED = 8, 32, 0


def run(folds):
    import torch
    import eval_unified as EU
    from cached_pair_dataset import CachedPairDataset, patient_kfold
    from vqgan3d import VQGAN3D
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "per_pair.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done.add((r["fold"], r["idx"]))
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck = torch.load(EU.VQGAN, map_location="cpu", weights_only=False)
    vq.load_state_dict(ck.get("gen", ck), strict=False)
    vq = vq.eval().to(EU.DEV)
    ds = CachedPairDataset(EU.CACHE)
    splits = list(patient_kfold(ds, k=5))
    ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=R, chunk=16, steps=20, ddim_steps=50)
    with open(path, "a") as f, torch.no_grad():
        for k in folds:
            net, _, info = EU.load_model(ns, k)                 # hash-verified frozen model
            todo = [i for i in splits[k][1] if (k, int(i)) not in done]
            print(f"[img] fold {k}: {len(todo)} pairs to go | {info['path']}", flush=True)
            t0 = time.time()
            for n, i in enumerate(todo):
                s = ds[i]
                z_a = s["z_a"][None].to(EU.DEV).float(); z_b = s["z_b"][None].to(EU.DEV).float()
                dt = s["dt"].view(1).to(EU.DEV).float()
                brain = s["mask_b"].to(EU.DEV).bool().reshape(*s["mask_b"].shape[-3:])
                torch.manual_seed(SEED * 100003 + int(i))
                samp = EU.sample_ensemble(ns, net, None, z_a, dt)
                x_fc = torch.stack([vq.decode(samp[r])[0] for r in range(K_DECODE)]).mean(0)
                x_a, x_b = vq.decode(z_a)[0], vq.decode(z_b)[0]
                rec = dict(fold=int(k), idx=int(i), patient=str(s["patient"]),
                           psnr_forecast=float(EU.psnr_masked(x_fc, x_b, brain)),
                           ssim_forecast=float(EU.ssim_masked(x_fc, x_b, brain)),
                           psnr_persistence=float(EU.psnr_masked(x_a, x_b, brain)),
                           ssim_persistence=float(EU.ssim_masked(x_a, x_b, brain)))
                f.write(json.dumps(rec) + "\n"); f.flush()
                if (n + 1) % 10 == 0 or n == 0:
                    el = time.time() - t0
                    print(f"    fold {k}: {n + 1}/{len(todo)} | {el / (n + 1):.1f}s/pair | "
                          f"ETA {el / (n + 1) * (len(todo) - n - 1) / 60:.0f} min", flush=True)
                    torch.cuda.empty_cache()


def summarize():
    recs = [json.loads(l) for l in open(os.path.join(OUT, "per_pair.jsonl"))]
    byp = {}
    for r in recs:
        byp.setdefault(r["patient"], []).append(r)
    keys = ("psnr_forecast", "ssim_forecast", "psnr_persistence", "ssim_persistence")
    pm = {k: float(np.mean([np.mean([r[k] for r in v]) for v in byp.values()])) for k in keys}
    S = dict(n_pairs=len(recs), n_patients=len(byp), folds=sorted({r["fold"] for r in recs}), **pm)
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"[img] {S['n_pairs']} pairs, {S['n_patients']} patients, folds {S['folds']}")
    print(f"      forecast    PSNR {pm['psnr_forecast']:.2f} dB  SSIM {pm['ssim_forecast']:.3f}")
    print(f"      persistence PSNR {pm['psnr_persistence']:.2f} dB  SSIM {pm['ssim_persistence']:.3f}")
    if S["n_pairs"] < 469:
        print("      WARNING: not all 469 pairs done yet")
    print(f"\nLaTeX: ... were {pm['psnr_forecast']:.1f}\\,dB and {pm['ssim_forecast']:.3f} in LUMIERE "
          f"(persistence: {pm['psnr_persistence']:.1f}\\,dB and {pm['ssim_persistence']:.3f}), ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    summarize() if args.summarize else run(args.folds)


if __name__ == "__main__":
    main()
