#!/usr/bin/env python3
"""
autoencoder_fidelity.py — encode-decode fidelity of the frozen VQGAN per cohort
(Table I "Autoencoder PSNR" and Section V-E).

For 50 randomly chosen complete timepoints per cohort (seed 2026), the preprocessed input x is
rebuilt exactly as during caching (LUMIERE: cache_latents._timepoint_tensors; external cohorts:
the saved per-cohort or per-patient registration transform, then transforms.preprocess), and
compared with D(E(x)). Brain-voxel PSNR uses the definition of the ingestion gates
(peak = intensity range of x within the brain mask; MSE over brain voxels, all four contrasts);
whole-volume PSNR uses vqgan3d.compute_psnr.

Run:  CUDA_VISIBLE_DEVICES=0 python autoencoder_fidelity.py        (~10-15 min)
Output: runs/autoencoder_fidelity.json
"""
from __future__ import annotations
import os, sys, json, glob, re
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
N_PER_COHORT, SEED = 50, 2026


def brain_psnr(x, xr, m):
    L = (x[0][:, m].max() - x[0][:, m].min()).clamp_min(1e-6)
    mse = ((xr[0] - x[0]) ** 2)[:, m].mean().clamp_min(1e-12)
    return float(10 * torch.log10(L * L / mse))


def plain(t):
    return t.as_tensor() if hasattr(t, "as_tensor") else t


@torch.no_grad()
def measure(vq, items, build):
    from vqgan3d import compute_psnr
    rows = []
    for it in items:
        img, mask = build(it)
        x = img.unsqueeze(0).to("cuda:0")
        xr = vq.decode(vq.encode(x))
        m = mask.to("cuda:0").bool()
        rows.append(dict(item=str(it), brain=brain_psnr(x, xr, m), whole=float(compute_psnr(x, xr))))
    return rows


def main():
    import cache_latents as CL
    from lumiere_dataset import discover, CHANNEL_ORDER
    from transforms import preprocess
    import ingest_mu as IM
    import ingest_alptdg as IA
    import ants

    vq = CL.load_vqgan(CL.VQGAN_CKPT)
    rng = np.random.RandomState(SEED)
    out = {}

    # LUMIERE
    tps = [t for ts in discover(CL.LUMIERE_ROOT).values() for t in ts if t.complete()]
    pick = [tps[i] for i in rng.choice(len(tps), min(N_PER_COHORT, len(tps)), replace=False)]
    def build_lum(tp):
        pp = CL._timepoint_tensors(tp)
        return plain(pp["image"]), plain(pp["mask"])
    out["LUMIERE"] = measure(vq, pick, build_lum)

    # external cohorts: rebuild with the saved transforms
    ref = ants.image_read(IM.os.path.join(HERE, "external", "mu", "lumiere_ref_T1.nii.gz"))

    def rebuild(paths, seg_path, fwd):
        img = np.stack([IM.warp(ref, p, fwd, "linear").numpy() for p in paths]).astype(np.float32)
        seg = np.zeros(img.shape[1:], dtype=np.int64)
        mask = (img != 0).any(0)
        pp = preprocess(torch.from_numpy(img), torch.from_numpy(seg), torch.from_numpy(mask))
        return plain(pp["image"]), plain(pp["mask"])

    mu_log = json.load(open(os.path.join(HERE, "external", "mu", "ingest_summary.json")))["registration_log"]
    mu_items = []
    for f in glob.glob(os.path.join(HERE, "latent_cache_mu", "MU-*_tp*.pt")):
        m = re.search(r"MU-(\d+)_tp(\d+)\.pt$", f)
        mu_items.append((int(m.group(1)), int(m.group(2))))
    mu_items = sorted(mu_items)
    mu_pick = [mu_items[i] for i in rng.choice(len(mu_items), N_PER_COHORT, replace=False)]
    def build_mu(item):
        pid, tp = item
        meth = mu_log[str(pid)]["method"]
        tag = "cohort" if meth == "cohort" else f"p{pid:04d}"
        fwd = [os.path.join(HERE, "external", "mu", f"mu_to_lumiere_{tag}_0.mat")]
        paths = [IM.mu_path(pid, tp, IM.SUFFIX[c]) for c in CHANNEL_ORDER]
        return rebuild(paths, None, fwd)
    out["MU-Glioma-Post"] = measure(vq, mu_pick, build_mu)

    al_log = json.load(open(os.path.join(HERE, "external", "alptdg", "ingest_summary.json")))["registration_log"]
    al_items = sorted((p, k) for p in al_log for k in (1, 2))
    al_pick = [al_items[i] for i in rng.choice(len(al_items), N_PER_COHORT, replace=False)]
    def build_al(item):
        pid, k = item
        meth = al_log[pid]["method"]
        tag = "cohort" if meth == "cohort" else f"p{pid}"
        fwd = [os.path.join(HERE, "external", "alptdg", f"alptdg_to_lumiere_{tag}_0.mat")]
        paths = [IA.tp_path(pid, k, IA.SUFFIX[c]) for c in CHANNEL_ORDER]
        return rebuild(paths, None, fwd)
    out["UCSF-ALPTDG"] = measure(vq, al_pick, build_al)

    summary = {}
    for name, rows in out.items():
        b = np.array([r["brain"] for r in rows]); w = np.array([r["whole"] for r in rows])
        summary[name] = dict(n=len(rows), brain_mean=float(b.mean()), brain_sd=float(b.std(ddof=1)),
                             whole_mean=float(w.mean()), whole_sd=float(w.std(ddof=1)))
        print(f"[fidelity] {name:15s} n={len(rows)}  brain PSNR {b.mean():.2f} ± {b.std(ddof=1):.2f} dB | "
              f"whole {w.mean():.2f} ± {w.std(ddof=1):.2f} dB")
    json.dump(dict(summary=summary, rows=out, seed=SEED), open(os.path.join(HERE, "runs", "autoencoder_fidelity.json"), "w"), indent=1)
    print("[fidelity] saved runs/autoencoder_fidelity.json")


if __name__ == "__main__":
    main()
