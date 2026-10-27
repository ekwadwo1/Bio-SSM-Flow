#!/usr/bin/env python3
"""
eval_unified.py — the SINGLE evaluation protocol behind every number in Results.

For every validation pair of every fold (fixed seed per pair):
  * sample an R-member ensemble (flow: midpoint ODE with M steps; ddpm: DDIM)
  * point metrics of the ensemble mean vs Copy vs capped-Linear (MSE, cosine) over:
      whole latent | brain | tumor ROI (GT ET∪NETC at baseline∪follow-up, dilated)
      | ROI-WT sensitivity (GT ET∪NETC∪SNFH)
  * distributional metrics on a GRID of temperatures T (widening about the mean):
      CRPS, coverage at nominal levels, rank histogram — over whole / brain / ROI.
      No temperature is chosen here: aggregate_unified.py cross-fits T on held-out
      patients, so every reported coverage number is out-of-sample.
  * error–uncertainty Spearman (token level) over whole / brain / ROI
  * image metrics for ALL pairs, on brain-mask voxels only (PSNR, SSIM with a 7^3
      uniform window): decoded forecast vs decoded follow-up, and Copy likewise
  * segmentation: decoded forecast / decoded follow-up (ceiling) / GT baseline (copy)
      vs GT follow-up (Dice per region), plus volumes for growth-direction accuracy

Models come from runs/FROZEN_MODELS.json; their SHA-256 is verified before use.

Outputs (runs/eval_unified/<tag>/): per_pair_fold{k}.json, arrays_fold{k}.npz, meta.json
Resumable: folds whose outputs already exist are skipped (--overwrite to redo).

Smoke test : CUDA_VISIBLE_DEVICES=0 python eval_unified.py --transport flow --folds 0 --max_pairs 3 --tag smoke
Flow (all) : CUDA_VISIBLE_DEVICES=0 python eval_unified.py --transport flow
DDPM (all) : CUDA_VISIBLE_DEVICES=0 python eval_unified.py --transport ddpm
DDPM long  : CUDA_VISIBLE_DEVICES=0 python eval_unified.py --transport ddpm --folds 0 \
                 --ckpt_pattern "runs/ddpm_long_fold{k}/best.pt" --tag ddpm_long
"""
from __future__ import annotations
import os, sys, re, glob, json, time, hashlib, argparse, datetime
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path:
    sys.path.insert(0, _P1)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet

CACHE = os.path.join(HERE, "latent_cache_lumiere")
VQGAN = ("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/"
         "UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt")
SEG = os.path.join(HERE, "runs", "seghead_img_fold0", "imgseg_best.pt")
MANIFEST = os.path.join(HERE, "runs", "FROZEN_MODELS.json")
DEV = "cuda:0"

T_GRID = np.round(np.arange(0.50, 3.5001, 0.05), 2)
NOMINAL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
DIST_REGIONS = ("whole", "brain", "roi")
SEG_REGIONS = {"ET": (3,), "NETC": (1,), "SNFH": (2,), "TC": (1, 3), "WT": (1, 2, 3)}
CORE, WHOLE_TUMOR = (1, 3), (1, 2, 3)
DILATE = 3


# ───────────────────────────── utilities ─────────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def label_mask(lab, classes):
    m = torch.zeros_like(lab, dtype=torch.bool)
    for c in classes:
        m |= (lab == c)
    return m


def roi_latent(seg_a, seg_b, classes):
    """GT tumor at baseline OR follow-up, dilated DILATE voxels, pooled to latent tokens."""
    m = (label_mask(seg_a, classes) | label_mask(seg_b, classes)).float()
    m = m.reshape(1, 1, *m.shape[-3:])
    m = F.max_pool3d(m, 2 * DILATE + 1, 1, DILATE)
    return (F.avg_pool3d(m, 4) > 0.1)[0, 0]


def masked_mse_cos(a, b, tok_mask):
    """a,b: [1,c,h,w,d]; tok_mask: [h,w,d] bool or None (whole)."""
    if tok_mask is None:
        av, bv = a.reshape(-1), b.reshape(-1)
    else:
        if int(tok_mask.sum()) == 0:
            return float("nan"), float("nan")
        mm = tok_mask[None].expand(a.shape[1], *tok_mask.shape)
        av, bv = a[0][mm], b[0][mm]
    return F.mse_loss(av, bv).item(), F.cosine_similarity(av[None], bv[None]).item()


@torch.no_grad()
def dist_metrics(S, y, gen):
    """
    S: [R,n] ensemble, y: [n] truth. For every T in T_GRID (widening about the mean):
      fair CRPS, coverage of central intervals at NOMINAL levels, rank histogram.
    Widening is affine and increasing, so quantiles / sorted order transform exactly.
    """
    R, n = S.shape
    mean = S.mean(0)
    Ss, _ = S.sort(0)
    w = 2 * torch.arange(1, R + 1, device=S.device, dtype=S.dtype) - R - 1
    spread = (w[:, None] * Ss).sum(0) / (R * (R - 1))        # 0.5*E|X-X'| (fair estimator)
    qs = torch.tensor(np.concatenate([(1 - NOMINAL) / 2, 1 - (1 - NOMINAL) / 2]),
                      device=S.device, dtype=S.dtype)
    Q = torch.quantile(S, qs, dim=0)
    lo, hi = Q[:len(NOMINAL)], Q[len(NOMINAL):]
    u = torch.rand(n, generator=gen, device=S.device)         # random tie-breaking for ranks
    crps = np.empty(len(T_GRID))
    cov = np.empty((len(T_GRID), len(NOMINAL)))
    rank = np.empty((len(T_GRID), R + 1))
    for ti, T in enumerate(T_GRID):
        T = float(T)
        Xt = mean + T * (S - mean)
        crps[ti] = ((Xt - y).abs().mean(0) - T * spread).mean().item()
        loT, hiT = mean + T * (lo - mean), mean + T * (hi - mean)
        cov[ti] = ((y >= loT) & (y <= hiT)).float().mean(1).cpu().numpy()
        lt = (Xt < y).sum(0)
        eq = (Xt == y).sum(0)
        rk = (lt + torch.floor(u * (eq + 1)).long()).clamp(max=R)
        rank[ti] = torch.bincount(rk, minlength=R + 1).cpu().numpy()
    return crps, cov, rank


def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm() + 1e-12)).item()


def psnr_masked(x, y, mask):
    """x,y: [C,D,H,W]; mask: [D,H,W] bool. Per-contrast PSNR on brain voxels, averaged."""
    vals = []
    for c in range(x.shape[0]):
        xv, yv = x[c][mask], y[c][mask]
        L = (yv.max() - yv.min()).clamp_min(1e-6)
        mse = F.mse_loss(xv, yv).clamp_min(1e-12)
        vals.append((10 * torch.log10(L * L / mse)).item())
    return float(np.mean(vals))


def ssim_masked(x, y, mask, win=7):
    """SSIM map with a win^3 uniform window, averaged over brain voxels, per contrast."""
    k = torch.ones(1, 1, win, win, win, device=x.device) / win ** 3
    p = win // 2
    vals = []
    for c in range(x.shape[0]):
        a, b = x[c][None, None], y[c][None, None]
        yv = y[c][mask]
        L = (yv.max() - yv.min()).clamp_min(1e-6)
        C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
        mu_a, mu_b = F.conv3d(a, k, padding=p), F.conv3d(b, k, padding=p)
        saa = F.conv3d(a * a, k, padding=p) - mu_a ** 2
        sbb = F.conv3d(b * b, k, padding=p) - mu_b ** 2
        sab = F.conv3d(a * b, k, padding=p) - mu_a * mu_b
        smap = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
        vals.append(smap[0, 0][mask].mean().item())
    return float(np.mean(vals))


def dice(pred, target):
    ps, ts = int(pred.sum()), int(target.sum())
    if ps + ts == 0:
        return float("nan")                                   # region absent in both: excluded
    return 2.0 * int((pred & target).sum()) / (ps + ts)


# ─────────────────────────── models & baselines ───────────────────────────
def resolve_ckpt(args, k):
    if args.ckpt_pattern:
        return args.ckpt_pattern.format(k=k), None
    key = f"{'forecast' if args.transport == 'flow' else 'ddpm'}_fold{k}"
    man = json.load(open(MANIFEST))
    if key not in man:
        raise KeyError(f"{key} not found in {MANIFEST}")
    p = man[key]["path"]
    return (p if os.path.isabs(p) else os.path.join(HERE, p)), man[key]["sha256"]


def load_model(args, k):
    path, expected = resolve_ckpt(args, k)
    got = sha256(path)
    if expected is None:
        print(f"[eval] WARNING: {path} is NOT in the frozen manifest (override); sha={got[:12]}")
    elif got != expected:
        raise RuntimeError(f"[eval] HASH MISMATCH for {path}: manifest {expected[:12]} vs file "
                           f"{got[:12]}. The checkpoint changed after freezing; stop and investigate.")
    st = torch.load(path, map_location=DEV, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                        use_checkpoint=False).to(DEV)
    net.load_state_dict(st.get("ema", st.get("net")))
    net.eval(); net.set_week(None)
    info = dict(path=path, sha256=got, step=st.get("step"), verified=expected is not None)
    if args.transport == "flow":
        return net, None, info
    from ddpm_transport import DDPMTransport
    return net, DDPMTransport(net, T=1000, device=DEV).to(DEV), info


@torch.no_grad()
def sample_ensemble(args, net, ddpm, z_a, dt):
    """Returns [R,1,c,h,w,d]."""
    if args.transport == "flow":
        outs = []
        for c0 in range(0, args.R, args.chunk):
            n = min(args.chunk, args.R - c0)
            za = z_a.expand(n, *z_a.shape[1:]).contiguous()
            out = generate(lambda z, t, a, d: net(z, t, a, d), za, dt.expand(n).contiguous(),
                           steps=args.steps, n_samples=1)[0]      # [n,c,h,w,d]; members = batch
            outs.append(out)
        return torch.cat(outs, 0).unsqueeze(1)
    samp = ddpm.ddim_sample(z_a, dt, steps=args.ddim_steps, R=args.R)
    if samp.dim() == z_a.dim():
        samp = samp.unsqueeze(1)
    assert samp.shape[0] == args.R, f"unexpected DDPM sample shape {tuple(samp.shape)}"
    return samp


def build_prev_index():
    byp = {}
    for f in glob.glob(os.path.join(CACHE, "*.pt")):
        m = re.search(r"(Patient-\d+)_week(\d+)\.pt$", os.path.basename(f))
        if m:
            byp.setdefault(m.group(1), set()).add(int(m.group(2)))
    prev = {}
    for p, ws in byp.items():
        ws = sorted(ws)
        for a, b in zip(ws[:-1], ws[1:]):
            prev[(p, b)] = a
    return prev


def linear_forecast(s, z_a, prev, cap):
    """z_a + min(dt_cur/dt_prev, cap) * (z_a - z_prev); copy if no earlier timepoint."""
    p = s["patient"]
    wa, wb = int(s["week_pair"][0]), int(s["week_pair"][1])
    wp = prev.get((p, wa))
    if wp is None or wp >= wa:
        return z_a, False
    f = os.path.join(CACHE, f"{p}_week{wp}.pt")
    if not os.path.exists(f):
        return z_a, False
    z_prev = torch.load(f, weights_only=False)["z"].float().to(DEV).view_as(z_a)
    ratio = min((wb - wa) / max(wa - wp, 1), cap)
    return z_a + ratio * (z_a - z_prev), True


# ─────────────────────────────── main loop ───────────────────────────────
@torch.no_grad()
def eval_fold(args, k, ds, folds, prev, vq, seg, meta):
    out_json = os.path.join(args.outdir, f"per_pair_fold{k}.json")
    out_npz = os.path.join(args.outdir, f"arrays_fold{k}.npz")
    if os.path.exists(out_json) and os.path.exists(out_npz) and not args.overwrite:
        print(f"[eval] fold {k}: outputs exist -> skip (use --overwrite to redo)")
        return
    net, ddpm, info = load_model(args, k)
    meta["models"][f"fold{k}"] = info
    val = list(folds[k][1])
    if args.max_pairs:
        val = val[:args.max_pairs]
    print(f"[eval] fold {k}: {len(val)} pairs | {info['path']} (step {info['step']}, "
          f"{'verified' if info['verified'] else 'UNVERIFIED'})", flush=True)

    recs = []
    arr = {f"{q}_{r}": [] for q in ("crps", "cov", "rank") for r in DIST_REGIONS}
    t0 = time.time()
    for n_done, i in enumerate(val):
        s = ds[i]
        z_a = s["z_a"][None].to(DEV).float()
        z_b = s["z_b"][None].to(DEV).float()
        dt = s["dt"].view(1).to(DEV).float()
        C = z_b.shape[1]
        seg_a = s["seg_a"].to(DEV).long().reshape(*s["seg_a"].shape[-3:])
        seg_b = s["seg_b"].to(DEV).long().reshape(*s["seg_b"].shape[-3:])
        brain = s["mask_b"].to(DEV).bool().reshape(*s["mask_b"].shape[-3:])
        brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
        roi_tok = roi_latent(seg_a, seg_b, CORE)
        roiwt_tok = roi_latent(seg_a, seg_b, WHOLE_TUMOR)
        regions = {"whole": None, "brain": brain_tok, "roi": roi_tok, "roiwt": roiwt_tok}

        torch.manual_seed(args.seed * 100003 + int(i))
        samp = sample_ensemble(args, net, ddpm, z_a, dt)
        mean = samp.mean(0)
        z_lin, has_prev = linear_forecast(s, z_a, prev, args.linear_cap)

        rec = dict(fold=int(k), idx=int(i), patient=str(s["patient"]),
                   week_a=int(s["week_pair"][0]), week_b=int(s["week_pair"][1]),
                   dt=float(s["dt"]), has_prev=bool(has_prev),
                   brain_tokens=int(brain_tok.sum()), roi_tokens=int(roi_tok.sum()),
                   roiwt_tokens=int(roiwt_tok.sum()),
                   diversity=float(samp.std(0).mean().item()))

        # point metrics (ensemble mean vs baselines)
        for name, pred in (("model", mean), ("copy", z_a), ("linear", z_lin)):
            for reg, tm in regions.items():
                mse, cos = masked_mse_cos(pred, z_b, tm)
                rec[f"mse_{reg}_{name}"] = mse
                rec[f"cos_{reg}_{name}"] = cos

        # distributional metrics over the T grid
        S_all = samp[:, 0].reshape(args.R, -1)
        y_all = z_b[0].reshape(-1)
        g = torch.Generator(device=DEV).manual_seed(args.seed * 100003 + int(i))
        for reg in DIST_REGIONS:
            tm = regions[reg]
            if tm is None:
                S, y = S_all, y_all
            else:
                em = tm[None].expand(C, *tm.shape).reshape(-1)
                if int(em.sum()) == 0:
                    arr[f"crps_{reg}"].append(np.full(len(T_GRID), np.nan))
                    arr[f"cov_{reg}"].append(np.full((len(T_GRID), len(NOMINAL)), np.nan))
                    arr[f"rank_{reg}"].append(np.zeros((len(T_GRID), args.R + 1)))
                    continue
                S, y = S_all[:, em], y_all[em]
            c_, v_, r_ = dist_metrics(S, y, g)
            arr[f"crps_{reg}"].append(c_)
            arr[f"cov_{reg}"].append(v_)
            arr[f"rank_{reg}"].append(r_)

        # error–uncertainty rank correlation (token level)
        err_tok = (mean - z_b).pow(2).mean(1)[0].sqrt()
        unc_tok = samp.var(0).mean(1)[0].sqrt()
        for reg in DIST_REGIONS:
            tm = regions[reg]
            if tm is None:
                rec[f"errunc_{reg}"] = spearman(err_tok.reshape(-1), unc_tok.reshape(-1))
            elif int(tm.sum()) < 10:
                rec[f"errunc_{reg}"] = float("nan")
            else:
                rec[f"errunc_{reg}"] = spearman(err_tok[tm], unc_tok[tm])

        # image + segmentation metrics
        if not args.no_image:
            x_b, x_fc, x_a = vq.decode(z_b), vq.decode(mean), vq.decode(z_a)
            for name, x in (("model", x_fc), ("copy", x_a)):
                rec[f"psnr_{name}"] = psnr_masked(x[0], x_b[0], brain)
                rec[f"ssim_{name}"] = ssim_masked(x[0], x_b[0], brain)
            lab_fc = seg(x_fc).argmax(1)[0]
            lab_zb = seg(x_b).argmax(1)[0]
            lab_za = seg(x_a).argmax(1)[0]
            for reg, cls in SEG_REGIONS.items():
                t = label_mask(seg_b, cls)
                rec[f"dice_forecast_{reg}"] = dice(label_mask(lab_fc, cls), t)
                rec[f"dice_ceiling_{reg}"] = dice(label_mask(lab_zb, cls), t)
                rec[f"dice_copy_{reg}"] = dice(label_mask(seg_a, cls), t)
            for reg in ("ET", "TC", "WT"):
                cls = SEG_REGIONS[reg]
                for nm, lab in (("gt_a", seg_a), ("gt_b", seg_b), ("seg_za", lab_za), ("seg_fc", lab_fc)):
                    rec[f"vol_{reg}_{nm}"] = int(label_mask(lab, cls).sum())
            del x_b, x_fc, x_a

        recs.append(rec)
        del samp, mean
        if (n_done + 1) % 10 == 0 or n_done == 0:
            el = time.time() - t0
            print(f"    fold {k}: {n_done + 1}/{len(val)} pairs | {el / (n_done + 1):.1f}s/pair | "
                  f"ETA {el / (n_done + 1) * (len(val) - n_done - 1) / 60:.0f} min", flush=True)
            torch.cuda.empty_cache()

    np.savez_compressed(out_npz, **{key: np.stack(v).astype(np.int32 if key.startswith("rank") else np.float32)
                                    for key, v in arr.items()})
    with open(out_json, "w") as f:
        json.dump(recs, f)
    print(f"[eval] fold {k} done in {(time.time() - t0) / 60:.1f} min -> {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["flow", "ddpm"], required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20, help="flow midpoint steps (NFE = 2*steps)")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=16, help="ensemble members per forward batch (flow)")
    ap.add_argument("--linear_cap", type=float, default=1.0,
                    help="max extrapolation ratio dt_cur/dt_prev for the Linear baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=0, help=">0: only the first N pairs per fold (smoke test)")
    ap.add_argument("--no_image", action="store_true", help="skip decoding / image / segmentation metrics")
    ap.add_argument("--ckpt_pattern", type=str, default="", help='override, e.g. "runs/ddpm_long_fold{k}/best.pt"')
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.outdir = os.path.join(HERE, "runs", "eval_unified", args.tag or args.transport)
    os.makedirs(args.outdir, exist_ok=True)

    protocol = {k: getattr(args, k) for k in ("transport", "R", "steps", "ddim_steps", "linear_cap",
                                               "seed", "max_pairs", "no_image", "ckpt_pattern")}
    meta_path = os.path.join(args.outdir, "meta.json")
    if os.path.exists(meta_path) and not args.overwrite:
        meta = json.load(open(meta_path))
        if meta["protocol"] != protocol:
            raise RuntimeError(f"[eval] {args.outdir} was produced with a different protocol:\n"
                               f"  existing {meta['protocol']}\n  now      {protocol}\n"
                               f"Use a different --tag or --overwrite.")
    else:
        meta = dict(protocol=protocol, T_GRID=T_GRID.tolist(), NOMINAL=NOMINAL.tolist(),
                    created=datetime.datetime.now().isoformat(timespec="seconds"),
                    torch=torch.__version__, models={})

    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck = torch.load(VQGAN, map_location="cpu", weights_only=False)
    res = vq.load_state_dict(ck.get("gen", ck), strict=False)
    missing, unexpected = list(getattr(res, "missing_keys", [])), list(getattr(res, "unexpected_keys", []))
    print(f"[eval] VQGAN loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[eval] WARNING: VQGAN missing keys (first 5): {missing[:5]}")
    vq.eval().to(DEV)
    seg = ImgSegUNet().to(DEV)
    seg.load_state_dict(torch.load(SEG, map_location=DEV, weights_only=False)["net"])
    seg.eval()
    for p in list(vq.parameters()) + list(seg.parameters()):
        p.requires_grad_(False)
    meta["vqgan"] = dict(path=VQGAN, missing_keys=len(missing), unexpected_keys=len(unexpected))
    meta["seg_head"] = dict(path=SEG, sha256=sha256(SEG),
                            note="trained on the fold-0 split: Dice is held-out only for fold 0")

    ds = CachedPairDataset(CACHE)
    folds = list(patient_kfold(ds, k=5))
    prev = build_prev_index()
    for k in args.folds:
        eval_fold(args, k, ds, folds, prev, vq, seg, meta)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"[eval] all requested folds done -> {args.outdir}")


if __name__ == "__main__":
    main()
