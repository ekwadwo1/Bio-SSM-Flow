#!/usr/bin/env python3
"""
baseline_probreg.py — deep probabilistic regressor with the IDENTICAL backbone of Bio-SSM-Flow.

Question it answers: does the generative transport add anything beyond a deep network with the
same spatial context? The regressor uses the same VelocityNetV2 (per-voxel path + axis-cycled
state-space network, same width/depths, lambda initialized at 0), the same two-phase objective
(MSE warm-up for 2,000 iterations, then the heteroscedastic Gaussian likelihood of Eq. (6) with the
network's own per-token log-variance head), the same recipe (AdamW, lr, warm-up, cosine schedule,
batch 16, EMA 0.999, gradient clipping, bf16), and the same checkpoint selection (validation CRPS
every 1,000 iterations, 8 samples). The ONLY difference: the network predicts the follow-up latent
directly in one pass,
        (mu, log sigma^2) = net(state = z_a, flow time = 1, baseline = z_a, interval),
instead of a velocity field integrated from noise. The predictive distribution is
N(mu, sigma^2) per latent element (sigma shared across channels of a token, as in Eq. (6)).

Evaluation is identical to every other forecaster (copied from baseline_ar1.py): analytic mean as
point forecast, R = 32 Gaussian samples with the per-pair seed for distributional metrics, written to
runs/eval_unified/probreg/{per_pair_fold*.json, arrays_fold*.npz, meta.json}, so aggregate_unified.py
applies the same cross-fitted temperature, patient-level tests, and Holm adjustment.

Usage (from the Bio-SSM-Flow folder):
  CUDA_VISIBLE_DEVICES=0 python baseline_probreg.py train --fold 0        (one fold per call; resumable by fold)
  CUDA_VISIBLE_DEVICES=0 python baseline_probreg.py eval --folds 0 1 2 3 4
  python baseline_probreg.py compare                                      (Table II row + tests vs Bio-SSM-Flow)
"""
from __future__ import annotations
import os, sys, json, math, copy, time, argparse, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_unified as EU
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import HeteroscedasticFMLoss
from velocity_net_v2 import VelocityNetV2
import metrics as M

DEV = EU.DEV
NAME = "probreg"
TRAIN_DIR = os.path.join(HERE, "runs", "baseline_probreg")
OUT_DIR = os.path.join(HERE, "runs", "eval_unified", NAME)
TAU_FIXED = 1.0          # constant flow-time input: the network is used as a one-pass regressor


# ───────────────────────────── network and recipe (identical to the flow model) ─────────────────────────────
def build_net(checkpointing=False):
    return VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                         use_checkpoint=checkpointing).to(DEV)


def load_recipe(fold):
    rc = torch.load(os.path.join(HERE, "runs", f"forecast_fold{fold}", "best.pt"),
                    map_location="cpu", weights_only=False).get("cfg", {}) or {}
    d = dict(batch=16, epochs=300, lr=2e-4, warmup_steps=500, weight_decay=1e-4, var_warmup=2000,
             ema_decay=0.999, grad_clip=1.0, gen_samples=8, k_folds=5)
    return {k: rc.get(k, v) for k, v in d.items()}


def lr_at(step, cfg, total):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * step / max(1, cfg["warmup_steps"])
    p = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


class EMA:
    def __init__(self, model, decay):
        self.decay, self.shadow = decay, copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def collate(batch):
    return {k: (torch.stack([b[k] for b in batch]) if torch.is_tensor(batch[0][k]) else [b[k] for b in batch])
            for k in batch[0]}


def logvar_volume(logvar, mu):
    """The network returns one log-variance per latent token, flattened as [B, h*w*d];
    reshape it to [B, 1, h, w, d] in row-major token order (verified by check_logvar_layout)."""
    lv = logvar.float()
    if lv.dim() == 2:
        lv = lv.view(mu.shape[0], 1, *mu.shape[2:])
    elif lv.dim() == 4:
        lv = lv.unsqueeze(1)
    return lv


def predict(net, z_a, dt):
    """One-pass predictive mean and per-element standard deviation."""
    tau = torch.full((z_a.shape[0],), TAU_FIXED, device=z_a.device)
    mu, logvar = net(z_a, tau, z_a, dt)
    sigma = torch.exp(0.5 * logvar_volume(logvar, mu).clamp(-10, 10)).expand_as(mu)
    return mu.float(), sigma


@torch.no_grad()
def check_logvar_layout(net, loss_fn, batch):
    """Recompute Eq. (6) by hand with the row-major token order and compare with HeteroscedasticFMLoss.
    A match proves that predict() places each token's variance on the right latent voxel."""
    z_a, z_b, dt = batch["z_a"].to(DEV), batch["z_b"].to(DEV), batch["dt"].to(DEV).float()
    net.set_week(None)
    tau = torch.full((z_a.shape[0],), TAU_FIXED, device=DEV)
    mu, logvar = net(z_a, tau, z_a, dt)
    mu, logvar = mu.float(), logvar.float()
    ref = float(loss_fn(z_b.float(), mu, logvar)[0])

    def manual(lv):
        w = lv.clamp(-10, 10)
        e = (z_b.float() - mu).pow(2).mean(1, keepdim=True)          # channel-averaged squared error per token
        return float((0.5 * (e * torch.exp(-w) + w)).mean() + 1e-4 * (w ** 2).mean())

    lv_rm = logvar_volume(logvar, mu)
    perm = torch.randperm(lv_rm[0, 0].numel(), generator=torch.Generator().manual_seed(0)).to(DEV)
    lv_perm = lv_rm.flatten(2)[:, :, perm].view_as(lv_rm)
    m_rm, m_perm = manual(lv_rm), manual(lv_perm)
    tol = 1e-4 * max(1.0, abs(ref))
    print(f"[probreg] log-variance layout check: loss {ref:.6f} | row-major {m_rm:.6f} | shuffled {m_perm:.6f} "
          f"| logvar shape {tuple(logvar.shape)}", flush=True)
    ok_rm, ok_perm = abs(m_rm - ref) <= tol, abs(m_perm - ref) <= tol
    if ok_rm and not ok_perm:
        print("[probreg] layout OK: row-major token order reproduces the training loss exactly", flush=True)
        return True
    if ok_rm and ok_perm:
        print("[probreg] formula OK; variances still nearly uniform, so token order is re-checked on the trained model", flush=True)
        return False
    if ok_perm:
        raise SystemExit("[probreg] STOP: the loss uses a non-row-major token order; paste HeteroscedasticFMLoss from flow.py")
    raise SystemExit("[probreg] STOP: Eq. (6) by hand does not reproduce HeteroscedasticFMLoss "
                     "(different constants or reduction); paste HeteroscedasticFMLoss from flow.py")


@torch.no_grad()
def validate(net, shadow, loader, cfg):
    backup = copy.deepcopy(net.state_dict())
    net.load_state_dict(shadow); net.eval(); net.set_week(None)
    torch.manual_seed(1234)
    crps = []
    for b in loader:
        z_a, z_b, dt = b["z_a"].to(DEV), b["z_b"].to(DEV), b["dt"].to(DEV).float()
        mu, sigma = predict(net, z_a, dt)
        samp = mu[None] + sigma[None] * torch.randn(cfg["gen_samples"], *mu.shape, device=DEV)
        crps.append(M.ensemble_crps(samp, z_b))
    net.load_state_dict(backup); net.train()
    return float(np.mean(crps))


def train(args):
    k = args.fold
    out = os.path.join(TRAIN_DIR, f"fold{k}")
    os.makedirs(out, exist_ok=True)
    cfg = load_recipe(k)
    torch.manual_seed(0)
    ds = CachedPairDataset(EU.CACHE)
    tr, va = list(patient_kfold(ds, k=cfg["k_folds"]))[k]
    total = cfg["epochs"] * (len(tr) // cfg["batch"])
    max_steps = args.max_steps or total
    net = build_net(checkpointing=True)
    print(f"[probreg] fold {k}: recipe {cfg} | {len(tr)} train / {len(va)} val pairs | schedule {total} steps, "
          f"max {max_steps} | early stop after {args.patience} validations without improvement", flush=True)
    dl = DataLoader(Subset(ds, tr), cfg["batch"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    vl = DataLoader(Subset(ds, va), cfg["batch"], shuffle=False, num_workers=2, collate_fn=collate)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = EMA(net, cfg["ema_decay"])
    loss_fn = HeteroscedasticFMLoss()
    check_logvar_layout(net, loss_fn, next(iter(vl)))
    net.train()
    best, best_step, step, stale, t0 = float("inf"), 0, 0, 0, time.time()
    log = open(os.path.join(out, "val_log.jsonl"), "w")
    net.train()
    done = False
    while not done:
        for b in dl:
            z_a, z_b, dt = b["z_a"].to(DEV), b["z_b"].to(DEV), b["dt"].to(DEV).float()
            net.set_week(None)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, total)
            tau = torch.full((z_a.shape[0],), TAU_FIXED, device=DEV)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mu, logvar = net(z_a, tau, z_a, dt)
            if step < cfg["var_warmup"]:
                loss = F.mse_loss(mu.float(), z_b.float())
            else:
                loss, _, _ = loss_fn(z_b.float(), mu.float(), logvar.float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step(); ema.update(net); step += 1
            if step % 100 == 0:
                print(f"  step {step}/{max_steps} | loss {float(loss):.4f} | {(time.time() - t0) / step * 100:.0f}s/100", flush=True)
            if step % args.val_every == 0 or step == max_steps:
                c = validate(net, ema.shadow, vl, cfg)
                log.write(json.dumps({"step": step, "crps": c}) + "\n"); log.flush()
                tag = ""
                if c < best:
                    best, best_step, stale = c, step, 0
                    torch.save({"ema": ema.shadow, "step": step, "val_crps": c, "cfg": cfg, "fold": k,
                                "variant": NAME, "tau_fixed": TAU_FIXED}, os.path.join(out, "best.pt"))
                    tag = "  <- best (saved)"
                else:
                    stale += 1
                print(f"  [val {step}] crps={c:.4f}{tag}", flush=True)
                if stale >= args.patience:
                    print(f"  early stop: no improvement for {args.patience} validations", flush=True)
                    done = True
            if step >= max_steps:
                done = True
            if done:
                break
    log.close()
    print(f"[probreg] fold {k} done in {(time.time() - t0) / 3600:.2f} h; best val CRPS {best:.4f} at step {best_step}")


# ───────────────────────────── evaluation (identical to baseline_ar1.py) ─────────────────────────────
@torch.no_grad()
def eval_fold(k, ds, folds, prev, args, meta):
    out_json = os.path.join(OUT_DIR, f"per_pair_fold{k}.json")
    out_npz = os.path.join(OUT_DIR, f"arrays_fold{k}.npz")
    if os.path.exists(out_json) and os.path.exists(out_npz) and not args.overwrite:
        print(f"[{NAME}] fold {k}: exists -> skip"); return
    ck = torch.load(os.path.join(TRAIN_DIR, f"fold{k}", "best.pt"), map_location="cpu", weights_only=False)
    net = build_net()
    net.load_state_dict(ck["ema"]); net.eval(); net.set_week(None)
    meta["checkpoints"][f"fold{k}"] = dict(step=int(ck["step"]), val_crps=float(ck["val_crps"]))
    _, va = folds[k]
    if not check_logvar_layout(net, HeteroscedasticFMLoss(), collate([ds[i] for i in va[:8]])):
        raise SystemExit("[probreg] STOP: token order of the log-variance could not be verified on the trained model")
    print(f"[{NAME}] fold {k}: checkpoint step {ck['step']} (val CRPS {ck['val_crps']:.4f}) | {len(va)} val pairs", flush=True)
    recs = []
    arr = {f"{q}_{r}": [] for q in ("crps", "cov", "rank") for r in EU.DIST_REGIONS}
    for i in va:
        s = ds[i]
        z_a = s["z_a"][None].to(DEV).float()
        z_b = s["z_b"][None].to(DEV).float()
        dt = s["dt"].view(1).to(DEV).float()
        C = z_b.shape[1]
        seg_a = s["seg_a"].to(DEV).long().reshape(*s["seg_a"].shape[-3:])
        seg_b = s["seg_b"].to(DEV).long().reshape(*s["seg_b"].shape[-3:])
        brain = s["mask_b"].to(DEV).bool().reshape(*s["mask_b"].shape[-3:])
        brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
        roi_tok = EU.roi_latent(seg_a, seg_b, EU.CORE)
        roiwt_tok = EU.roi_latent(seg_a, seg_b, EU.WHOLE_TUMOR)
        regions = {"whole": None, "brain": brain_tok, "roi": roi_tok, "roiwt": roiwt_tok}

        mean, sig = predict(net, z_a, dt)                                    # [1,C,h,w,d] each
        sigma = sig[0]
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
    print(f"[{NAME}] fold {k} done", flush=True)


def evaluate(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = CachedPairDataset(EU.CACHE)
    folds = list(patient_kfold(ds, k=5))
    prev = EU.build_prev_index()
    mp = os.path.join(OUT_DIR, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else dict(
        protocol=dict(transport=NAME, R=args.R, linear_cap=args.linear_cap, seed=args.seed,
                      point_forecast="analytic predictive mean (one-pass regressor, same backbone)"),
        T_GRID=EU.T_GRID.tolist(), NOMINAL=EU.NOMINAL.tolist(),
        created=datetime.datetime.now().isoformat(timespec="seconds"), checkpoints={})
    for k in args.folds:
        eval_fold(k, ds, folds, prev, args, meta)
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)


# ───────────────────────────── comparison with Bio-SSM-Flow ─────────────────────────────
def compare(args):
    import aggregate_unified as AG
    import make_table2 as MT
    runs = {n: AG.prepare(n, os.path.join(HERE, "runs", "eval_unified", n), "brain") for n in ("flow", NAME)}
    R = runs[NAME]["recs"]
    c = MT.calibration(runs[NAME])
    cells = [MT.pmean(R, "mse_whole_model"), MT.pmean(R, "mse_brain_model"), MT.pmean(R, "mse_roi_model"),
             MT.pmean(R, "cos_roi_model"), MT.pmean(R, "crps_cal_brain"), MT.pmean(R, "crps_cal_roi"),
             c["calerr_cal"], c["cov90_cal"]]
    fmt = ["{:.3f}"] * 6 + ["{:.4f}", "{:.3f}"]
    print("% Table II row (patient-level; same definitions as make_table2.py)")
    print("Probabilistic regressor$^{d}$ & " + " & ".join(f.format(v) for f, v in zip(fmt, cells)) + " \\\\")
    print(f"   raw calibration error {c['calerr_raw']:.4f} -> calibrated {c['calerr_cal']:.4f} | "
          f"T range {c['T_range'][0]:.2f}-{c['T_range'][1]:.2f} | patients {c['n_patients']}")

    def by_p(recs, key):
        return {p: v for p, (_, v) in AG.by_patient(recs, key).items()}
    print("\n% Bio-SSM-Flow vs probabilistic regressor (positive = Bio-SSM-Flow better; unadjusted p)")
    for label, key, higher in (("Tumor-region MSE", "mse_roi_model", False), ("Brain MSE", "mse_brain_model", False),
                               ("Whole-volume MSE", "mse_whole_model", False), ("Tumor-region cosine", "cos_roi_model", True),
                               ("Tumor-region CRPS (cal.)", "crps_cal_roi", False), ("Brain CRPS (cal.)", "crps_cal_brain", False)):
        da, db = by_p(runs["flow"]["recs"], key), by_p(R, key)
        common = sorted(set(da) & set(db))
        a = np.array([da[p] for p in common], float); b = np.array([db[p] for p in common], float)
        ok = np.isfinite(a) & np.isfinite(b)
        t = AG.paired_test(a[ok], b[ok])
        rel, lo, hi = t["rel_improvement_pct"], t["ci95_pct"][0], t["ci95_pct"][1]
        if higher:
            rel, lo, hi = -rel, -hi, -lo
        print(f"   {label:26s} flow {a[ok].mean():.4f} vs reg {b[ok].mean():.4f} | {rel:+5.1f}% [{lo:+.1f},{hi:+.1f}] "
              f"p={t['p']:.1e} n={int(ok.sum())}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--max_steps", type=int, default=0)
    t.add_argument("--val_every", type=int, default=1000)
    t.add_argument("--patience", type=int, default=3)
    e = sub.add_parser("eval")
    e.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    e.add_argument("--R", type=int, default=32)
    e.add_argument("--linear_cap", type=float, default=1.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--overwrite", action="store_true")
    sub.add_parser("compare")
    args = ap.parse_args()
    {"train": train, "eval": evaluate, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
