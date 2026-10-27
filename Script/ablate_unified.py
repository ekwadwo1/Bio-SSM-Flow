#!/usr/bin/env python3
"""
ablate_unified.py — model-analysis ablations under the UNIFIED evaluation protocol (Table III).

Every variant is evaluated exactly like the main flow model in eval_unified.py: R = 32 ensemble
members, 20 midpoint steps, latent metrics on whole / brain / tumor-ROI tokens, coverage and rank
arrays, written in the eval_unified format to runs/eval_unified/<variant>/. aggregate_unified.py
then cross-fits a temperature for each variant separately and compares it with the full model
(runs/eval_unified/flow) on the SAME patients with patient-level paired tests.

Variants
  ssm_off     frozen full models (hash-verified); state-space branch disabled at inference
              (lambda = refine_scale set to 0). No training.
  ssm_only    per-voxel path removed (its output multiplied by 0), lambda initialised to 1;
              retrained with the frozen recipe, the original schedule length, and the same
              validation-CRPS checkpoint selection as the reported models.
  dt_shuffle  frozen full models; true intervals permuted within each validation fold.
  dt_const    frozen full models; every interval set to 84 days (LUMIERE median).
The "no calibration" row needs no run: it is computed from the flow outputs (raw vs. calibrated).

The per-voxel head is never guessed by name: with lambda = 0 the network output must equal the
head's output exactly; only a module passing that numerical test is ablated.

Usage (from the Bio-SSM-Flow folder)
  python ablate_unified.py identify                                  # check the head is found (seconds)
  CUDA_VISIBLE_DEVICES=0 python ablate_unified.py train_ssm_only --fold 0   # ~4 h
  CUDA_VISIBLE_DEVICES=0 python ablate_unified.py eval --variant ssm_only --folds 0
  CUDA_VISIBLE_DEVICES=1 python ablate_unified.py eval --variant ssm_off  --folds 0 1 2 3 4
  CUDA_VISIBLE_DEVICES=1 python ablate_unified.py eval --variant dt_shuffle --folds 0 1 2 3 4
  CUDA_VISIBLE_DEVICES=1 python ablate_unified.py eval --variant dt_const   --folds 0 1 2 3 4
  python ablate_unified.py summary                                   # Table III rows (LaTeX)
"""
from __future__ import annotations
import os, sys, json, math, copy, time, argparse, datetime
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import eval_unified as EU
import aggregate_unified as AG
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss, generate
from velocity_net_v2 import VelocityNetV2
import metrics as M

DEV = EU.DEV
EVAL_DIR = os.path.join(HERE, "runs", "eval_unified")
TRAIN_DIR = os.path.join(HERE, "runs", "ablate_unified")
DT_CONST = 84.0
VARIANTS = ("ssm_off", "ssm_only", "dt_shuffle", "dt_const")
LABELS = {"ssm_only": "State-space only", "ssm_off": "State-space off at test",
          "dt_shuffle": "Interval shuffled", "dt_const": "Interval fixed (84 d)"}


# ───────────────────────────── network surgery ─────────────────────────────
def build_net(checkpointing=False):
    return VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                         use_checkpoint=checkpointing).to(DEV)


def refine_param(net):
    hits = [(n, p) for n, p in net.named_parameters() if n.split(".")[-1] == "refine_scale"]
    if len(hits) != 1:
        raise SystemExit(f"[ablate] expected exactly one 'refine_scale' parameter, found {[n for n, _ in hits]}")
    return hits[0]


class ZeroOut(nn.Module):
    """Wraps the per-voxel head; keeps its output shape, contributes exactly zero."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *a, **k):
        return self.m(*a, **k) * 0.0


def _conv1_only(mod):
    convs = [m for m in mod.modules() if isinstance(m, nn.Conv3d)]
    return bool(convs) and all(tuple(c.kernel_size) == (1, 1, 1) for c in convs) and convs[0].in_channels == 8


@torch.no_grad()
def find_local_head(net):
    """Name of the per-voxel head, identified numerically on a TRAINED network."""
    torch.manual_seed(0)
    z = torch.randn(1, 4, 32, 32, 32, device=DEV)
    za = torch.randn_like(z)
    tau = torch.full((1,), 0.5, device=DEV)
    dt = torch.tensor([DT_CONST], device=DEV)
    _, rs = refine_param(net)
    saved = rs.data.clone()
    net.eval(); net.set_week(None)
    rs.data.zero_()
    v0 = net(z, tau, za, dt)[0].float()
    rs.data.copy_(saved)
    cands = sorted([(n, m) for n, m in net.named_modules() if n and _conv1_only(m)], key=lambda t: len(t[0]))
    for n, m in cands:
        for x in (torch.cat([z, za], 1), torch.cat([za, z], 1)):
            try:
                out = m(x)
            except Exception:
                continue
            if torch.is_tensor(out) and out.shape == v0.shape and torch.allclose(out.float(), v0, atol=1e-4, rtol=1e-3):
                return n
    print("[ablate] no module reproduced the lambda=0 output. Top-level modules:")
    for n, m in net.named_children():
        print(f"   {n}: {m.__class__.__name__}")
    raise SystemExit("[ablate] stopping: per-voxel head not identified; nothing was ablated.")


def zero_local_head(net, name):
    parent = net.get_submodule(".".join(name.split(".")[:-1])) if "." in name else net
    leaf = name.split(".")[-1]
    setattr(parent, leaf, ZeroOut(getattr(parent, leaf)))


def frozen_flow(k):
    ns = SimpleNamespace(transport="flow", ckpt_pattern="")
    net, _, info = EU.load_model(ns, k)                    # hash-verified against FROZEN_MODELS.json
    return net, info


def identify():
    net, info = frozen_flow(0)
    name = find_local_head(net)
    _, rs = refine_param(net)
    print(f"[ablate] per-voxel head = '{name}' | trained lambda (refine_scale) fold 0 = {float(rs):+.4f}")
    return name


# ───────────────────────────── training (ssm_only) ─────────────────────────────
def load_recipe(fold):
    rc = torch.load(os.path.join(HERE, "runs", f"forecast_fold{fold}", "best.pt"),
                    map_location="cpu", weights_only=False).get("cfg", {}) or {}
    d = dict(batch=16, epochs=300, lr=2e-4, warmup_steps=500, weight_decay=1e-4, var_warmup=2000,
             ema_decay=0.999, grad_clip=1.0, gen_steps=20, gen_samples=8, k_folds=5)
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


@torch.no_grad()
def validate(net, shadow, loader, cfg):
    backup = copy.deepcopy(net.state_dict())
    net.load_state_dict(shadow); net.eval(); net.set_week(None)
    torch.manual_seed(1234)
    crps = []
    for b in loader:
        z_a, z_b, dt = b["z_a"].to(DEV), b["z_b"].to(DEV), b["dt"].to(DEV).float()
        samp = generate(lambda z, t, za, d: net(z, t, za, d), z_a, dt,
                        steps=cfg["gen_steps"], n_samples=cfg["gen_samples"])
        crps.append(M.ensemble_crps(samp, z_b))
    net.load_state_dict(backup); net.train()
    return float(np.mean(crps))


def train_ssm_only(args):
    k = args.fold
    out = os.path.join(TRAIN_DIR, f"ssm_only_fold{k}")
    os.makedirs(out, exist_ok=True)
    name = args.local_attr or identify()
    cfg = load_recipe(k)
    torch.manual_seed(0)
    ds = CachedPairDataset(EU.CACHE)
    tr, va = list(patient_kfold(ds, k=cfg["k_folds"]))[k]
    total = cfg["epochs"] * (len(tr) // cfg["batch"])
    max_steps = args.max_steps or total
    net = build_net(checkpointing=True)
    zero_local_head(net, name)
    _, rs = refine_param(net)
    rs.data.fill_(1.0)                                    # the state-space network forms the field from the start
    print(f"[ssm_only] fold {k}: recipe {cfg} | head '{name}' zeroed | lambda init 1.0 | "
          f"{len(tr)} train / {len(va)} val pairs | schedule {total} steps, training {max_steps}")
    dl = DataLoader(Subset(ds, tr), cfg["batch"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    vl = DataLoader(Subset(ds, va), cfg["batch"], shuffle=False, num_workers=2, collate_fn=collate)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = EMA(net, cfg["ema_decay"])
    loss_fn = HeteroscedasticFMLoss()
    best, step, t0 = float("inf"), 0, time.time()
    log = open(os.path.join(out, "val_log.jsonl"), "w")
    net.train()
    while step < max_steps:
        for b in dl:
            z_a, z_b, dt = b["z_a"].to(DEV), b["z_b"].to(DEV), b["dt"].to(DEV).float()
            net.set_week(None)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, total)
            tau = sample_tau(z_a.shape[0], DEV)
            eps = torch.randn_like(z_b)
            z_tau, u = noise_interpolate(z_b, tau, eps)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v, logvar = net(z_tau, tau, z_a, dt)
            if step < cfg["var_warmup"]:
                loss = F.mse_loss(v.float(), u.float())
            else:
                loss, _, _ = loss_fn(u, v.float(), logvar.float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step(); ema.update(net); step += 1
            if step % 100 == 0:
                print(f"  step {step}/{max_steps} | loss {float(loss):.4f} | lambda {float(rs):+.3f} | "
                      f"{(time.time() - t0) / step * 100:.0f}s/100", flush=True)
            if step % args.val_every == 0 or step == max_steps:
                c = validate(net, ema.shadow, vl, cfg)
                log.write(json.dumps({"step": step, "crps": c}) + "\n"); log.flush()
                tag = ""
                if c < best:
                    best = c
                    torch.save({"ema": ema.shadow, "step": step, "val_crps": c, "local_attr": name,
                                "cfg": cfg, "fold": k, "variant": "ssm_only"}, os.path.join(out, "best.pt"))
                    tag = "  <- best (saved)"
                print(f"  [val {step}] crps={c:.4f}{tag}", flush=True)
            if step >= max_steps:
                break
    log.close()
    print(f"[ssm_only] done in {(time.time() - t0) / 3600:.2f} h; best val CRPS {best:.4f} -> {out}/best.pt")


# ───────────────────────────── evaluation (unified format) ─────────────────────────────
def variant_net(variant, k):
    if variant == "ssm_only":
        path = os.path.join(TRAIN_DIR, f"ssm_only_fold{k}", "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"[eval] {path} missing — run train_ssm_only --fold {k} first")
        st = torch.load(path, map_location=DEV, weights_only=False)
        net = build_net()
        zero_local_head(net, st["local_attr"])
        net.load_state_dict(st["ema"])
        info = dict(path=path, sha256=EU.sha256(path), step=st["step"], val_crps=st["val_crps"])
    else:
        net, info = frozen_flow(k)
        if variant == "ssm_off":
            _, rs = refine_param(net)
            info["lambda_trained"] = float(rs)
            rs.data.zero_()
    net.eval(); net.set_week(None)
    return net, info


def week_of(f):
    return int(os.path.basename(f).split("_week")[1][:-3])


def dt_map(ds, idx, variant, k, seed):
    true = {i: float((week_of(ds.pairs[i][1]) - week_of(ds.pairs[i][0])) * 7) for i in idx}
    if variant == "dt_const":
        return {i: DT_CONST for i in idx}
    if variant == "dt_shuffle":
        vals = [true[i] for i in idx]
        perm = np.random.RandomState(seed * 1000 + k).permutation(len(idx))
        return {i: vals[perm[j]] for j, i in enumerate(idx)}
    return true


@torch.no_grad()
def eval_fold(variant, k, args, ds, folds, prev):
    outdir = os.path.join(EVAL_DIR, variant)
    os.makedirs(outdir, exist_ok=True)
    out_json, out_npz = os.path.join(outdir, f"per_pair_fold{k}.json"), os.path.join(outdir, f"arrays_fold{k}.npz")
    if os.path.exists(out_json) and os.path.exists(out_npz) and not args.overwrite:
        print(f"[eval] {variant} fold {k}: exists -> skip"); return None
    net, info = variant_net(variant, k)
    ns = SimpleNamespace(transport="flow", ckpt_pattern="", R=args.R, chunk=args.chunk, steps=20, ddim_steps=50)
    va = folds[k][1]
    dts = dt_map(ds, va, variant, k, args.seed)
    print(f"[eval] {variant} fold {k}: {len(va)} pairs | {info}", flush=True)
    recs, t0 = [], time.time()
    arr = {f"{q}_{r}": [] for q in ("crps", "cov", "rank") for r in EU.DIST_REGIONS}
    for n_done, i in enumerate(va):
        s = ds[i]
        z_a = s["z_a"][None].to(DEV).float(); z_b = s["z_b"][None].to(DEV).float()
        dt_val = dts[i]
        dt = torch.tensor([dt_val], device=DEV)
        C = z_b.shape[1]
        seg_a = s["seg_a"].to(DEV).long().reshape(*s["seg_a"].shape[-3:])
        seg_b = s["seg_b"].to(DEV).long().reshape(*s["seg_b"].shape[-3:])
        brain = s["mask_b"].to(DEV).bool().reshape(*s["mask_b"].shape[-3:])
        brain_tok = (F.avg_pool3d(brain.float()[None, None], 4) > 0.5)[0, 0]
        roi_tok = EU.roi_latent(seg_a, seg_b, EU.CORE)
        regions = {"whole": None, "brain": brain_tok, "roi": roi_tok,
                   "roiwt": EU.roi_latent(seg_a, seg_b, EU.WHOLE_TUMOR)}

        torch.manual_seed(args.seed * 100003 + int(i))
        samp = EU.sample_ensemble(ns, net, None, z_a, dt)
        mean = samp.mean(0)
        z_lin, has_prev = EU.linear_forecast(s, z_a, prev, 1.0)
        rec = dict(fold=int(k), idx=int(i), patient=str(s["patient"]), week_a=int(s["week_pair"][0]),
                   week_b=int(s["week_pair"][1]), dt=float(s["dt"]), dt_used=dt_val, has_prev=bool(has_prev),
                   brain_tokens=int(brain_tok.sum()), roi_tokens=int(roi_tok.sum()),
                   roiwt_tokens=int(regions["roiwt"].sum()), diversity=float(samp.std(0).mean()))
        for name, pred in (("model", mean), ("copy", z_a), ("linear", z_lin)):
            for reg, tm in regions.items():
                rec[f"mse_{reg}_{name}"], rec[f"cos_{reg}_{name}"] = EU.masked_mse_cos(pred, z_b, tm)
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
            else:
                rec[f"errunc_{reg}"] = EU.spearman(err_tok[tm], unc_tok[tm]) if int(tm.sum()) >= 10 else float("nan")
        recs.append(rec)
        if (n_done + 1) % 10 == 0 or n_done == 0:
            el = time.time() - t0
            print(f"    {variant} fold {k}: {n_done + 1}/{len(va)} | {el / (n_done + 1):.1f}s/pair | "
                  f"ETA {el / (n_done + 1) * (len(va) - n_done - 1) / 60:.0f} min", flush=True)
            torch.cuda.empty_cache()
    np.savez_compressed(out_npz, **{key: np.stack(v).astype(np.int32 if key.startswith("rank") else np.float32)
                                    for key, v in arr.items()})
    with open(out_json, "w") as f:
        json.dump(recs, f)
    meta_path = os.path.join(outdir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else dict(
        protocol=dict(transport="flow", variant=variant, R=args.R, steps=20, seed=args.seed),
        T_GRID=EU.T_GRID.tolist(), NOMINAL=EU.NOMINAL.tolist(), models={})
    meta["models"][f"fold{k}"] = info
    meta["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[eval] {variant} fold {k} done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ───────────────────────────── summary (Table III) ─────────────────────────────
def summary():
    flow = AG.prepare("flow", os.path.join(EVAL_DIR, "flow"), "brain")
    keys = [("mse_roi_model", "ROI MSE"), ("crps_cal_brain", "brain CRPS (cal.)"), ("mse_whole_model", "whole MSE")]
    res, lines = {}, []

    def fmt(t):
        return f"{t['rel_improvement_pct']:.1f} [{t['ci95_pct'][0]:.1f}, {t['ci95_pct'][1]:.1f}]".replace("-", "$-$")

    for v in VARIANTS:
        d = os.path.join(EVAL_DIR, v)
        if not any(f.startswith("per_pair_fold") for f in (os.listdir(d) if os.path.isdir(d) else [])):
            print(f"[summary] {v}: no results yet"); continue
        R = AG.prepare(v, d, "brain")
        res[v] = {}
        for key, lab in keys:
            a, b, n = AG.aligned_across(flow["recs"], R["recs"], key)   # a = full model, b = variant
            t = AG.paired_test(a, b)                                     # positive = full model better
            t["n_common_pairs"] = n
            res[v][key] = t
        r, c = res[v]["mse_roi_model"], res[v]["crps_cal_brain"]
        print(f"  {LABELS[v]:26s} n={r['n_patients']:>3} | ROI MSE {r['rel_improvement_pct']:+5.1f}% "
              f"[{r['ci95_pct'][0]:+.1f},{r['ci95_pct'][1]:+.1f}] p={r['p']:.1e} | CRPS {c['rel_improvement_pct']:+5.1f}% "
              f"[{c['ci95_pct'][0]:+.1f},{c['ci95_pct'][1]:+.1f}] p={c['p']:.1e}")
        lines.append(f"{LABELS[v]} ({r['n_patients']}) & {fmt(r)} & {fmt(c)} \\\\")
    a, b = AG.aligned_within(flow["recs"], "crps_cal_brain", "crps_raw_brain")
    t = AG.paired_test(a, b)
    res["no_calibration"] = t
    print(f"  {'No calibration':26s} n={t['n_patients']:>3} | brain CRPS {t['rel_improvement_pct']:+5.1f}% "
          f"[{t['ci95_pct'][0]:+.1f},{t['ci95_pct'][1]:+.1f}] p={t['p']:.1e}")
    lines.append(f"No calibration ({t['n_patients']}) & 0 & {fmt(t)} \\\\")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    with open(os.path.join(TRAIN_DIR, "table3.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("\n% ---- Table III rows (advantage of the full model, %, 95% CI) ----")
    print("\n".join(lines))
    print(f"\n[saved] {TRAIN_DIR}/table3.json  (p-values unadjusted; Table III is descriptive)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["identify", "train_ssm_only", "eval", "summary"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--local_attr", type=str, default="")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = full original schedule")
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "identify":
        identify()
    elif args.mode == "train_ssm_only":
        train_ssm_only(args)
    elif args.mode == "eval":
        if not args.variant:
            raise SystemExit("--variant is required for eval")
        folds_to_run = args.folds or ([0] if args.variant == "ssm_only" else [0, 1, 2, 3, 4])
        ds = CachedPairDataset(EU.CACHE)
        folds = list(patient_kfold(ds, k=5))
        prev = EU.build_prev_index()
        for k in folds_to_run:
            eval_fold(args.variant, k, args, ds, folds, prev)
    else:
        summary()


if __name__ == "__main__":
    main()
