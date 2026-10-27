#!/usr/bin/env python3
"""
train_pilot_lan.py — pilot retraining on LESION-AWARE latents, post-operative pairs only.

Variants (architecture and recipe otherwise identical to the reported V2 model):
  lan      : conditional flow noise -> z_b                       (only the data changed)
  lan_res  : conditional flow noise -> Δ,  Δ = (z_b - z_a) / s    (copy-anchored residual)
             forecast = z_a + s * Δ_generated. s = std of (z_b - z_a) over the TRAINING
             pairs of this fold (stored in the checkpoint). Leak-free: z_a is conditioning
             and the anchor, never inside the noise interpolant together with the target.

Controlled design:
  * recipe (batch, LR, warmups, EMA, clip) read from runs/forecast_fold{k}/best.pt cfg;
  * LR schedule keeps the ORIGINAL fold's total length (computed from its full train set);
  * FIXED budget: training stops at MAX_STEPS (3000, where every V2 fold peaked) and the
    final EMA weights are saved — no checkpoint selection on validation data;
  * same patient split as the original folds (asserted); week-0 (pre-operative) baselines
    are excluded from both training and validation.

Run (one process per variant/fold):
  CUDA_VISIBLE_DEVICES=0 python train_pilot_lan.py --variant lan     --fold 0
  CUDA_VISIBLE_DEVICES=1 python train_pilot_lan.py --variant lan_res --fold 0
Output: runs/pilot_lan/<variant>_fold<k>/final.pt, val_log.jsonl
"""
from __future__ import annotations
import os, sys, json, math, copy, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss, generate
from velocity_net_v2 import VelocityNetV2
import metrics as M

CACHE_OLD = os.path.join(HERE, "latent_cache_lumiere")
CACHE_LAN = os.path.join(HERE, "latent_cache_lumiere_lan")
DEV = "cuda:0"


def postop(ds, idx):
    return [i for i in idx if not os.path.basename(ds.pairs[i][0]).endswith("_week0.pt")]


def load_recipe(fold):
    rc = torch.load(os.path.join(HERE, "runs", f"forecast_fold{fold}", "best.pt"),
                    map_location="cpu", weights_only=False).get("cfg", {}) or {}
    d = dict(batch=16, epochs=300, lr=2e-4, warmup_steps=500, weight_decay=1e-4, var_warmup=2000,
             ema_decay=0.999, grad_clip=1.0, gen_steps=20, gen_samples=8, k_folds=5)
    return {k: rc.get(k, v) for k, v in d.items()}, [k for k in d if k not in rc]


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

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def collate(batch):
    return {k: (torch.stack([b[k] for b in batch]) if torch.is_tensor(batch[0][k]) else [b[k] for b in batch])
            for k in batch[0]}


@torch.no_grad()
def validate(net, ema, loader, cfg, res_scale):
    backup = copy.deepcopy(net.state_dict()); ema.copy_to(net); net.eval(); net.set_week(None)
    torch.manual_seed(1234)
    crps, mse = [], []
    for b in loader:
        z_a, z_b, dt = b["z_a"].to(DEV), b["z_b"].to(DEV), b["dt"].to(DEV).float()
        x = generate(lambda z, t, za, d: net(z, t, za, d), z_a, dt,
                     steps=cfg["gen_steps"], n_samples=cfg["gen_samples"])
        samp = z_a.unsqueeze(0) + res_scale * x if res_scale else x
        crps.append(M.ensemble_crps(samp, z_b))
        mse.append(F.mse_loss(samp.mean(0), z_b).item())
    net.load_state_dict(backup); net.train()
    return float(np.mean(crps)), float(np.mean(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["lan", "lan_res"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--val_every", type=int, default=1000)
    args = ap.parse_args()
    out = os.path.join(HERE, "runs", "pilot_lan", f"{args.variant}_fold{args.fold}")
    os.makedirs(out, exist_ok=True)

    cfg, missing = load_recipe(args.fold)
    print(f"[pilot-lan] recipe from forecast_fold{args.fold}: {cfg}")
    if missing:
        print(f"[pilot-lan] WARNING: defaults used for {missing}")

    torch.manual_seed(0)
    ds = CachedPairDataset(CACHE_LAN)
    ds_old = CachedPairDataset(CACHE_OLD)
    assert ds.patients == ds_old.patients, "LAN cache patients differ from the original cache — fold split would change"
    tr_full, va_full = list(patient_kfold(ds, k=cfg["k_folds"]))[args.fold]
    tr, va = postop(ds, tr_full), postop(ds, va_full)
    total = cfg["epochs"] * (len(tr_full) // cfg["batch"])          # original schedule length
    print(f"[pilot-lan] fold {args.fold}: train {len(tr)}/{len(tr_full)} post-op pairs, "
          f"val {len(va)}/{len(va_full)} | LR schedule {total} steps, stop at {args.max_steps}")

    res_scale = None
    if args.variant == "lan_res":
        d = torch.stack([ds[i]["z_b"] - ds[i]["z_a"] for i in tr])
        res_scale = float(d.std())
        print(f"[pilot-lan] residual scale s = std(z_b - z_a) over training pairs = {res_scale:.4f}")

    dl = DataLoader(Subset(ds, tr), cfg["batch"], shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    vl = DataLoader(Subset(ds, va), cfg["batch"], shuffle=False, num_workers=2, collate_fn=collate)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                        use_checkpoint=True).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = EMA(net, cfg["ema_decay"])
    loss_fn = HeteroscedasticFMLoss()
    logf = open(os.path.join(out, "val_log.jsonl"), "w")
    step, t0 = 0, time.time()
    net.train()
    while step < args.max_steps:
        for b in dl:
            z_a, z_b, dt = b["z_a"].to(DEV), b["z_b"].to(DEV), b["dt"].to(DEV).float()
            net.set_week(None)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, total)
            x1 = (z_b - z_a) / res_scale if res_scale else z_b
            tau = sample_tau(z_a.shape[0], DEV)
            eps = torch.randn_like(x1)
            z_tau, u = noise_interpolate(x1, tau, eps)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v, logvar = net(z_tau, tau, z_a, dt)
            if step < cfg["var_warmup"]:
                loss = F.mse_loss(v.float(), u.float()); vel = float(loss.detach())
            else:
                loss, _, lg = loss_fn(u, v.float(), logvar.float()); vel = lg["vel_mse"]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step(); ema.update(net); step += 1
            if step % 100 == 0:
                print(f"  step {step}/{args.max_steps} | loss {float(loss):.4f} | vel_mse {vel:.4f} | "
                      f"{(time.time() - t0) / step * 100:.0f}s/100", flush=True)
            if step % args.val_every == 0:
                c, m = validate(net, ema, vl, cfg, res_scale)
                print(f"  [val {step}] crps={c:.4f} mse={m:.4f}  (monitoring only; no selection)", flush=True)
                logf.write(json.dumps({"step": step, "crps": c, "latent_mse": m}) + "\n"); logf.flush()
            if step >= args.max_steps:
                break
    logf.close()
    torch.save({"ema": ema.shadow, "net": net.state_dict(), "step": step, "cfg": cfg,
                "variant": args.variant, "fold": args.fold, "res_scale": res_scale,
                "cache": CACHE_LAN, "post_op_only": True},
               os.path.join(out, "final.pt"))
    print(f"[pilot-lan] done in {(time.time() - t0) / 3600:.2f}h -> {out}/final.pt")


if __name__ == "__main__":
    main()
