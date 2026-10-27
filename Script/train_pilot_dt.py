#!/usr/bin/env python3
"""
train_pilot_dt.py — fold-0 PILOT: does the model use Δt once the pathway is fixed?

Controlled comparison against the reported fold-0 model (VelocityNetV2):
  * ONLY the architecture differs (VelocityNetV3: Δt into LocalHead, non-zero gates).
  * Recipe (batch, LR, warmup, var-warmup, EMA, clip, validation R/steps) is read from
    the saved cfg in runs/forecast_fold0/best.pt. The LR schedule keeps its ORIGINAL
    total length (epochs x steps/epoch); training just stops at MAX_STEPS, so the
    schedule is identical to the reported run up to that point.
  * Same patient split, same validation set.

At the end it re-runs, on the best pilot checkpoint, the exact Δt tests already run on
V2 (same 40 pairs, same seeds, same shuffle permutation), and prints a verdict.

Run on GPU 0:
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=0 setsid nohup python train_pilot_dt.py > runs/pilot_dt_fold0.log 2>&1 &
"""
from __future__ import annotations
import os, sys, json, math, copy, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss, generate
from velocity_net_v3 import VelocityNetV3
import metrics as M

FOLD = 0
MAX_STEPS = 4000          # every V2 fold peaked at 3000; 4000 shows whether V3 peaks later
EVAL_EVERY = 500
COND_INIT, REFINE_INIT = 0.5, 0.1
CACHE = "latent_cache_lumiere"
REF = f"runs/forecast_fold{FOLD}/best.pt"
OUT = f"runs/pilot_dt_fold{FOLD}"
DEV = "cuda:0"

# Reference numbers for the CURRENT V2 fold-0 model (from the runs you already did)
REF_BEST_VAL_CRPS = 0.1738                                   # train-time val, step 3000
REF_DT = dict(true=0.1686, shuffled=0.1689, median=0.1690, pct=57.0, sens=0.0188)


def load_recipe():
    rc = torch.load(REF, map_location="cpu", weights_only=False).get("cfg", {}) or {}
    defaults = dict(batch=16, epochs=300, lr=2e-4, warmup_steps=500, weight_decay=1e-4,
                    var_warmup=2000, ema_decay=0.999, grad_clip=1.0, gen_steps=20,
                    gen_samples=8, k_folds=5)
    cfg = {k: rc.get(k, v) for k, v in defaults.items()}
    missing = [k for k in defaults if k not in rc]
    return cfg, missing


def lr_at(step, cfg, total):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * step / max(1, cfg["warmup_steps"])
    p = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

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
    return {k: (torch.stack([b[k] for b in batch]) if torch.is_tensor(batch[0][k])
                else [b[k] for b in batch]) for k in batch[0]}


def make_net(use_checkpoint=True):
    return VelocityNetV3(latent_dim=4, d_h=128, stage_depths=(2, 2, 4), bottleneck_depth=4,
                         use_checkpoint=use_checkpoint, cond_init=COND_INIT,
                         refine_init=REFINE_INIT).to(DEV)


@torch.no_grad()
def validate(net, ema, loader, cfg):
    backup = copy.deepcopy(net.state_dict()); ema.copy_to(net); net.eval(); net.set_week(None)
    torch.manual_seed(1234)
    crps, mse, div = [], [], []
    for b in loader:
        z_a = b["z_a"].to(DEV); z_b = b["z_b"].to(DEV); dt = b["dt"].to(DEV).float()
        samp = generate(lambda z, t, za, d: net(z, t, za, d), z_a, dt,
                        steps=cfg["gen_steps"], n_samples=cfg["gen_samples"])
        crps.append(M.ensemble_crps(samp, z_b))
        mse.append(F.mse_loss(samp.mean(0), z_b).item())
        div.append(M.sample_diversity(samp))
        del samp; torch.cuda.empty_cache()
    net.load_state_dict(backup); net.train()
    return dict(crps=float(np.mean(crps)), latent_mse=float(np.mean(mse)),
                diversity=float(np.mean(div)))


@torch.no_grad()
def dt_tests(net, ds, val_idx):
    """Identical protocol to the test already run on V2."""
    net.eval(); net.set_week(None)
    sub = val_idx[:40]
    dts = torch.stack([ds[i]["dt"] for i in sub]).float()
    perm = dts[torch.randperm(len(dts), generator=torch.Generator().manual_seed(0))]
    med = dts.median()
    res = {"true": [], "shuffled": [], "median": []}
    for j, i in enumerate(sub):
        s = ds[i]; z_a = s["z_a"][None].to(DEV); z_b = s["z_b"][None].to(DEV)
        for name, d in [("true", dts[j]), ("shuffled", perm[j]), ("median", med)]:
            torch.manual_seed(1000 + i)
            samp = generate(lambda z, t, za, dd: net(z, t, za, dd), z_a, d.view(1).to(DEV),
                            steps=20, n_samples=16)
            res[name].append(M.ensemble_crps(samp, z_b))
    r = {k: np.array(v) for k, v in res.items()}

    sens = []
    for i in val_idx[:20]:
        s = ds[i]; z_a = s["z_a"][None].to(DEV)
        torch.manual_seed(i); eps = torch.randn_like(z_a)
        def g(d):
            return generate(lambda z, t, za, dd: net(z, t, za, dd), z_a,
                            torch.tensor([d], device=DEV), steps=20, n_samples=1, eps0=eps)[0]
        y = g(float(s["dt"])); y30 = g(30.0); y300 = g(300.0)
        sens.append((y30 - y300).abs().mean().item() / y.abs().mean().item())

    return dict(true=float(r["true"].mean()), shuffled=float(r["shuffled"].mean()),
                median=float(r["median"].mean()),
                pct=float(100 * np.mean(r["true"] < r["shuffled"])),
                sens=float(np.mean(sens)))


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg, missing = load_recipe()
    print(f"[pilot] recipe copied from {REF}:\n        {cfg}")
    if missing:
        print(f"[pilot] WARNING: not found in saved cfg, defaults used for {missing}. "
              f"Confirm these match the reported run before trusting the comparison.")

    torch.manual_seed(0)
    ds = CachedPairDataset(CACHE)
    tr_idx, va_idx = list(patient_kfold(ds, k=cfg["k_folds"]))[FOLD]
    tr = DataLoader(Subset(ds, tr_idx), cfg["batch"], shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True)
    va = DataLoader(Subset(ds, va_idx), cfg["batch"], shuffle=False, num_workers=2,
                    collate_fn=collate)
    total = cfg["epochs"] * len(tr)
    print(f"[pilot] fold {FOLD}: train={len(tr_idx)} val={len(va_idx)} | "
          f"LR schedule length {total} steps (as original), stopping at {MAX_STEPS}")

    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = EMA(net, cfg["ema_decay"])
    loss_fn = HeteroscedasticFMLoss()
    step, best, t0 = 0, float("inf"), time.time()
    logf = open(os.path.join(OUT, "val_log.jsonl"), "w")

    net.train()
    while step < MAX_STEPS:
        for b in tr:
            z_a = b["z_a"].to(DEV); z_b = b["z_b"].to(DEV); dt = b["dt"].to(DEV).float()
            net.set_week(None)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg, total)
            tau = sample_tau(z_a.shape[0], DEV)
            eps = torch.randn_like(z_b)
            z_tau, u = noise_interpolate(z_b, tau, eps)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v, logvar = net(z_tau, tau, z_a, dt)
            if step < cfg["var_warmup"]:
                loss = F.mse_loss(v.float(), u.float()); vel_mse = float(loss.detach())
            else:
                loss, _, lg = loss_fn(u, v.float(), logvar.float()); vel_mse = lg["vel_mse"]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step(); ema.update(net); step += 1

            if step % 100 == 0:
                print(f"  step {step}/{MAX_STEPS} | loss {float(loss):.4f} | vel_mse {vel_mse:.4f} | "
                      f"cond {float(net.cond_scale):+.3f} refine {float(net.refine_scale):+.3f} | "
                      f"{(time.time() - t0) / step * 100:.0f}s/100", flush=True)

            if step % EVAL_EVERY == 0:
                r = validate(net, ema, va, cfg)
                r.update(step=step, cond_scale=float(net.cond_scale),
                         refine_scale=float(net.refine_scale))
                print(f"  [val {step}] crps={r['crps']:.4f} mse={r['latent_mse']:.4f} "
                      f"div={r['diversity']:.4f}", flush=True)
                logf.write(json.dumps(r) + "\n"); logf.flush()
                if r["crps"] < best:
                    best = r["crps"]
                    torch.save({"net": net.state_dict(), "ema": ema.shadow, "step": step,
                                "best_crps": best, "cfg": cfg, "arch": "VelocityNetV3",
                                "cond_init": COND_INIT, "refine_init": REFINE_INIT},
                               os.path.join(OUT, "best.pt"))
                    print(f"  [val] new best CRPS {best:.4f} -> best.pt", flush=True)
            if step >= MAX_STEPS:
                break
    logf.close()
    print(f"[pilot] training done in {(time.time() - t0) / 3600:.2f}h; running Δt tests...", flush=True)

    st = torch.load(os.path.join(OUT, "best.pt"), map_location=DEV, weights_only=False)
    ev = make_net(use_checkpoint=False); ev.load_state_dict(st["ema"])
    d = dt_tests(ev, ds, va_idx)

    functional = (d["shuffled"] - d["true"]) >= 0.002 and d["pct"] >= 60
    not_worse = best <= REF_BEST_VAL_CRPS + 0.002
    print("\n" + "=" * 70)
    print(f"  PILOT (V3) vs REPORTED (V2), fold {FOLD}")
    print("=" * 70)
    print(f"  best val CRPS (R={cfg['gen_samples']}):  V3 {best:.4f} @ step {st['step']}   "
          f"V2 {REF_BEST_VAL_CRPS:.4f} @ step 3000")
    print(f"  Δt test CRPS true/shuffled/median:  V3 {d['true']:.4f}/{d['shuffled']:.4f}/{d['median']:.4f}"
          f"   V2 {REF_DT['true']:.4f}/{REF_DT['shuffled']:.4f}/{REF_DT['median']:.4f}")
    print(f"  true Δt better than shuffled:       V3 {d['pct']:.0f}%   V2 {REF_DT['pct']:.0f}%")
    print(f"  output sensitivity 30d vs 300d:     V3 {d['sens']:.4f}   V2 {REF_DT['sens']:.4f}")
    print("-" * 70)
    print(f"  Δt functional (gap>=0.002 & >=60% pairs): {'YES' if functional else 'NO'}")
    print(f"  accuracy not worse than V2 (+0.002):      {'YES' if not_worse else 'NO'}")
    if functional and not_worse:
        print("  VERDICT: adopt V3 -> retrain all 5 folds with this recipe.")
    elif not functional:
        print("  VERDICT: Δt carries no usable signal even with a direct path -> keep V2,")
        print("           drop the interval-conditioning claim, report this test as the finding.")
    else:
        print("  VERDICT: Δt is used but accuracy dropped -> discuss before retraining.")
    print("=" * 70)
    json.dump(dict(best_crps=best, best_step=st["step"], dt_test=d, ref_v2=REF_DT,
                   functional=functional, not_worse=not_worse, cfg=cfg),
              open(os.path.join(OUT, "pilot_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
