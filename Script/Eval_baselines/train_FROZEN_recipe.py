#!/usr/bin/env python3
"""
train.py — real training loop for conditional generative glioma forecasting.

Trains VelocityNetV2 as a conditional flow: noise -> z_b | (z_a, dt), on the full
469 LUMIERE pairs, with the stability fixes established during bring-up:
  - patient-level train/val split (NO patient leaks across the split)
  - VARIANCE WARMUP: pure velocity-MSE for the first `var_warmup` steps, then the
    heteroscedastic NLL (prevents the sigma-shortcut collapse we diagnosed)
  - LR warmup -> cosine decay
  - EMA of weights (evaluated with EMA; generative models are noisy otherwise)
  - grad clip, bf16 autocast with fp32 loss, gradient checkpointing
  - resumable checkpoints (step/opt/ema) every ckpt_every, plus best-by-CRPS
  - periodic validation computing the TMI metric suite (metrics.py)

Recommended device: cuda:1 (per nvidia-smi GPU 0 was at 100% util; GPU 1 idle).
Run:  CUDA_VISIBLE_DEVICES=1 python train.py
Resume: set RESUME to a checkpoint path.
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import os, sys, time, math, copy, json
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)

from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss, generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
import metrics as M
import calibration as CAL

# ─────────────────────────── CONFIG ───────────────────────────
CFG = dict(
    cache   = "/.../Bio-SSM-Flow/latent_cache_lumiere",
    vqgan   = "/.../checkpoints/vqgan_r4/best_vqgan.pt",
    out_dir = "/.../runs/forecast_fold0",
    device  = "cuda:0",             # with CUDA_VISIBLE_DEVICES=1 this maps to physical GPU 1
    fold    = 0, k_folds = 5,
    batch   = 16, epochs = 300,
    lr = 2e-4, warmup_steps = 500, weight_decay = 1e-4,
    var_warmup = 2000,              # pure-MSE steps before switching on heteroscedastic NLL
    ema_decay = 0.999,
    grad_clip = 1.0,
    val_every = 1000, ckpt_every = 1000,
    gen_steps = 20, gen_samples = 8,     # ODE steps / ensemble size at validation
    val_image_metrics = True,            # decode a small val subset for PSNR/SSIM
    val_image_n = 8,
    RESUME = None,                       # path to resume, or None
)

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
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return out


@torch.no_grad()
def validate(net, ema, loader, vqgan, cfg, device):
    backup = copy.deepcopy(net.state_dict()); ema.copy_to(net); net.eval()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)
    agg = {}; n_img = 0
    per_samp, per_y = [], []
    for bi, b in enumerate(loader):
        z_a = b["z_a"].to(device); z_b = b["z_b"].to(device); dt = b["dt"].to(device)
        net.set_week(None)
        samples = generate(v_fn, z_a, dt, steps=cfg["gen_steps"], n_samples=cfg["gen_samples"])
        # store per-pair ensembles for temperature fitting (split batch into singletons)
        for j in range(z_b.shape[0]):
            per_samp.append(samples[:, j:j+1]); per_y.append(z_b[j:j+1])
        mean = samples.mean(0)
        for k, v in M.latent_fidelity(mean, z_b).items(): agg.setdefault(k, []).append(v)
        agg.setdefault("crps", []).append(M.ensemble_crps(samples, z_b))
        agg.setdefault("picp90_raw", []).append(M.picp(samples, z_b, 0.9))
        agg.setdefault("diversity", []).append(M.sample_diversity(samples))
        s2 = samples.var(0).mean(1).reshape(z_b.shape[0], -1).clamp_min(1e-8)
        for k, v in M.uncertainty_block(mean, z_b, s2).items(): agg.setdefault(k, []).append(v)
        if cfg["val_image_metrics"] and n_img < cfg["val_image_n"]:
            for i in range(z_b.shape[0]):
                if n_img >= cfg["val_image_n"]: break
                xp = vqgan.decode(mean[i:i+1]); xt = vqgan.decode(z_b[i:i+1])
                mk = b["mask_b"][i:i+1].to(device)
                for k, v in M.image_fidelity(xp, xt, mk).items(): agg.setdefault(k, []).append(v)
                del xp, xt; torch.cuda.empty_cache(); n_img += 1
        del samples, mean; torch.cuda.empty_cache()
    # FIT temperature on validation ensembles, then report calibrated coverage
    T = CAL.fit_temperature(per_samp, per_y, verbose=False)
    picp90_cal = float(sum(CAL._picp(CAL.calibrate(s, T), y, 0.9) for s, y in zip(per_samp, per_y)) / max(len(per_y),1))
    cov_err = CAL.coverage_error(per_samp, per_y, T)
    net.load_state_dict(backup)
    out = {k: float(sum(v) / len(v)) for k, v in agg.items()}
    out["temperature"] = float(T); out["picp90_cal"] = picp90_cal; out["cov_err_cal"] = cov_err
    return out


def main():
    cfg = CFG; os.makedirs(cfg["out_dir"], exist_ok=True)
    device = cfg["device"]
    torch.manual_seed(0)

    ds = CachedPairDataset(cfg["cache"])
    folds = list(patient_kfold(ds, k=cfg["k_folds"]))
    tr_idx, va_idx = folds[cfg["fold"]]
    tr = DataLoader(Subset(ds, tr_idx), batch_size=cfg["batch"], shuffle=True,
                    num_workers=4, collate_fn=collate, drop_last=True)
    va = DataLoader(Subset(ds, va_idx), batch_size=cfg["batch"], shuffle=False,
                    num_workers=2, collate_fn=collate)
    print(f"[train] fold {cfg['fold']}: train_pairs={len(tr_idx)} val_pairs={len(va_idx)}")

    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                        bottleneck_depth=4, use_checkpoint=True).to(device)
    # frozen VQGAN for decode-based image metrics
    vqgan = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck = torch.load(cfg["vqgan"], map_location="cpu", weights_only=False)
    vqgan.load_state_dict(ck.get("gen", ck), strict=False); vqgan.eval().to(device)
    for p in vqgan.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema = EMA(net, cfg["ema_decay"])
    loss_fn = HeteroscedasticFMLoss()
    total_steps = cfg["epochs"] * len(tr)
    step = 0; best_crps = float("inf")

    if cfg["RESUME"]:
        st = torch.load(cfg["RESUME"], map_location=device, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        ema.shadow = st["ema"]; step = st["step"]; best_crps = st.get("best_crps", best_crps)
        print(f"[train] resumed at step {step}")

    print(f"[train] total_steps={total_steps} var_warmup={cfg['var_warmup']} device={device}")
    net.train(); t0 = time.time()
    for epoch in range(cfg["epochs"]):
        for b in tr:
            z_a = b["z_a"].to(device); z_b = b["z_b"].to(device); dt = b["dt"].to(device)
            net.set_week(None)
            for g in opt.param_groups: g["lr"] = lr_at(step, cfg, total_steps)

            tau = sample_tau(z_a.shape[0], device)
            eps = torch.randn_like(z_b)
            z_tau, u = noise_interpolate(z_b, tau, eps)
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                v, logvar = net(z_tau, tau, z_a, dt)
            if step < cfg["var_warmup"]:
                loss = F.mse_loss(v.float(), u.float())          # pure velocity MSE
                logs = {"vel_mse": float(loss.detach()), "sigma2_mean": float("nan")}
            else:
                loss, _, logs = loss_fn(u, v.float(), logvar.float())

            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step(); ema.update(net); step += 1

            if step % 100 == 0:
                el = time.time() - t0
                print(f"  step {step:6d}/{total_steps} | loss {float(loss):8.4f} | "
                      f"vel_mse {logs['vel_mse']:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"{el/step*100:.1f}s/100st")

            if step % cfg["val_every"] == 0:
                vm = validate(net, ema, va, vqgan, cfg, device)
                print(f"  [val step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in vm.items()))
                with open(os.path.join(cfg["out_dir"], "val_log.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, **vm}) + "\n")
                if vm.get("crps", 9e9) < best_crps:
                    best_crps = vm["crps"]
                    torch.save({"net": net.state_dict(), "ema": ema.shadow, "step": step,
                                "best_crps": best_crps, "temperature": vm.get("temperature",1.0), "cfg": cfg},
                               os.path.join(cfg["out_dir"], "best.pt"))
                    print(f"  [val] new best CRPS {best_crps:.4f} -> best.pt")
                net.train()

            if step % cfg["ckpt_every"] == 0:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                            "ema": ema.shadow, "step": step, "best_crps": best_crps, "cfg": cfg},
                           os.path.join(cfg["out_dir"], "last.pt"))
    print(f"[train] done in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    main()
