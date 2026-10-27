#!/usr/bin/env python3
"""
overfit_test.py — can the flow engine MEMORIZE 4 real LUMIERE pairs?

This is the decisive end-to-end check. It wires together:
  cached_pair_dataset (real z_a,z_b) -> flow.interpolate -> VelocityNet
  -> HeteroscedasticFMLoss -> backward -> flow.forecast (few-step ODE).

Pass criteria (on 4 pairs, a few hundred steps, single GPU):
  1. FM loss drops sharply (velocity MSE -> near 0): the net learns the transport.
  2. Forecast reconstruction cosine(z_b_hat, z_b) -> ~1 and latent-MSE -> small:
     integrating the learned velocity from z_a actually lands on z_b.
If both happen, the learning machinery is correct end-to-end. If the loss is
flat, something upstream is broken (grad flow, conditioning, rescale) — and we
debug THAT, not a 10-hour training run.

Run:  CUDA_VISIBLE_DEVICES=0 python overfit_test.py
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
import flow as flowmod
from flow import interpolate, sample_s, HeteroscedasticFMLoss, forecast, calibrate_vel_scale
from velocity_net import VelocityNet

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"
N_PAIRS = 4
STEPS = 200                    # zero-init+scaling should show clear convergence by here
LR = 1e-3                      # raised from 2e-4: 4-pair overfit wants a hotter LR
LOSS_MODE = "mse"             # "mse" (diagnostic: pure velocity regression) or "hetero"
VEL_SCALE = 0.23              # standardize target u to ~unit magnitude (mean|u|~0.23).
                              # network predicts u/VEL_SCALE; multiply back at inference.


def main():
    torch.manual_seed(0)
    ds = CachedPairDataset(CACHE)
    assert len(ds) >= N_PAIRS, f"need >= {N_PAIRS} pairs, have {len(ds)}"

    # grab 4 fixed pairs, move to GPU once (tiny — latents are [4,32,32,32])
    batch = [ds[i] for i in range(N_PAIRS)]
    z_a = torch.stack([b["z_a"] for b in batch]).to(DEVICE)      # [4,4,32,32,32]
    z_b = torch.stack([b["z_b"] for b in batch]).to(DEVICE)
    dt  = torch.stack([b["dt"] for b in batch]).to(DEVICE)       # [4]
    print(f"[overfit] {N_PAIRS} pairs | z_a {tuple(z_a.shape)} | "
          f"dt(days) {[round(float(x)) for x in dt]}")
    print(f"[overfit] latent scale: z_b std={z_b.std().item():.3f} "
          f"mean|z_b-z_a|={ (z_b-z_a).abs().mean().item():.3f}")

    # calibrate the velocity standardization scale from the real pairs
    flowmod.VEL_SCALE = calibrate_vel_scale(z_a, z_b)
    print(f"[overfit] VEL_SCALE (std of z_b-z_a) = {flowmod.VEL_SCALE:.4f} "
          f"-> net predicts unit-scale velocity")

    net = VelocityNet(latent_dim=4, d_h=128, stage_depths=(2, 2, 4),
                      bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    # Zero-init the velocity head: start at v=0 and build up (standard flow/diffusion
    # practice). Prevents the overshoot->punish->collapse-to-zero dynamic we observed.
    torch.nn.init.zeros_(net.denoiser.eps_head.weight)
    if net.denoiser.eps_head.bias is not None:
        torch.nn.init.zeros_(net.denoiser.eps_head.bias)
    net.set_week(None)
    loss_fn = HeteroscedasticFMLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[overfit] trainable params: {n_params:,}\n")

    net.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        s = sample_s(N_PAIRS, DEVICE)
        z_s, u = interpolate(z_a, z_b, s)
        u_scaled = u / flowmod.VEL_SCALE              # ~unit-magnitude target (calibrated from data)
        # fp32 for the overfit diagnostic: bf16's ~3 sig-digits underflow the
        # small-magnitude velocity target (mean|u|~0.23) and stall learning.
        v, logvar = net(z_s, s, z_a, dt)
        if LOSS_MODE == "mse":
            loss = F.mse_loss(v.float(), u_scaled.float())
            logs = {"vel_mse": float(loss.detach()), "sigma2_mean": float("nan")}
        else:
            loss, _, logs = loss_fn(u_scaled, v, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # instrument: is gradient actually reaching the output head?
        gh = net.denoiser.eps_head.weight.grad
        gh_norm = float(gh.norm()) if gh is not None else -1.0
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 50 == 0:
            vmag = float(v.detach().abs().mean())
            print(f"  step {step:4d} | loss {float(loss):8.4f} | "
                  f"vel_mse {logs['vel_mse']:.5f} | |v|~{vmag:.4f} (tgt {float(u_scaled.abs().mean()):.4f}) | "
                  f"eps_head.grad {gh_norm:.3e}")

    # ── Forecast reconstruction: integrate learned velocity from z_a ──
    net.eval()
    def vfn(z, s, za, dtt):
        v, lv = net(z, s, za, dtt)
        return v * VEL_SCALE, lv          # undo target standardization for integration
    zb_hat, _ = forecast(vfn, z_a, dt, steps=8, ensemble=1)
    zb_hat = zb_hat.float()
    mse = F.mse_loss(zb_hat, z_b).item()
    cos = F.cosine_similarity(zb_hat.flatten(1), z_b.flatten(1)).mean().item()
    # baseline: how far z_a already is from z_b (what the model had to cross)
    base_mse = F.mse_loss(z_a, z_b).item()

    print(f"\n[overfit] done in {time.time()-t0:.1f}s")
    print(f"[overfit] forecast latent MSE  : {mse:.5f}   (baseline z_a vs z_b: {base_mse:.5f})")
    print(f"[overfit] forecast cosine sim   : {cos:.4f}   (target -> ~1.0)")
    ok = (cos > 0.9) and (mse < 0.5 * base_mse)
    print(f"[overfit] VERDICT: {'PASS — engine learns the transport' if ok else 'INVESTIGATE — loss/forecast not converging'}")
    if not ok:
        print("  If vel_mse fell but forecast is poor -> ODE step count / rescale.")
        print("  If vel_mse stayed flat -> grad flow / conditioning wiring.")


if __name__ == "__main__":
    main()
