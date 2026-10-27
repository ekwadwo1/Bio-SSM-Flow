#!/usr/bin/env python3
"""
probe_null.py — the decisive fork: is it the SSM DENOISER, or the DATA/LOSS?

Every denoiser-side fix (init, scale, decay, hook, out_norm, precision) produced
the SAME molasses crawl. That means the cause is common to all of them. Two
possibilities remain, and this test separates them in ~15s:

  Replace the ENTIRE SSM denoiser with a trivial 3-layer 3D conv net and overfit
  the SAME 2 pairs with the SAME loss/interpolant/scaling.

  - convnet OVERFITS (loss->~0, cosine->~1)  => the data+loss are FINE; the SSM
    denoiser architecture is what can't learn this task. (Fork: adapt/replace it.)
  - convnet ALSO STALLS                       => the problem is upstream of the
    model: the cached latents or the pairing (z_a,z_b not actually a learnable
    pair). We re-examine the data itself.

A 3-layer convnet MUST be able to memorize 2 samples. If it can't, the data is wrong.

Run:  CUDA_VISIBLE_DEVICES=0 python probe_null.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s

CACHE = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
DEVICE = "cuda:0"; N = 2; STEPS = 400; LR = 3e-3


class TinyVel(nn.Module):
    """Trivial velocity net: cat[z_s, z_a] + s-embed -> 3 convs -> velocity. No SSM."""
    def __init__(self, c=4, d=64):
        super().__init__()
        self.in_conv = nn.Conv3d(2 * c, d, 3, padding=1)
        self.s_emb = nn.Linear(1, d)
        self.mid = nn.Sequential(nn.SiLU(), nn.Conv3d(d, d, 3, padding=1),
                                 nn.SiLU(), nn.Conv3d(d, d, 3, padding=1), nn.SiLU())
        self.out = nn.Conv3d(d, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, z_s, s, z_a):
        h = self.in_conv(torch.cat([z_s, z_a], 1))
        h = h + self.s_emb(s.view(-1, 1)).view(h.shape[0], -1, 1, 1, 1)
        return self.out(self.mid(h))


def main():
    ds = CachedPairDataset(CACHE)
    b = [ds[i] for i in range(N)]
    z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
    z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
    vel_scale = float((z_b - z_a).std().clamp_min(1e-6))
    print(f"VEL_SCALE={vel_scale:.4f} | z_a std {z_a.std():.3f} | "
          f"mean|z_b-z_a| {(z_b-z_a).abs().mean():.3f}")
    # SANITY on the data: are the 2 pairs actually distinct and non-degenerate?
    print(f"pair0 |z_b-z_a|={ (z_b[0]-z_a[0]).abs().mean():.3f}  "
          f"pair1 |z_b-z_a|={ (z_b[1]-z_a[1]).abs().mean():.3f}  "
          f"cos(a0,a1)={F.cosine_similarity(z_a[0].flatten(0,-1)[None],z_a[1].flatten(0,-1)[None]).item():.3f}")

    torch.manual_seed(0)
    net = TinyVel().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    net.train(); t0 = time.time()
    for step in range(1, STEPS + 1):
        s = sample_s(N, DEVICE)
        z_s, u = interpolate(z_a, z_b, s)
        target = u / vel_scale
        v = net(z_s, s * 1000.0, z_a)
        loss = F.mse_loss(v, target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d} | loss {float(loss):.5f} | |v|~{float(v.detach().abs().mean()):.4f} (tgt {float(target.abs().mean()):.3f})")

    net.eval()
    with torch.no_grad():
        z = z_a.clone(); M = 8
        for m in range(M):
            v = net(z, torch.full((N,), m/M*1000., device=DEVICE), z_a)
            z_mid = z + (0.5/M)*(v*vel_scale)
            v2 = net(z_mid, torch.full((N,), (m+0.5)/M*1000., device=DEVICE), z_a)
            z = z + (1.0/M)*(v2*vel_scale)
    cos = F.cosine_similarity(z.flatten(1), z_b.flatten(1)).mean().item()
    base = F.mse_loss(z_a, z_b).item(); mse = F.mse_loss(z, z_b).item()
    print(f"\n{time.time()-t0:.0f}s | forecast MSE {mse:.5f} (base {base:.5f}) | cosine {cos:.4f}")
    if cos > 0.95 and mse < 0.2 * base:
        print("VERDICT: convnet OVERFITS -> data+loss FINE; the SSM denoiser is the problem.")
    elif cos > 0.9:
        print("VERDICT: convnet mostly learns -> data OK; SSM denoiser struggles (optimization).")
    else:
        print("VERDICT: convnet ALSO STALLS -> problem is UPSTREAM (cached latents / pairing). Re-examine data.")


if __name__ == "__main__":
    main()
