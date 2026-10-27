#!/usr/bin/env python3
"""
bisect.py — irreducible localization of the "learns but ~5x too slow" bug.

No LUMIERE, no VelocityNet, no SSM. Four self-contained stages, each adds ONE
real component. Run stages in order; the FIRST stage that crawls is the culprit.

  STAGE 1: synthetic tensors + tiny MLP  -> proves the x1 flow loop converges at all
  STAGE 2: REAL cached latents + same tiny MLP -> tests if the data is hard to fit
  STAGE 3: synthetic tensors + the SSM denoiser -> tests if the denoiser is the slow part
  STAGE 4: real latents + SSM denoiser (= current setup) -> confirms

Each stage prints steps-to-reach mse<0.05 (or 'DID NOT REACH'). A healthy stage
hits it in << 300 steps. Set STAGE below.

Run:  CUDA_VISIBLE_DEVICES=0 python bisect.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)

DEVICE = "cuda:0"; N = 2; C = 4; H = 32; STEPS = 400; LR = 3e-3
STAGE = 4        # <-- set 1,2,3,4 and rerun each

def get_data(real: bool):
    if real:
        from cached_pair_dataset import CachedPairDataset
        ds = CachedPairDataset("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere")
        b = [ds[i] for i in range(N)]
        z_a = torch.stack([x["z_a"] for x in b]).to(DEVICE)
        z_b = torch.stack([x["z_b"] for x in b]).to(DEVICE)
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        z_a = torch.randn(N, C, H, H, H, generator=g).to(DEVICE) * 0.6
        z_b = (z_a + 0.24 * torch.randn(N, C, H, H, H, generator=g).to(DEVICE))
    return z_a, z_b

class TinyMLP(nn.Module):
    """per-voxel MLP on cat[z_s,z_a] + s -> x1. No spatial mixing needed to overfit."""
    def __init__(self, c=C, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(2*c+1, d, 1), nn.SiLU(),
                                  nn.Conv3d(d, d, 1), nn.SiLU(), nn.Conv3d(d, c, 1))
    def forward(self, z_s, s, z_a):
        sm = s.view(-1,1,1,1,1).expand(-1,1,z_s.shape[2],z_s.shape[3],z_s.shape[4])
        return self.net(torch.cat([z_s, z_a, sm], 1))

def get_model(ssm: bool):
    if ssm:
        from velocity_net import VelocityNet
        net = VelocityNet(latent_dim=C, d_h=128, stage_depths=(1,1,1),
                          bottleneck_depth=1, use_checkpoint=False).to(DEVICE)
        net.set_week(None)
        def fwd(z_s, s, z_a):
            x1, _ = net(z_s, s, z_a, torch.zeros(z_s.shape[0], device=DEVICE)); return x1
        return net, fwd
    else:
        net = TinyMLP().to(DEVICE)
        return net, (lambda z_s, s, z_a: net(z_s, s, z_a))

def main():
    real = STAGE in (2, 4)
    ssm  = STAGE in (3, 4)
    print(f"STAGE {STAGE}: data={'REAL' if real else 'synthetic'}  model={'SSM denoiser' if ssm else 'tiny MLP'}")
    z_a, z_b = get_data(real)
    net, fwd = get_model(ssm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    hit = None; t0 = time.time()
    for step in range(1, STEPS+1):
        s = torch.rand(N, device=DEVICE).clamp(1e-3, 1-1e-3)
        s_ = s.view(-1,1,1,1,1)
        z_s = (1-s_)*z_a + s_*z_b
        x1 = fwd(z_s, s, z_a)
        loss = F.mse_loss(x1.float(), z_b.float())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step == 1 or step % 50 == 0:
            print(f"  step {step:3d} | mse {float(loss):.5f}")
        if hit is None and float(loss) < 0.05:
            hit = step
    print(f"  {time.time()-t0:.0f}s | reached mse<0.05 at step: {hit if hit else 'DID NOT REACH'}")
    print(f"  {'HEALTHY' if hit and hit < 300 else 'SLOW/STUCK — culprit is in this stage''s added component'}")

if __name__ == "__main__":
    main()
