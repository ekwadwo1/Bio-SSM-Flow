#!/usr/bin/env python3
"""
test_forecast_modes.py — training works; is the ODE SAMPLER the remaining bug?

velocity_net_v2 learned z_b (mse<0.05 @ step 29) but the 8-step ODE forecast gave
cosine 0.68. With x1-prediction we don't need an ODE: the net predicts z_b directly.
This trains the same net, then compares THREE forecast readouts:

  (A) single-shot at s=0 : x1_hat(z_a, s=0, z_a)      -> predicted z_b directly
  (B) single-shot at s=0.5 (feed true interpolant midpoint as a probe)
  (C) 8-step ODE (the current forecast())            -> for comparison

If (A) gives high cosine, the model is fine and the ODE sampler was the bug.
Run:  CUDA_VISIBLE_DEVICES=0 python test_forecast_modes.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s, forecast
from velocity_net_v2 import VelocityNetV2

DEV="cuda:0"; N=2; STEPS=400; LR=1e-3
ds = CachedPairDataset("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere")
b=[ds[i] for i in range(N)]
z_a=torch.stack([x["z_a"] for x in b]).to(DEV)
z_b=torch.stack([x["z_b"] for x in b]).to(DEV)
dt =torch.stack([x["dt"] for x in b]).to(DEV)

torch.manual_seed(0)
net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(1,1,1),bottleneck_depth=1,use_checkpoint=False).to(DEV)
net.set_week(None)
opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.0)

net.train()
for step in range(1,STEPS+1):
    s=sample_s(N,DEV); z_s,_=interpolate(z_a,z_b,s)
    x1,_=net(z_s,s,z_a,dt); loss=F.mse_loss(x1.float(),z_b.float())
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    if step%100==0: print(f"  step {step} | mse {float(loss):.5f}")

net.eval()
base=F.mse_loss(z_a,z_b).item()
def cosmse(zh):
    return (F.cosine_similarity(zh.flatten(1),z_b.flatten(1)).mean().item(),
            F.mse_loss(zh.float(),z_b).item())
with torch.no_grad():
    # (A) single-shot at s=0: feed z_a as the interpolant at s=0
    xA,_=net(z_a, torch.zeros(N,device=DEV), z_a, dt)
    cA=cosmse(xA)
    # (B) single-shot at s=0.5 using the TRUE midpoint interpolant (upper bound probe)
    zmid,_=interpolate(z_a,z_b,torch.full((N,),0.5,device=DEV))
    xB,_=net(zmid, torch.full((N,),0.5,device=DEV), z_a, dt)
    cB=cosmse(xB)
    # (C) the 8-step ODE
    xC,_=forecast(lambda z,s,za,d: net(z,s,za,d), z_a, dt, steps=8)
    cC=cosmse(xC)

print(f"\nbaseline z_a->z_b MSE: {base:.5f}")
print(f"(A) single-shot s=0   : cosine {cA[0]:.4f}  MSE {cA[1]:.5f}")
print(f"(B) single-shot s=0.5*: cosine {cB[0]:.4f}  MSE {cB[1]:.5f}   (*uses true midpoint, upper bound)")
print(f"(C) 8-step ODE        : cosine {cC[0]:.4f}  MSE {cC[1]:.5f}")
print(f"\nVERDICT: {'(A) works -> ODE sampler was the bug; forecast with single-shot s=0' if (cA[0]>0.9 and cA[1]<0.5*base) else 'even single-shot fails -> prediction at s=0 differs from training regime'}")
