#!/usr/bin/env python3
"""
test_generative.py — the CORRECT success test for a conditional generative forecaster.

The old "overfit to cosine 1.0" was wrong: the deterministic map z_a->z_b does not
exist (that's why pseudoprogression exists). Success is now THREE criteria:

  (1) FIXED-eps reconstruction: with the noise fixed, the ODE from eps must
      reconstruct z_b (proves the transport + training are correct).
  (2) z_a-DEPENDENCE (leak-check / conditioning works): generated futures must
      CHANGE when z_a changes. If swapping z_a doesn't move the output, the model
      ignores its conditioning (unconditional) -> useless forecaster.
  (3) DISTRIBUTION: with random eps, samples spread (nonzero variance) and their
      MEAN is closer to z_b than z_a is (the forecast is informative).

Trains conditional noise->z_b CFM on 4 pairs (a real distribution to condition on),
then checks all three. Fast rig.
Run:  CUDA_VISIBLE_DEVICES=0 python test_generative.py
"""
from __future__ import annotations
import os, sys, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss, generate
from velocity_net_v2 import VelocityNetV2

DEV="cuda:0"; N=4; STEPS=800; LR=1e-3

def main():
    torch.manual_seed(0)
    ds = CachedPairDataset("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere")
    b=[ds[i] for i in range(N)]
    z_a=torch.stack([x["z_a"] for x in b]).to(DEV)
    z_b=torch.stack([x["z_b"] for x in b]).to(DEV)
    dt =torch.stack([x["dt"] for x in b]).to(DEV)
    print(f"[gen] {N} pairs")

    net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(1,1,1),bottleneck_depth=1,use_checkpoint=False).to(DEV)
    net.set_week(None)
    loss_fn=HeteroscedasticFMLoss()
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4)

    def v_fn(z,tau,za,dtt): return net(z,tau,za,dtt)

    net.train(); t0=time.time()
    for step in range(1,STEPS+1):
        tau=sample_tau(N,DEV)
        eps=torch.randn_like(z_b)
        z_tau,u=noise_interpolate(z_b,tau,eps)
        v,logvar=net(z_tau,tau,z_a,dt)
        loss,_,logs=loss_fn(u,v,logvar)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        if step==1 or step%100==0:
            print(f"  step {step:4d} | loss {float(loss):8.4f} | vel_mse {logs['vel_mse']:.4f}")

    net.eval()
    base=F.mse_loss(z_a,z_b).item()
    def cm(zh): return (F.cosine_similarity(zh.flatten(1),z_b.flatten(1)).mean().item(), F.mse_loss(zh.float(),z_b).item())

    # (1) fixed-eps reconstruction
    fixed_eps=torch.randn_like(z_b)
    rec=generate(v_fn,z_a,dt,steps=20,n_samples=1,eps0=fixed_eps)[0]
    c1,m1=cm(rec)
    print(f"\n(1) fixed-eps recon : cosine {c1:.4f}  MSE {m1:.5f}  (base {base:.5f})")

    # (2) z_a-dependence: generate with correct z_a vs SHUFFLED z_a, same eps
    z_a_shuf = z_a[torch.tensor([1,2,3,0],device=DEV)]
    g_true=generate(v_fn,z_a,     dt,steps=20,n_samples=1,eps0=fixed_eps)[0]
    g_shuf=generate(v_fn,z_a_shuf,dt,steps=20,n_samples=1,eps0=fixed_eps)[0]
    za_effect=(g_true-g_shuf).abs().mean().item()
    out_scale=g_true.abs().mean().item()
    print(f"(2) z_a-dependence  : |gen(z_a)-gen(shuffled z_a)| = {za_effect:.4f} "
          f"(vs output scale {out_scale:.4f}; ratio {za_effect/max(out_scale,1e-8):.2f}) "
          f"-> {'USES z_a' if za_effect > 0.2*out_scale else 'IGNORES z_a!'}")

    # (3) distribution: R random samples -> spread + mean informativeness
    R=8
    samples=generate(v_fn,z_a,dt,steps=20,n_samples=R)   # [R,B,c,h,w,d]
    mean=samples.mean(0)
    spread=samples.std(0).mean().item()
    cM,mM=cm(mean)
    print(f"(3) distribution    : sample spread {spread:.4f} | mean forecast cosine {cM:.4f} "
          f"MSE {mM:.5f} (base {base:.5f})")

    print(f"\n[gen] {time.time()-t0:.0f}s")
    ok1 = c1>0.9 and m1<0.5*base
    ok2 = za_effect > 0.2*out_scale
    ok3 = spread>1e-3 and mM < base
    print(f"CRITERIA: (1)recon {'PASS' if ok1 else 'FAIL'} | (2)uses-z_a {'PASS' if ok2 else 'FAIL'} | (3)informative-dist {'PASS' if ok3 else 'FAIL'}")
    print(f"VERDICT: {'ENGINE WORKS — conditional generative forecaster is alive' if (ok1 and ok2 and ok3) else 'see which criterion failed'}")

if __name__=="__main__":
    main()
