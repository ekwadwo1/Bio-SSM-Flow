#!/usr/bin/env python3
"""
biophysics_ablate.py — is the weak Fisher-KPP fit a MEASUREMENT artifact or REAL?

Tests the three fixable suspects against the 4.2% baseline:
  (S1) SHORT intervals only (dt <= 60d): less nonlinearity per step
  (S2) SMOOTHED density: gaussian-blur the density field (approximate continuous c)
  (S3) SHORT + SMOOTHED combined
If any lifts variance-explained substantially (>>4%), the physics has real signal
hidden under measurement noise -> worth keeping stronger. If all stay ~5%, the
biophysics is genuinely a weak prior for post-treatment LUMIERE -> reframe as soft.

CPU-ok. Run: python biophysics_ablate.py
"""
from __future__ import annotations
import os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from density import from_labels
from physics_operators import laplacian3d

CACHE="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
H=1.42; N=150

def smooth(c, sigma=1.0):
    # separable-ish gaussian blur via avg pool proxy (cheap): 3x3x3 mean, repeated
    k=3
    w=torch.ones(1,1,k,k,k)/(k**3)
    c2=c
    for _ in range(2):
        c2=F.conv3d(F.pad(c2,(1,1,1,1,1,1),mode='replicate'),w)
    return c2

def fit(c_a,c_b,dt):
    c_bar=0.5*(c_a+c_b); dcdt=(c_b-c_a)/max(dt,1.0)
    lap=laplacian3d(c_bar,H); react=c_bar*(1-c_bar)
    m=(c_bar[0,0]>0.03)
    if m.sum()<50: return None
    y=dcdt[0,0][m].reshape(-1)
    A=torch.stack([lap[0,0][m].reshape(-1),react[0,0][m].reshape(-1)],1)
    sol,*_=torch.linalg.lstsq(A,y.unsqueeze(1))
    r0=float(y.pow(2).mean().sqrt()); r1=float((y-(A@sol.squeeze())).pow(2).mean().sqrt())
    return r0,r1

def run(tag, short, sm):
    ds=CachedPairDataset(CACHE); rb,ra=[],[]; kept=0; tot=0
    for i in range(min(N,len(ds))):
        s=ds[i]; dt=float(s["dt"])
        if short and dt>60: continue
        tot+=1
        ca=from_labels(s["seg_a"]); cb=from_labels(s["seg_b"])
        if sm: ca,cb=smooth(ca),smooth(cb)
        o=fit(ca,cb,dt)
        if o: rb.append(o[0]); ra.append(o[1]); kept+=1
    rb,ra=np.array(rb),np.array(ra)
    if len(rb)==0: print(f"  [{tag}] no fittable pairs"); return
    drop=100*(1-ra.mean()/max(rb.mean(),1e-12))
    print(f"  [{tag:16s}] pairs={tot:3d} fittable={kept:3d} | var_explained={drop:5.1f}%")

print("baseline was 4.2% (all intervals, hard density)\n")
run("short_only",     True,  False)
run("smoothed_only",  False, True)
run("short+smoothed", True,  True)
print("\n>>4% anywhere -> keep physics stronger; all ~5% -> soft prior reframe.")
