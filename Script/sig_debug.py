#!/usr/bin/env python3
"""
sig_debug.py — diagnostic significance run. Prints progress EVERY pair so a hang is
localized immediately, loads each model once, no DataLoader (direct indexing, no
worker deadlock). Processes ALL folds but flushes output continuously.

Run:  CUDA_VISIBLE_DEVICES=0 python -u sig_debug.py    (the -u forces unbuffered output)
"""
from __future__ import annotations
import os, sys, json, time
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
from ddpm_transport import DDPMTransport
import calibration as CAL
import metrics as M

CACHE="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
BASE="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs"
VQGAN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
DEV="cuda:0"; K=5; R=16; STEPS=20; TUMOR=(1,3); DIL=3

def log(m): print(m, flush=True)   # force flush so we SEE progress

log("[sig] loading VQGAN + seg (once)...")
vq=VQGAN3D(in_channels=4,latent_dim=4,downsample_factor=4)
vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEV)
seg=ImgSegUNet().to(DEV); seg.load_state_dict(torch.load(SEG,map_location=DEV,weights_only=False)["net"]); seg.eval()
for p in list(vq.parameters())+list(seg.parameters()): p.requires_grad_(False)
log("[sig] VQGAN+seg loaded OK")
ds=CachedPairDataset(CACHE)

def roi_lat(z_a,z_b):
    def tum(z):
        lab=seg(vq.decode(z)).argmax(1,keepdim=True); return sum((lab==c) for c in TUMOR)>0
    r=(tum(z_a)|tum(z_b)).float(); r=F.max_pool3d(r,2*DIL+1,1,DIL)>0.5
    return F.avg_pool3d(r.float(),4)>0.1
def rmse(a,b,m):
    mm=m.expand_as(a); return F.mse_loss(a[mm],b[mm]).item() if mm.sum()>0 else np.nan

rows=defaultdict(lambda: defaultdict(list))
t0=time.time()
for fold in range(K):
    log(f"[sig] fold {fold}: loading flow + ddpm checkpoints...")
    sf=torch.load(f"{BASE}/forecast_fold{fold}/best.pt",map_location=DEV,weights_only=False)
    flow=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=False).to(DEV)
    flow.load_state_dict(sf.get("ema",sf.get("net"))); flow.eval(); flow.set_week(None); Tf=float(sf.get("temperature",1.9))
    sd=torch.load(f"{BASE}/ddpm_fold{fold}/best.pt",map_location=DEV,weights_only=False)
    dn=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=False).to(DEV)
    dn.load_state_dict(sd.get("ema",sd.get("net"))); dn.eval(); dn.set_week(None)
    ddpm=DDPMTransport(dn,T=1000,device=DEV).to(DEV); Td=float(sd.get("temperature",1.4))
    log(f"[sig] fold {fold}: checkpoints loaded, generating...")
    _,val=list(patient_kfold(ds,k=K))[fold]
    with torch.no_grad():
        for n,i in enumerate(val):
            s=ds[i]; z_a=s["z_a"].unsqueeze(0).to(DEV); z_b=s["z_b"].unsqueeze(0).to(DEV); dt=s["dt"].view(1).to(DEV)
            roi=roi_lat(z_a,z_b)
            torch.manual_seed(i)
            sf_e=CAL.calibrate(generate(lambda z,t,za,d: flow(z,t,za,d),z_a,dt,steps=STEPS,n_samples=R),Tf)
            zf=sf_e.mean(0)
            torch.manual_seed(i)
            sd_e=CAL.calibrate(ddpm.ddim_sample(z_a,dt,steps=50,R=R),Td)
            zd=sd_e.mean(0)
            rows["region"]["MODEL"].append(rmse(zf,z_b,roi)); rows["region"]["COPY"].append(rmse(z_a,z_b,roi)); rows["region"]["DDPM"].append(rmse(zd,z_b,roi))
            rows["crps"]["MODEL"].append(M.ensemble_crps(sf_e,z_b)); rows["crps"]["DDPM"].append(M.ensemble_crps(sd_e,z_b))
            if n%10==0: log(f"    fold {fold} pair {n}/{len(val)}  ({time.time()-t0:.0f}s)")
            del sf_e,sd_e; torch.cuda.empty_cache()
    log(f"[sig] fold {fold} DONE")

def rb(x,y):
    d=np.array(x)-np.array(y); d=d[d!=0]
    if not len(d): return 0.0
    r=np.argsort(np.argsort(np.abs(d)))+1
    return float((r[d>0].sum()-r[d<0].sum())/r.sum())
def T(metric,a,b):
    xa=np.array(rows[metric][a],float); xb=np.array(rows[metric][b],float)
    m=~(np.isnan(xa)|np.isnan(xb)); xa,xb=xa[m],xb[m]
    _,p=wilcoxon(xa,xb); return len(xa),np.median(xa),np.median(xb),p,rb(xa,xb)

comps=[("region","MODEL","COPY"),("region","MODEL","DDPM"),("crps","MODEL","DDPM")]
log("\n==== PAIRED SIGNIFICANCE (Wilcoxon, Bonferroni k=%d) ===="%len(comps))
out={}
for metric,a,b in comps:
    n,ma,mb,p,r=T(metric,a,b); pc=min(p*len(comps),1.0)
    sig="***" if pc<1e-3 else "**" if pc<1e-2 else "*" if pc<0.05 else "ns"
    out[f"{metric}:{a}_vs_{b}"]={"n":n,"med_a":ma,"med_b":mb,"p":p,"p_corr":pc,"sig":sig,"r_rb":r}
    log(f"  {metric:>7} {a} vs {b:<5} n={n} med {ma:.4f} vs {mb:.4f} p={p:.2e} p_corr={pc:.2e} {sig} r_rb={r:+.3f}")
json.dump(out,open(f"{BASE}/significance_tests.json","w"),indent=2)
log(f"\n[saved] significance_tests.json")
