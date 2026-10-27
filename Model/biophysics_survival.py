#!/usr/bin/env python3
"""
biophysics_survival.py — does forecast-derived D/rho stratify SURVIVAL?

The make-or-break test for Pillar 2. Re-fits (D,rho) per PATIENT from forecasts
(averaging over that patient's pairs), joins to LUMIERE overall survival, and tests:
  - Spearman correlation of D/rho, invasion speed, D, rho vs overall survival
  - median-split KM-style: median survival in high vs low D/rho groups
If invasion speed or D/rho correlates with survival (even |rho_s|>0.2, p<0.1 on this
small n), biophysics becomes a prognostic contribution. If flat, report as null.

Edit CLIN path if the survival CSV differs. Run: python biophysics_survival.py
"""
import os, sys, glob, re
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0,_P1)
from cached_pair_dataset import CachedPairDataset
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
from density import from_labels, from_probs
from physics_operators import laplacian3d
import calibration as CAL

CACHE="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
RUN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/forecast_fold0/best.pt"
VQGAN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
# EDIT if needed — find it: find .../LUMIERE* -iname "*Demograph*"
CLIN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/LUMIERE_Imaging/LUMIERE-Demographics_Pathology.csv"
DEV="cuda:0"; H=1.42; T_CAL=2.0; R=16

def smooth(c,r=2):
    w=torch.ones(1,1,3,3,3,device=c.device)/27
    for _ in range(r): c=F.conv3d(F.pad(c,(1,)*6,mode='replicate'),w)
    return c
def fit(c_a,c_b,dt):
    cb=0.5*(c_a+c_b); dcdt=(c_b-c_a)/max(dt,1.); lap=laplacian3d(cb,H); rc=cb*(1-cb)
    m=(cb[0,0]>0.03)
    if m.sum()<50: return None
    y=dcdt[0,0][m].reshape(-1); A=torch.stack([lap[0,0][m].reshape(-1),rc[0,0][m].reshape(-1)],1)
    sol,*_=torch.linalg.lstsq(A,y.unsqueeze(1)); return float(sol[0]),float(sol[1])

@torch.no_grad()
def main():
    # --- load survival CSV ---
    if not os.path.exists(CLIN):
        print(f"[surv] clinical CSV not found at {CLIN}")
        print("       run:  find /media/image522/8TPan1 -iname '*Demograph*' -o -iname '*LUMIERE*.csv'")
        return
    import pandas as pd
    df=pd.read_csv(CLIN)
    print("[surv] clinical columns:", list(df.columns))
    # heuristics for id + survival columns; adjust after seeing the printout
    idcol=[c for c in df.columns if re.search(r'patient|subject|id',c,re.I)]
    survcol=[c for c in df.columns if re.search(r'surviv|os|overall|death|days|months',c,re.I)]
    print("[surv] guessed id col:", idcol, "| survival col candidates:", survcol)
    if not idcol or not survcol:
        print("[surv] could not auto-detect columns — tell me the id + survival column names.")
        return

    # --- fit per-patient (D,rho) from forecasts ---
    ds=CachedPairDataset(CACHE)
    st=torch.load(RUN,map_location=DEV,weights_only=False)
    net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=False).to(DEV)
    net.load_state_dict(st.get("ema",st.get("net"))); net.eval(); net.set_week(None)
    vq=VQGAN3D(in_channels=4,latent_dim=4,downsample_factor=4)
    vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEV)
    seg=ImgSegUNet().to(DEV); seg.load_state_dict(torch.load(SEG,map_location=DEV,weights_only=False)["net"]); seg.eval()
    def vfn(z,t,za,dt): return net(z,t,za,dt)

    perpat=defaultdict(list)
    for i in range(len(ds)):
        s=ds[i]; z_a=s["z_a"].unsqueeze(0).to(DEV); dt=s["dt"].view(1).to(DEV)
        samp=CAL.calibrate(generate(vfn,z_a,dt,steps=20,n_samples=R),T_CAL); zp=samp.mean(0)
        c_a=smooth(from_labels(s["seg_a"]).to(DEV))
        c_b=smooth(from_probs(F.softmax(seg(vq.decode(zp)),1)).to(DEV))
        o=fit(c_a,c_b,float(s["dt"]))
        if o and 0<o[0]<5 and 0<o[1]<1:
            perpat[s["patient"]].append(o)
    # median (D,rho) per patient -> D/rho, invasion
    rows=[]
    for p,vals in perpat.items():
        D=np.median([v[0] for v in vals]); rho=np.median([v[1] for v in vals])
        rows.append((p, D, rho, D/max(rho,1e-6), 2*np.sqrt(max(rho,0)*max(D,0))))
    print(f"[surv] fitted params for {len(rows)} patients")

    # --- join to survival ---
    # normalize patient ids (LUMIERE 'Patient-063' vs CSV format may differ)
    def norm(x): 
        m=re.search(r'(\d+)',str(x)); return m.group(1).lstrip('0') if m else str(x)
    surv_map={}
    ic, sc = idcol[0], survcol[0]
    for _,r in df.iterrows():
        try: surv_map[norm(r[ic])]=float(r[sc])
        except: pass
    joined=[(D,rho,dr,inv,surv_map[norm(p)]) for (p,D,rho,dr,inv) in rows if norm(p) in surv_map]
    print(f"[surv] joined {len(joined)} patients to survival (col='{sc}')")
    if len(joined)<8:
        print("[surv] too few joined — id formats likely differ. Print a few:")
        print("  fitted ids:", [norm(p) for p,*_ in rows[:5]])
        print("  csv ids:", list(surv_map.keys())[:5])
        return
    J=np.array(joined)  # cols: D,rho,D/rho,invasion,survival
    from scipy.stats import spearmanr
    names=["D","rho","D/rho","invasion_speed"]
    print("\n=== SURVIVAL CORRELATIONS (Spearman) ===")
    for k in range(4):
        rho_s,p=spearmanr(J[:,k],J[:,4])
        flag="  <-- signal" if abs(rho_s)>0.2 and p<0.15 else ""
        print(f"  {names[k]:16s} vs survival: rho={rho_s:+.3f} p={p:.3f}{flag}")
    # median split on D/rho
    med=np.median(J[:,2]); hi=J[J[:,2]>med,4]; lo=J[J[:,2]<=med,4]
    print(f"\n  median survival: high D/rho={np.median(hi):.1f}  low D/rho={np.median(lo):.1f}")
    print("\nREAD: any |rho|>0.2 with p<0.15 on this small n = a prognostic signal worth")
    print("      reporting. All flat = report as null (params plausible but not prognostic).")

if __name__=="__main__":
    main()
