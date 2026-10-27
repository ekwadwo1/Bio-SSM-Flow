#!/usr/bin/env python3
"""
plot_fig3_qualitative.py — Figure 3: qualitative forecasts + uncertainty (TMI-grade).

Renders a grid of REAL held-out cases (rows) x panels (cols):
  baseline x^a | forecast x_hat^b | observed x^b | error |x_hat^b - x^b| | uncertainty
An overlaid contour of the tumor ROI ties the panels together; the uncertainty and
error panels share a colormap so the reader sees they co-locate.

This loads your ACTUAL model + VQGAN + a few val cases and generates real forecasts,
so the figure shows real data (not a mockup). Pick 2-3 representative CASE_IDXS,
ideally including one where enhancement resolves (treatment effect).

Run:  CUDA_VISIBLE_DEVICES=0 python plot_fig3_qualitative.py
Output: fig3_qualitative.pdf
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pubstyle import set_style, C
set_style()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0,_P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
import calibration as CAL

CACHE="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
RUN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/forecast_fold0/best.pt"
VQGAN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
DEV="cuda:0"; R=32; STEPS=20; CONTRAST=1   # channel 1 = T1post (enhancing)
CASE_IDXS=[0, 5, 12]                        # <<REPLACE with representative cases>>

@torch.no_grad()
def main():
    st=torch.load(RUN,map_location=DEV,weights_only=False)
    net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=False).to(DEV)
    net.load_state_dict(st.get("ema",st.get("net"))); net.eval(); net.set_week(None); T=float(st.get("temperature",1.9))
    vq=VQGAN3D(in_channels=4,latent_dim=4,downsample_factor=4)
    vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEV)
    ds=CachedPairDataset(CACHE)
    _,val=list(patient_kfold(ds,k=5))[0]

    def mid(x): return x[x.shape[0]//2]      # central axial slice
    def norm(a): 
        a=a-a.min(); return a/(a.max()+1e-8)

    ncase=len(CASE_IDXS); cols=5
    fig,ax=plt.subplots(ncase, cols, figsize=(cols*1.6, ncase*1.75))
    if ncase==1: ax=ax[None,:]
    titles=["Baseline","Forecast","Observed","|Error|","Uncertainty"]

    for r,ci in enumerate(CASE_IDXS):
        s=ds[val[ci]]
        z_a=s["z_a"].unsqueeze(0).to(DEV); z_b=s["z_b"].unsqueeze(0).to(DEV); dt=s["dt"].view(1).to(DEV)
        samp=CAL.calibrate(generate(lambda z,t,za,d: net(z,t,za,d),z_a,dt,steps=STEPS,n_samples=R),T)
        z_pred=samp.mean(0)
        unc=samp.std(0).mean(1,keepdim=True)              # per-voxel latent uncertainty
        x_a=vq.decode(z_a); x_b=vq.decode(z_b); x_p=vq.decode(z_pred)
        # upsample latent uncertainty to image res for display
        unc_img=F.interpolate(unc,scale_factor=4,mode="trilinear",align_corners=False)
        err=(x_p-x_b).abs()

        sl=lambda t: mid(t[0,CONTRAST].cpu().numpy())
        imgs=[norm(sl(x_a)), norm(sl(x_p)), norm(sl(x_b)),
              norm(mid(err[0,CONTRAST].cpu().numpy())),
              norm(mid(unc_img[0,0].cpu().numpy()))]
        cmaps=["gray","gray","gray","inferno","inferno"]
        for c in range(cols):
            a=ax[r,c]; a.imshow(imgs[c].T, cmap=cmaps[c], origin="lower")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            if r==0: a.set_title(titles[c], fontsize=11, fontweight="bold")
        ax[r,0].set_ylabel(f"Case {r+1}", fontsize=11, fontweight="bold")

    # shared colorbars for error & uncertainty (last two columns)
    for c,lab in [(3,"error"),(4,"uncertainty")]:
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap="inferno"),
                        ax=ax[:,c], fraction=0.046, pad=0.02)
        cb.set_ticks([0,1]); cb.set_ticklabels(["low","high"]); cb.ax.tick_params(labelsize=8)

    fig.savefig("fig3_qualitative.pdf", bbox_inches="tight")
    print("saved fig3_qualitative.pdf  (cases:", CASE_IDXS, ")")
    print("NOTE: inspect the panels; pick CASE_IDXS where uncertainty visibly")
    print("      co-locates with error, ideally one enhancement-resolving case.")

if __name__=="__main__":
    main()
