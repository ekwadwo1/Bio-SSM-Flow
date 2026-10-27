import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0,_P1)
from cached_pair_dataset import CachedPairDataset
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
from bio_coupling import BioRegularizer
DEV="cuda:0"
VQGAN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
vq=VQGAN3D(in_channels=4,latent_dim=4,downsample_factor=4)
vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEV)
seg=ImgSegUNet().to(DEV); seg.load_state_dict(torch.load(SEG,map_location=DEV,weights_only=False)["net"]); seg.eval()
for p in list(vq.parameters())+list(seg.parameters()): p.requires_grad_(False)
ds=CachedPairDataset("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere")
# sweep residual_scale to find the one giving L_bio ~ O(1) on real data
for rs in [3.6e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
    reg=BioRegularizer(vq,seg,DEV,residual_scale=rs,use_param_head=True)
    Ls=[]
    for i in range(8):
        s=ds[i]
        z_a=s["z_a"].unsqueeze(0).to(DEV); z_b=s["z_b"].unsqueeze(0).to(DEV)
        seg_a=s["seg_a"].unsqueeze(0).to(DEV); dt=s["dt"].view(1).to(DEV)
        L,log=reg(z_a, z_b, seg_a, dt, s["mask_b"].unsqueeze(0).to(DEV))  # use z_b as stand-in forecast
        Ls.append(float(L))
    import numpy as np
    print(f"residual_scale={rs:.1e} -> L_bio median={np.median(Ls):.3f}  (want ~1)")
