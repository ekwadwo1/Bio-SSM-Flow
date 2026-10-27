import os, sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0,_P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from torch.utils.data import DataLoader, Subset
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
from bio_coupling import BioRegularizer
DEV="cuda:0"; LAM=0.05; BIO_BATCH=2
VQGAN="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
CACHE="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"

ds=CachedPairDataset(CACHE); tr,_=list(patient_kfold(ds,k=5))[0]
def coll(b):
    return {k:(torch.stack([x[k] for x in b]) if torch.is_tensor(b[0][k]) else [x[k] for x in b]) for k in b[0]}
dl=DataLoader(Subset(ds,tr),batch_size=8,shuffle=True,collate_fn=coll,drop_last=True)

net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=True).to(DEV)
vq=VQGAN3D(in_channels=4,latent_dim=4,downsample_factor=4)
vq.load_state_dict(torch.load(VQGAN,map_location="cpu",weights_only=False).get("gen"),strict=False); vq.eval().to(DEV)
seg=ImgSegUNet().to(DEV); seg.load_state_dict(torch.load(SEG,map_location=DEV,weights_only=False)["net"]); seg.eval()
for p in list(vq.parameters())+list(seg.parameters()): p.requires_grad_(False)
reg=BioRegularizer(vq,seg,DEV,residual_scale=1e-3,use_param_head=True)
opt=torch.optim.AdamW(list(net.parameters())+reg.parameters(),lr=2e-4)
lf=HeteroscedasticFMLoss()

net.train(); step=0; vlist=[]
for b in dl:
    z_a=b["z_a"].to(DEV); z_b=b["z_b"].to(DEV); dt=b["dt"].to(DEV); net.set_week(None)
    tau=sample_tau(z_a.shape[0],DEV); eps=torch.randn_like(z_b)
    z_tau,u=noise_interpolate(z_b,tau,eps)
    v,lv=net(z_tau,tau,z_a,dt)
    loss,_,logs=lf(u,v.float(),lv.float())
    vel=logs["vel_mse"]; biostr=""
    if step%5==0:
        zi=z_a[:BIO_BATCH]; dt2=dt[:BIO_BATCH]
        sa=b["seg_a"][:BIO_BATCH].to(DEV); mk=b["mask_b"][:BIO_BATCH].to(DEV)
        e=torch.randn_like(zi); ta=torch.zeros(BIO_BATCH,device=DEV)
        vpred,_=net(e,ta,zi,dt2)              # single grad-carrying velocity eval
        zc=e+vpred                            # one-step forecast estimate
        Lb,bl=reg(zi,zc,sa,dt2,mk)            # ONE decode graph, batch 2
        loss=loss+LAM*Lb
        biostr=f" | L_bio={float(Lb):.3f} D={bl['D_med']:.4f} rho={bl['rho_med']:.4f}"
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    vlist.append(vel)
    if step%5==0:
        mem=torch.cuda.max_memory_allocated()/1024**3
        print(f"step {step:3d} | vel_mse {vel:.4f}{biostr} | peakVRAM {mem:.1f}GB")
    step+=1
    if step>=40: break
print(f"\nvel_mse: first5={np.mean(vlist[:5]):.4f} last5={np.mean(vlist[-5:]):.4f}")
print("PASS if L_bio~O(1), D/rho in [0.005,0.1], vel_mse stable, VRAM<20GB, no NaN")
