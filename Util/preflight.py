import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0,_P1)
from cached_pair_dataset import CachedPairDataset
from flow import sample_tau, noise_interpolate, HeteroscedasticFMLoss
from velocity_net_v2 import VelocityNetV2
import torch.nn.functional as F

DEV="cuda:0"  # physical GPU 1 via CUDA_VISIBLE_DEVICES=1
ds=CachedPairDataset("/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere")
B=4; b=[ds[i] for i in range(B)]
z_a=torch.stack([x["z_a"] for x in b]).to(DEV)
z_b=torch.stack([x["z_b"] for x in b]).to(DEV)
dt =torch.stack([x["dt"] for x in b]).to(DEV)

torch.cuda.reset_peak_memory_stats()
net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=True).to(DEV)
net.set_week(None)
opt=torch.optim.AdamW(net.parameters(),lr=2e-4)
print("params:", sum(p.numel() for p in net.parameters() if p.requires_grad))
loss_fn=HeteroscedasticFMLoss()
net.train(); t0=time.time()
for i in range(3):
    tau=sample_tau(B,DEV); eps=torch.randn_like(z_b)
    z_tau,u=noise_interpolate(z_b,tau,eps)
    with torch.amp.autocast(DEV,dtype=torch.bfloat16):
        v,logvar=net(z_tau,tau,z_a,dt)
    loss=F.mse_loss(v.float(),u.float())
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    print(f"  step {i+1} loss {float(loss):.4f}")
torch.cuda.synchronize()
print(f"3 full-depth steps (B=4): {time.time()-t0:.1f}s total, {(time.time()-t0)/3:.2f}s/step")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
