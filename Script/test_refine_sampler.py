import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import CachedPairDataset
from flow import interpolate, sample_s
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
print(f"trained. final train mse {float(loss):.5f}")

net.eval()
base=F.mse_loss(z_a,z_b).item()
def cm(zh): return (F.cosine_similarity(zh.flatten(1),z_b.flatten(1)).mean().item(), F.mse_loss(zh.float(),z_b).item())

@torch.no_grad()
def refine(s_star, K):
    x1 = z_a.clone()
    for _ in range(K):
        s_vec = torch.full((N,), s_star, device=DEV)
        z_mid = (1-s_star)*z_a + s_star*x1
        x1, _ = net(z_mid, s_vec, z_a, dt)
    return x1

print(f"\nbaseline MSE {base:.5f}")
for s_star in [0.3, 0.5, 0.7]:
    for K in [1, 3, 5, 10]:
        c, m = cm(refine(s_star, K))
        flag = "  <== " if (c>0.9 and m<0.5*base) else ""
        print(f"  s*={s_star} K={K:2d} | cosine {c:.4f} MSE {m:.5f}{flag}")
print("\nIf any config clears cosine>0.9 & MSE<0.5*base, THAT is the inference method.")
