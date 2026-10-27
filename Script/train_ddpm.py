#!/usr/bin/env python3
"""
train_ddpm.py — train the DIFFUSION-transport ablation on the 5 folds.

Identical to the flow recipe EXCEPT the transport is DDPM (train_ddpm uses
DDPMTransport; flow used noise_interpolate + velocity regression). Same
VelocityNetV2 backbone, same VQGAN, same data/splits, same heteroscedastic loss
structure, same variance-warmup, EMA, LR schedule, and post-hoc temperature.
This is the controlled flow-vs-diffusion comparison.

Run one fold:  CUDA_VISIBLE_DEVICES=0 python train_ddpm.py --fold 1 --out_dir runs/ddpm_fold1
Loop all folds via a bash loop (like run_campaign.sh), pointing at runs/ddpm_fold{k}.
"""
from __future__ import annotations
import os, sys, math, copy, json, time, argparse
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import HeteroscedasticFMLoss            # reuse identical loss structure
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from ddpm_transport import DDPMTransport
import calibration as CAL
import metrics as M

CFG = dict(
    cache="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere",
    vqgan="/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt",
    device="cuda:0", fold=0, k_folds=5, batch=16, epochs=300,
    lr=2e-4, warmup=500, wd=1e-4, var_warmup=2000, ema=0.999, clip=1.0,
    T_diffusion=1000, ddim_steps=50, gen_R=8, val_every=1000, ckpt_every=1000,
    out_dir="runs/ddpm_fold0",
)

def lr_at(step, total):
    if step < CFG["warmup"]: return CFG["lr"]*step/max(1,CFG["warmup"])
    p=(step-CFG["warmup"])/max(1,total-CFG["warmup"]); return CFG["lr"]*0.5*(1+math.cos(math.pi*min(p,1.0)))

class EMA:
    def __init__(s,m,d): s.d=d; s.sh=copy.deepcopy(m.state_dict())
    @torch.no_grad()
    def upd(s,m):
        for k,v in m.state_dict().items():
            s.sh[k].mul_(s.d).add_(v,alpha=1-s.d) if v.dtype.is_floating_point else s.sh[k].copy_(v)
    def to(s,m): m.load_state_dict(s.sh,strict=True)

def collate(b):
    return {k:(torch.stack([x[k] for x in b]) if torch.is_tensor(b[0][k]) else [x[k] for x in b]) for k in b[0]}

@torch.no_grad()
def validate(ddpm, net, ema, loader, cfg, device):
    bk=copy.deepcopy(net.state_dict()); ema.to(net); net.eval()
    agg={}; per_s=[]; per_y=[]
    for b in loader:
        z_a=b["z_a"].to(device); z_b=b["z_b"].to(device); dt=b["dt"].to(device); net.set_week(None)
        samp=ddpm.ddim_sample(z_a, dt, steps=cfg["ddim_steps"], R=cfg["gen_R"])
        mean=samp.mean(0)
        for k,v in M.latent_fidelity(mean,z_b).items(): agg.setdefault(k,[]).append(v)
        agg.setdefault("crps",[]).append(M.ensemble_crps(samp,z_b))
        agg.setdefault("picp90_raw",[]).append(M.picp(samp,z_b,0.9))
        for j in range(z_b.shape[0]): per_s.append(samp[:,j:j+1]); per_y.append(z_b[j:j+1])
    net.load_state_dict(bk)
    T=CAL.fit_temperature(per_s,per_y,verbose=False)
    picp_cal=float(sum(CAL._picp(CAL.calibrate(s,T),y,0.9) for s,y in zip(per_s,per_y))/max(len(per_y),1))
    out={k:float(sum(v)/len(v)) for k,v in agg.items()}
    out.update({"temperature":float(T),"picp90_cal":picp_cal,"cov_err_cal":CAL.coverage_error(per_s,per_y,T)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fold",type=int,default=None); ap.add_argument("--out_dir",type=str,default=None)
    a,_=ap.parse_known_args()
    if a.fold is not None: CFG["fold"]=a.fold
    if a.out_dir is not None: CFG["out_dir"]=a.out_dir
    cfg=CFG; os.makedirs(cfg["out_dir"],exist_ok=True); dev=cfg["device"]; torch.manual_seed(0)

    ds=CachedPairDataset(cfg["cache"]); tr_i,va_i=list(patient_kfold(ds,k=cfg["k_folds"]))[cfg["fold"]]
    tr=DataLoader(Subset(ds,tr_i),cfg["batch"],shuffle=True,num_workers=4,collate_fn=collate,drop_last=True)
    va=DataLoader(Subset(ds,va_i),cfg["batch"],shuffle=False,num_workers=2,collate_fn=collate)
    print(f"[ddpm] fold {cfg['fold']}: train={len(tr_i)} val={len(va_i)}")

    net=VelocityNetV2(latent_dim=4,d_h=128,stage_depths=(2,2,4),bottleneck_depth=4,use_checkpoint=True).to(dev)
    ddpm=DDPMTransport(net,T=cfg["T_diffusion"],device=dev).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=cfg["wd"]); ema=EMA(net,cfg["ema"])
    loss_fn=HeteroscedasticFMLoss()
    total=cfg["epochs"]*len(tr); step=0; best=1e9; t0=time.time()
    print(f"[ddpm] total_steps={total} (diffusion transport)")
    net.train()
    for ep in range(cfg["epochs"]):
        for b in tr:
            z_a=b["z_a"].to(dev); z_b=b["z_b"].to(dev); dt=b["dt"].to(dev); net.set_week(None)
            for g in opt.param_groups: g["lr"]=lr_at(step,total)
            with torch.amp.autocast(dev,dtype=torch.bfloat16):
                loss,logs=ddpm.training_loss(z_b,z_a,dt,loss_fn,
                                             warmup=(step<cfg['var_warmup']))
            # variance warmup: before var_warmup, drop the NLL log-var term to pure eps-MSE
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),cfg["clip"]); opt.step(); ema.upd(net); step+=1
            if step%100==0:
                print(f"  step {step}/{total} | loss {float(loss):.4f} | {(time.time()-t0)/step*100:.1f}s/100st")
            if step%cfg["val_every"]==0:
                vm=validate(ddpm,net,ema,va,cfg,dev)
                print(f"  [val {step}] "+" ".join(f"{k}={v:.4f}" for k,v in vm.items()))
                open(os.path.join(cfg["out_dir"],"val_log.jsonl"),"a").write(json.dumps({"step":step,**vm})+"\n")
                if vm.get("crps",9e9)<best:
                    best=vm["crps"]; torch.save({"net":net.state_dict(),"ema":ema.sh,"step":step,"best_crps":best,"temperature":vm["temperature"],"cfg":cfg},os.path.join(cfg["out_dir"],"best.pt"))
                    print(f"  [val] new best CRPS {best:.4f}")
                net.train()
            if step%cfg["ckpt_every"]==0:
                torch.save({"net":net.state_dict(),"opt":opt.state_dict(),"ema":ema.sh,"step":step,"best_crps":best,"cfg":cfg},os.path.join(cfg["out_dir"],"last.pt"))
    print(f"[ddpm] done in {(time.time()-t0)/3600:.2f}h")

if __name__=="__main__":
    main()
