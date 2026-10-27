#!/usr/bin/env python3
"""
train_seghead_img.py — train the IMAGE-space seg head on DECODED cached latents.

decode(frozen VQGAN)(z) -> 128^3 image -> ImgSegUNet -> seg. Same patient split as
the forecaster fold 0. Dice+CE, silver-aware. Decoding on the fly (VQGAN frozen).
Heavier than latent-space (full-res UNet) but should recover thin ET/NETC.

Run:  CUDA_VISIBLE_DEVICES=1 python train_seghead_img.py
"""
from __future__ import annotations
import os, sys, glob, re, time
from collections import defaultdict
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import patient_kfold, CachedPairDataset
from seg_head_img import ImgSegUNet
from seg_head import DiceCELoss
from vqgan3d import VQGAN3D
import metrics as M

CACHE  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
VQGAN  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
OUT    = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0"
DEVICE = "cuda:0"
EPOCHS = 60; BATCH = 4; LR = 3e-4; FOLD, K = 0, 5


class LatentSegDS(Dataset):
    def __init__(self, cache):
        self.files = sorted(glob.glob(os.path.join(cache, "*.pt")))
        self.pat = [re.search(r"(Patient-\d+)_week", os.path.basename(f)).group(1) for f in self.files]
    def __len__(s): return len(s.files)
    def __getitem__(s, i):
        d = torch.load(s.files[i], weights_only=False)
        return {"z": d["z"].float(), "seg": d["seg"].long(), "q": 0.5}

def collate(b):
    return {"z": torch.stack([x["z"] for x in b]), "seg": torch.stack([x["seg"] for x in b]),
            "q": torch.tensor([x["q"] for x in b])}

def load_vqgan():
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck = torch.load(VQGAN, map_location="cpu", weights_only=False)
    vq.load_state_dict(ck.get("gen", ck), strict=False); vq.eval().to(DEVICE)
    for p in vq.parameters(): p.requires_grad_(False)
    return vq

@torch.no_grad()
def evaluate(net, vq, loader):
    net.eval(); acc = defaultdict(list)
    for b in loader:
        img = vq.decode(b["z"].to(DEVICE))
        pred = net(img).argmax(1)
        for k, v in M.segmentation_metrics(pred, b["seg"].to(DEVICE), 4).items():
            acc[k].append(v)
    net.train()
    return {k: float(np.nanmean(v)) for k, v in acc.items()}

def main():
    os.makedirs(OUT, exist_ok=True)
    ds = LatentSegDS(CACHE)
    fds = CachedPairDataset(CACHE)
    _, vp = list(patient_kfold(fds, k=K))[FOLD]
    holdout = {fds.patient_of(i) for i in vp}
    tr_i = [i for i in range(len(ds)) if ds.pat[i] not in holdout]
    va_i = [i for i in range(len(ds)) if ds.pat[i] in holdout]
    tr = DataLoader(Subset(ds, tr_i), BATCH, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    va = DataLoader(Subset(ds, va_i), BATCH, shuffle=False, num_workers=2, collate_fn=collate)
    print(f"[imgseg] train={len(tr_i)} val={len(va_i)} (holdout {len(holdout)} patients)")

    vq = load_vqgan()
    net = ImgSegUNet().to(DEVICE); loss_fn = DiceCELoss().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    print(f"[imgseg] params: {sum(p.numel() for p in net.parameters()):,}")
    best = -1; t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        for b in tr:
            with torch.no_grad():
                img = vq.decode(b["z"].to(DEVICE))       # frozen decode
            with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
                logits = net(img)
            L, _ = loss_fn(logits.float(), b["seg"].to(DEVICE), b["q"].to(DEVICE))
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
        if (ep+1) % 5 == 0 or ep == EPOCHS-1:
            m = evaluate(net, vq, va)
            md = np.mean([v for k, v in m.items() if k.startswith("dice")])
            print(f"  ep {ep+1:3d} | loss {float(L):.4f} | " +
                  " ".join(f"{k}={v:.3f}" for k, v in m.items() if k.startswith('dice')) +
                  f" | meanDice {md:.3f}")
            if md > best:
                best = md; torch.save({"net": net.state_dict(), "epoch": ep, "meanDice": md}, os.path.join(OUT, "imgseg_best.pt"))
    print(f"[imgseg] done in {(time.time()-t0)/60:.1f}min | best meanDice {best:.3f} -> imgseg_best.pt")

if __name__ == "__main__":
    main()
