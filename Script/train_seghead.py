#!/usr/bin/env python3
"""
train_seghead.py — train the latent->segmentation head on cached (latent, seg) pairs.

Trains SegHead on ALL cached LUMIERE timepoints (real encoded latents + harmonized
DeepBraTumIA segs). Patient-level split (same fold-0 patients held out as the
forecaster's val, to avoid cross-task leakage). Silver-label aware. Reports Dice/HD95
per subregion via metrics.py. Fast (~20 min).

Run:  CUDA_VISIBLE_DEVICES=1 python train_seghead.py
"""
from __future__ import annotations
import os, sys, glob, re, time, copy
from collections import defaultdict
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import patient_kfold, CachedPairDataset
from seg_head import SegHead, DiceCELoss
import metrics as M

CACHE  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
OUT    = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_fold0"
DEVICE = "cuda:0"
EPOCHS = 60; BATCH = 8; LR = 3e-4; FOLD, K = 0, 5


class LatentSegDS(Dataset):
    """All cached timepoints as (z, seg) with patient id for splitting."""
    def __init__(self, cache):
        self.files = sorted(glob.glob(os.path.join(cache, "*.pt")))
        self.pat = [re.search(r"(Patient-\d+)_week", os.path.basename(f)).group(1) for f in self.files]
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = torch.load(self.files[i], weights_only=False)
        return {"z": d["z"].float(), "seg": d["seg"].long(), "patient": self.pat[i],
                "quality": 0.5}   # silver (DeepBraTumIA) -> down-weight

def split_by_patient(ds, holdout_patients):
    tr = [i for i in range(len(ds)) if ds.pat[i] not in holdout_patients]
    va = [i for i in range(len(ds)) if ds.pat[i] in holdout_patients]
    return tr, va

def collate(b):
    return {"z": torch.stack([x["z"] for x in b]),
            "seg": torch.stack([x["seg"] for x in b]),
            "quality": torch.tensor([x["quality"] for x in b])}

@torch.no_grad()
def evaluate(net, loader, device):
    net.eval(); dices = defaultdict(list); hds = defaultdict(list)
    for b in loader:
        z = b["z"].to(device); seg = b["seg"].to(device)
        pred = net(z).argmax(1)
        m = M.segmentation_metrics(pred, seg, n_classes=4)
        for k, v in m.items():
            (dices if k.startswith("dice") else hds)[k].append(v)
    net.train()
    out = {k: float(np.nanmean(v)) for k, v in {**dices, **hds}.items()}
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    ds = LatentSegDS(CACHE)
    # hold out the SAME patients as the forecaster's fold-0 val (no cross-task leak)
    fds = CachedPairDataset(CACHE)
    _, val_pair_idx = list(patient_kfold(fds, k=K))[FOLD]
    holdout = {fds.patient_of(i) for i in val_pair_idx}
    tr_idx, va_idx = split_by_patient(ds, holdout)
    tr = DataLoader(Subset(ds, tr_idx), BATCH, shuffle=True, num_workers=4, collate_fn=collate, drop_last=True)
    va = DataLoader(Subset(ds, va_idx), BATCH, shuffle=False, num_workers=2, collate_fn=collate)
    print(f"[seg] timepoints: train={len(tr_idx)} val={len(va_idx)} (holdout {len(holdout)} patients)")

    net = SegHead().to(DEVICE); loss_fn = DiceCELoss().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    best = -1; t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        for b in tr:
            z = b["z"].to(DEVICE); seg = b["seg"].to(DEVICE); w = b["quality"].to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
                logits = net(z)
            L, _ = loss_fn(logits.float(), seg, w)
            opt.zero_grad(set_to_none=True); L.backward(); opt.step()
        if (ep+1) % 5 == 0 or ep == EPOCHS-1:
            m = evaluate(net, va, DEVICE)
            md = np.mean([v for k, v in m.items() if k.startswith("dice")])
            print(f"  ep {ep+1:3d} | loss {float(L):.4f} | " +
                  " ".join(f"{k}={v:.3f}" for k, v in m.items()) + f" | meanDice {md:.3f}")
            if md > best:
                best = md; torch.save({"net": net.state_dict(), "epoch": ep, "meanDice": md}, os.path.join(OUT, "seghead_best.pt"))
    print(f"[seg] done in {(time.time()-t0)/60:.1f}min | best meanDice {best:.3f} -> seghead_best.pt")

if __name__ == "__main__":
    main()
