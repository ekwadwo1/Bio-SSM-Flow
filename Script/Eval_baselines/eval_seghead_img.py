import os, sys, glob, re
from collections import defaultdict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cached_pair_dataset import patient_kfold, CachedPairDataset
from seg_head_img import ImgSegUNet
from vqgan3d import VQGAN3D
import metrics as M

CACHE  = "/.../Bio-SSM-Flow/latent_cache_lumiere"
VQGAN  = "/.../checkpoints/vqgan_r4/best_vqgan.pt"
CKPT   = "/.../Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
DEVICE = "cuda:0"; FOLD, K = 0, 5

# same fold-0 holdout patients as training
fds = CachedPairDataset(CACHE)
_, vp = list(patient_kfold(fds, k=K))[FOLD]
holdout = {fds.patient_of(i) for i in vp}
files = sorted(glob.glob(os.path.join(CACHE, "*.pt")))
val_files = [f for f in files if re.search(r"(Patient-\d+)_week", os.path.basename(f)).group(1) in holdout]
print(f"[eval] {len(val_files)} val timepoints ({len(holdout)} holdout patients)")

vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
ck = torch.load(VQGAN, map_location="cpu", weights_only=False)
vq.load_state_dict(ck.get("gen", ck), strict=False); vq.eval().to(DEVICE)
for p in vq.parameters(): p.requires_grad_(False)

net = ImgSegUNet().to(DEVICE)
st = torch.load(CKPT, map_location=DEVICE, weights_only=False)
net.load_state_dict(st["net"]); net.eval()
print(f"[eval] loaded imgseg_best.pt (saved epoch {st.get('epoch')}, meanDice {st.get('meanDice'):.3f})")

acc = defaultdict(list)
with torch.no_grad():
    for f in val_files:
        d = torch.load(f, weights_only=False)
        z = d["z"].float().unsqueeze(0).to(DEVICE)
        seg = d["seg"].long().unsqueeze(0).to(DEVICE)
        img = vq.decode(z)
        pred = net(img).argmax(1)
        for k, v in M.segmentation_metrics(pred, seg, 4).items():
            acc[k].append(v)
res = {k: float(np.nanmean(v)) for k, v in acc.items()}
print("\n=== IMAGE-SPACE SEG HEAD — per-class results (fold 0 val) ===")
for k in sorted(res): print(f"  {k:12s} = {res[k]:.4f}")
dm = np.mean([v for k,v in res.items() if k.startswith('dice')])
print(f"  {'meanDice':12s} = {dm:.4f}")
