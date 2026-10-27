#!/usr/bin/env python3
"""
eval_region.py — tumor-REGION-restricted forecasting evaluation (fold 0).

WHY: whole-volume MSE/PSNR are dominated by unchanged brain, so "copy baseline"
(predict no change) looks deceptively strong and the model's real advantage is
hidden. The clinically meaningful question is: in the TUMOR REGION, does the
forecast beat copy? This evaluates forecasting metrics restricted to a tumor ROI.

Region definition (dilated union of tumor classes) comes from the seg head applied
to the BASELINE and FOLLOW-UP decoded volumes (union, then dilated) -> the ROI where
change actually happens. We report, IN-REGION and WHOLE-VOLUME, for:
  - our model (generative mean, T-calibrated ensemble)
  - copy baseline (z_b_hat = z_a)
in latent space (mse/mae/cos) and image space (psnr/ssim within ROI).

Also reports the KEY number: relative improvement of model over copy, IN-REGION.
This is the forecasting result that goes in the paper.

Run:  CUDA_VISIBLE_DEVICES=1 python eval_region.py
"""
from __future__ import annotations
import os, sys, glob, re
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_P1 = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt"
if _P1 not in sys.path: sys.path.insert(0, _P1)
from cached_pair_dataset import CachedPairDataset, patient_kfold
from flow import generate
from velocity_net_v2 import VelocityNetV2
from vqgan3d import VQGAN3D
from seg_head_img import ImgSegUNet
import calibration as CAL
import metrics as M

CACHE  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/latent_cache_lumiere"
RUN    = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/forecast_fold0/best.pt"
VQGAN  = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/u101prjt/data/UR_SSM_DIFF_DATASETS/UR_SSM_Diff_Outputs/checkpoints/vqgan_r4/best_vqgan.pt"
SEG    = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs/seghead_img_fold0/imgseg_best.pt"
DEVICE = "cuda:0"; FOLD, K = 0, 5
R = 32; GEN_STEPS = 20; N_VAL = 60; TUMOR_CLASSES = (1, 3); DILATE = 3


def load_all():
    st = torch.load(RUN, map_location=DEVICE, weights_only=False)
    net = VelocityNetV2(latent_dim=4, d_h=128, stage_depths=(2,2,4),
                        bottleneck_depth=4, use_checkpoint=False).to(DEVICE)
    net.load_state_dict(st.get("ema", st.get("net"))); net.eval(); net.set_week(None)
    T = float(st.get("temperature", 2.0))
    vq = VQGAN3D(in_channels=4, latent_dim=4, downsample_factor=4)
    ck = torch.load(VQGAN, map_location="cpu", weights_only=False)
    vq.load_state_dict(ck.get("gen", ck), strict=False); vq.eval().to(DEVICE)
    seg = ImgSegUNet().to(DEVICE)
    seg.load_state_dict(torch.load(SEG, map_location=DEVICE, weights_only=False)["net"]); seg.eval()
    for m in (net, vq, seg):
        for p in m.parameters(): p.requires_grad_(False)
    print(f"[region] loaded forecaster (T={T}), VQGAN, seg head")
    return net, vq, seg, T


def tumor_roi(seg_logits_a, seg_logits_b):
    """union of tumor-class predictions at a & b, dilated. -> bool [1,1,H,W,D]."""
    def tumor(logits):
        lab = logits.argmax(1, keepdim=True)
        return sum((lab == c) for c in TUMOR_CLASSES) > 0
    roi = (tumor(seg_logits_a) | tumor(seg_logits_b)).float()
    k = DILATE
    roi = F.max_pool3d(roi, kernel_size=2*k+1, stride=1, padding=k)   # dilation
    return roi > 0.5


def masked(a, b, m):
    """mse/mae/cos of a vs b within boolean mask m (broadcast over channels)."""
    mm = m.expand_as(a)
    if mm.sum() == 0: return None
    av, bv = a[mm], b[mm]
    mse = F.mse_loss(av, bv).item(); mae = (av-bv).abs().mean().item()
    cos = F.cosine_similarity(av.reshape(1,-1), bv.reshape(1,-1)).item()
    return mse, mae, cos


@torch.no_grad()
def main():
    ds = CachedPairDataset(CACHE)
    _, val_idx = list(patient_kfold(ds, k=K))[FOLD]; val_idx = val_idx[:N_VAL]
    net, vq, seg, T = load_all()
    def v_fn(z, tau, za, dt): return net(z, tau, za, dt)

    acc = defaultdict(list)
    for i in val_idx:
        s = ds[i]
        z_a = s["z_a"].unsqueeze(0).to(DEVICE); z_b = s["z_b"].unsqueeze(0).to(DEVICE)
        dt = s["dt"].view(1).to(DEVICE)
        samp = generate(v_fn, z_a, dt, steps=GEN_STEPS, n_samples=R)
        samp = CAL.calibrate(samp, T)
        z_pred = samp.mean(0)                      # model forecast (latent)
        z_copy = z_a                              # copy baseline

        # decode for image metrics + ROI
        x_b   = vq.decode(z_b); x_pred = vq.decode(z_pred); x_copy = vq.decode(z_copy)
        roi_img = tumor_roi(seg(vq.decode(z_a)), seg(x_b))     # [1,1,128^3]
        # latent-space ROI (downsample the image ROI to latent res)
        roi_lat = (F.avg_pool3d(roi_img.float(), 4) > 0.1)

        # ---- IN-REGION (latent) ----
        mp = masked(z_pred, z_b, roi_lat); mc = masked(z_copy, z_b, roi_lat)
        if mp and mc:
            acc["region_mse_model"].append(mp[0]); acc["region_mse_copy"].append(mc[0])
            acc["region_cos_model"].append(mp[2]); acc["region_cos_copy"].append(mc[2])
        # ---- WHOLE (latent) ----
        acc["whole_mse_model"].append(F.mse_loss(z_pred, z_b).item())
        acc["whole_mse_copy"].append(F.mse_loss(z_copy, z_b).item())
        # ---- IN-REGION (image PSNR within ROI) ----
        def psnr_roi(xp):
            e = ((xp - x_b)**2)[roi_img.expand_as(xp)]
            if e.numel()==0: return None
            mse = e.mean().item(); dr = float((x_b.max()-x_b.min()).clamp(min=1e-6))
            return 10*np.log10(dr**2/max(mse,1e-12))
        pm, pc = psnr_roi(x_pred), psnr_roi(x_copy)
        if pm and pc: acc["region_psnr_model"].append(pm); acc["region_psnr_copy"].append(pc)

    r = {k: float(np.mean(v)) for k, v in acc.items() if v}
    def imp(m, c): return 100*(1 - r[m]/r[c]) if c in r and r[c] else float('nan')
    print("\n" + "="*66)
    print("TUMOR-REGION-RESTRICTED FORECASTING (fold 0 val, model vs copy)")
    print("="*66)
    print(f"  WHOLE-VOLUME latent MSE : model {r.get('whole_mse_model',0):.4f}  "
          f"copy {r.get('whole_mse_copy',0):.4f}  -> improvement {imp('whole_mse_model','whole_mse_copy'):.1f}%")
    print(f"  IN-REGION   latent MSE : model {r.get('region_mse_model',0):.4f}  "
          f"copy {r.get('region_mse_copy',0):.4f}  -> improvement {imp('region_mse_model','region_mse_copy'):.1f}%")
    print(f"  IN-REGION   latent cos : model {r.get('region_cos_model',0):.4f}  "
          f"copy {r.get('region_cos_copy',0):.4f}")
    print(f"  IN-REGION   image PSNR : model {r.get('region_psnr_model',0):.2f}  "
          f"copy {r.get('region_psnr_copy',0):.2f}  dB")
    print("="*66)
    print("KEY: if IN-REGION improvement >> WHOLE-VOLUME improvement, the model's")
    print("     advantage is concentrated where it matters (the tumor). That's the")
    print("     forecasting result for the paper.")


if __name__ == "__main__":
    main()
