#!/usr/bin/env python3
"""
src/metrics.py — TMI-grade evaluation suite for conditional generative glioma forecasting.

Organized by the four categories a TMI reviewer expects for a longitudinal
generative + uncertainty paper. Each metric notes WHAT it measures and WHY it matters.

  A. FORECAST FIDELITY  (predicted follow-up vs observed follow-up)
       latent: MSE, MAE, cosine ; image (decoded, in-brain): PSNR, SSIM
  B. UNCERTAINTY CALIBRATION  (does predicted sigma match actual error?)
       Gaussian NLL ; regression-ECE ; AUSE (sparsification) ; err-vs-unc Spearman
  C. GENERATIVE / DISTRIBUTIONAL  (is the predicted DISTRIBUTION good?)
       ensemble CRPS (proper score) ; PICP coverage ; sample diversity
  D. SEGMENTATION + VOLUMETRIC  (needs a seg head on the forecast; enable when ready)
       Dice / HD95 / NSD per subregion ; tumor-volume error ; growth-direction acc

Uses MONAI for standard image/seg metrics; custom implementations for the
uncertainty and generative scores (which MONAI does not provide).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

try:
    from monai.metrics import SSIMMetric, PSNRMetric, DiceMetric, HausdorffDistanceMetric
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False


# ======================================================================
# A. FORECAST FIDELITY
# ======================================================================
def latent_fidelity(z_pred: torch.Tensor, z_true: torch.Tensor) -> Dict[str, float]:
    """z_pred,z_true: [B,c,h,w,d]. Point-forecast accuracy in latent space."""
    mse = F.mse_loss(z_pred, z_true).item()
    mae = (z_pred - z_true).abs().mean().item()
    cos = F.cosine_similarity(z_pred.flatten(1), z_true.flatten(1)).mean().item()
    return {"latent_mse": mse, "latent_mae": mae, "latent_cos": cos}


def image_fidelity(x_pred: torch.Tensor, x_true: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    x_pred,x_true: [B,C,H,W,D] decoded volumes. PSNR + SSIM within brain mask.
    These are the headline restoration-style numbers TMI expects.
    """
    if mask is not None:
        m = mask.unsqueeze(1).float()
        x_pred = x_pred * m
        x_true = x_true * m
    out = {}
    if _HAS_MONAI:
        dr = float((x_true.max() - x_true.min()).clamp(min=1e-6))
        psnr = PSNRMetric(max_val=dr)(x_pred, x_true).mean().item()
        ssim = SSIMMetric(spatial_dims=3, data_range=dr)(x_pred, x_true).mean().item()
        out.update({"psnr": psnr, "ssim": ssim})
    else:
        mse = F.mse_loss(x_pred, x_true).item()
        out["psnr"] = 10 * np.log10(1.0 / max(mse, 1e-12))
    return out


# ======================================================================
# B. UNCERTAINTY CALIBRATION
# ======================================================================
def gaussian_nll(err: torch.Tensor, sigma2: torch.Tensor) -> float:
    """Mean Gaussian NLL. err=(pred-true) per voxel; sigma2 predicted variance."""
    nll = 0.5 * (err.pow(2) / sigma2.clamp_min(1e-8) + torch.log(sigma2.clamp_min(1e-8))
                 + np.log(2 * np.pi))
    return nll.mean().item()


def regression_ece(err: torch.Tensor, sigma: torch.Tensor, n_bins: int = 15) -> float:
    """
    Regression ECE: bin voxels by predicted sigma; in each bin compare predicted
    sigma to observed RMSE. |predicted - observed| averaged over bins. 0 = perfect.
    """
    err = err.flatten(); sigma = sigma.flatten()
    order = torch.argsort(sigma)
    err, sigma = err[order], sigma[order]
    bins = torch.chunk(torch.arange(len(err)), n_bins)
    ece, tot = 0.0, len(err)
    for idx in bins:
        if len(idx) == 0:
            continue
        pred_sig = sigma[idx].mean().item()
        obs_rmse = err[idx].pow(2).mean().sqrt().item()
        ece += (len(idx) / tot) * abs(pred_sig - obs_rmse)
    return ece


def ause(err: torch.Tensor, sigma: torch.Tensor, n_frac: int = 20) -> float:
    """
    Area Under the Sparsification Error curve. Remove most-uncertain voxels
    progressively; a good uncertainty ranks errors so removing high-sigma voxels
    drops RMSE as fast as removing high-error voxels (oracle). AUSE = gap area. Lower better.
    """
    err = err.flatten().abs(); sigma = sigma.flatten()
    n = len(err)
    fr = torch.linspace(0, 0.9, n_frac)
    by_unc = err[torch.argsort(sigma, descending=True)]
    by_err = err[torch.argsort(err, descending=True)]
    def curve(sorted_err):
        out = []
        for f in fr:
            k = int(f * n)
            rem = sorted_err[k:]
            out.append(rem.pow(2).mean().sqrt().item() if len(rem) else 0.0)
        return np.array(out)
    spars = curve(by_unc); oracle = curve(by_err)
    denom = spars[0] if spars[0] > 0 else 1.0
    return float(np.trapz(np.abs(spars - oracle), fr.numpy()) / denom)


def err_unc_correlation(err: torch.Tensor, sigma: torch.Tensor,
                        n_sample: int = 20000) -> float:
    """Spearman corr between per-voxel |error| and predicted sigma. Higher = better."""
    e = err.flatten().abs(); s = sigma.flatten()
    if len(e) > n_sample:
        idx = torch.randperm(len(e))[:n_sample]
        e, s = e[idx], s[idx]
    def rank(x): return torch.argsort(torch.argsort(x)).float()
    re, rs = rank(e), rank(s)
    re, rs = re - re.mean(), rs - rs.mean()
    return (re @ rs / (re.norm() * rs.norm() + 1e-8)).item()


# ======================================================================
# C. GENERATIVE / DISTRIBUTIONAL  (samples: [R,B,c,h,w,d])
# ======================================================================
def ensemble_crps(samples: torch.Tensor, y: torch.Tensor) -> float:
    """
    Empirical CRPS (proper scoring rule for distributional forecasts):
        CRPS = E|X - y| - 0.5 E|X - X'|
    Averaged over voxels. Lower is better. THE key generative-forecast metric.
    """
    R = samples.shape[0]
    term1 = (samples - y.unsqueeze(0)).abs().mean(0)                 # E|X-y|
    # pairwise E|X-X'| (subsample pairs if R large)
    diffs = 0.0; cnt = 0
    for i in range(R):
        for j in range(i + 1, R):
            diffs = diffs + (samples[i] - samples[j]).abs(); cnt += 1
    term2 = (diffs / max(cnt, 1))
    crps = (term1 - 0.5 * term2).mean().item()
    return crps


def picp(samples: torch.Tensor, y: torch.Tensor, nominal: float = 0.9) -> float:
    """
    Prediction Interval Coverage Probability: fraction of true voxels within the
    central `nominal` interval of the ensemble. Should ~= nominal if calibrated.
    """
    lo = torch.quantile(samples, (1 - nominal) / 2, dim=0)
    hi = torch.quantile(samples, 1 - (1 - nominal) / 2, dim=0)
    inside = ((y >= lo) & (y <= hi)).float().mean().item()
    return inside


def sample_diversity(samples: torch.Tensor) -> float:
    """Mean per-voxel std across ensemble members. Nonzero => genuinely generative."""
    return samples.std(0).mean().item()


# ======================================================================
# D. SEGMENTATION + VOLUMETRIC  (enable once a seg head predicts on forecasts)
# ======================================================================
def segmentation_metrics(seg_pred: torch.Tensor, seg_true: torch.Tensor,
                         n_classes: int = 4, spacing=(1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    seg_pred, seg_true: [B,H,W,D] integer labels (unified: 0=bg,1=NETC,2=SNFH,3=ET).
    Dice + HD95 per foreground class. Needs MONAI.
    """
    if not _HAS_MONAI:
        return {}
    out = {}
    names = {1: "NETC", 2: "SNFH", 3: "ET"}
    def onehot(x): return F.one_hot(x.long(), n_classes).permute(0, 4, 1, 2, 3).float()
    p, t = onehot(seg_pred), onehot(seg_true)
    dice = DiceMetric(include_background=False, reduction="none")(p, t)          # [B,C-1]
    try:
        hd = HausdorffDistanceMetric(include_background=False, percentile=95,
                                     reduction="none")(p, t)
    except Exception:
        hd = None
    for ci, cls in enumerate(range(1, n_classes)):
        nm = names.get(cls, str(cls))
        out[f"dice_{nm}"] = float(dice[:, ci].nanmean())
        if hd is not None:
            out[f"hd95_{nm}"] = float(hd[:, ci].nanmean())
    return out


def volume_metrics(seg_pred: torch.Tensor, seg_true: torch.Tensor,
                   seg_baseline: torch.Tensor, tumor_classes=(1, 3)) -> Dict[str, float]:
    """
    Clinically meaningful: predicted vs true tumor VOLUME, and whether the model gets
    the GROWTH DIRECTION right (grow vs shrink relative to baseline).
    seg_*: [B,H,W,D]. tumor = union of enhancing+core by default.
    """
    def vol(seg): return sum((seg == c).float().sum((1, 2, 3)) for c in tumor_classes)
    vp, vt, vb = vol(seg_pred), vol(seg_true), vol(seg_baseline)
    rel_err = ((vp - vt).abs() / vt.clamp_min(1.0)).mean().item()
    dir_true = torch.sign(vt - vb); dir_pred = torch.sign(vp - vb)
    dir_acc = (dir_true == dir_pred).float().mean().item()
    return {"volume_rel_err": rel_err, "growth_dir_acc": dir_acc}


# ======================================================================
# Convenience: full uncertainty block from (pred, true, sigma2)
# ======================================================================
def uncertainty_block(z_pred: torch.Tensor, z_true: torch.Tensor,
                      sigma2: torch.Tensor) -> Dict[str, float]:
    """sigma2: [B,N] per-token variance; broadcast to per-voxel error via channel-mean."""
    B, c = z_pred.shape[:2]
    err = (z_pred - z_true).pow(2).reshape(B, c, -1).mean(1).sqrt()   # [B,N] per-token RMSE
    sigma = sigma2.clamp_min(1e-8).sqrt()
    return {
        "nll": gaussian_nll(err, sigma2),
        "reg_ece": regression_ece(err, sigma),
        "ause": ause(err, sigma),
        "err_unc_corr": err_unc_correlation(err, sigma),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, c, h = 2, 4, 8; N = h * h * h
    z_true = torch.randn(B, c, h, h, h)
    z_pred = z_true + 0.1 * torch.randn_like(z_true)
    sig2 = torch.rand(B, N) * 0.05 + 0.01
    print("latent:", latent_fidelity(z_pred, z_true))
    print("uncert:", uncertainty_block(z_pred, z_true, sig2))
    samples = z_true.unsqueeze(0) + 0.1 * torch.randn(6, B, c, h, h, h)
    print("crps:", round(ensemble_crps(samples, z_true), 4),
          "picp90:", round(picp(samples, z_true, 0.9), 3),
          "diversity:", round(sample_diversity(samples), 4))
    print("metrics.py self-test passed.")
