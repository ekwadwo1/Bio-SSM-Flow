#!/usr/bin/env python3
"""
00_verify_env.py — Environment & GPU sanity gate for Bio-SSM-Flow (Phase 0).

Run this FIRST, on the target machine, before writing/importing anything else.
It catches the three failures that otherwise surface mid-training:
  (1) mamba-ssm / causal-conv1d CUDA kernels not actually working;
  (2) picking a GPU that a co-tenant is already hammering (your box is shared);
  (3) bf16 unsupported or a package missing.

Jupyter-safe: no argparse. Just run it.
"""
from __future__ import annotations
import importlib, os, sys


def _hr(t=""): print(f"\n{'='*68}\n  {t}\n{'='*68}" if t else "-"*68)


def check_core():
    _hr("CORE")
    import torch
    print(f"  python           {sys.version.split()[0]}")
    print(f"  torch            {torch.__version__}")
    print(f"  cuda available   {torch.cuda.is_available()}")
    print(f"  torch cuda build {torch.version.cuda}")
    assert torch.cuda.is_available(), "CUDA not available — stop here."
    return torch


def pick_gpu(torch, min_free_gb=20.0):
    """Auto-select the EMPTIEST visible GPU. Warns if the best is contended."""
    _hr("GPU (shared-box aware)")
    n = torch.cuda.device_count()
    best, best_free = None, -1.0
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        free_gb, total_gb = free/1024**3, total/1024**3
        used_gb = total_gb - free_gb
        name = torch.cuda.get_device_name(i)
        flag = ""
        if used_gb > 2.0:
            flag = "  <-- CO-TENANT ACTIVE (someone else is using this card)"
        print(f"  cuda:{i}  {name[:28]:28s}  free {free_gb:5.1f} / {total_gb:4.0f} GB{flag}")
        if free_gb > best_free:
            best, best_free = i, free_gb
    print(f"\n  --> recommended device: cuda:{best}  ({best_free:.1f} GB free)")
    if best_free < min_free_gb:
        print(f"  !! WARNING: best GPU has < {min_free_gb:.0f} GB free. "
              f"Training may OOM against co-tenant jobs.")
    print(f"  Set it explicitly:  export CUDA_VISIBLE_DEVICES={best}")
    print(f"  DO NOT hard-code DDP on this machine — GPU 1 is contended.")
    return best


def check_mamba(torch, device):
    """The real test: build a Mamba block and run its CUDA kernel."""
    _hr("MAMBA-SSM KERNELS (the usual install trap)")
    try:
        import mamba_ssm, causal_conv1d
        print(f"  mamba_ssm        {getattr(mamba_ssm,'__version__','?')}")
        print(f"  causal_conv1d    {getattr(causal_conv1d,'__version__','?')}")
        from mamba_ssm import Mamba
        m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).to(f"cuda:{device}")
        x = torch.randn(1, 64, 16, device=f"cuda:{device}")
        y = m(x)
        assert y.shape == x.shape
        print(f"  selective-scan forward  {tuple(x.shape)} -> {tuple(y.shape)}  OK")
        print(f"  >> kernels compiled & running. This is the thing that usually breaks.")
    except Exception as e:
        print(f"  !! MAMBA CHECK FAILED: {type(e).__name__}: {e}")
        print(f"  Fix before anything else. Reuse your Paper-1 env if it worked there,")
        print(f"  or reinstall with:  pip install mamba-ssm[causal-conv1d] --no-build-isolation")
        raise


def check_amp(torch, device):
    _hr("MIXED PRECISION (SSMs are fp16-fragile)")
    ok = torch.cuda.is_bf16_supported()
    print(f"  bfloat16 supported  {ok}")
    print(f"  RULE: use AMP with fp32 master params (bf16 autocast). "
          f"Never store SSM params in fp16 — the recurrence destabilizes.")
    if not ok:
        print("  !! bf16 unsupported — fall back to fp32 (slower) rather than fp16.")


def check_packages():
    _hr("PROJECT PACKAGES")
    want = {
        "monai": "data/transforms/metrics",
        "nibabel": "NIfTI I/O",
        "pytorch_lightning": "orchestration (recommended)",
        "hydra": "config (import name: hydra)",
        "omegaconf": "config backend",
        "SimpleITK": "resampling / registration fallback",
        "ants": "ANTsPy registration (import name: ants; pip: antspyx)",
        "wandb": "logging (optional)",
    }
    for mod, why in want.items():
        try:
            importlib.import_module(mod)
            print(f"  [ok ] {mod:20s} {why}")
        except Exception:
            print(f"  [ -- ] {mod:20s} MISSING  ({why})")
    print("\n  Missing 'ants'/'SimpleITK' is fine IF the data is already co-registered")
    print("  (verify that with 01_inspect_data.py before installing anything).")


def _run():
    torch = check_core()
    dev = pick_gpu(torch)
    check_mamba(torch, dev)
    check_amp(torch, dev)
    check_packages()
    _hr("ENV GATE PASSED" )
    print(f"  Proceed to 01_inspect_data.py.  Use device cuda:{dev}.")


if __name__ == "__main__":
    _run()

