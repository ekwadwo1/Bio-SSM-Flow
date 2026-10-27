#!/usr/bin/env python3
"""
run_folds.py — run the FROZEN recipe across folds 1-4 (fold 0 already done).

Sequential (single shared GPU). Each fold -> its own run dir with val_log.jsonl,
best.pt (with fitted temperature), last.pt (resumable). Durable logging so a reboot
doesn't lose results. Skips a fold if its best.pt already exists (resume-friendly).

Run:  CUDA_VISIBLE_DEVICES=1 nohup python run_folds.py > /tmp/folds.log 2>&1 &
      tail -f /tmp/folds.log
"""
import os, sys, time, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FOLDS = [1, 2, 3, 4]
BASE_OUT = "/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs"

def run_fold(fold):
    out = f"{BASE_OUT}/forecast_fold{fold}"
    best = f"{out}/best.pt"
    if os.path.exists(best):
        print(f"[folds] fold {fold} already has best.pt -> skip"); return
    print(f"\n{'='*60}\n[folds] FOLD {fold} starting {time.strftime('%H:%M:%S')}\n{'='*60}")
    # import train fresh, override its CFG for this fold, call main()
    import train
    importlib.reload(train)
    train.CFG["fold"] = fold
    train.CFG["out_dir"] = out
    train.CFG["RESUME"] = f"{out}/last.pt" if os.path.exists(f"{out}/last.pt") else None
    t0 = time.time()
    train.main()
    print(f"[folds] fold {fold} done in {(time.time()-t0)/3600:.2f}h")

if __name__ == "__main__":
    print(f"[folds] frozen-recipe campaign, folds {FOLDS}, started {time.strftime('%Y-%m-%d %H:%M')}")
    for f in FOLDS:
        try:
            run_fold(f)
        except Exception as e:
            print(f"[folds] FOLD {f} FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            print(f"[folds] continuing to next fold...")
    print(f"\n[folds] campaign complete {time.strftime('%Y-%m-%d %H:%M')}")
    # summary: collect best CRPS per fold
    import json
    print("\n=== CAMPAIGN SUMMARY ===")
    for f in [0,1,2,3,4]:
        p = f"{BASE_OUT}/forecast_fold{f}/best.pt"
        if os.path.exists(p):
            import torch
            d = torch.load(p, map_location="cpu", weights_only=False)
            print(f"  fold {f}: step={d.get('step')} CRPS={d.get('best_crps'):.4f} T={d.get('temperature')}")
        else:
            print(f"  fold {f}: (no checkpoint)")
