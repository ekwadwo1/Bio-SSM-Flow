#!/bin/bash
BASE=/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs
for F in 0 1 2 3 4; do
  OUT=$BASE/ddpm_fold$F
  if [ -f "$OUT/best.pt" ]; then echo "[$(date +%H:%M)] ddpm fold $F done -> skip"; continue; fi
  echo "[$(date +%H:%M)] ===== DDPM FOLD $F starting ====="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train_ddpm.py --fold $F --out_dir "$OUT" > /tmp/ddpm_fold$F.log 2>&1
  echo "[$(date +%H:%M)] ===== DDPM FOLD $F finished (exit $?) ====="
done
echo "[$(date +%H:%M)] DDPM CAMPAIGN COMPLETE"
