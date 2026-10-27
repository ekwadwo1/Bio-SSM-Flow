#!/bin/bash
# frozen-recipe campaign: folds 1-4, sequential, each to its own dir + log
BASE=/media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow/runs
for F in 1 2 3 4; do
  OUT=$BASE/forecast_fold$F
  if [ -f "$OUT/best.pt" ]; then
    echo "[$(date +%H:%M)] fold $F already done -> skip"; continue
  fi
  echo "[$(date +%H:%M)] ===== FOLD $F starting ====="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python train.py --fold $F --out_dir "$OUT" > /tmp/fold$F.log 2>&1
  echo "[$(date +%H:%M)] ===== FOLD $F finished (exit $?) ====="
done
echo "[$(date +%H:%M)] CAMPAIGN COMPLETE"
# summary
for F in 0 1 2 3 4; do
  P=$BASE/forecast_fold$F/best.pt
  [ -f "$P" ] && python3 -c "import torch;d=torch.load('$P',map_location='cpu',weights_only=False);print(f'fold $F: CRPS {d[\"best_crps\"]:.4f} step {d[\"step\"]}')"
done
