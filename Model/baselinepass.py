1. Smoke test first (about 5 minutes)

cd /media/image522/8TPan1/image522/Ernest_Tri-Mamba-Net/NBPY-FILES/Bio-SSM-Flow
CUDA_VISIBLE_DEVICES=0 python baseline_probreg.py train --fold 0 --max_steps 200 --val_every 100
rm -rf runs/baseline_probreg/fold0          # discard the smoke-test checkpoint


CUDA_VISIBLE_DEVICES=0 python baseline_probreg.py train --fold 0 --max_steps 200 --val_every 100
rm -rf runs/baseline_probreg/fold0


CUDA_VISIBLE_DEVICES=0 python baseline_probreg.py train --fold 0 --max_steps 200 --val_every 100
rm -rf runs/baseline_probreg/fold0


grep -n "class HeteroscedasticFMLoss" -A 25 flow.py


2. Train all five folds (≈ 9–10 h on two GPUs)


export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0 setsid nohup sh -c 'for k in 0 1 2; do python baseline_probreg.py train --fold $k; done' > runs/probreg_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup sh -c 'for k in 3 4;   do python baseline_probreg.py train --fold $k; done' > runs/probreg_gpu1.log 2>&1 &




export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0 setsid nohup sh -c 'for k in 0 1 2; do python baseline_probreg.py train --fold $k; done' > runs/probreg_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 setsid nohup sh -c 'for k in 3 4;   do python baseline_probreg.py train --fold $k; done' > runs/probreg_gpu1.log 2>&1 &

3. Evaluate and compare (≈ 15 minutes)

CUDA_VISIBLE_DEVICES=0 python baseline_probreg.py eval
python baseline_probreg.py compare
