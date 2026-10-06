#!/bin/bash
# usage: chair_seed_sweep.sh GPU TAG MODE SEED   (TAG = orig | epoch7000 | epoch8000; MODE = vae | trainbank)
cd /fsx/weicyang/shix/HMC-DiT3D/hmc_dit3d
G=$1; TAG=$2; M=$3; SEED=$4
O=results/chair_seed_sweep; mkdir -p $O; T=${TAG}_${M}_seed${SEED}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=/fsx/weicyang/shix/emd_build:src OMP_NUM_THREADS=8
if [ $TAG = orig ]; then CK=../checkpoints/dit/chair.pt; else CK=results/retrain_chair_h100_formal_bottleneck_bs512/epoch_${TAG#epoch}.pt; fi
[ -f $O/$T.pt ] || ../.venv/bin/python scripts/diag/diag_gen.py --config results/retrain_epoch_sweep_n664_batch8/runtime_configs/chair_orig_seed0.yaml \
  --checkpoint $CK --vae ../checkpoints/vae/chair.pt --mode $M ${EXTRA:-} --bank results/cov_diag/banks/chair_train.pt --seed $SEED --out $O/$T.pt > $O/$T.log 2>&1 || exit 1
../.venv/bin/python scripts/diag/val_eval_mmd.py --perms 3 --out $O/val_$T.json $O/$T.pt >> $O/$T.log 2>&1
