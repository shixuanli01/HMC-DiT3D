#!/bin/bash
# usage: train_vaes.sh CAT "tag beta z hidden epochs" ...
cd /workspace/HMC-DiT3D/hmc_dit3d && source /venv/main/bin/activate >/dev/null
CAT=$1; shift; V=results/vae_retrain/$CAT/vae; mkdir -p $V
for spec in "$@"; do
  set -- $spec
  [ -f $V/${CAT}_$1.pt ] && { echo "skip $1"; continue; }
  t0=$(date +%s)
  python -m hmc_dit3d.train.train_condition_vae --bank results/cov_diag/banks/${CAT}_train.pt --output $V/${CAT}_$1.pt \
    --beta $2 --latent-dim $3 --hidden-dims $4 --epochs $5 --device cuda > $V/train_$1.log 2>&1 || { echo "FAILED $1"; tail -3 $V/train_$1.log; continue; }
  echo "$CAT $1 done in $(( $(date +%s) - t0 ))s | $(grep 'epoch=' $V/train_$1.log | tail -1 | sed 's/.*| loss/loss/')"
done
