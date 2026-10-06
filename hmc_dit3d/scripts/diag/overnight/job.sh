#!/bin/bash
# usage: job.sh CAT TAG MODE SEED VAE [extra diag_gen args]
# fast generation -> fixed size correction -> val-protocol score of the corrected payload (RAW_EVAL=1 also scores the raw one)
set -u
cd /workspace/HMC-DiT3D/hmc_dit3d && source /venv/main/bin/activate >/dev/null
CAT=$1; TAG=$2; MODE=$3; SEED=$4; VAE=$5; shift 5
O=results/vae_retrain/$CAT; mkdir -p $O; T=${CAT}_${TAG}_seed${SEED}; R=${CAT}_${TAG}_rescaled_seed${SEED}
export PYTHONPATH=/workspace/emd_build:scripts/diag:src EMD_BUILD=/workspace/emd_build SHAPENET_ROOT=/workspace/HMC-DiT3D/ShapeNetCore.v2.PC15k
[ -f $O/val_$R.json ] && { echo "skip $R"; exit 0; }
[ -f $O/$T.pt ] || timeout 60m python scripts/diag/diag_gen.py --config configs/train_${CAT}_h100_formal_bottleneck.yaml \
  --checkpoint ../checkpoints/dit/$CAT.pt --vae $VAE --mode $MODE --bank results/cov_diag/banks/${CAT}_train.pt \
  --seed $SEED --fast "$@" --out $O/$T.pt > $O/$T.log 2>&1 || { echo "GEN FAILED $T: $(tail -1 $O/$T.log | cut -c1-200)"; exit 0; }
python scripts/diag/apply_rescale.py $CAT $O/$T.pt >> $O/$T.log 2>&1 || { echo "RESCALE FAILED $T"; exit 0; }
timeout 45m python scripts/diag/val_eval_fast.py --perms 3 --out $O/val_$R.json $O/$R.pt >> $O/$T.log 2>&1 || { echo "EVAL FAILED $R"; exit 0; }
[ "${RAW_EVAL:-0}" = 1 ] && timeout 45m python scripts/diag/val_eval_fast.py --perms 3 --out $O/val_$T.json $O/$T.pt >> $O/$T.log 2>&1
grep "$R.pt" $O/$T.log | tail -1 | sed 's|results/vae_retrain/[a-z]*/||' | cut -c1-170
