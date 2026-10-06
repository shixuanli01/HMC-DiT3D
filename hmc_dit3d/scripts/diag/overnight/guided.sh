#!/bin/bash
# usage: guided.sh CAT G SEED VAE VAETAG   -> calibrate size factors at guidance G (trainbank, same seed), then score trainbank and knn-interp at G
cd /workspace/HMC-DiT3D/hmc_dit3d && source /venv/main/bin/activate >/dev/null
CAT=$1; G=$2; SEED=$3; VAE=$4; VT=$5; O=results/vae_retrain/$CAT
export PYTHONPATH=/workspace/emd_build:scripts/diag:src
if [ ! -f $O/${CAT}_rescale_factors_g$G.pt ]; then
  [ -f $O/${CAT}_trainbank_g${G}_seed$SEED.pt ] || timeout 90m python scripts/diag/diag_gen.py --config configs/train_${CAT}_h100_formal_bottleneck.yaml --checkpoint ../checkpoints/dit/$CAT.pt \
    --vae ../checkpoints/vae/$CAT.pt --mode trainbank --bank results/cov_diag/banks/${CAT}_train.pt --seed $SEED --fast --guidance $G --out $O/${CAT}_trainbank_g${G}_seed$SEED.pt > $O/${CAT}_trainbank_g${G}_seed$SEED.log 2>&1
  python scripts/diag/calib.py $CAT trainbank_g$G $SEED _g$G
fi
cd /workspace/HMC-DiT3D/hmc_dit3d/scripts/diag/overnight
RESCALE_SUFFIX=_g$G ./job.sh $CAT trainbank_g${G}c trainbank $SEED ../checkpoints/vae/$CAT.pt --guidance $G
RESCALE_SUFFIX=_g$G ./job.sh $CAT ${VT}_knn5_l0.5_g$G knninterp $SEED $VAE --knn-k 5 --interp-max 0.5 --guidance $G
