#!/usr/bin/env bash
# Retrain-sweep of the condition VAE for one category; evaluates each variant
# under both the prior (unpaired) and reconstruction (paired) protocols.
set -uo pipefail

CATEGORY=$1
CHECKPOINT=$2
RUN="micromamba run -n hmc-dit3d-py312"
CFG=configs/train_${CATEGORY}_h100_formal_bottleneck.yaml
DIR=results/train_${CATEGORY}_h100_formal_bottleneck/vae
EVAL_BANK=results/train_${CATEGORY}_h100_formal_bottleneck/test_eval/${CATEGORY}_train_bank_eval128.pt

# tag | beta | latent | hidden | epochs
VARIANTS=(
  "big_b0.05_z64   0.05 64  4096,2048,1024 3000"
  "big_b0.02_z64   0.02 64  4096,2048,1024 3000"
  "big_b0.1_z64    0.1  64  4096,2048,1024 3000"
  "base_b0.05_z128 0.05 128 2048,1024,512  4000"
)

for spec in "${VARIANTS[@]}"; do
  read -r tag beta z hidden epochs <<<"$spec"
  vae=$DIR/${CATEGORY}_vae_${tag}.pt
  $RUN python -m hmc_dit3d.train.train_condition_vae \
    --bank "$DIR/${CATEGORY}_train_bank_full.pt" --output "$vae" \
    --epochs "$epochs" --beta "$beta" --latent-dim "$z" --hidden-dims "$hidden" \
    --device cuda > "$DIR/train_${tag}.log" 2>&1
  rg 'Prior descriptor|scale\[2\]' "$DIR/train_${tag}.log" | tail -2

  out=$DIR/sweep2_prior_${tag}.pt
  $RUN python -m hmc_dit3d.train.sample --config "$CFG" --checkpoint "$CHECKPOINT" \
    --condition-source vae --vae "$vae" --split test --limit 64 \
    --vae-sequence-threshold 0.5 --vae-temperature 1.0 --output "$out" > /dev/null 2>&1
  echo "===== [$CATEGORY] $tag PRIOR"
  $RUN python -m hmc_dit3d.train.evaluate --samples "$out" --batch-size 8 \
    --device cuda --jsd-resolution 8 2>&1 | grep -E '1-NN|lgan_cov|lgan_mmd-|JSD'

  out=$DIR/sweep2_recon_${tag}.pt
  $RUN python scripts/sample_vae_reconstruction.py --config "$CFG" \
    --checkpoint "$CHECKPOINT" --bank "$EVAL_BANK" --vae "$vae" --limit 64 \
    --sequence-threshold 0.5 --output "$out" > /dev/null 2>&1
  echo "===== [$CATEGORY] $tag RECON"
  $RUN python -m hmc_dit3d.train.evaluate --samples "$out" --batch-size 8 \
    --device cuda --jsd-resolution 8 2>&1 | grep -E '1-NN|lgan_cov|lgan_mmd-|JSD'
done
echo "########## [$CATEGORY] sweep done"
