#!/usr/bin/env bash
# Focused VAE retune for airplane/car: lower beta, fine-scale CE emphasis,
# top-k occupancy matching for paired reconstruction.
set -euo pipefail

CATEGORY=$1
CHECKPOINT=$2
RUN="micromamba run -n hmc-dit3d-py312"
CFG=configs/train_${CATEGORY}_h100_formal_bottleneck.yaml
DIR=results/train_${CATEGORY}_h100_formal_bottleneck/vae
BANK=$DIR/${CATEGORY}_train_bank_full.pt
EVAL_BANK=results/train_${CATEGORY}_h100_formal_bottleneck/test_eval/${CATEGORY}_train_bank_eval128.pt
TUNE=$DIR/tune
mkdir -p "$TUNE"

# tag | beta | z | seq_weights | epochs
CONFIGS=(
  "b0.005_z128_w112 0.005 128 1,1,2 2500"
  "b0.002_z128_w113 0.002 128 1,1,3 2500"
  "b0.005_z256_w112 0.005 256 1,1,2 2500"
  "b0.01_z128_w112 0.01 128 1,1,2 2500"
)

eval_one() {
  local path=$1
  $RUN python -m hmc_dit3d.train.evaluate --samples "$path" \
    --batch-size 8 --device cuda --jsd-resolution 8 2>&1 \
    | grep -E '1-NN|lgan_cov|lgan_mmd-|JSD'
}

for row in "${CONFIGS[@]}"; do
  set -- $row
  TAG=$1; BETA=$2; Z=$3; SW=$4; EP=$5
  VAE=$TUNE/${CATEGORY}_condition_vae_${TAG}.pt
  echo "########## [$CATEGORY] train $TAG"
  $RUN python -m hmc_dit3d.train.train_condition_vae \
    --bank "$BANK" --output "$VAE" \
    --epochs "$EP" --beta "$BETA" --latent-dim "$Z" \
    --hidden-dims 2048,1024,512 \
    --sequence-weights "$SW" \
    --device cuda --log-every 500

  for MODE in thr0 topk; do
    OUT=$TUNE/vae_recon_${TAG}_${MODE}.pt
    EXTRA=()
    if [[ "$MODE" == "topk" ]]; then
      EXTRA+=(--topk-occupancy --sequence-threshold 0)
    else
      EXTRA+=(--sequence-threshold 0)
    fi
    echo "########## [$CATEGORY] recon $TAG / $MODE"
    $RUN python scripts/sample_vae_reconstruction.py \
      --config "$CFG" --checkpoint "$CHECKPOINT" \
      --bank "$EVAL_BANK" --vae "$VAE" --limit 64 \
      "${EXTRA[@]}" --output "$OUT"
    echo "===== [$CATEGORY] recon $TAG $MODE"
    eval_one "$OUT"
  done

  # prior with topk-style sparsification via threshold; also thr0
  for THR in 0.0 0.1; do
    OUT=$TUNE/vae_prior_${TAG}_thr${THR}.pt
    echo "########## [$CATEGORY] prior $TAG thr=$THR"
    $RUN python -m hmc_dit3d.train.sample \
      --config "$CFG" --checkpoint "$CHECKPOINT" \
      --condition-source vae --vae "$VAE" \
      --split test --limit 64 \
      --vae-sequence-threshold "$THR" --vae-temperature 1.0 \
      --output "$OUT"
    echo "===== [$CATEGORY] prior $TAG thr=$THR"
    eval_one "$OUT"
  done
done

echo "########## [$CATEGORY] tune done"
