#!/usr/bin/env bash
# Reproduce one main-table row: generate 664 samples with bank- or VAE-sampled HMC conditions and score them
# under the validation protocol (val split reference, 8-vs-8 groups, 3 permutations, exact EMD).
#
# usage: scripts/evaluate.sh CATEGORY CONFIG CHECKPOINT [--mode trainbank|vae] [--vae VAE.pt] [--seeds "0 11 12 13"] [--tag NAME]
#                             [--category CLASS] [--bank BANK.pt]
#   CATEGORY   chair | airplane | car | lamp | vessel
#   CONFIG     the training YAML of the checkpoint
#   CHECKPOINT the DiT checkpoint (.pt)
#   --category for multi-class checkpoints: restrict bank entries, labels and references to CLASS (e.g. chair)
#   --bank     train-condition bank to draw from (default results/banks/<CATEGORY>_train.pt, built from CONFIG on first use;
#              for multi-class checkpoints pass the single-category bank built from that category's config)
# Requires EMD_BUILD (directory with TopoDiT's emd_cuda*.so); SHAPENET_ROOT defaults to <repo>/ShapeNetCore.v2.PC15k.
# Outputs: results/<CATEGORY>/<CATEGORY>_<tag>_<mode>_seed<S>.pt and results/<CATEGORY>/val_<tag>_<mode>.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
CAT=$1; CFG=$2; CKPT=$3; shift 3
MODE=trainbank; VAE=""; SEEDS="0"; TAG="main"; CATEGORY=""; BANK=""
while [[ $# -gt 0 ]]; do case $1 in
  --mode) MODE=$2; shift 2;; --vae) VAE=$2; shift 2;; --seeds) SEEDS=$2; shift 2;; --tag) TAG=$2; shift 2;;
  --category) CATEGORY=$2; shift 2;; --bank) BANK=$2; shift 2;;
  *) echo "unknown option $1"; exit 1;; esac; done
[[ "$MODE" == "vae" && -z "$VAE" ]] && { echo "--vae is required for --mode vae"; exit 1; }
export SHAPENET_ROOT="${SHAPENET_ROOT:-$HERE/ShapeNetCore.v2.PC15k}"
export PYTHONPATH="${EMD_BUILD:?set EMD_BUILD to the emd_cuda build directory}:$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
BANK="${BANK:-results/banks/${CAT}_train.pt}"; mkdir -p results/banks "results/$CAT"
[[ -f "$BANK" ]] || python -m hmc_dit3d.train.build_bank --config "$CFG" --split train --output "$BANK"
PAYLOADS=()
for S in $SEEDS; do
  OUT="results/$CAT/${CAT}_${TAG}_${MODE}_seed${S}.pt"; PAYLOADS+=("$OUT")
  [[ -f "$OUT" ]] && { echo "skip $OUT"; continue; }
  python scripts/generate.py --config "$CFG" --checkpoint "$CKPT" --mode "$MODE" --bank "$BANK" ${VAE:+--vae "$VAE"} --seed "$S" --fast ${CATEGORY:+--category "$CATEGORY"} --out "$OUT"
done
python scripts/evaluate_val_protocol.py --perms 3 --out "results/$CAT/val_${TAG}_${MODE}.json" "${PAYLOADS[@]}"
