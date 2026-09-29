#!/bin/bash
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --output="airplane_test_%j.txt"
#SBATCH --error="airplane_test_%j.txt"

# End-to-end airplane eval (matches training: configs/train_airplane_h100_formal_bottleneck.yaml).
# Repo root: ./test_airplane.sh   (or: sbatch test_airplane.sh)
#
# Outputs go to results/train_airplane_h100_formal_bottleneck/test_eval/ (under hmc_dit3d/).
# Optional env:
#   SHAPENET_ROOT=/path/to/ShapeNetCore.v2.PC15k  (default: <repo>/ShapeNetCore.v2.PC15k)
#   CHECKPOINT=/path/to.pt  (overrides auto-pick below)
#   AIRPLANE_RUN_DIR=<dir>  (default: <repo>/hmc_results/train_airplane_h100_formal_bottleneck — searched first for best_train.pt / best_val.pt)
#   SAMPLE_LIMIT=<int>|all  (default 64; ref+bank count each, or "all" = entire test split for reference; bank matches ref count)
#   e.g. SAMPLE_LIMIT=128 ./test_airplane.sh
#        SAMPLE_LIMIT=all ./test_airplane.sh
#
# After sample: each *.pt contains model_ids + synset_ids; save_sample_payload also writes *_manifest.json (same stem).
# Mitsuba (optional): from repo root, after this script finishes:
#   RENDER_MITSUBA=1 ./test_airplane.sh
# uses MITSUBA_OUT_REF / MITSUBA_OUT_BANK (defaults under <repo>/final_imgs/) and render_mitsuba_samples.py with --output-name-mode auto.
#
# Default weights (first existing wins): AIRPLANE_RUN_DIR/best_train.pt, best_val.pt, then hmc_airplane_*.pt, then hmc_dit3d/results/...
#
# Sensitivity sweeps: set CHECKPOINT explicitly (or category-specific sweep scripts if added).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/hmc_dit3d"

export PATH="${HOME}/.local/bin:${PATH}"

ENV_NAME="hmc-dit3d-py312"
CONFIG_SRC="configs/train_airplane_h100_formal_bottleneck.yaml"

TRAIN_RESULT_DIR="results/train_airplane_h100_formal_bottleneck"
TEST_EVAL_DIR="${TRAIN_RESULT_DIR}/test_eval_dit3d"
mkdir -p "${TEST_EVAL_DIR}"

SHAPENET_ROOT="${SHAPENET_ROOT:-${SCRIPT_DIR}/ShapeNetCore.v2.PC15k}"
CONFIG_RUNTIME="$(mktemp /tmp/hmc_test_airplane_config.XXXXXX.yaml)"
cleanup() { rm -f "${CONFIG_RUNTIME}"; }
trap cleanup EXIT

if [[ -d "${SHAPENET_ROOT}" ]]; then
  export SHAPENET_ROOT
  micromamba run -n "${ENV_NAME}" python - <<PY
import os
import yaml
from pathlib import Path

src = Path("${CONFIG_SRC}")
dst = Path("${CONFIG_RUNTIME}")
root = Path(os.environ["SHAPENET_ROOT"]).resolve()
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["data"]["root_dir"] = str(root)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
  CONFIG="${CONFIG_RUNTIME}"
  echo "[INFO] Using absolute data.root_dir -> ${SHAPENET_ROOT}"
else
  CONFIG="${CONFIG_SRC}"
  echo "[WARN] SHAPENET_ROOT not found: ${SHAPENET_ROOT} — using yaml relative root_dir (must match training layout)."
fi

# Prefer checkpoints under <repo>/hmc_results/... (e.g. symlink or copy from training machine).
AIRPLANE_RUN_DIR="${AIRPLANE_RUN_DIR:-${SCRIPT_DIR}/hmc_results/hmc_airplane}"
CKPT_HMC_RESULTS_TRAIN="${AIRPLANE_RUN_DIR}/best_val.pt"
CKPT_HMC_RESULTS_VAL="${AIRPLANE_RUN_DIR}/best_val.pt"

CKPT_REPO_TRAIN="${SCRIPT_DIR}/hmc_airplane_best_val.pt"
CKPT_REPO_VAL="${SCRIPT_DIR}/hmc_airplane_best_train.pt"
CKPT_BEST_TRAIN="${SCRIPT_DIR}/hmc_dit3d/${TRAIN_RESULT_DIR}/best_train.pt"
CKPT_BEST_VAL="${SCRIPT_DIR}/hmc_dit3d/${TRAIN_RESULT_DIR}/best_val.pt"
CKPT_FINAL="${SCRIPT_DIR}/hmc_dit3d/${TRAIN_RESULT_DIR}/train_checkpoint.pt"

if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ -f "${CKPT_HMC_RESULTS_TRAIN}" ]]; then
    CHECKPOINT="${CKPT_HMC_RESULTS_TRAIN}"
  elif [[ -f "${CKPT_HMC_RESULTS_VAL}" ]]; then
    CHECKPOINT="${CKPT_HMC_RESULTS_VAL}"
  elif [[ -f "${CKPT_REPO_TRAIN}" ]]; then
    CHECKPOINT="${CKPT_REPO_TRAIN}"
  elif [[ -f "${CKPT_REPO_VAL}" ]]; then
    CHECKPOINT="${CKPT_REPO_VAL}"
  elif [[ -f "${CKPT_BEST_TRAIN}" ]]; then
    CHECKPOINT="${CKPT_BEST_TRAIN}"
  elif [[ -f "${CKPT_BEST_VAL}" ]]; then
    CHECKPOINT="${CKPT_BEST_VAL}"
  elif [[ -f "${CKPT_FINAL}" ]]; then
    CHECKPOINT="${CKPT_FINAL}"
  else
    CHECKPOINT="${CKPT_HMC_RESULTS_TRAIN}"
  fi
fi

if [[ "${CHECKPOINT}" != /* ]]; then
  if [[ -f "${PWD}/${CHECKPOINT}" ]]; then
    CHECKPOINT="${PWD}/${CHECKPOINT}"
  elif [[ -f "${SCRIPT_DIR}/${CHECKPOINT}" ]]; then
    CHECKPOINT="${SCRIPT_DIR}/${CHECKPOINT}"
  fi
fi

BANK_LIMIT="${BANK_LIMIT:-128}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-64}"
if [[ "${SAMPLE_LIMIT}" == "all" ]]; then
  SAMPLE_TAG="full"
else
  SAMPLE_TAG="${SAMPLE_LIMIT}"
fi
JSD_RESOLUTION="${JSD_RESOLUTION:-8}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

BANK_FILE="${BANK_FILE:-${TEST_EVAL_DIR}/airplane_train_bank_eval${BANK_LIMIT}.pt}"
REF_SAMPLES="${REF_SAMPLES:-${TEST_EVAL_DIR}/reference_samples_test${SAMPLE_TAG}.pt}"
BANK_SAMPLES="${BANK_SAMPLES:-${TEST_EVAL_DIR}/bank_samples_test${SAMPLE_TAG}.pt}"
REF_MANIFEST="${REF_SAMPLES%.pt}_manifest.json"
BANK_MANIFEST="${BANK_SAMPLES%.pt}_manifest.json"

RENDER_MITSUBA="${RENDER_MITSUBA:-0}"
MITSUBA_OUT_REF="${MITSUBA_OUT_REF:-${SCRIPT_DIR}/final_imgs/hmc_airplane_ref_mitsuba}"
MITSUBA_OUT_BANK="${MITSUBA_OUT_BANK:-${SCRIPT_DIR}/final_imgs/hmc_airplane_bank_mitsuba}"

run_py() {
  micromamba run -n "${ENV_NAME}" python "$@"
}

echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Train result dir (checkpoints): ${TRAIN_RESULT_DIR}"
echo "[INFO] Airplane run dir (hmc_results fallback): ${AIRPLANE_RUN_DIR}"
echo "[INFO] Test outputs (bank/samples/png): ${TEST_EVAL_DIR}"
echo "[INFO] Checkpoint: ${CHECKPOINT}"
if [[ "${SAMPLE_LIMIT}" == "all" ]]; then
  echo "[INFO] Sample count: all test-split shapes (reference: no --limit; bank: same N as ref)"
else
  echo "[INFO] Sample count (ref + bank each): ${SAMPLE_LIMIT}"
fi
echo "[INFO] BANK_FILE (build_bank ->): ${BANK_FILE}"
echo "[INFO] REF_SAMPLES (reference cond ->): ${REF_SAMPLES}"
echo "[INFO] REF_MANIFEST (ids for Mitsuba naming ->): ${REF_MANIFEST}"
echo "[INFO] BANK_SAMPLES (bank cond ->): ${BANK_SAMPLES}"
echo "[INFO] BANK_MANIFEST (ids for Mitsuba naming ->): ${BANK_MANIFEST}"
echo "[INFO] Requested eval device: ${EVAL_DEVICE}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  echo "[HINT] Expected e.g. ${CKPT_HMC_RESULTS_TRAIN} or set CHECKPOINT=/path/to.pt" >&2
  echo "       Also tried: ${CKPT_BEST_TRAIN} | ${CKPT_BEST_VAL} | ${CKPT_REPO_TRAIN} | ${CKPT_REPO_VAL}" >&2
  exit 1
fi

if [[ "${EVAL_DEVICE}" == "cuda" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1 {print $1}')"
    echo "[INFO] Detected GPU memory: ${GPU_MEM_MIB} MiB"
  else
    echo "[WARN] nvidia-smi not found. Falling back to conservative eval batch size."
    GPU_MEM_MIB=""
  fi
else
  GPU_MEM_MIB=""
fi

if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  EVAL_BATCH="${EVAL_BATCH_SIZE}"
elif [[ -n "${GPU_MEM_MIB}" ]] && [[ "${GPU_MEM_MIB}" -ge 70000 ]]; then
  EVAL_BATCH=64
elif [[ -n "${GPU_MEM_MIB}" ]] && [[ "${GPU_MEM_MIB}" -ge 35000 ]]; then
  EVAL_BATCH=32
elif [[ -n "${GPU_MEM_MIB}" ]] && [[ "${GPU_MEM_MIB}" -ge 20000 ]]; then
  EVAL_BATCH=16
else
  EVAL_BATCH=8
fi

echo "[INFO] Eval batch size: ${EVAL_BATCH}"
echo "[INFO] Bank limit: ${BANK_LIMIT}, Sample limit: ${SAMPLE_LIMIT}, JSD resolution: ${JSD_RESOLUTION}"

run_py -m hmc_dit3d.train.build_bank \
  --config "${CONFIG}" \
  --split train \
  --limit "${BANK_LIMIT}" \
  --output "${BANK_FILE}"

if [[ "${SAMPLE_LIMIT}" == "all" ]]; then
  run_py -m hmc_dit3d.train.sample \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --condition-source reference \
    --split test \
    --output "${REF_SAMPLES}" \
    --clip-denoised
  EFFECTIVE_N="$(run_py -c "
import torch
from pathlib import Path
p = Path('${REF_SAMPLES}').resolve()
try:
    d = torch.load(p, map_location='cpu', weights_only=False)
except TypeError:
    d = torch.load(p, map_location='cpu')
print(int(d['samples'].shape[0]))
")"
  echo "[INFO] Full test split: N=${EFFECTIVE_N} (using same N for bank-conditioned sample)"
else
  EFFECTIVE_N="${SAMPLE_LIMIT}"
  run_py -m hmc_dit3d.train.sample \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --condition-source reference \
    --split test \
    --limit "${SAMPLE_LIMIT}" \
    --output "${REF_SAMPLES}" \
    --clip-denoised
fi

run_py -m hmc_dit3d.train.sample \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --condition-source bank \
  --bank "${BANK_FILE}" \
  --bank-categories airplane \
  --limit "${EFFECTIVE_N}" \
  --output "${BANK_SAMPLES}" \
  --clip-denoised

if [[ -f "${REF_MANIFEST}" ]]; then
  echo "[INFO] Reference sample id manifest present: ${REF_MANIFEST}"
else
  echo "[WARN] Expected manifest missing (older hmc_dit3d?): ${REF_MANIFEST}" >&2
fi
if [[ -f "${BANK_MANIFEST}" ]]; then
  echo "[INFO] Bank sample id manifest present: ${BANK_MANIFEST}"
else
  echo "[WARN] Expected manifest missing: ${BANK_MANIFEST}" >&2
fi

echo "[INFO] Full metrics (reference-conditioned samples)"
run_py -m hmc_dit3d.train.evaluate \
  --samples "${REF_SAMPLES}" \
  --batch-size "${EVAL_BATCH}" \
  --device "${EVAL_DEVICE}" \
  --jsd-resolution "${JSD_RESOLUTION}"

echo "[INFO] Full metrics (bank-conditioned samples)"
run_py -m hmc_dit3d.train.evaluate \
  --samples "${BANK_SAMPLES}" \
  --batch-size "${EVAL_BATCH}" \
  --device "${EVAL_DEVICE}" \
  --jsd-resolution "${JSD_RESOLUTION}"

REF_PNG="${TEST_EVAL_DIR}/reference_samples_test${SAMPLE_TAG}.png"
BANK_PNG="${TEST_EVAL_DIR}/bank_samples_test${SAMPLE_TAG}.png"
COMBINED_PNG="${TEST_EVAL_DIR}/airplane_visualization_test${SAMPLE_TAG}.png"

echo "[INFO] Rendering sample visualizations (non-fatal on failure)"
if ! run_py -m hmc_dit3d.train.visualize \
  --samples "${REF_SAMPLES}" \
  --labels airplane-ref \
  --num-samples 4 \
  --output "${REF_PNG}"; then
  echo "[WARN] Failed to render reference visualization: ${REF_PNG}" >&2
fi

if ! run_py -m hmc_dit3d.train.visualize \
  --samples "${BANK_SAMPLES}" \
  --labels airplane-bank \
  --num-samples 4 \
  --output "${BANK_PNG}"; then
  echo "[WARN] Failed to render bank visualization: ${BANK_PNG}" >&2
fi

if ! run_py -m hmc_dit3d.train.visualize \
  --samples "${REF_SAMPLES}" "${BANK_SAMPLES}" \
  --labels airplane-ref airplane-bank \
  --num-samples 4 \
  --output "${COMBINED_PNG}"; then
  echo "[WARN] Failed to render combined visualization: ${COMBINED_PNG}" >&2
fi

echo "[DONE] PNGs: ${REF_PNG} | ${BANK_PNG} | ${COMBINED_PNG}"
echo "[DONE] Sample tensors + ids: ${REF_SAMPLES} | ${BANK_SAMPLES}"
echo "[DONE] Id manifests: ${REF_MANIFEST} | ${BANK_MANIFEST}"

if [[ "${RENDER_MITSUBA}" == "1" ]]; then
  # render_mitsuba_samples.py lives at repo root; paths must be absolute for a stable --samples-path.
  _abs() {
    local p="$1"
    if [[ "${p}" == /* ]]; then
      printf '%s' "${p}"
    else
      printf '%s' "${PWD}/${p}"
    fi
  }
  REF_ABS="$(_abs "${REF_SAMPLES}")"
  BANK_ABS="$(_abs "${BANK_SAMPLES}")"
  echo "[INFO] RENDER_MITSUBA=1 -> Mitsuba PNGs (filenames = synset_id_model_id when ids present)"
  mkdir -p "${MITSUBA_OUT_REF}" "${MITSUBA_OUT_BANK}"
  micromamba run -n "${ENV_NAME}" python "${SCRIPT_DIR}/render_mitsuba_samples.py" \
    --samples-path "${REF_ABS}" \
    --output-dir "${MITSUBA_OUT_REF}" \
    --output-name-mode auto \
    || echo "[WARN] Mitsuba ref render failed (non-fatal)" >&2
  micromamba run -n "${ENV_NAME}" python "${SCRIPT_DIR}/render_mitsuba_samples.py" \
    --samples-path "${BANK_ABS}" \
    --output-dir "${MITSUBA_OUT_BANK}" \
    --output-name-mode auto \
    || echo "[WARN] Mitsuba bank render failed (non-fatal)" >&2
  echo "[DONE] Mitsuba dirs: ${MITSUBA_OUT_REF} | ${MITSUBA_OUT_BANK}"
fi
