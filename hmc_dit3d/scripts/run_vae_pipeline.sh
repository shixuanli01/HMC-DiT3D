#!/usr/bin/env bash
# Full HMC-condition-VAE pipeline for one category:
#   bank build -> VAE training -> prior sampling eval -> paired reconstruction eval
#   -> TopoDiT comparison against the same paired references.
set -euo pipefail

CATEGORY=$1        # airplane | car | chair
CHECKPOINT=$2      # diffusion checkpoint path
RUN="micromamba run -n hmc-dit3d-py312"
CFG=configs/train_${CATEGORY}_h100_formal_bottleneck.yaml
DIR=results/train_${CATEGORY}_h100_formal_bottleneck/vae
EVAL_BANK=results/train_${CATEGORY}_h100_formal_bottleneck/test_eval/${CATEGORY}_train_bank_eval128.pt
TOPO_SAMPLES=${TOPO_SAMPLES:-../TopoDiT-3D/checkpoints/output/test_S4_${CATEGORY}/syn/samples.pth}
mkdir -p "$DIR"

echo "########## [$CATEGORY] 1/5 build full train bank"
$RUN python -m hmc_dit3d.train.build_bank --config "$CFG" --split train \
  --output "$DIR/${CATEGORY}_train_bank_full.pt"

echo "########## [$CATEGORY] 2/5 train condition VAE (beta=0.05 z=64)"
$RUN python -m hmc_dit3d.train.train_condition_vae \
  --bank "$DIR/${CATEGORY}_train_bank_full.pt" \
  --output "$DIR/${CATEGORY}_condition_vae_b0.05_z64.pt" \
  --epochs 2000 --beta 0.05 --latent-dim 64 --device cuda

echo "########## [$CATEGORY] 3/5 prior sampling (64, thr=0.1, T=1.2) + eval"
$RUN python -m hmc_dit3d.train.sample --config "$CFG" --checkpoint "$CHECKPOINT" \
  --condition-source vae --vae "$DIR/${CATEGORY}_condition_vae_b0.05_z64.pt" \
  --split test --limit 64 --vae-sequence-threshold 0.1 --vae-temperature 1.2 \
  --output "$DIR/vae_samples_test64.pt"
echo "===== [$CATEGORY] VAE prior (unpaired test refs)"
$RUN python -m hmc_dit3d.train.evaluate --samples "$DIR/vae_samples_test64.pt" \
  --batch-size 8 --device cuda --jsd-resolution 8 2>&1 | grep -E '1-NN|lgan_cov|lgan_mmd-|JSD'

echo "########## [$CATEGORY] 4/5 VAE reconstruction (paired refs) + eval"
$RUN python scripts/sample_vae_reconstruction.py --config "$CFG" --checkpoint "$CHECKPOINT" \
  --bank "$EVAL_BANK" --vae "$DIR/${CATEGORY}_condition_vae_b0.05_z64.pt" \
  --limit 64 --output "$DIR/vae_recon_samples_test64.pt"
echo "===== [$CATEGORY] VAE reconstruction (paired refs, comparable to paper row)"
$RUN python -m hmc_dit3d.train.evaluate --samples "$DIR/vae_recon_samples_test64.pt" \
  --batch-size 8 --device cuda --jsd-resolution 8 2>&1 | grep -E '1-NN|lgan_cov|lgan_mmd-|JSD'

echo "########## [$CATEGORY] 5/5 TopoDiT vs same paired refs"
$RUN python - "$DIR" "$TOPO_SAMPLES" <<'PY'
import sys, torch
out_dir, topo_path = sys.argv[1], sys.argv[2]
recon = torch.load(f"{out_dir}/vae_recon_samples_test64.pt", map_location="cpu", weights_only=False)
topo = torch.load(topo_path, map_location="cpu", weights_only=False)
topo_samples = topo["samples"] if isinstance(topo, dict) else topo
g = torch.Generator().manual_seed(42)
idx = torch.randperm(topo_samples.shape[0], generator=g)[:64]
torch.save(
    {"samples": topo_samples[idx].float(), "references": recon["references"].float()},
    f"{out_dir}/topodit_vs_pairedrefs64.pt",
)
print("topodit subset saved")
PY
echo "===== [$CATEGORY] TopoDiT (same paired refs)"
$RUN python -m hmc_dit3d.train.evaluate --samples "$DIR/topodit_vs_pairedrefs64.pt" \
  --batch-size 8 --device cuda --jsd-resolution 8 2>&1 | grep -E '1-NN|lgan_cov|lgan_mmd-|JSD'

echo "########## [$CATEGORY] done"
