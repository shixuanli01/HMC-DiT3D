# HMC-DiT3D

Research code for **Hilbert–multifractal conditioning of 3D diffusion
transformers**. The current system extracts multi-scale HMC descriptors from a
point cloud, serializes voxel measures along a 3D Hilbert curve, encodes them as
conditioning tokens, and generates 2,048-point ShapeNet point clouds with a
DiT-S/4-scale denoiser. A condition VAE provides an unpaired prior so generation
does not require a test point cloud at inference time.

This repository is the clean handoff of the active experiment. Datasets and
checkpoints are hosted on Google Drive; generated samples, logs, and caches are
intentionally excluded from Git.

## What we are testing

The central question is whether HMC conditioning can match or exceed
TopoDiT-3D on ShapeNet **Chair, Airplane, and Car**, especially coverage (COV),
without leaking test geometry into generation.

The active comparison protocol is:

- 664 generated samples per category;
- generation batch size 8;
- seed 0 unless a multi-seed experiment is requested;
- VAE prior temperature 1.2 and sequence threshold 0.1;
- 1-NNA is better when closer to 50%; COV is better when higher;
- metric order: 1-NNA-CD, 1-NNA-EMD, COV-CD, COV-EMD.

Current seed-0 results under the batch-8 TopoDiT reporting protocol are:

| Category | 1-NNA-CD | 1-NNA-EMD | COV-CD | COV-EMD |
|---|---:|---:|---:|---:|
| Chair | 50.53 | 48.49 | 51.36 | 50.90 |
| Airplane | 54.22 | 52.41 | 47.14 | 50.30 |
| Car | 57.98 | 49.02 | 46.99 | 52.11 |

The immediate research bottleneck is COV. Recent experiments showed that a
larger/lower-beta VAE can improve a small N=64 probe but may become unstable at
N=664. The safest baseline remains the first `beta=0.05, latent=64` VAE for all
three categories. Do not replace it without a matched-seed N=664 comparison.

## Repository layout

```text
.
├── README.md
├── AGENTS.md
├── CHECKPOINTS.md
├── CHECKPOINTS.sha256
├── download_checkpoints.sh
├── checkpoints/                 # created after downloading; gitignored
│   ├── dit/
│   └── vae/
├── ShapeNetCore.v2.PC15k/       # created after downloading; gitignored
├── hmc_dit3d/
│   ├── configs/
│   ├── scripts/
│   ├── src/hmc_dit3d/
│   ├── tests/
│   ├── environment.yml
│   ├── requirements.txt
│   └── pyproject.toml
├── train_{chair,airplane,car}.sh
└── test_{chair,airplane,car}.sh
```

## 1. Clone and create the environment

Python 3.12 and a recent CUDA-enabled PyTorch build are recommended. The tested
environment uses CUDA 12.8 and an RTX 5090; H100/A100-class GPUs are also
appropriate.

```bash
git clone https://github.com/shixuanli01/HMC-DiT3D.git
cd HMC-DiT3D

cd hmc_dit3d
micromamba create -y -f environment.yml
micromamba run -n hmc-dit3d-py312 python -m pip install -e .
micromamba run -n hmc-dit3d-py312 python -c \
  "import torch, hmc_dit3d; print(torch.__version__, torch.cuda.is_available())"
cd ..
```

A normal virtual environment also works:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r hmc_dit3d/requirements.txt
python -m pip install -e hmc_dit3d
```

## 2. Download ShapeNetCore.v2.PC15k

The prepared dataset is hosted on Google Drive. From the repository root:

```bash
python -m pip install gdown
gdown 1GgmM4dRbXUei4dKtgQKGLHLibuaghPep -O ShapeNetCore.v2.PC15k.zip
unzip ShapeNetCore.v2.PC15k.zip
```

The expected result is:

```text
HMC-DiT3D/ShapeNetCore.v2.PC15k/
```

The committed configs resolve the dataset relative to `hmc_dit3d/configs`, so
this root-level placement works without editing YAML files. If the archive
contains an extra outer directory, move the directory containing the category
folders to exactly `ShapeNetCore.v2.PC15k/`.

## 3. Download checkpoints

The six verified files are in this public
[Google Drive folder](https://drive.google.com/drive/folders/1HUuCpNJ1QBtyVwzJLoWDmIWAngdpZfrp?usp=sharing).
Download and rename them into the paths expected by the launch scripts with:

```bash
bash download_checkpoints.sh
```

See [CHECKPOINTS.md](CHECKPOINTS.md) for individual Google Drive file IDs,
SHA-256 hashes, roles, epochs, and manual download commands. The final layout
is:

```text
checkpoints/
├── dit/
│   ├── chair.pt
│   ├── airplane.pt
│   └── car.pt
└── vae/
    ├── chair.pt
    ├── airplane.pt
    └── car.pt
```

These are the exact DiT and first/best VAE checkpoints used for the current
seed-0 N=664 table.

## 4. Validate the installation

```bash
cd hmc_dit3d
micromamba run -n hmc-dit3d-py312 python -m pytest -q
micromamba run -n hmc-dit3d-py312 python -m ruff check .
cd ..
```

For a quick data-path check:

```bash
cd hmc_dit3d
micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.build_bank \
  --config configs/train_chair_h100_formal_bottleneck.yaml \
  --split train --limit 8 \
  --output results/smoke/chair_train_bank_8.pt
cd ..
```

## 5. Reproduce generation

The sampling seed is read from the YAML config. To reproduce the published
seed-0 run without modifying a tracked file, create temporary configs:

```bash
cd hmc_dit3d
for category in chair airplane car; do
  python - "$category" <<'PY'
import sys
from pathlib import Path
import yaml

category = sys.argv[1]
src = Path(f"configs/train_{category}_h100_formal_bottleneck.yaml")
dst = Path(f"/tmp/hmc_{category}_seed0.yaml")
cfg = yaml.safe_load(src.read_text())
cfg["train"]["seed"] = 0
cfg["data"]["root_dir"] = str(Path("../ShapeNetCore.v2.PC15k").resolve())
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(dst)
PY
done
```

Generate 664 samples per category, eight at a time:

```bash
mkdir -p results/reproduce_n664/{chair,airplane,car}

for category in chair airplane car; do
  micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.sample \
    --config "/tmp/hmc_${category}_seed0.yaml" \
    --checkpoint "../checkpoints/dit/${category}.pt" \
    --condition-source vae \
    --vae "../checkpoints/vae/${category}.pt" \
    --split test \
    --limit 664 \
    --generation-batch-size 8 \
    --vae-temperature 1.2 \
    --vae-sequence-threshold 0.1 \
    --output "results/reproduce_n664/${category}/vae_prior_seed0.pt"
done
```

On an RTX 5090, one category takes roughly 17 minutes with 1,000 diffusion
steps. The generated payload contains both samples and the matched test
references, plus checkpoint/VAE provenance.

## 6. Evaluate

The repository contains two evaluators:

- `python -m hmc_dit3d.train.evaluate`: project metrics, including a
  Sinkhorn-based EMD approximation;
- `scripts/eval_topodit_protocol.py`: exact TopoDiT squared-Chamfer math for
  CD-based 1-NNA/COV/MMD plus JSD.

Example:

```bash
cd hmc_dit3d
micromamba run -n hmc-dit3d-py312 python scripts/eval_topodit_protocol.py \
  --samples results/reproduce_n664/airplane/vae_prior_seed0.pt \
  --device cuda --ref-chunk 8

micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.evaluate \
  --samples results/reproduce_n664/airplane/vae_prior_seed0.pt \
  --batch-size 8 --device cuda --jsd-resolution 8
```

Important: TopoDiT's reported EMD uses its compiled CUDA `approxmatch`
extension. The built-in project evaluator uses Sinkhorn EMD and must not be
presented as numerically identical. Also distinguish full-set metrics from the
TopoDiT paper's average of independent 8-vs-8 batches.

## 7. Train from scratch

The active formal configs are:

| Category | Config | Current HMC scales / q orders |
|---|---|---|
| Chair | `configs/train_chair_h100_formal_bottleneck.yaml` | `[2,3,4] / [1,2]` |
| Airplane | `configs/train_airplane_h100_formal_bottleneck.yaml` | `[3,4,5] / [1,2,3]` |
| Car | `configs/train_car_h100_formal_bottleneck.yaml` | `[3,4,5] / [1,2,3]` |

All use a bottleneck model with voxel size 32, patch size 4, model dimension
384, depth 12, six heads, 2,048 output points, AMP, and 10,000 epochs.

```bash
cd hmc_dit3d
micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.train \
  --config configs/train_chair_h100_formal_bottleneck.yaml
```

The trainer writes `latest.pt`, `best_train.pt`, `best_val.pt`, periodic
`epoch_*.pt`, `train_metrics.jsonl`, and `evolution.jsonl` under
`hmc_dit3d/results/<experiment_name>/`. Training supports resume, EMA, Min-SNR
weighting, HMC dropout, and optional frozen-VAE condition reconstruction; see
`train/config.py` and `configs/train_airplane_vae_recon_retrain.yaml`.

## 8. Train the condition VAE

First build a train-only HMC condition bank, then train the VAE:

```bash
cd hmc_dit3d
micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.build_bank \
  --config configs/train_airplane_h100_formal_bottleneck.yaml \
  --split train \
  --output results/train_airplane_h100_formal_bottleneck/vae/airplane_train_bank_full.pt

micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.train_condition_vae \
  --bank results/train_airplane_h100_formal_bottleneck/vae/airplane_train_bank_full.pt \
  --output results/train_airplane_h100_formal_bottleneck/vae/airplane_condition_vae_b0.05_z64.pt \
  --latent-dim 64 --hidden-dims 2048,1024,512 \
  --beta 0.05 --epochs 2000 --device cuda
```

The bank must come from the **train split**. Test references are used only for
evaluation.

## Notes for the next agent

Read [AGENTS.md](AGENTS.md) before changing training or evaluation. The most
important invariants are: never use a test HMC bank for VAE prior generation,
keep N=664/batch=8/seed explicit in comparisons, record checkpoint hashes, and
do not compare full-set metrics with averages of 8-vs-8 metric batches.
