# HMC-DiT3D

**Hilbert-multifractal conditioning for 3D point-cloud diffusion transformers.**

HMC-DiT3D generates 2,048-point ShapeNet shapes with a DiT-S/4-scale denoiser conditioned on a
*Hilbert-multifractal condition* (HMC): the point cloud is voxelised at several dyadic scales, the per-voxel
measures are serialised along a 3D Hilbert curve into one sequence per scale, and generalised dimensions
D(q) summarise the multifractal statistics. A windowed transformer encodes the sequences into condition
tokens, which enter the denoiser through a bottleneck (Perceiver-style) resampler. At inference the condition
comes either from a small condition VAE (unpaired prior, **Ours-VAE**) or from a bank of real train-split
conditions (**Ours-Bank**, a non-parametric condition library).

```text
.
├── README.md
├── CHECKPOINTS.md            released checkpoint, hash and download link
├── src/hmc_dit3d/            the package
│   ├── hmc/                  HMC extraction (extractor.py, config.py), condition encoder, condition VAE
│   ├── models/hmc_dit.py     DiT-3D backbone with bottleneck-resampler fusion
│   ├── data/                 ShapeNet loader, train-condition bank, global point normalisation
│   ├── train/                trainer (resume / EMA / AMP), diffusion schedule, samplers, VAE training
│   └── metrics/              exact-EMD batch-8 metrics
├── configs/                  one training config per category, plus the two multi-class configs
├── scripts/                  generate.py, evaluate_val_protocol.py, evaluate.sh
└── tests/                    pytest suite
```

## 1. Installation

Python 3.12 and a CUDA build of PyTorch >= 2.4 (tested: torch 2.11 + CUDA 12.8 on an H100 80 GB).

```bash
git clone git@github.com:shixuanli01/HMC-DiT3D.git && cd HMC-DiT3D
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # choose the wheel matching your driver
pip install -e ".[dev]"
python -m pytest -q
```

### Exact EMD

The reported metrics use TopoDiT-3D's CUDA `approxmatch` EMD. Build it once and export `EMD_BUILD`:

```bash
git clone https://github.com/Zechao-Guan/TopoDiT-3D.git /tmp/TopoDiT-3D
cd /tmp/TopoDiT-3D/metrics/PyTorchEMD
# three source fixes for current PyTorch (the math is unchanged)
sed -i 's|#include <THC/THC.h>|#include <c10/cuda/CUDAException.h>|; s|THCudaCheck(cudaGetLastError());|C10_CUDA_CHECK(cudaGetLastError());|; s|x.type().is_cuda()|x.is_cuda()|' cuda/emd_kernel.cu
sed -i 's|CHECK_EQ(\(.*\), \(.*\));|TORCH_CHECK((\1) == (\2));|' cuda/emd_kernel.cu
TORCH_CUDA_ARCH_LIST="9.0" python setup.py build_ext --build-lib /path/to/emd_build   # H100: 9.0, B200: 10.0, RTX 5090: 12.0, A100: 8.0
export EMD_BUILD=/path/to/emd_build
```

The built-in `python -m hmc_dit3d.train.evaluate` uses a Sinkhorn approximation whose EMD values are several
times larger than the exact ones; it is a quick check only and must not be compared with published numbers.

## 2. Data

Download the PointFlow release of ShapeNetCore.v2.PC15k (about 7.7 GB) into the repository root:

```bash
pip install gdown
gdown 1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX -O ShapeNetCore.v2.PC15k.zip
unzip -q ShapeNetCore.v2.PC15k.zip && rm ShapeNetCore.v2.PC15k.zip
```

Expected layout: `HMC-DiT3D/ShapeNetCore.v2.PC15k/<synset>/{train,val,test}/*.npy`
(chair 4612/662/1317, airplane 2832/405/808, car 2458/352/704, lamp 1620/232/464, vessel 1356/194/388).
The configs resolve the data root relative to `configs/`, so no path editing is needed.

## 3. Checkpoint

See `CHECKPOINTS.md`. Put the airplane DiT at `checkpoints/dit/airplane_norm_epoch8000.pt`. Every checkpoint
carries the global point normalisation it was trained with, so sampling returns raw ShapeNet coordinates
without any post-processing.

## 4. Evaluate (validation protocol)

Reference = ShapeNet val split of the category (points `npy[10000:12048]`); N = 664 samples generated 166 at a
time in fp16; independent 8-vs-8 groups; three random permutations averaged; Chamfer distance and exact EMD.
1-NNA is best at 50 %, COV is better when higher. `scripts/evaluate.sh` runs the whole pipeline:

```bash
export EMD_BUILD=/path/to/emd_build          # SHAPENET_ROOT defaults to ./ShapeNetCore.v2.PC15k

# Ours-Bank: real train-split HMC conditions (seed 0)
scripts/evaluate.sh airplane configs/retrain_airplane_norm_bs512.yaml checkpoints/dit/airplane_norm_epoch8000.pt

# Ours-VAE: conditions sampled from the airplane condition VAE (seed 42)
scripts/evaluate.sh airplane configs/retrain_airplane_norm_bs512.yaml checkpoints/dit/airplane_norm_epoch8000.pt \
  --mode vae --vae checkpoints/vae/airplane.pt --seeds "42"
```

The script builds the train-condition bank on first use (`results/banks/airplane_train.pt`, train split only),
writes the payloads to `results/airplane/` and the scores to `results/airplane/val_main_<mode>.json`
(mean and std over the three permutations). Generation takes about 10 minutes per seed on an H100 and
scoring about 3 minutes. Expected numbers for this checkpoint (1-NNA-CD / 1-NNA-EMD / COV-CD / COV-EMD):

| Row | seed | expected |
|---|---|---|
| Ours-Bank | 0 | 54.79 / 52.83 / 57.92 / 58.58 |
| Ours-VAE | 42 | 56.79 / 53.25 / 58.17 / 59.25 |

Seed-to-seed standard deviations are about 1-1.5 points per metric; fp16 generation is non-deterministic, so
expect differences of a few tenths. Two reporting rules: Ours-Bank uses real train conditions and is labelled as
a non-parametric condition library, never as a learned prior; and no test-split condition is ever fed to the
generator.

## 5. Train

```bash
python -m hmc_dit3d.train.train --config configs/retrain_airplane_norm_bs512.yaml
```

The per-category configs (`configs/retrain_<category>_norm_bs512.yaml`) share one recipe: HMC scales {3,4,5}
with q in {1,2,3}, a resampler depth of 4, batch size 512, 10,000 epochs and AMP. The
denoising target is whitened with the train-split global mean and scalar std (`data.normalize_points: true`);
the statistics are computed at start-up and stored in every checkpoint. The trainer writes `latest.pt` every
10 epochs (used by `auto_resume`), `epoch_N.pt` every 1,000 epochs, `best_train.pt`, `best_val.pt` and the
final `train_checkpoint.pt`. One category takes 6-11 hours on a single H100 (about 52 GB of GPU memory; chair has the most steps per epoch). The per-epoch validation loss is a noisy single-timestep estimate that saturates long
before sample quality does, so select checkpoints by sampling rather than by `best_val`.

### Condition VAE (for Ours-VAE)

```bash
python -m hmc_dit3d.train.build_bank --config configs/retrain_airplane_norm_bs512.yaml --split train --output results/banks/airplane_train.pt
python -m hmc_dit3d.train.train_condition_vae --bank results/banks/airplane_train.pt \
  --output checkpoints/vae/airplane.pt --latent-dim 64 --hidden-dims 2048,1024,512 --beta 0.05 --epochs 2000 --device cuda
```

### Multi-class training

One model can be trained on several categories at once; the class label is the index of the category in
`data.categories` (chair 0, car 1, airplane 2) and enters the DiT through its class embedding, while the HMC
condition is still extracted per shape. Two configs are provided, with the same recipe as the single-category ones:

```bash
python -m hmc_dit3d.train.train --config configs/retrain_chaircar_norm_bs512.yaml           # {chair, car},            7,070 shapes
python -m hmc_dit3d.train.train --config configs/retrain_chaircarairplane_norm_bs512.yaml   # {chair, car, airplane},  9,902 shapes
```

Each run needs about 52 GB of GPU memory, so the two runs fit together on one B200 (or one each on two H100s) but
not on a single 80 GB card. Alone on an H100 they take roughly 16 and 24 hours; running both at once on one B200
takes about 18-22 hours in total. HMC features are extracted on the CPU by the dataloader workers, so give each run
`num_workers: 16` when the machine has 32 or more cores (the configs default to 8). The normalisation statistics are computed over the union of the training classes.

To test a multi-class checkpoint on one class, draw that class's conditions from its **single-category** bank
and pass `--category` so that the labels and reference shapes are restricted to it:

```bash
# chair bank from the chair config (same HMC settings), then evaluate the 3-class model on chair
python -m hmc_dit3d.train.build_bank --config configs/retrain_chair_norm_bs512.yaml --split train --output results/banks/chair_train.pt
scripts/evaluate.sh chair configs/retrain_chaircarairplane_norm_bs512.yaml results/retrain_chaircarairplane_norm_bs512/epoch_10000.pt \
  --category chair --bank results/banks/chair_train.pt --seeds "0 11 12 13" --tag mc3
scripts/evaluate.sh chair configs/retrain_chaircarairplane_norm_bs512.yaml results/retrain_chaircarairplane_norm_bs512/epoch_10000.pt \
  --category chair --bank results/banks/chair_train.pt --mode vae --vae checkpoints/vae/chair.pt --seeds "0 11 12 13" --tag mc3
```

`checkpoints/vae/chair.pt` is the chair condition VAE trained on the chair bank (see "Condition VAE" above).

## 6. Scripts

- `scripts/generate.py` - 664 samples from a checkpoint with `--mode vae | trainbank`; `--fast` for fp16,
  `--category` for multi-class checkpoints.
- `scripts/evaluate_val_protocol.py` - the validation-protocol scorer (batched exact EMD, cached reference matrices).
- `scripts/evaluate.sh` - bank build + generation + scoring for one checkpoint; `--category` and `--bank` for multi-class checkpoints.

## Citation

<add the paper citation here>
