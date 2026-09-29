# Agent handoff

## Objective

Improve ShapeNet Chair/Airplane/Car coverage while keeping 1-NNA near 50 under
the TopoDiT-style N=664, generation-batch=8 protocol. The current limitation is
COV, not merely reconstruction quality.

## Active model

- 3D DiT-S/4-scale point-cloud diffusion model.
- HMC condition = multi-scale voxel measures serialized by a 3D Hilbert curve,
  plus generalized-dimension descriptors.
- Bottleneck/resampler fusion is the active architecture.
- A train-split HMC condition VAE supplies an unconditional prior.
- The stable baseline VAE is beta 0.05, latent 64, hidden widths
  2048/1024/512, 2,000 epochs, temperature 1.2, threshold 0.1.

## Non-negotiable evaluation rules

1. Never generate from a test HMC bank. VAE training/fitting uses train only.
2. Report category, sample count, random seed, generation batch size, metric
   grouping, checkpoint SHA-256, VAE SHA-256, temperature, and threshold.
3. The target formal run is 664 samples and generation batch size 8.
4. 1-NNA is best near 50%; COV is maximized.
5. TopoDiT paper numbers average metrics computed independently on 8-vs-8
   batches. Full 664-vs-664 numbers are a different protocol.
6. Built-in EMD is Sinkhorn. Do not label it exact TopoDiT EMD; exact EMD needs
   the TopoDiT CUDA approxmatch extension.

## Current evidence

- Seed-0 official-style baseline:
  - Chair: 50.53 / 48.49 / 51.36 / 50.90
  - Airplane: 54.22 / 52.41 / 47.14 / 50.30
  - Car: 57.98 / 49.02 / 46.99 / 52.11
- A beta=0.02, z=64 wider Airplane VAE looked better in one N=64 matched-seed
  probe but degraded badly at N=664 for seed 42.
- A beta=0.005, z=128 VAE also degraded N=664 coverage. Better reconstruction
  alone does not imply a usable standard-normal prior or higher point-cloud COV.
- Before adopting any new VAE, run matched seeds with the same DiT and test
  references. Prefer a five-seed screen, then N=664 confirmation.

Metric order above is always: 1-NNA-CD, 1-NNA-EMD, COV-CD, COV-EMD.

## Important code paths

- HMC extraction: `hmc_dit3d/src/hmc_dit3d/hmc/`
- DiT model: `hmc_dit3d/src/hmc_dit3d/models/hmc_dit.py`
- Train/sample/evaluate: `hmc_dit3d/src/hmc_dit3d/train/`
- Condition VAE: `hmc_dit3d/src/hmc_dit3d/hmc/condition_vae.py`
- TopoDiT CD evaluator: `hmc_dit3d/scripts/eval_topodit_protocol.py`
- Active configs: `hmc_dit3d/configs/train_*_h100_formal_bottleneck.yaml`

## Safe first actions

1. Download data/checkpoints as described in `README.md`.
2. Verify hashes in `CHECKPOINTS.md`.
3. Run the test suite.
4. Reproduce one N=64 sample/evaluation before spending time on N=664.
5. Never overwrite an existing result payload; use a new directory tagged with
   VAE, seed, N, and batch size.

