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

## Training pipeline (perf/fast-training)

- HMC features are extracted inside DataLoader workers by `HMCBatchCollator`
  (`train/smoke.py`); the training loop reuses them through
  `resolve_hmc_batch` and falls back to in-process extraction when a batch has
  no precomputed features. Extraction is deterministic, so tensors are
  identical to the old `prepare_hmc_batch` path.
- The validation loader is materialized once per run on device
  (`_cache_eval_batches` in `train/train.py`); it is not re-read or
  re-extracted every epoch.
- Checkpoints are written by `AsyncCheckpointWriter` on a background thread.
  A pending write to the same path is replaced by the newer snapshot, so only
  the newest `best_train.pt` / `latest.pt` state is written. Uniquely named
  `epoch_N.pt` files are never dropped. `writer.close()` flushes at exit.
- `train.deterministic` (default `true`) controls `set_global_seed`. When
  `false`, seeds are still set but cuDNN autotuning and flash / memory-efficient
  attention are allowed. Runs with `deterministic: false` are not bit-for-bit
  reproducible across reruns; say so when reporting them.
- The `retrain_*_h100_formal_bottleneck_bs512.yaml` configs are the formal
  bottleneck configs with `num_workers: 16`, `latest_every: 10`,
  `auto_resume: true`, and `deterministic: false`. Use them only on nodes with
  enough CPU cores for 16 workers.
- No before/after throughput measurement is recorded. Metrics files carry no
  timestamps, so speedups must be measured with a fresh timed run.

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
7. The `trainbank` prior draws real train-split HMC conditions at random. It
   is not a learned generative prior; label it as such in any report.
8. `diag_gen.py --mode testref` feeds paired test conditions to the DiT. It is
   an oracle diagnostic only and must never be reported as a result.
9. Two evaluation protocols are in use. The formal one is the test-split
   N=664 / batch-8 protocol above. The val protocol (`scripts/diag/`) uses the
   val split as reference with points `npy[10000:12048]`, independent 8-vs-8
   groups, 3 random permutations averaged, exact CUDA EMD, and adds MMD.
   Numbers from the two protocols are not comparable.

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
- Val-protocol findings (2026-10-04, details in `PROGRESS_2026-10-04.md`):
  - Chair, 10 seeds each, mean COV-CD / COV-EMD: trainbank prior about
    56.6 / 56.7 with 1-NNA near 49-50; N(0,I) VAE prior about 53.9 / 53.6;
    GMM prior (K=32 on train posterior means) about 54.1 / 53.8. The
    `epoch7000` and `epoch8000` bs512 retrain checkpoints match the released
    checkpoint within seed noise.
  - Airplane: vae prior 62.04 / 54.21 / 58.67 / 60.08, trainbank
    58.08 / 54.21 / 60.08 / 59.83. Car: vae 61.98 / 52.51 / 46.69 / 53.03,
    trainbank 58.38 / 50.47 / 51.33 / 56.82 (seed 0, released checkpoints).
  - GMM sampling closes the N(0,I) prior hole (|z| about 1.95, matching the
    posterior means) but does not raise COV. Conclusion: the VAE decoder, not
    the prior, costs about 3 COV points on Chair.
- Open next steps: (1) confirm by feeding encode-decode reconstructions of
  train conditions to the DiT and checking COV drops to about 54; (2) retrain
  the VAE with lower beta and pair it with the GMM prior; (3) bypass the
  decoder by generating directly in condition space (kNN interpolation or a
  small flow / diffusion prior). The runbook for (1) and (2), with commands
  and acceptance criteria, is `NEXT_STEPS_VAE_RETRAIN.md`. The DiT stays
  fixed at `checkpoints/dit/{cat}.pt` for that work; bs512 retraining of the
  DiT did not beat it.

Metric order above is always: 1-NNA-CD, 1-NNA-EMD, COV-CD, COV-EMD.

## Important code paths

- HMC extraction: `hmc_dit3d/src/hmc_dit3d/hmc/`
- DiT model: `hmc_dit3d/src/hmc_dit3d/models/hmc_dit.py`
- Train/sample/evaluate: `hmc_dit3d/src/hmc_dit3d/train/`
- Condition VAE: `hmc_dit3d/src/hmc_dit3d/hmc/condition_vae.py`
- TopoDiT CD evaluator: `hmc_dit3d/scripts/eval_topodit_protocol.py`
- Active configs: `hmc_dit3d/configs/train_*_h100_formal_bottleneck.yaml`
- bs512 retrain configs:
  `hmc_dit3d/configs/retrain_*_h100_formal_bottleneck_bs512.yaml`
- Epoch sweep under the formal protocol:
  `hmc_dit3d/scripts/sweep_retrain_epochs.py` (one worker per GPU, resumable)
- Val-protocol diagnostics: `hmc_dit3d/scripts/diag/` (`diag_gen.py` with
  `--mode vae|trainbank|gmm|trainrecon|testref`, `gmm_prior.py`,
  `val_eval_mmd.py`,
  `chair_seed_sweep.sh`, `rescore_missing.sh`, `summarize_chair_sweep.py`).
  These scripts hard-code the cluster paths `/fsx/weicyang/shix/...` and the
  EMD build at `/fsx/weicyang/shix/emd_build`; edit them before use elsewhere.

## Formal multi-seed queue

- Runner: `hmc_dit3d/scripts/run_multiseed_n664.py`
- Phase 1 seeds: `0,1,2,3,4`
- Phase 2 seeds: `5,6,7,8,9,10,11,12,13,42`
- Each seed evaluates Chair, Airplane, and Car with N=664 and generation
  batch size 8 using the stable beta=0.05, latent-64 VAE checkpoints.
- Results: `hmc_dit3d/results/vae_prior_multiseed_n664_batch8/`
- The runner is resumable: rerun the same command from `README.md`; valid
  sample and metric payloads are checked and skipped instead of overwritten.

## Safe first actions

1. Download data/checkpoints as described in `README.md`.
2. Verify hashes in `CHECKPOINTS.md`.
3. Run the test suite.
4. Reproduce one N=64 sample/evaluation before spending time on N=664.
5. Never overwrite an existing result payload; use a new directory tagged with
   VAE, seed, N, and batch size.
