# Checkpoints

| File | Role | Training classes | HMC scales / q | Checkpoint | SHA-256 |
|---|---|---|---|---|---|
| `checkpoints/dit/chair345_norm_epoch10000.pt` | Chair DiT | chair | {3,4,5} / {1,2,3} | epoch 10000 of `configs/retrain_chair_norm_bs512.yaml` | `58b9eae2cfdd07ae375dc2806dd579d8b7ee458e506eeba7d273a75d789d6452` |
| `checkpoints/dit/airplane_norm_epoch8000.pt` | Airplane DiT | airplane | {3,4,5} / {1,2,3} | epoch 8000 of `configs/retrain_airplane_norm_bs512.yaml` | `49e1dbf5aed0d93c907af91e6652f5ce098182e530c463c6baaa2932eb26ec4c` |
| `checkpoints/vae/chair345_b0.05_z64.pt` | Chair condition VAE | chair train bank | {3,4,5} / {1,2,3} | beta 0.05, latent 64, 2000 epochs, seed 42 | `ce461e4e1c4955bed12bea3639fc594f6705efd963b881be84685a3bf3ed5fbc` |

**Download:** <ADD-URL-HERE>

Both DiTs were trained with the normalized recipe (batch size 512, 10,000 epochs, AMP, one H100) and store
the train-split point normalization they were trained with (`point_normalization`: per-axis mean, scalar std),
which the sampler undoes automatically; no size correction is applied. Verify with:

```bash
sha256sum -c - <<'EOF'
58b9eae2cfdd07ae375dc2806dd579d8b7ee458e506eeba7d273a75d789d6452  checkpoints/dit/chair345_norm_epoch10000.pt
49e1dbf5aed0d93c907af91e6652f5ce098182e530c463c6baaa2932eb26ec4c  checkpoints/dit/airplane_norm_epoch8000.pt
ce461e4e1c4955bed12bea3639fc594f6705efd963b881be84685a3bf3ed5fbc  checkpoints/vae/chair345_b0.05_z64.pt
EOF
```

Ours-VAE for airplane additionally needs the airplane condition VAE (beta 0.05, latent 64, trained for 2000
epochs on the airplane train bank); train it with `python -m hmc_dit3d.train.train_condition_vae` as described
in the README and place it at `checkpoints/vae/airplane.pt`. The chair condition VAE is released above.
