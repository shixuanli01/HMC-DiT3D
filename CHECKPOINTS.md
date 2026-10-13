# Checkpoints

| File | Role | Training classes | HMC scales / q | Checkpoint | SHA-256 |
|---|---|---|---|---|---|
| `checkpoints/dit/chair345_norm_epoch10000.pt` | Chair DiT | chair | {3,4,5} / {1,2,3} | epoch 10000 of `configs/retrain_chair_norm_bs512.yaml` | `58b9eae2cfdd07ae375dc2806dd579d8b7ee458e506eeba7d273a75d789d6452` |
| `checkpoints/dit/airplane_norm_epoch8000.pt` | Airplane DiT | airplane | {3,4,5} / {1,2,3} | epoch 8000 of `configs/retrain_airplane_norm_bs512.yaml` | `49e1dbf5aed0d93c907af91e6652f5ce098182e530c463c6baaa2932eb26ec4c` |

**Download:** <ADD-URL-HERE>

Both DiTs were trained with the normalized recipe (batch size 512, 10,000 epochs, AMP, one H100) and store
the train-split point normalization they were trained with (`point_normalization`: per-axis mean, scalar std),
which the sampler undoes automatically; no size correction is applied. Verify with:

```bash
sha256sum -c - <<'EOF'
58b9eae2cfdd07ae375dc2806dd579d8b7ee458e506eeba7d273a75d789d6452  checkpoints/dit/chair345_norm_epoch10000.pt
49e1dbf5aed0d93c907af91e6652f5ce098182e530c463c6baaa2932eb26ec4c  checkpoints/dit/airplane_norm_epoch8000.pt
EOF
```

The condition VAEs are not released: Ours-VAE needs the category's condition VAE (beta 0.05, latent 64,
2000 epochs on the category's train bank, seed 42), which trains in a few minutes with
`python -m hmc_dit3d.train.train_condition_vae` as described in the README ("Condition VAE"); place them at
`checkpoints/vae/chair.pt` and `checkpoints/vae/airplane.pt`.
