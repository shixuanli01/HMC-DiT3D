# Checkpoints

| File | Role | Training classes | HMC scales / q | Checkpoint | SHA-256 |
|---|---|---|---|---|---|
| `checkpoints/dit/airplane_norm_epoch8000.pt` | Airplane DiT | airplane | {3,4,5} / {1,2,3} | epoch 8000 of `configs/retrain_airplane_norm_bs512.yaml` | `49e1dbf5aed0d93c907af91e6652f5ce098182e530c463c6baaa2932eb26ec4c` |

**Download:** <ADD-URL-HERE>

The checkpoint was trained with the normalized recipe (batch size 512, 10,000 epochs, AMP, one H100) and stores
the train-split point normalization it was trained with (`point_normalization`: per-axis mean, scalar std), which
the sampler undoes automatically; no size correction is applied. Verify with:

```bash
echo "49e1dbf5aed0d93c907af91e6652f5ce098182e530c463c6baaa2932eb26ec4c  checkpoints/dit/airplane_norm_epoch8000.pt" | sha256sum -c
```

Ours-VAE additionally needs the airplane condition VAE (beta 0.05, latent 64, trained for 2000 epochs on the
airplane train bank); train it with `python -m hmc_dit3d.train.train_condition_vae` as described in the README
and place it at `checkpoints/vae/airplane.pt`.
