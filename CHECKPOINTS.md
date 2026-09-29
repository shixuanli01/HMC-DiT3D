# Checkpoints

The six files below are the exact DiT and condition-VAE weights used for the
current seed-0 N=664 result table. Google Drive file IDs will be filled in after
the upload is verified.

| Destination | Role | Size (bytes) | SHA-256 | Google Drive ID |
|---|---|---:|---|---|
| `checkpoints/dit/chair.pt` | Chair DiT, best train, epoch 9429 | 570033994 | `14585893d01a9925b2dfa6945712228ea7edfb8abcfbc843d4cc75cb46106b9f` | `PENDING` |
| `checkpoints/dit/airplane.pt` | Airplane DiT, best val, epoch 9463 | 612613533 | `c6b09816c7b37e31956782a6159689b259fec4bc26b5584a9d66e738a2afebde` | `PENDING` |
| `checkpoints/dit/car.pt` | Car DiT, best val, epoch 8597 | 612613533 | `ed3675825445f736554d0a6d1477e90b3b1bd3eae893cb28d0dbe38d4f306052` | `PENDING` |
| `checkpoints/vae/chair.pt` | Chair condition VAE, beta 0.05, z64 | 98002133 | `69b0e48767b86285c1667e09ec2dd7ad7c0b3e1fa023925de99db58e908de13d` | `PENDING` |
| `checkpoints/vae/airplane.pt` | Airplane condition VAE, beta 0.05, z64 | 633971829 | `f79b0d87c75c61d86c92a466c9461fbed3a67f137bb005b9caa552e536a62fd0` | `PENDING` |
| `checkpoints/vae/car.pt` | Car condition VAE, beta 0.05, z64 | 633971605 | `a75e26f0a4c3d66ff54ac571fdc54a8322ce61431605326a3709a8d735295bc9` | `PENDING` |

After downloading, verify all files from the repository root:

```bash
sha256sum checkpoints/dit/*.pt checkpoints/vae/*.pt
```

