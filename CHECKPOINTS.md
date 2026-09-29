# Checkpoints

The six files below are the exact DiT and condition-VAE weights used for the
current seed-0 N=664 result table. The upload was verified byte-for-byte using
the file size and MD5 reported by Google Drive. The public folder is
[HMC-DiT3D-checkpoints](https://drive.google.com/drive/folders/1HUuCpNJ1QBtyVwzJLoWDmIWAngdpZfrp?usp=sharing).

| Destination | Role | Size (bytes) | SHA-256 | Google Drive ID |
|---|---|---:|---|---|
| `checkpoints/dit/chair.pt` | Chair DiT, best train, epoch 9429 | 570033994 | `14585893d01a9925b2dfa6945712228ea7edfb8abcfbc843d4cc75cb46106b9f` | `1CKOuCABUwbi4eKckN0Y5WGcPMNs722L5` |
| `checkpoints/dit/airplane.pt` | Airplane DiT, best val, epoch 9463 | 612613533 | `c6b09816c7b37e31956782a6159689b259fec4bc26b5584a9d66e738a2afebde` | `1G7eEv-9CoogeAfLDLP7V2A-qmSnSEz_Y` |
| `checkpoints/dit/car.pt` | Car DiT, best val, epoch 8597 | 612613533 | `ed3675825445f736554d0a6d1477e90b3b1bd3eae893cb28d0dbe38d4f306052` | `1NAgscvTFaD2DerHgJuYx8UFfMxCkMnAz` |
| `checkpoints/vae/chair.pt` | Chair condition VAE, beta 0.05, z64 | 98002133 | `69b0e48767b86285c1667e09ec2dd7ad7c0b3e1fa023925de99db58e908de13d` | `1uvpnwo2gcGhsz35SrsuZo155NGdwNeLO` |
| `checkpoints/vae/airplane.pt` | Airplane condition VAE, beta 0.05, z64 | 633971829 | `f79b0d87c75c61d86c92a466c9461fbed3a67f137bb005b9caa552e536a62fd0` | `1pzSX2oBW649YA-zN8NqNkU7h3vfuGD9F` |
| `checkpoints/vae/car.pt` | Car condition VAE, beta 0.05, z64 | 633971605 | `a75e26f0a4c3d66ff54ac571fdc54a8322ce61431605326a3709a8d735295bc9` | `1o4i7IiJyamGW-1LGyg2_j7D1H9QxZ5y1` |

The recommended download verifies every SHA-256 automatically:

```bash
bash download_checkpoints.sh
```

For a manual download, use `gdown FILE_ID -O DESTINATION`, taking `FILE_ID`
and `DESTINATION` from the table above. To verify files separately, run:

```bash
sha256sum -c CHECKPOINTS.sha256
```
