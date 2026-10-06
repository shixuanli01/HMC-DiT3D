#!/bin/bash
# score any chair_seed_sweep payload that has a .pt but no val json
cd /fsx/weicyang/shix/HMC-DiT3D/hmc_dit3d; O=results/chair_seed_sweep
export CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=/fsx/weicyang/shix/emd_build:src
for p in $O/*.pt; do T=$(basename $p .pt); [ -f $O/val_$T.json ] || ../.venv/bin/python scripts/diag/val_eval_mmd.py --perms 3 --out $O/val_$T.json $p >> $O/$T.log 2>&1; done
