#!/bin/bash
#SBATCH -N 1
#SBATCH -t 70:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output="airplane_%j.txt"
#SBATCH --error="airplane_%j.txt"

cd hmc_dit3d
micromamba run -n hmc-dit3d-py312 python -m hmc_dit3d.train.train --config configs/train_airplane_h100_formal_bottleneck.yaml