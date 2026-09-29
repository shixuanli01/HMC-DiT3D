#!/usr/bin/env bash
set -euo pipefail

if ! command -v gdown >/dev/null 2>&1; then
  python -m pip install gdown
fi

mkdir -p checkpoints/dit checkpoints/vae

gdown 1CKOuCABUwbi4eKckN0Y5WGcPMNs722L5 -O checkpoints/dit/chair.pt
gdown 1G7eEv-9CoogeAfLDLP7V2A-qmSnSEz_Y -O checkpoints/dit/airplane.pt
gdown 1NAgscvTFaD2DerHgJuYx8UFfMxCkMnAz -O checkpoints/dit/car.pt

gdown 1uvpnwo2gcGhsz35SrsuZo155NGdwNeLO -O checkpoints/vae/chair.pt
gdown 1pzSX2oBW649YA-zN8NqNkU7h3vfuGD9F -O checkpoints/vae/airplane.pt
gdown 1o4i7IiJyamGW-1LGyg2_j7D1H9QxZ5y1 -O checkpoints/vae/car.pt

sha256sum -c CHECKPOINTS.sha256
