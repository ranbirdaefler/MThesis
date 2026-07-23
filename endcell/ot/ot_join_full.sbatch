#!/bin/bash
#SBATCH --job-name=ot_join
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ot_join_%j.out
# STAGE 2/3: attach the shipped scVI latent to the cache by barcode. scvi_venv (needs h5py).
set -euo pipefail
mkdir -p logs
PY=/data/BuffaF-Projetcs/florian_c2s/envs/scvi_venv/bin/python
cd ~/tahoe
$PY join_latent.py \
    --cache_dir /data/BuffaF-Projetcs/florian_c2s/ot_cache \
    --adata /data/BuffaF-Projetcs/florian_c2s/SCVI_MODEL/adata.h5ad
echo "STAGE 2 done -> /data/BuffaF-Projetcs/florian_c2s/ot_cache/scvi_latent.npz"
