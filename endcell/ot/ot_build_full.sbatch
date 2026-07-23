#!/bin/bash
#SBATCH --job-name=ot_build
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/ot_build_%j.out
# STAGE 1/3: full panel-expression + PCA cache (treated + dedicated DMSO control pass). c2s env.
set -euo pipefail
mkdir -p logs
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
$PY build_embeddings.py --panel l1000_panel.json \
    --num_shards 80 --cells_per_condition 100 --max_cells 600000 \
    --control_shards 220 --control_per_group 150 --pca_dim 50 \
    --out_dir /data/BuffaF-Projetcs/florian_c2s/ot_cache
echo "STAGE 1 done -> /data/BuffaF-Projetcs/florian_c2s/ot_cache"
