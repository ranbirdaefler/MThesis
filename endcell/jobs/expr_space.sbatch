#!/bin/bash
#SBATCH --job-name=expr_space
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/expr_space_%j.out
# CPU streaming job: pulls TRUE expression from Tahoe-100M and runs the
# rank-vs-expression discrimination test. No GPU needed.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)

# Call the env's Python directly — robust whether or not `conda` is on PATH.
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big

# The test is model-free and data-level, so held-out-ness is irrelevant — we stream
# a broad shard sample. Add --held_out_drugs_file $DATA/held_out_drugs.json to restrict
# to tier2 (unseen) drugs if you want strict comparability with the grading runs.
$PY endcell/analysis/expression_space_discrimination.py \
    --out RESULTS/expr_space_discrimination.json \
    --panel_file $DATA/l1000_panel.json \
    --num_shards 8 --rows_per_shard 150000 --shard_seed 7 \
    --cells_per_drug 40 --min_cells_per_drug 30 \
    --n_celllines 30 --drugs_per_cellline 25 \
    --pb_sizes 1,15 --n_trials 200 --n_boot 1000 --n_perm 2000 \
    --collect_controls --seed 42
