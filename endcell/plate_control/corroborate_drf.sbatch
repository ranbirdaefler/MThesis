#!/bin/bash
#SBATCH --job-name=corrob_drf
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/corrob_drf_%j.out
# Corroborate the DRF metric-calibration claim (NIR +0.80) WITHIN-PLATE only.
# Cross-plate +0.80 is already on record, so we don't re-stream it — the whole budget goes to the
# within-plate number. Streams true expression from Tahoe (~150 cells/drug), interpolated-duplicate
# positive control, full metric suite; groups by (cell_line, plate) so NIR's other-drug comparison
# set and the leave-one-out negative control hold the plate signature constant.
#
# Same-plate groups hold ~4-12 drugs -> --min_drugs_per_cl 3, --n_celllines 200 (now # plate-groups).
# HF streaming needs the token + xet disabled. Flat paths for the cluster layout.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs

# High cell count: the 12-shard run only reached ~53 cells/drug (same-plate thins the pool) and
# NIR-DRF rises with cells, so we stream more (32 shards x 250k rows) to reach ~120 cells/drug for
# the DEFINITIVE within-plate magnitude.
$PY calibration_eval.py \
    --same_plate_only \
    --num_shards 32 --rows_per_shard 250000 --cells_per_drug 150 --min_cells_per_drug 40 \
    --n_celllines 300 --min_drugs_per_cl 3 --de_k 50 --seed 42 \
    --out RESULTS/drf_sameplate_highn.json

echo "done -> RESULTS/drf_sameplate_highn.json"
