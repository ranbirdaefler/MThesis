#!/bin/bash
#SBATCH --job-name=spikein_sp
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/spikein_sp_%j.out
# WITHIN-PLATE spike-in ONLY (the cross-plate number is already reproduced on record:
# panel_tau 0.991 / de_delta 0.949 / topn_tau 0.921 at spike=0). Running only the same-plate half
# so the whole 12h budget is for the one number we still need. Reads jsonl, no streaming.
# n_celllines 60 keeps it well inside the wall.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big

$PY spikein_metric_benchmark.py \
    --data_dir "$DATA" --sources train,eval_tier1_seen_conditions --same_plate_only \
    --sample_size 15 --n_trials 200 --spike_fracs 0,0.3,1.0 \
    --n_celllines 60 --drug_pairs_per_cellline 20 --de_k 50 --topn 100 \
    --modes tail_max --n_boot 200 --seed 42 \
    --out RESULTS/spikein_sameplate.json

echo "done -> RESULTS/spikein_sameplate.json"
