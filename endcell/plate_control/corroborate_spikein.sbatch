#!/bin/bash
#SBATCH --job-name=corrob_spikein
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/corrob_spikein_%j.out
# Corroborate the spike-in discrimination claim (~0.93-0.99) under within-plate comparison sets.
# Reads jsonl (train+tier1) — NO streaming. The previous run died on time because it used 3 modes x
# 7 spike fracs x 400 trials x 1000-boot x 160 groups. This is a LEAN config aimed at the one number
# that matters: discrimination at spike=0. One absent-gene convention (tail_max = Federico's; the
# discrimination number is convention-robust to <=0.004), 3 spike points, lighter trials/bootstrap.
#
# Runs cross-plate first (reproduce the number on record) then within-plate (the corroborated one).
# Flat paths for the cluster layout.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big

COMMON="--data_dir $DATA --sources train,eval_tier1_seen_conditions \
  --sample_size 15 --n_trials 200 --spike_fracs 0,0.3,1.0 \
  --drug_pairs_per_cellline 20 --de_k 50 --topn 100 \
  --modes tail_max --n_boot 200 --seed 42"

# (A) cross-plate — reproduce the ~0.95-0.99 on record
$PY spikein_metric_benchmark.py $COMMON --n_celllines 40 \
    --out RESULTS/spikein_crossplate.json

# (B) within-plate — the corroborated number (quote this)
$PY spikein_metric_benchmark.py $COMMON --same_plate_only --n_celllines 80 \
    --out RESULTS/spikein_sameplate.json

echo "done -> RESULTS/spikein_{crossplate,sameplate}.json"
