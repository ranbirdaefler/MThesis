#!/bin/bash
#SBATCH --job-name=drug_atlas
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/drug_atlas_%j.out
# TEST 1 — drug difficulty atlas (NO model, NO streaming; pure CPU, fast).
# Answers the advisor: do all drugs induce a substantial change, and are some drugs near-identical?
# Per drug x cell line: effect size vs DMSO, replicate noise, SNR, nearest-other-drug distance,
# isolation, and the PER-DRUG ceiling NIR (the distribution hiding behind the aggregate 0.88).
#
# NOTE: paths are FLAT (~/tahoe/<script>.py) to match the cluster layout.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big

$PY drug_difficulty_atlas.py \
    --eval_dir "$DATA" \
    --tiers tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos,tier4_dose_interpolation \
    --min_cells 8 --n_celllines 30 --identifiable_nir 0.8 \
    --out RESULTS/drug_atlas.json \
    --csv RESULTS/drug_atlas.csv \
    --profiles RESULTS/drug_atlas_profiles.npz \
    --seed 42

echo "done -> RESULTS/drug_atlas.json (+ .csv, + profiles.npz)"
