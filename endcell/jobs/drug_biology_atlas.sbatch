#!/bin/bash
#SBATCH --job-name=drug_bio_atlas
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/drug_bio_atlas_%j.out
# The definitive drug-biology atlas answering the advisor's question, on CLEAN ground:
#   streams ~120 cells/drug from Tahoe (real effect-vs-noise), compares drugs only WITHIN the same
#   (cell_line, plate) (real drug signal, not batch). Reports, with proper statistics:
#     (a) potency vs plate-matched DMSO (permutation test + BH-FDR DEG count) -> which drugs are inert
#     (b) identifiability (same-plate ceiling NIR) + redundancy (isolation) -> which are distinguishable
#     (c) MoA structure  (d) drug x cell-line interaction  (e) dose design
#     (f) chemical-similarity gate (Arm-2 go/no-go; skips gracefully if rdkit absent)
#     (g) ceiling-NIR vs cells/drug sweep
# CPU streaming, 8 CPUs (QOS cap). Flat paths for the cluster layout.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs

$PY drug_biology_atlas.py \
    --num_shards 24 --rows_per_shard 250000 --cells_per_drug 120 \
    --min_cells 30 --min_drugs_per_group 3 --n_perm 200 --fdr_q 0.05 \
    --identifiable_nir 0.8 --cell_sweep 10,20,40,80,120 --chem_gate \
    --out RESULTS/drug_biology_atlas.json \
    --csv RESULTS/drug_biology_atlas.csv \
    --seed 42

echo "done -> RESULTS/drug_biology_atlas.json (+ .csv)"
