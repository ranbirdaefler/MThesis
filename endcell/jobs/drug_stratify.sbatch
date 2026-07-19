#!/bin/bash
#SBATCH --job-name=drug_stratify
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/drug_stratify_%j.out
# TESTS 2 + 3 — per-drug stratified NIR, and drug confusion geometry.
#   (1) nir_benchmark (PATCHED: now also stores per-drug rows + per-drug pseudobulk profiles).
#       Aggregate numbers are computed exactly as before — this only adds output.
#   (2) drug_stratify_geometry: decomposes the aggregate by per-drug difficulty (Test 2) and
#       compares the real vs predicted drug x drug geometry (Test 3).
# Run drug_atlas.sbatch first (CPU, in parallel is fine) so --atlas can join effect size/isolation.
#
# NOTE: paths are FLAT (~/tahoe/<script>.py) to match the cluster layout.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
SCRAM=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big_scram
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final

# (1) NIR benchmark: per-drug rows + profiles + the SCRAMBLE arm (same control, wrong drug token)
#     and the CONTROL-COPY leakage baseline. Scramble doubles the generation, so this is ~2x slower.
$PY nir_benchmark.py \
    --eval_dir "$DATA" --scram_dir "$SCRAM" --model_path "$CKPT" --train_file "$DATA/train.jsonl" \
    --tiers tier2_unseen_drugs,tier3_unseen_combos,tier4_dose_interpolation \
    --k_samples 8 --temperature 0.8 --top_p 0.9 \
    --min_cells 20 --min_drugs_per_cl 6 --n_celllines 20 --bf16 \
    --out RESULTS/nir_benchmark_perdrug.json \
    --profiles RESULTS/nir_profiles.npz \
    --seed 42

# (2) Tests 2 + 3 (CPU, seconds)
$PY drug_stratify_geometry.py \
    --nir RESULTS/nir_benchmark_perdrug.json \
    --profiles RESULTS/nir_profiles.npz \
    --atlas RESULTS/drug_atlas.json \
    --metric nir_expr --identifiable_nir 0.8 \
    --out RESULTS/stratify_geometry.json \
    --csv RESULTS/per_drug.csv

echo "done -> RESULTS/stratify_geometry.json (+ per_drug.csv)"
