#!/bin/bash
#SBATCH --job-name=drug_lookup
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/drug_lookup_%j.out
# ============================================================================================
# A-03 DRUG-SPECIFIC BASELINE — the yardstick the model must beat.
#
# Adds `drug_lookup` = control pseudobulk + the drug's TRAIN-ONLY mean displacement
# (per drug x cell_line; pooled-drug fallback = cross-context transfer). This is the simplest
# predictor that actually USES drug identity. Our standing claim is "the LLM adds no measurable
# skill over drug-AGNOSTIC baselines" — this asks the sharper question: does it beat a drug LOOKUP?
#
# TIER LOGIC (the baseline is only legitimate where the drug is in train):
#   tier1 seen conditions      -> lookup fully legitimate (drug x cell_line seen)
#   tier3 unseen combos        -> drug seen, new cell line  -> pooled-drug displacement = transfer test
#   tier4 dose interpolation   -> drug seen at other doses
#   tier2 unseen drugs         -> NO lookup possible (reported NA) — itself an informative bound:
#                                 without chemical features, no drug-specific baseline exists there.
#
# SPARSITY: Q12 found tier1 within-plate NIR yields n=0 at default thresholds (~2.4 cells per
# (cell_line,plate,drug)). We relax --min_cells/--min_drugs_per_cl and let the log report which
# tiers actually produce scorable groups. If within-plate yields nothing anywhere, rerun WITHOUT
# --same_plate_only and read control-copy as the leakage indicator (it is scored alongside).
#
# LEAK GUARD: displacements are fitted from train.jsonl ONLY; eval tiers are separate files, so no
# evaluation response can contribute to any fitted delta (the auditor's explicit requirement).
# ============================================================================================
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
SCRAM=${DATA}_scram
MODEL=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final   # canonical single-cell

echo "=== A-03 drug-specific lookup baseline vs the model (within-plate) ==="
$PY nir_benchmark.py --same_plate_only \
    --eval_dir "$DATA" --scram_dir "$SCRAM" --model_path "$MODEL" \
    --train_file "$DATA/train.jsonl" \
    --drug_lookup --lookup_min_cells 5 \
    --tiers tier1_seen_conditions,tier3_unseen_combos,tier4_dose_interpolation,tier2_unseen_drugs \
    --k_samples 8 --temperature 0.8 --top_p 0.9 \
    --min_cells 4 --min_drugs_per_cl 3 --max_groups 200 --bf16 --seed 42 \
    --out RESULTS/nir_drug_lookup.json --profiles RESULTS/nir_drug_lookup_profiles.npz

echo "=== per-drug decomposition (which drugs the lookup beats the model on) ==="
$PY drug_stratify_geometry.py --nir RESULTS/nir_drug_lookup.json \
    --profiles RESULTS/nir_drug_lookup_profiles.npz --metric nir_expr --identifiable_nir 0.8 \
    --out RESULTS/stratify_drug_lookup.json --csv RESULTS/per_drug_lookup.csv

echo "done. KEY COMPARISON = model vs drug_lookup on the SEEN-drug tiers (tier1/3/4)."
