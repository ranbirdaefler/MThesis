#!/bin/bash
#SBATCH --job-name=scramble_sweep
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scramble_sweep_%j.out
# ============================================================================================
# SCRAMBLE VALIDITY — is `model ~= scramble` real, or an artifact of swapping to SIMILAR drugs?
# Sweeps model - scramble as a function of the RESPONSE DISTANCE between the real and swapped drug.
#   flat ~0 across the FAR tail  -> drug-blindness confirmed at the hardest setting (claim HARDENED)
#   rises with distance          -> the model discriminates dissimilar drugs (claim CORRECTED)
#
# NOTE: no --same_plate_only. The scramble DIFFERENCE cancels control/batch leakage (both arms use the
# SAME control cell), so it is already leak-immune; grouping by cell line only gives the widest pool of
# swap-to drugs and the fullest distance range. Runs on the canonical single-cell model (whose
# drug-blindness we are validating) and the OT T2 model.
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
SINGLE=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final
T2=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_ot_T2/final

echo "=== [1] single-cell model (the foundational drug-blindness claim), tier2 unseen drugs ==="
$PY scramble_distance_sweep.py --eval_dir "$DATA" --model_path "$SINGLE" \
    --tier tier2_unseen_drugs --n_drugs 40 --n_bins 5 \
    --k_samples 8 --temperature 0.8 --top_p 0.9 --min_cells 8 --min_drugs_per_cl 4 --bf16 --seed 42 \
    --out RESULTS/scramble_sweep_single.json

echo "=== [2] OT T2 model, tier2 unseen drugs (does OT change the distance behaviour?) ==="
$PY scramble_distance_sweep.py --eval_dir "$DATA" --model_path "$T2" \
    --tier tier2_unseen_drugs --n_drugs 40 --n_bins 5 \
    --k_samples 8 --temperature 0.8 --top_p 0.9 --min_cells 8 --min_drugs_per_cl 4 --bf16 --seed 42 \
    --out RESULTS/scramble_sweep_T2.json

echo "done. KEY = the model-scramble-vs-distance curve + TREND corr in each log block."
