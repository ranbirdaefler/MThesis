#!/bin/bash
#SBATCH --job-name=sameplate_final
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sameplate_final_%j.out
# THE definitive, leak-free model benchmark. Replaces every contaminated model number in one job.
#
# Drug and PLATE are confounded by the experimental design (each drug sits on its own plate) and
# controls are plate-matched, so cross-plate comparisons let batch identity stand in for drug
# identity (a zero-drug-info control scored 0.551 cross-plate vs 0.510 same-plate). Every arm below
# is scored with the comparison set restricted to the SAME (cell_line, plate), which makes that leak
# structurally impossible.
#
# ARMS: ceiling | model | scramble | linear | control | mean
#   control ~0.50 is the CLEANLINESS CHECK — if it is above chance, the run is still leaking and
#   nothing below it can be trusted.
#   model - scramble  = causal drug use  (leak-immune: identical control, only the drug token differs)
#   model - linear    = now measured leak-free (this is what replaces the contaminated +0.141)
#
# min_cells 8 / min_drugs_per_cl 3: same-plate groups hold ~3-12 drugs and ~8-20 cells per drug.
# The same-plate ceiling at n=8 is ~0.579 (headroom +0.079) and is still RISING with cells, so treat
# these as a lower bound on the achievable bar, not the final target.

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

# (1) all arms, same-plate, per-drug rows + profiles
$PY nir_benchmark.py \
    --same_plate_only \
    --eval_dir "$DATA" --scram_dir "$SCRAM" --model_path "$CKPT" \
    --train_file "$DATA/train.jsonl" \
    --tiers tier2_unseen_drugs \
    --k_samples 8 --temperature 0.8 --top_p 0.9 \
    --min_cells 8 --min_drugs_per_cl 3 --max_groups 80 --bf16 \
    --out RESULTS/nir_sameplate.json \
    --profiles RESULTS/nir_sameplate_profiles.npz \
    --seed 42

# (2) stratified decomposition + clustered CIs + scramble/control verdicts (CPU, seconds)
$PY drug_stratify_geometry.py \
    --nir RESULTS/nir_sameplate.json \
    --profiles RESULTS/nir_sameplate_profiles.npz \
    --metric nir_expr --identifiable_nir 0.8 \
    --out RESULTS/stratify_sameplate.json \
    --csv RESULTS/per_drug_sameplate.csv

echo "done -> RESULTS/stratify_sameplate.json"
