#!/bin/bash
#SBATCH --job-name=ot_eval
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ot_eval_%j.out
# Leak-immune two-axis eval of an OT arm (ARM=T0/T1/T2). Same protocol as arm1a_eval_consensus:
#   within-plate NIR + SCRAMBLE (same control, wrong drug token) + control-copy, vs the FIXED
#   single-cell linear baseline. Decisive drug-use number = model - scramble (clustered CI).
#   Pick arm:  ARM=T2 sbatch ot_eval.sbatch
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
ARM=${ARM:-T2}
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_ot_${ARM}/final
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
SCRAM=${DATA}_scram
test -d "$CKPT" || { echo "[FATAL] $CKPT not found"; exit 1; }

COMMON="--same_plate_only --eval_dir $DATA --scram_dir $SCRAM --train_file $DATA/train.jsonl \
  --tiers tier1_seen_conditions,tier2_unseen_drugs \
  --k_samples 8 --temperature 0.8 --top_p 0.9 \
  --min_cells 8 --min_drugs_per_cl 3 --max_groups 80 --bf16 --seed 42"

# ---- (0) VALIDITY GATE: does the OT model still emit sane cell sentences? --------------------
echo "=== [0] validity gate (ARM=$ARM) ==="
$PY evaluate_endcell.py --mode model \
    --eval_dir "$DATA" --model_path "$CKPT" \
    --tiers tier1_seen_conditions --max_eval 200 --max_new_tokens 1200 --bf16 \
    --out RESULTS/ot_${ARM}_validity.json --seed 42
echo "^^ check emits_end_cell ~0.99, sane len_ratio, hallucination ~0 before trusting NIR below"

# ---- (1) NIR + scramble + control-copy on the OT model --------------------------------------
echo "=== [1] NIR + scramble + control (ARM=$ARM) ==="
$PY nir_benchmark.py $COMMON --model_path "$CKPT" \
    --out RESULTS/nir_ot_${ARM}.json --profiles RESULTS/nir_ot_${ARM}_profiles.npz

# ---- (2) stratified decomposition + causal verdict ------------------------------------------
echo "=== [2] stratified decomposition + verdict (ARM=$ARM) ==="
$PY drug_stratify_geometry.py --nir RESULTS/nir_ot_${ARM}.json \
    --profiles RESULTS/nir_ot_${ARM}_profiles.npz --metric nir_expr --identifiable_nir 0.8 \
    --out RESULTS/stratify_ot_${ARM}.json --csv RESULTS/per_drug_ot_${ARM}.csv

echo "done (ARM=$ARM). PRIMARY NUMBER = 'model - scramble' in RESULTS/stratify_ot_${ARM}.json"
