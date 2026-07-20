#!/bin/bash
#SBATCH --job-name=arm1a_eval
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/arm1a_eval_%j.out
# ============================================================================================
# ARM 1a EVALUATION — did denoised (consensus) targets teach the model to USE the drug?
#
# PRIMARY TEST (internal, needs no baseline and no matched training length):
#     model - scramble   on tier1 (seen drugs)
#   Same checkpoint, same control cell, ONLY the drug token changes -> training length, dataset
#   size and model identity all cancel. If this is > 0 with a clustered CI excluding zero, the
#   consensus model genuinely uses the drug.
#
# WHY THE TWO MODELS ARE COMPARABLE AT ALL:
#   nir_benchmark scores every model the same way — generate K samples from the held-out prompts,
#   pseudobulk the generations, compare that profile to the real pseudobulk truths. The single-cell
#   model contributes individually-predicted cells that are then pooled; the consensus model
#   contributes consensus-like cells that are then pooled. Identical output interface, identical
#   scoring. Only the TRAINING TARGET differs (the independent variable).
#
# CAVEATS THIS JOB IS DESIGNED AROUND:
#   * consensus stopped at ~25.5k/42k steps (~60% epoch) vs a full epoch for single-cell. If
#     consensus WINS, undertraining only strengthens it; if it ties, undertraining is a live
#     alternative explanation -> that is why the primary test is the internal one.
#   * we had NO tier1 numbers for the single-cell model, so it is re-run here on tier1+tier2 to give
#     a matched head-to-head on identical tiers/protocol.
#
# PROTOCOL (same as the validated sameplate_final run):
#   --same_plate_only  : plate/batch identity cannot stand in for drug identity
#   --scram_dir        : the decisive causal arm
#   control-copy       : automatic; the honest null (NOT 0.5)
#   --train_file       : the FIXED single-cell linear baseline, identical for both models
#   NOTE: no --exclude_manifest — strength/distinctiveness now come from TRAIN cells, which are
#   disjoint from eval by construction, so no eval cells need to be sacrificed.
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
CONSENSUS=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell_consensus/checkpoint-25500
SINGLE=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final

COMMON="--same_plate_only --eval_dir $DATA --scram_dir $SCRAM --train_file $DATA/train.jsonl \
  --tiers tier1_seen_conditions,tier2_unseen_drugs \
  --k_samples 8 --temperature 0.8 --top_p 0.9 \
  --min_cells 8 --min_drugs_per_cl 3 --max_groups 80 --bf16 --seed 42"

# ---- (0) VALIDITY GATE: is the consensus model still generating sane cell sentences? -----------
# If consensus training degenerated generation, every NIR number downstream is meaningless.
echo "=== [0] validity gate (consensus checkpoint) ==="
$PY evaluate_endcell.py --mode model \
    --eval_dir "$DATA" --model_path "$CONSENSUS" \
    --tiers tier1_seen_conditions --max_eval 200 --max_new_tokens 1200 --bf16 \
    --out RESULTS/consensus_validity.json --seed 42
echo "^^ check emits_end_cell ~0.99, sane len_ratio, hallucination ~0 before trusting anything below"

# ---- (1) CONSENSUS model: the primary arm --------------------------------------------------
echo "=== [1] consensus model — NIR + scramble + control ==="
$PY nir_benchmark.py $COMMON --model_path "$CONSENSUS" \
    --out RESULTS/nir_consensus.json --profiles RESULTS/nir_consensus_profiles.npz

# ---- (2) SINGLE-CELL model on the SAME tiers/protocol: the matched head-to-head -------------
echo "=== [2] single-cell model — same tiers, same protocol ==="
$PY nir_benchmark.py $COMMON --model_path "$SINGLE" \
    --out RESULTS/nir_singlecell_t1t2.json --profiles RESULTS/nir_singlecell_t1t2_profiles.npz

# ---- (3) clustered CIs + verdicts for both --------------------------------------------------
echo "=== [3] stratified decomposition + causal verdicts ==="
$PY drug_stratify_geometry.py --nir RESULTS/nir_consensus.json \
    --profiles RESULTS/nir_consensus_profiles.npz --metric nir_expr --identifiable_nir 0.8 \
    --out RESULTS/stratify_consensus.json --csv RESULTS/per_drug_consensus.csv
$PY drug_stratify_geometry.py --nir RESULTS/nir_singlecell_t1t2.json \
    --profiles RESULTS/nir_singlecell_t1t2_profiles.npz --metric nir_expr --identifiable_nir 0.8 \
    --out RESULTS/stratify_singlecell_t1t2.json --csv RESULTS/per_drug_singlecell_t1t2.csv

echo "done. PRIMARY NUMBER = 'model - scramble' on tier1 in RESULTS/stratify_consensus.json"
