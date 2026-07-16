#!/bin/bash
#SBATCH --job-name=endcell_gpu
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/endcell_gpu_%j.out
# GPU-bound [END_CELL] work, sequential in one allocation:
#   (1) causal drug-steering probe   (~1h, high value -> runs FIRST so a later overrun can't lose it)
#   (2) standard eval (model) on tier1/2/3, both absent-gene conventions, DE-Δr sweep + panel-τ + validity + breakdowns
#   (3) scramble arm on tier1/2 -> DE-Δr(real) vs DE-Δr(scramble)

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
SCRAM=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big_scram
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final

# (1) causal probe
$PY endcell/analysis/causal_drug_probe.py \
    --eval_dir $DATA --model_path $CKPT --tier tier2_unseen_drugs \
    --n_contexts 4 --n_pairs 8 --layers 4,8,12 --scales 1,2,4 \
    --temperature 0.8 --top_p 0.9 --topn 100 --max_new_tokens 1200 --bf16 \
    --out RESULTS/causal_drug_probe.json --seed 42

# (2)+(3) standard eval (real) + scramble arm, both absent-gene conventions
# NOTE: confirm the tier3 filename in $DATA is eval_tier3_unseen_combos.jsonl; a missing tier is
# skipped with a warning (the job still completes tier1/tier2), so adjust --tiers if it differs.
$PY endcell/eval/evaluate_endcell.py --mode model,scramble \
    --eval_dir $DATA --scram_dir $SCRAM --model_path $CKPT \
    --tiers tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos \
    --scram_tiers tier1_seen_conditions,tier2_unseen_drugs \
    --max_eval 400 --max_new_tokens 1200 --bf16 \
    --out RESULTS/eval_endcell_model.json --seed 42
