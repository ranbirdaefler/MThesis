#!/usr/bin/env bash
# Freeze canonical Tier-2 support for the admitted step-41301 Gemma checkpoint.
# Run this script inside srun on defq; it performs no login-node computation.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

REPO="${REPO:-$HOME/tahoe}"
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
MODEL_ID=vandijklab/C2S-Scale-Gemma-2-2B
REVISION=5ddf28b8f1c81b7ab7a9be192924da82b6c5d512
GEMMA_CHECKPOINT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell/checkpoint-41301-mb660816
PYTHIA_CHECKPOINT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
SCRAM=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big_scram
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
FP="$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
ADMIT="$REPO/endcell/jobs/gemma2_partial_checkpoint_admission.py"
RUNTIME="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_RUNTIME_CONTRACT.json"

cd "$REPO"
source "$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh"
"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL_ID" \
    --revision "$REVISION" --verify-current-environment --runtime-contract-out "$RUNTIME"

GEMMA_FP=$("$PY" "$ADMIT" --checkpoint "$GEMMA_CHECKPOINT" \
    --runtime-contract "$RUNTIME" --fingerprint-tool "$FP" \
    --out "$REPO/RESULTS/gemma2_partial_checkpoint_admission.json" --digest-only)
PYTHIA_FP=$("$PY" "$FP" --checkpoint "$PYTHIA_CHECKPOINT" --digest_only)

mapfile -t BINDING < <("$PY" - "$RUNTIME" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("snapshot_path", "preflight_certificate_sha256", "snapshot_inventory_sha256",
            "authoritative_revision_files_sha256"):
    print(d[key])
PY
)
[[ "${#BINDING[@]}" -eq 4 ]]
GEMMA_PARENT="${BINDING[0]}"
PREFLIGHT_SHA256="${BINDING[1]}"
SNAPSHOT_SHA256="${BINDING[2]}"
AUTHORITATIVE_SHA256="${BINDING[3]}"

PYTHIA_PARENT=$("$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    "vandijklab/C2S-Scale-Pythia-1b-pt",
    revision="830e4689d4238bbb5e2ec9a89b76e6a6d48061db",
    cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True))
PY
)
GEMMA_PARENT_FP=$("$PY" "$FP" --checkpoint "$GEMMA_PARENT" --require_gemma_parent \
    --expected_model_id "$MODEL_ID" --expected_revision "$REVISION" \
    --expected_parent_snapshot "$GEMMA_PARENT" \
    --expected_preflight_certificate_sha256 "$PREFLIGHT_SHA256" \
    --expected_snapshot_inventory_sha256 "$SNAPSHOT_SHA256" \
    --expected_authoritative_files_sha256 "$AUTHORITATIVE_SHA256" --digest_only)
PYTHIA_PARENT_FP=$("$PY" "$FP" --checkpoint "$PYTHIA_PARENT" --digest_only)

"$PY" endcell/analysis/freeze_nir_manifest.py \
    --eval_dir "$DATA" --scram_dir "$SCRAM" --train_file "$DATA/train.jsonl" \
    --tier tier2_unseen_drugs --k_samples 8 --min_cells 8 \
    --min_drugs_per_group 3 --max_groups 80 --same_plate_only --seed 42 \
    --generation_contract_version nir-generation-v1 \
    --expected_rows 606 --expected_drugs 35 --expected_cell_lines 40 --expected_groups 80 \
    --model_fingerprint "$GEMMA_FP" --model_fingerprint "$PYTHIA_FP" \
    --validity_only_parent "gemma_parent=$GEMMA_PARENT_FP" \
    --validity_only_parent "pythia_parent=$PYTHIA_PARENT_FP" \
    --hash_named --out RESULTS/gemma2_standard_tier2_manifest.json \
    | tee RESULTS/gemma2_standard_manifest_freeze.log

echo "[PASS] partial-checkpoint Tier-2 manifest frozen"
