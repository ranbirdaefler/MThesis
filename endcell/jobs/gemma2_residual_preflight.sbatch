#!/bin/bash
# =============================================================================
# gemma2_residual_preflight.sbatch — Stage 3: THE MEASUREMENT STAGE (CPU only)
# =============================================================================
#
# WHAT THIS BUYS
#   Every failure it catches -- response truncation, unknown tokens, a sequence
#   over 8192, a missing sentinel -- would otherwise surface inside
#   C2SDataset.__getitem__ hours into an H200 allocation, because
#   --strict_token_contract raises at the offending ROW, not at startup. This is
#   that failure moved to a 30-50 minute CPU job.
#
# MODEL PROVENANCE VS DATA PROVENANCE
#   It calls the EXISTING standard `verify` subcommand WITHOUT
#   --require-generation-cap. verify then still checks snapshot inventory,
#   authoritative revision bytes, the protobuf tree, pip freeze, tokenizer
#   256000/256002 and sentinels 256000/256001 -- everything about the MODEL --
#   while 1800 and the five canonical data digests stay bound to the standard data.
#   Residual DATA is certified separately by gemma2_residual_data_contract.py.
#   NO constant in the standard contract changes, and
#   tests/test_gemma_phase1a_training_contract.py stays green.
#
#   --load-source is the exact resolved snapshot path and never the HF model id.
#   Loading the tokenizer by model id on this cluster once resolved a broken
#   5-token vocabulary.
#
# SUBMIT
#     cd "$HOME/tahoe"
#     sbatch endcell/jobs/gemma2_residual_preflight.sbatch
#
# PREREQUISITES (both must have exited 0)
#     RESULTS/gemma2_residual/RESOLVED_TGT.txt   <- Stage 1
#     RESULTS/gemma2_residual/row_audit.json     <- Stage 2
#
# WRITES
#     RESULTS/gemma2_residual/tokenizer_probe_residual.json
#     RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json
#
# THREE LOAD-BEARING NUMBERS ARE PRINTED AT THE END
#     max_response_tokens        replaces the standard target's 1699
#     residual_generation_cap    replaces 1800; feeds the Stage 6 eval
#     max_total_sequence_tokens  must be < 8192 or --max_length has to change,
#                                which changes the contract and hence the run
# =============================================================================

#SBATCH --job-name=gemma2_res_preflight
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gemma2_residual_preflight_%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
REPO="${REPO:-$HOME/tahoe}"
source "$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh"
MODEL=vandijklab/C2S-Scale-Gemma-2-2B
REVISION=5ddf28b8f1c81b7ab7a9be192924da82b6c5d512
PROBE="$REPO/endcell/train/gemma_tokenizer_probe.py"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
RESIDUAL_CONTRACT="$REPO/endcell/jobs/gemma2_residual_data_contract.py"
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
RESULT="$REPO/RESULTS/gemma2_residual"
PROBE_OUT="$RESULT/tokenizer_probe_residual.json"
RESIDUAL_CERT="$RESULT/RESIDUAL_DATA_PASSED.json"
mkdir -p "$REPO/logs" "$RESULT"
cd "$REPO"

[[ "${SLURM_JOB_PARTITION:-}" == "defq" ]] || {
    echo "[FATAL] submitted to '${SLURM_JOB_PARTITION:-unknown}', expected 'defq'." >&2
    echo "        This stage needs no GPU; do not spend a gpuh200 allocation on it." >&2
    exit 2
}

TGT_FILE="$RESULT/RESOLVED_TGT.txt"
[[ -f "$TGT_FILE" ]] || {
    echo "[FATAL] Stage 1 has not resolved a residual build: $TGT_FILE" >&2
    echo "        srun -p defq -c 2 --mem 8G bash endcell/jobs/gemma2_residual_inventory.sh" >&2
    exit 2
}
TGT="$(head -n 1 "$TGT_FILE")"
[[ -d "$TGT" ]] || { echo "[FATAL] resolved target dir does not exist: $TGT" >&2; exit 2; }
TRAIN_JSONL="$TGT/residual.jsonl"
VAL_JSONL="$TGT/residual_val.jsonl"
for f in "$TRAIN_JSONL" "$VAL_JSONL"; do
    [[ -f "$f" ]] || { echo "[FATAL] missing $f" >&2; exit 2; }
done
ROW_AUDIT="$RESULT/row_audit.json"
[[ -f "$ROW_AUDIT" ]] || {
    echo "[FATAL] Stage 2 row audit missing: $ROW_AUDIT" >&2
    echo "        Run gemma2_residual_data_audit.py under srun first." >&2
    exit 2
}
echo "[info] residual target: $TGT"

# A failed run must never leave a stale authorization at the canonical path.
if [[ -f "$RESIDUAL_CERT" ]]; then
    mv "$RESIDUAL_CERT" \
       "$RESULT/RESIDUAL_DATA_PASSED.previous.$(date -u +%Y%m%dT%H%M%SZ).json"
fi

# ---- MODEL provenance only. Note the absence of --require-generation-cap. ----
RUNTIME_CONTRACT="${SLURM_TMPDIR:-/tmp}/gemma2-residual-runtime-${SLURM_JOB_ID}.json"
"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL" \
    --revision "$REVISION" --verify-current-environment \
    --runtime-contract-out "$RUNTIME_CONTRACT"
mapfile -t RUNTIME_VALUES < <("$PY" - "$RUNTIME_CONTRACT" <<'PY'
import json, sys
document = json.load(open(sys.argv[1], encoding="utf-8"))
if document.get("schema_version") != 2:
    raise SystemExit("[FATAL] invalid preflight runtime-contract schema")
for key in ("snapshot_path", "preflight_certificate_sha256", "snapshot_inventory_sha256",
            "authoritative_revision_files_sha256"):
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"[FATAL] runtime contract lacks {key}")
    print(value)
PY
)
[[ "${#RUNTIME_VALUES[@]}" -eq 4 ]] || { echo "[FATAL] incomplete runtime contract" >&2; exit 2; }
SNAPSHOT_PATH="${RUNTIME_VALUES[0]}"
[[ -d "$SNAPSHOT_PATH" ]] || {
    echo "[FATAL] certificate returned no exact Gemma snapshot: $SNAPSHOT_PATH" >&2; exit 2;
}
echo "[PASS] model provenance verified; snapshot: $SNAPSHOT_PATH"
echo "[PASS] generation cap NOT required from the standard certificate; the residual"
echo "       cap is measured below rather than carried over from 1800."

# ---- Tokenize every row of both residual files under the run's own tokenizer --
# --max_examples 0 means every row. The probe registers both sentinels itself, so
# it measures the same 256002-length tokenizer the run will train under.
#
# --revision and --load-source are MODEL=VALUE MAPS, not scalars. The probe keys
# them by model name (gemma_tokenizer_probe.py:465 does revisions.get(model) and
# load_sources.get(model)) and _parse_revisions/_parse_load_sources raise
# ValueError on any item without '='. A bare value therefore dies at argument
# parse, before a single row is tokenized. This is the identical form used by the
# working standard preflight, gemma2_standard_preflight.sh:213.
"$PY" "$PROBE" \
    --model "$MODEL" \
    --revision "$MODEL=$REVISION" \
    --load-source "$MODEL=$SNAPSHOT_PATH" \
    --jsonl "$TRAIN_JSONL" \
    --jsonl "$VAL_JSONL" \
    --max_examples 0 \
    --parity_examples 512 \
    --max_length 8192 \
    --require_fast \
    --output "$PROBE_OUT" \
  > "$RESULT/tokenizer_probe_residual.stdout.log"
echo "[PASS] tokenizer probe complete -> $PROBE_OUT"

# ---- Residual-scoped validation, cap derivation, certificate -----------------
"$PY" "$RESIDUAL_CONTRACT" create \
    --target_dir "$TGT" \
    --probe "$PROBE_OUT" \
    --row_audit "$ROW_AUDIT" \
    --standard-certificate "$CERT" \
    --snapshot-path "$SNAPSHOT_PATH" \
    --model-id "$MODEL" \
    --revision "$REVISION" \
    --batch_size 1 --grad_accum 16 --num_epochs 1 --max_length 8192 \
    --output "$RESIDUAL_CERT"

echo
echo "[next] Stage 4: sbatch endcell/jobs/gemma2_residual_smoke.sbatch"
