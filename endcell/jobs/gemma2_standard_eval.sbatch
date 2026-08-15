#!/bin/bash
#SBATCH --job-name=gemma2_std_eval
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemma2_standard_eval_%j.out

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
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
SCRAM=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big_scram
MODEL_ID=vandijklab/C2S-Scale-Gemma-2-2B
REVISION=5ddf28b8f1c81b7ab7a9be192924da82b6c5d512
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
CHECKPOINT_ROLE="${CHECKPOINT_ROLE:-gemma_sft}"
MODEL="${MODEL_PATH:-/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell/final}"
VALIDITY="$REPO/endcell/eval/evaluate_endcell.py"
NIR="$REPO/endcell/analysis/nir_benchmark.py"
FINGERPRINT_TOOL="$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
MANIFEST="${TIER2_MANIFEST:?Set TIER2_MANIFEST to the immutable hash-named manifest from freeze_nir_manifest.py}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1800}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
case "$CHECKPOINT_ROLE" in
    gemma_sft) EXPECTED_MODEL_LABEL=gemma; EXPECTED_MODEL_TYPE=gemma2 ;;
    gemma_parent) EXPECTED_MODEL_LABEL=gemma_parent; EXPECTED_MODEL_TYPE=gemma2 ;;
    pythia_sft_legacy) EXPECTED_MODEL_LABEL=pythia; EXPECTED_MODEL_TYPE=gpt_neox ;;
    pythia_parent) EXPECTED_MODEL_LABEL=pythia_parent; EXPECTED_MODEL_TYPE=gpt_neox ;;
    *) echo "[FATAL] invalid CHECKPOINT_ROLE: $CHECKPOINT_ROLE" >&2; exit 2 ;;
esac
if [[ -n "${MODEL_LABEL:-}" && "$MODEL_LABEL" != "$EXPECTED_MODEL_LABEL" ]]; then
    echo "[FATAL] CHECKPOINT_ROLE=$CHECKPOINT_ROLE requires MODEL_LABEL=$EXPECTED_MODEL_LABEL; got $MODEL_LABEL" >&2
    exit 2
fi
MODEL_LABEL="$EXPECTED_MODEL_LABEL"
OUT="$REPO/RESULTS/gemma2_standard_eval/$MODEL_LABEL"
mkdir -p "$REPO/logs" "$OUT"
cd "$REPO"

RUNTIME_CONTRACT="${SLURM_TMPDIR:-/tmp}/gemma2-preflight-runtime-${SLURM_JOB_ID}.json"
"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL_ID" \
    --revision "$REVISION" --require-generation-cap "$MAX_NEW_TOKENS" \
    --verify-current-environment --runtime-contract-out "$RUNTIME_CONTRACT"
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
[[ "${#RUNTIME_VALUES[@]}" -eq 4 ]] || { echo "[FATAL] incomplete runtime contract" >&2; exit 20; }
SNAPSHOT_PATH="${RUNTIME_VALUES[0]}"
PREFLIGHT_SHA256="${RUNTIME_VALUES[1]}"
SNAPSHOT_INVENTORY_SHA256="${RUNTIME_VALUES[2]}"
AUTHORITATIVE_FILES_SHA256="${RUNTIME_VALUES[3]}"
[[ -d "$MODEL" && -f "$MODEL/config.json" ]] || {
    echo "[FATAL] local offline checkpoint directory missing: $MODEL" >&2; exit 20;
}
"$PY" - "$MODEL/config.json" "$CHECKPOINT_ROLE" "$EXPECTED_MODEL_TYPE" <<'PY'
import json, sys
config_path, role, expected = sys.argv[1:]
document = json.load(open(config_path, encoding="utf-8"))
observed = document.get("model_type")
if observed != expected:
    raise SystemExit(
        f"[FATAL] CHECKPOINT_ROLE={role} requires config.model_type={expected!r}; "
        f"observed={observed!r}")
print(f"[PASS] role/label/family bound: {role} -> {expected}")
PY
for required in "$VALIDITY" "$NIR" "$FINGERPRINT_TOOL" "$MANIFEST"; do
    [[ -f "$required" ]] || { echo "[FATAL] required evaluation artifact missing: $required" >&2; exit 20; }
done

# Production evaluators are fixed repo entrypoints.  Bind the exact paths selected
# for execution to the already verified preflight source hashes so inherited shell
# variables or a same-name replacement cannot redirect the scientific readout.
"$PY" - "$CERT" "$VALIDITY" "$NIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

certificate_path, validity_path, nir_path = sys.argv[1:]
certificate = json.load(open(certificate_path, encoding="utf-8"))

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for name, selected in (("evaluate_endcell", validity_path), ("nir_benchmark", nir_path)):
    entry = certificate.get("sources", {}).get(name, {})
    selected_resolved = str(Path(selected).resolve())
    certified_resolved = str(Path(entry.get("path", "")).resolve())
    if selected_resolved != certified_resolved:
        raise SystemExit(
            f"[FATAL] production evaluator path differs from preflight sources.{name}: "
            f"selected={selected_resolved}, certified={certified_resolved}")
    observed = sha256_file(selected_resolved)
    if observed != entry.get("sha256"):
        raise SystemExit(
            f"[FATAL] production evaluator hash differs from preflight sources.{name}: "
            f"observed={observed}, certified={entry.get('sha256')}")
print("[PASS] fixed production evaluator entrypoints match preflight path and SHA-256")
PY

FINGERPRINT_ARGS=()
if [[ "$CHECKPOINT_ROLE" == gemma_sft ]]; then
    FINGERPRINT_ARGS=(
        --require_complete_sft --expected_global_step 42198 --expected_epoch 1
        --expected_microbatch_position 0 --expected_accumulation_position 0
        --require_gemma_ancestry
        --expected_model_id "$MODEL_ID" --expected_revision "$REVISION"
        --expected_parent_snapshot "$SNAPSHOT_PATH"
        --expected_preflight_certificate_sha256 "$PREFLIGHT_SHA256"
        --expected_snapshot_inventory_sha256 "$SNAPSHOT_INVENTORY_SHA256"
        --expected_authoritative_files_sha256 "$AUTHORITATIVE_FILES_SHA256"
    )
elif [[ "$CHECKPOINT_ROLE" == gemma_parent ]]; then
    FINGERPRINT_ARGS=(
        --require_gemma_parent
        --expected_model_id "$MODEL_ID" --expected_revision "$REVISION"
        --expected_parent_snapshot "$SNAPSHOT_PATH"
        --expected_preflight_certificate_sha256 "$PREFLIGHT_SHA256"
        --expected_snapshot_inventory_sha256 "$SNAPSHOT_INVENTORY_SHA256"
        --expected_authoritative_files_sha256 "$AUTHORITATIVE_FILES_SHA256"
    )
elif [[ "$CHECKPOINT_ROLE" == pythia_sft_legacy ]]; then
    [[ "${ALLOW_LEGACY_PYTHIA_CHECKPOINT:-0}" == 1 ]] || {
        echo "[FATAL] Pythia lacks modern training_state; set ALLOW_LEGACY_PYTHIA_CHECKPOINT=1 explicitly" >&2
        exit 20
    }
    echo "[EXEMPTION] legacy Pythia SFT checkpoint has no completed training_state.pt"
fi

VALIDITY_HELP="$($PY "$VALIDITY" --help 2>&1)"
NIR_HELP="$($PY "$NIR" --help 2>&1)"
for flag in --generation_contract --model_fingerprint --prediction_cache --invalid_policy \
    --manifest_sha256 --parent_validity_only --attention_implementation; do
    grep -Fq -- "$flag" <<<"$VALIDITY_HELP" || {
        echo "[FATAL] validity evaluator lacks repaired flag $flag" >&2; exit 21;
    }
done
for flag in --manifest --prediction_cache --model_fingerprint --generation_contract_version \
    --invalid_policy --parent_validity_only --attention_implementation; do
    grep -Fq -- "$flag" <<<"$NIR_HELP" || {
        echo "[FATAL] NIR evaluator lacks repaired flag $flag" >&2; exit 21;
    }
done

"$PY" "$VALIDITY" --selftest --out "$OUT/validity_selftest.json"
"$PY" "$NIR" --selftest --out "$OUT/nir_selftest.json"
FINGERPRINT_TMP="$OUT/.${MODEL_LABEL}_checkpoint_fingerprint.${SLURM_JOB_ID}.tmp.json"
MODEL_FINGERPRINT="$($PY "$FINGERPRINT_TOOL" --checkpoint "$MODEL" \
    "${FINGERPRINT_ARGS[@]}" --out "$FINGERPRINT_TMP" --digest_only)"
[[ "$MODEL_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[FATAL] invalid checkpoint fingerprint: $MODEL_FINGERPRINT" >&2; exit 22;
}
if [[ "$CHECKPOINT_ROLE" == gemma_sft ]]; then
    echo "[PASS] Gemma manifest/content, terminal state, and certified ancestry validated"
elif [[ "$CHECKPOINT_ROLE" == gemma_parent ]]; then
    echo "[PASS] Gemma frozen-parent path and snapshot inventory validated"
fi
mv "$FINGERPRINT_TMP" "$OUT/${MODEL_LABEL}_checkpoint_fingerprint.json"
MANIFEST_SHA256="$($PY - "$MANIFEST" <<"PY"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])
PY
)"

"$PY" - "$MANIFEST" "$MODEL_FINGERPRINT" <<"PY"
import json, re, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"rows": 606, "drugs": 35, "cell_lines": 40, "groups": 80}
if manifest.get("support") != expected:
    raise SystemExit(f"[FATAL] manifest support {manifest.get('support')} != {expected}")
match = re.search(r"-([0-9a-f]{64})\.json$", Path(sys.argv[1]).name)
if not match or match.group(1) != manifest.get("manifest_sha256"):
    raise SystemExit("[FATAL] TIER2_MANIFEST is not the immutable hash-named artifact")
declared = set(manifest.get("model_fingerprints", []))
parents = {x.get("fingerprint") for x in manifest.get("model_declarations", [])}
if sys.argv[2] not in declared | parents:
    raise SystemExit("[FATAL] checkpoint fingerprint was not declared before Tier-2 output inspection")
print("[PASS] exact canonical support and predeclared checkpoint fingerprint")
PY

PARENT_ARGS=()
if [[ "$CHECKPOINT_ROLE" == gemma_parent || "$CHECKPOINT_ROLE" == pythia_parent ]]; then
    PARENT_ARGS=(--parent_validity_only)
fi

echo "=== greedy validity gate: native termination only ==="
"$PY" "$VALIDITY" --mode model --eval_dir "$DATA" --model_path "$MODEL" \
    --tiers tier2_unseen_drugs --max_eval 400 --max_new_tokens "$MAX_NEW_TOKENS" \
    --gen_batch_size "$GEN_BATCH_SIZE" --bf16 --seed 42 \
    --attention_implementation eager \
    --generation_contract deterministic --generation_contract_version endcell-validity-v1 \
    --model_fingerprint "$MODEL_FINGERPRINT" --manifest_sha256 "$MANIFEST_SHA256" \
    --prediction_cache "$OUT/${MODEL_LABEL}_greedy_cache.jsonl" \
    --invalid_policy validity_failure --min_complete_rate 0.95 --min_recognized_gene_rate 0.95 \
    "${PARENT_ARGS[@]}" --out "$OUT/${MODEL_LABEL}_greedy_validity.json"

set +e
"$PY" - "$OUT/${MODEL_LABEL}_greedy_validity.json" "$CHECKPOINT_ROLE" <<"PY"
import json, sys
document = json.load(open(sys.argv[1], encoding="utf-8"))
role = sys.argv[2]
result = document.get("model", {}).get("tier2_unseen_drugs")
if not result:
    if document.get("status") == "validity_failure" and role.endswith("_parent"):
        raise SystemExit(10)
    raise SystemExit(20)
if result.get("status") == "validity_failure" or not result.get("validity_gate", {}).get("passed"):
    raise SystemExit(10 if role.endswith("_parent") else 20)
raise SystemExit(0)
PY
VALIDITY_RC=$?
set -e
if [[ "$VALIDITY_RC" -eq 10 ]]; then
    echo "[EXPECTED VALIDITY FAILURE] frozen parent emitted no valid cell sentence; NIR is not computed"
    exit 0
elif [[ "$VALIDITY_RC" -ne 0 ]]; then
    echo "[FATAL] trained checkpoint failed greedy validity" >&2
    exit "$VALIDITY_RC"
fi

echo "=== canonical Tier 2: frozen rows and per-row/per-draw seeds ==="
"$PY" "$NIR" --same_plate_only --eval_dir "$DATA" --scram_dir "$SCRAM" \
    --model_path "$MODEL" --train_file "$DATA/train.jsonl" --tiers tier2_unseen_drugs \
    --k_samples 8 --temperature 0.8 --top_p 0.9 --max_new_tokens "$MAX_NEW_TOKENS" \
    --gen_batch_size "$GEN_BATCH_SIZE" --min_cells 8 --min_drugs_per_cl 3 --max_groups 80 \
    --bf16 --seed 42 --attention_implementation eager \
    --manifest "$MANIFEST" --prediction_cache "$OUT/${MODEL_LABEL}_tier2_cache.jsonl" \
    --model_fingerprint "$MODEL_FINGERPRINT" --generation_contract_version nir-generation-v1 \
    "${PARENT_ARGS[@]}" \
    --invalid_policy validity_failure --min_complete_rate 0.95 \
    --min_recognized_gene_rate 0.95 --min_recognized_genes 20 \
    --out "$OUT/${MODEL_LABEL}_tier2_nir.json" --profiles "$OUT/${MODEL_LABEL}_tier2_profiles.npz"

"$PY" - "$OUT/${MODEL_LABEL}_tier2_nir.json" "$CHECKPOINT_ROLE" <<"PY"
import json, sys
document = json.load(open(sys.argv[1], encoding="utf-8"))
if sys.argv[2].endswith("_parent"):
    if document.get("status") != "validity_failure" or document.get("nir_computed") is not False:
        raise SystemExit("[FATAL] validity-only parent unexpectedly produced NIR")
    print("[PASS] validity-only parent recorded without NIR")
    raise SystemExit(0)
if document.get("status") != "ok":
    raise SystemExit(f"[FATAL] sampled NIR status is {document.get('status')!r}, expected 'ok'")
if document.get("nir_computed") is False:
    raise SystemExit("[FATAL] NIR artifact reports nir_computed=false")
print("[PASS] sampled NIR status=ok")
PY
echo "[PASS] $MODEL_LABEL validity and canonical Tier-2 evaluation completed"
