#!/usr/bin/env bash
# Scheduler inspection may run on the login node; all model/data work runs in srun.
set -euo pipefail
unset PYTHONHOME

ACCOUNT="${ACCOUNT:-3180408}"
CPU_PARTITION="${CPU_PARTITION:-defq}"
CPU_TIME="${CPU_TIME:-06:00:00}"
GPU_PARTITION="${1:-${GPU_PARTITION:-gpuh200}}"
REPO="${REPO:-$HOME/tahoe}"
MODEL_ID="vandijklab/C2S-Scale-Gemma-2-2B"
REVISION="5ddf28b8f1c81b7ab7a9be192924da82b6c5d512"
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"

[[ "$GPU_PARTITION" == "gpuh200" ]] || {
    echo "[FATAL] canonical Gemma training partition is literal 'gpuh200', got '$GPU_PARTITION'" >&2
    exit 2
}

# Invalidate authorization before even inspecting the scheduler. Any failure in
# this attempt must leave no current certificate behind.
PREFLIGHT_OUT="$REPO/RESULTS/gemma2_standard_preflight"
PREFLIGHT_CERT="$PREFLIGHT_OUT/PREFLIGHT_PASSED.json"
PREFLIGHT_RUNTIME="$PREFLIGHT_OUT/PREFLIGHT_RUNTIME_CONTRACT.json"
mkdir -p "$PREFLIGHT_OUT"
if [[ -f "$PREFLIGHT_CERT" ]]; then
    mv "$PREFLIGHT_CERT" \
        "$PREFLIGHT_OUT/PREFLIGHT_PASSED.previous.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
fi
if [[ -f "$PREFLIGHT_RUNTIME" ]]; then
    mv "$PREFLIGHT_RUNTIME" \
        "$PREFLIGHT_OUT/PREFLIGHT_RUNTIME_CONTRACT.previous.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
fi

for command_name in scontrol sinfo srun sbatch; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "[FATAL] required scheduler command not found: $command_name" >&2
        exit 2
    }
done

echo "=== scheduler-only checks (login node) ==="
scontrol show partition "$GPU_PARTITION" || {
    echo "[FATAL] partition '$GPU_PARTITION' was not verified" >&2
    exit 2
}
PARTITION_VIEW="$(sinfo -p "$GPU_PARTITION" -N -h -o '%N|%l|%G|%t')"
printf '%s\n' "$PARTITION_VIEW"
grep -Eiq 'h200' <<<"$PARTITION_VIEW" || {
    echo "[FATAL] partition '$GPU_PARTITION' exposes no H200 GRES in sinfo" >&2
    exit 2
}
if sbatch --help 2>&1 | grep -Fq -- '--test-only'; then
    sbatch --test-only --partition="$GPU_PARTITION" \
        "$REPO/endcell/jobs/gemma2_standard_train.sbatch"
else
    echo "[WARN] sbatch --test-only unavailable; existence/GRES were verified, policy was not"
fi

export ACCOUNT CPU_PARTITION CPU_TIME GPU_PARTITION REPO MODEL_ID REVISION HF_HOME HF_HUB_CACHE
srun --account="$ACCOUNT" --partition="$CPU_PARTITION" --cpus-per-task=4 --mem=32G \
    --time="$CPU_TIME" --export=ALL bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
if [[ -r "$HOME/.hf_token" ]]; then export HF_TOKEN="$(<"$HOME/.hf_token")"; fi

PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
PROTOBUF_ENV="$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh"
source "$PROTOBUF_ENV"
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
OUT="$REPO/RESULTS/gemma2_standard_preflight"
CERT="$OUT/PREFLIGHT_PASSED.json"
TRAINER="$REPO/endcell/train/train_c2s_tahoe_endcell.py"
TOKENIZER_PROBE="$REPO/endcell/train/gemma_tokenizer_probe.py"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
CLI_CONTRACT="$REPO/endcell/jobs/gemma2_standard_cli_contract.py"
TEST_CONTRACT="$REPO/endcell/jobs/gemma2_standard_tests_contract.py"
TEST_CERT="$REPO/RESULTS/gemma2_standard_tests/TESTS_PASSED.json"
mkdir -p "$OUT" "$REPO/logs"
cd "$REPO"

command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock unavailable" >&2; exit 3; }
exec 8>"$OUT/.gemma2_standard_preflight.lock"
flock -n 8 || { echo "[FATAL] another Gemma preflight is active" >&2; exit 3; }

# Close the concurrent-attempt race: another preflight may have published after
# this attempt performed its required pre-scheduler invalidation.
if [[ -f "$CERT" ]]; then
    mv "$CERT" "$OUT/PREFLIGHT_PASSED.previous.postlock.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
fi

for required in "$PY" "$PROTOBUF_ENV" "$TRAINER" "$TOKENIZER_PROBE" "$CONTRACT" "$CLI_CONTRACT" \
    "$TEST_CONTRACT" "$TEST_CERT" \
    "$DATA/train.jsonl" "$DATA/eval_tier1_seen_conditions.jsonl" \
    "$DATA/eval_tier2_unseen_drugs.jsonl" "$DATA/eval_tier3_unseen_combos.jsonl" \
    "$DATA/eval_tier4_dose_interpolation.jsonl"; do
    [[ -e "$required" ]] || { echo "[FATAL] missing: $required" >&2; exit 3; }
done

echo "=== environment and disk ===" | tee "$OUT/environment.txt"
hostname | tee -a "$OUT/environment.txt"
if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$REPO" rev-parse HEAD | tee -a "$OUT/environment.txt"
else
    echo "source_bundle_git_commit=unavailable; provenance=certificate-bound-source-sha256" \
        | tee -a "$OUT/environment.txt"
fi
df -h /data/BuffaF-Projetcs/florian_c2s | tee -a "$OUT/environment.txt"
MIN_FREE_GB="${MIN_FREE_GB:-120}"
FREE_GB=$(df --output=avail -BG /data/BuffaF-Projetcs/florian_c2s | tail -1 | tr -dc "0-9")
[[ -n "$FREE_GB" && "$FREE_GB" -ge "$MIN_FREE_GB" ]] || {
    echo "[FATAL] ${FREE_GB:-unknown} GiB free; require at least $MIN_FREE_GB GiB" >&2
    exit 3
}
"$PY" - <<"PY" | tee -a "$OUT/environment.txt"
import platform, torch, transformers, tokenizers, huggingface_hub, google.protobuf
print("python", platform.python_version())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("protobuf", google.protobuf.__version__)
print("protobuf_file", google.protobuf.__file__)
print("cuda_available", torch.cuda.is_available())
PY
"$PY" -m pip freeze | LC_ALL=C sort > "$OUT/pip_freeze.txt"
"$PY" "$TEST_CONTRACT" verify --certificate "$TEST_CERT" \
    --verify-current-environment

echo "=== download and verify pinned full Gemma snapshot on defq ==="
SNAPSHOT_PATH=$("$PY" - "$MODEL_ID" "$REVISION" <<"PY"
import os, sys
from huggingface_hub import HfApi, snapshot_download
repo_id, revision = sys.argv[1:]
info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(repo_id, revision=revision)
if info.sha != revision:
    raise SystemExit(f"[FATAL] Hub resolved {revision} to unexpected commit {info.sha}")
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    cache_dir=os.environ["HF_HUB_CACHE"],
    token=os.environ.get("HF_TOKEN"),
)
print(path)
PY
)
[[ -d "$SNAPSHOT_PATH" ]] || { echo "[FATAL] snapshot missing: $SNAPSHOT_PATH" >&2; exit 4; }
printf "%s\n" "$SNAPSHOT_PATH" > "$OUT/model_snapshot_path.txt"

REQUIRED_MODEL_FILES=(
    config.json tokenizer.model tokenizer_config.json special_tokens_map.json
    generation_config.json model.safetensors.index.json
    model-00001-of-00002.safetensors model-00002-of-00002.safetensors
)
for file in "${REQUIRED_MODEL_FILES[@]}"; do
    [[ -s "$SNAPSHOT_PATH/$file" ]] || { echo "[FATAL] snapshot lacks $file" >&2; exit 4; }
done
"$PY" - "$SNAPSHOT_PATH/model.safetensors.index.json" <<"PY"
import json, sys
index = json.load(open(sys.argv[1], encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
expected = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
if shards != expected:
    raise SystemExit(f"[FATAL] unexpected model shard set: {shards}")
print("[PASS] model index names exactly two expected shards")
PY

echo "=== immutable data identities ==="
sha256sum \
    "$DATA/train.jsonl" "$DATA/eval_tier1_seen_conditions.jsonl" \
    "$DATA/eval_tier2_unseen_drugs.jsonl" "$DATA/eval_tier3_unseen_combos.jsonl" \
    "$DATA/eval_tier4_dose_interpolation.jsonl" | tee "$OUT/data_sha256.txt"
wc -l \
    "$DATA/train.jsonl" "$DATA/eval_tier1_seen_conditions.jsonl" \
    "$DATA/eval_tier2_unseen_drugs.jsonl" "$DATA/eval_tier3_unseen_combos.jsonl" \
    "$DATA/eval_tier4_dose_interpolation.jsonl" | tee "$OUT/data_rows.txt"
[[ "$(wc -l < "$DATA/train.jsonl")" -eq 675183 ]] || {
    echo "[FATAL] canonical train row count is not 675183" >&2; exit 4;
}

# Everything after the one explicit download is offline. This proves later jobs can be offline too.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"$PY" - "$SNAPSHOT_PATH" "$REPO/endcell/jobs" <<"PY" | tee "$OUT/autoconfig_offline.txt"
import sys
sys.path.insert(0, sys.argv[2])
from transformers import AutoConfig
from gemma2_standard_provenance import (
    guarded_snapshot_load, snapshot_inventory_sha256, verify_authoritative_snapshot)
snapshot_path = sys.argv[1]
verify_authoritative_snapshot(snapshot_path)
snapshot_digest = snapshot_inventory_sha256(snapshot_path)
config = guarded_snapshot_load(
    lambda: AutoConfig.from_pretrained(snapshot_path, local_files_only=True),
    snapshot_path, snapshot_digest, stage="preflight offline AutoConfig load")
if getattr(config, "model_type", None) != "gemma2":
    raise SystemExit(f"[FATAL] expected Gemma-2 config, got {config.model_type!r}")
print(f"[PASS] exact-snapshot offline AutoConfig load: {config.__class__.__name__}, "
      f"model_type={config.model_type}, source={snapshot_path}, inventory={snapshot_digest}")
PY
"$PY" "$CLI_CONTRACT" --repo "$REPO" | tee "$OUT/cli_contract.txt"
"$PY" "$TRAINER" --help > "$OUT/trainer_help.txt"
"$PY" "$TOKENIZER_PROBE" \
    --model "$MODEL_ID" --data_dir "$DATA" --output "$OUT/gemma_tokenizer_probe.json" \
    --max_length 8192 --max_examples 0 --parity_examples 512 --require_fast \
    --revision "$MODEL_ID=$REVISION" --load-source "$MODEL_ID=$SNAPSHOT_PATH"

SOURCES=(
    "trainer=$TRAINER"
    "tokenizer_probe=$TOKENIZER_PROBE"
    "protobuf_env=$PROTOBUF_ENV"
    "provenance=$REPO/endcell/jobs/gemma2_standard_provenance.py"
    "evaluate_endcell=$REPO/endcell/eval/evaluate_endcell.py"
    "nir_benchmark=$REPO/endcell/analysis/nir_benchmark.py"
    "freeze_manifest=$REPO/endcell/analysis/freeze_nir_manifest.py"
    "compare_backbones=$REPO/endcell/analysis/compare_backbones.py"
    "residual_eval=$REPO/endcell/analysis/residual_eval.py"
    "preflight=$REPO/endcell/jobs/gemma2_standard_preflight.sh"
    "contract=$CONTRACT"
    "cli_contract=$CLI_CONTRACT"
    "tests_contract=$TEST_CONTRACT"
    "tests_command=$REPO/endcell/jobs/gemma2_standard_tests.sh"
    "fingerprint=$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
    "smoke=$REPO/endcell/jobs/gemma2_standard_smoke.sbatch"
    "train_job=$REPO/endcell/jobs/gemma2_standard_train.sbatch"
    "eval_job=$REPO/endcell/jobs/gemma2_standard_eval.sbatch"
    "phase1a_test=$REPO/tests/test_gemma_phase1a_training_contract.py"
    "eval_test=$REPO/tests/test_gemma_eval_contract.py"
    "runbook=$REPO/docs/endcell/gemma2_standard_hpc_runbook.md"
)
DATA_ARGS=(
    "train=$DATA/train.jsonl"
    "tier1=$DATA/eval_tier1_seen_conditions.jsonl"
    "tier2=$DATA/eval_tier2_unseen_drugs.jsonl"
    "tier3=$DATA/eval_tier3_unseen_combos.jsonl"
    "tier4=$DATA/eval_tier4_dose_interpolation.jsonl"
)
CANDIDATE="$OUT/.PREFLIGHT_PASSED.candidate.$$.json"
trap "rm -f \"$CANDIDATE\"" EXIT
CREATE=("$PY" "$CONTRACT" create --certificate "$CANDIDATE" --model-id "$MODEL_ID" \
    --revision "$REVISION" --snapshot-path "$SNAPSHOT_PATH" \
    --hub-cache "$HF_HUB_CACHE" --tests-certificate "$TEST_CERT" \
    --tokenizer-probe "$OUT/gemma_tokenizer_probe.json" --generation-cap 1800 \
    --protobuf-root "$GEMMA_PROTOBUF_DIR" \
    --protobuf-tree-sha256 "$GEMMA_PROTOBUF_TREE_SHA256" \
    --environment "environment=$OUT/environment.txt" --environment "pip_freeze=$OUT/pip_freeze.txt")
for item in "${DATA_ARGS[@]}"; do CREATE+=(--data "$item"); done
for item in "${SOURCES[@]}"; do CREATE+=(--source "$item"); done
"${CREATE[@]}" > "$OUT/PREFLIGHT_PASSED.candidate.printed.json"

"$PY" "$CONTRACT" verify --certificate "$CANDIDATE" --model-id "$MODEL_ID" \
    --revision "$REVISION" --require-generation-cap 1800 --verify-current-environment
mv "$CANDIDATE" "$CERT"
"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL_ID" \
    --revision "$REVISION" --require-generation-cap 1800 --verify-current-environment \
    --runtime-contract-out "$OUT/PREFLIGHT_RUNTIME_CONTRACT.json"
echo "[PASS] atomic launch certificate created only after scheduler, download, environment, data and tokenizer checks"
'
