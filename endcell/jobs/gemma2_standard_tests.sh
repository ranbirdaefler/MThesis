#!/usr/bin/env bash
# Run inside a defq srun. Publishes TESTS_PASSED.json only after every gate passes.
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"

REPO="${REPO:-$HOME/tahoe}"
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
PROTOBUF_ENV="$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh"
source "$PROTOBUF_ENV"
TEST_DEPS="${GEMMA_TEST_DEPS:-/data/BuffaF-Projetcs/florian_c2s/test_deps/pytest-8.3.5}"
OUT="$REPO/RESULTS/gemma2_standard_tests"
CERT="$OUT/TESTS_PASSED.json"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_tests_contract.py"
COMMAND="$REPO/endcell/jobs/gemma2_standard_tests.sh"
mkdir -p "$OUT"
cd "$REPO"

command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock unavailable" >&2; exit 2; }
exec 8>"$OUT/.gemma2_standard_tests.lock"
flock -n 8 || { echo "[FATAL] another Gemma test gate is active" >&2; exit 2; }

if [[ -f "$CERT" ]]; then
    mv "$CERT" "$OUT/TESTS_PASSED.previous.$(date -u +%Y%m%dT%H%M%SZ).$$.json"
fi

SOURCES=(
    "trainer=$REPO/endcell/train/train_c2s_tahoe_endcell.py"
    "tokenizer_probe=$REPO/endcell/train/gemma_tokenizer_probe.py"
    "protobuf_env=$PROTOBUF_ENV"
    "provenance=$REPO/endcell/jobs/gemma2_standard_provenance.py"
    "freeze_manifest=$REPO/endcell/analysis/freeze_nir_manifest.py"
    "nir_benchmark=$REPO/endcell/analysis/nir_benchmark.py"
    "compare_backbones=$REPO/endcell/analysis/compare_backbones.py"
    "residual_eval=$REPO/endcell/analysis/residual_eval.py"
    "evaluate_endcell=$REPO/endcell/eval/evaluate_endcell.py"
    "fingerprint=$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
    "cli_contract=$REPO/endcell/jobs/gemma2_standard_cli_contract.py"
    "preflight_contract=$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
    "tests_contract=$CONTRACT"
    "preflight=$REPO/endcell/jobs/gemma2_standard_preflight.sh"
    "smoke=$REPO/endcell/jobs/gemma2_standard_smoke.sbatch"
    "train_job=$REPO/endcell/jobs/gemma2_standard_train.sbatch"
    "eval_job=$REPO/endcell/jobs/gemma2_standard_eval.sbatch"
    "phase1a_test=$REPO/tests/test_gemma_phase1a_training_contract.py"
    "eval_test=$REPO/tests/test_gemma_eval_contract.py"
    "runbook=$REPO/docs/endcell/gemma2_standard_hpc_runbook.md"
)
for item in "${SOURCES[@]}" "$COMMAND" "$PY"; do
    path="${item#*=}"
    [[ -f "$path" ]] || { echo "[FATAL] missing test input: $path" >&2; exit 2; }
done

bash -n \
    endcell/jobs/gemma2_standard_tests.sh \
    endcell/jobs/gemma2_standard_protobuf_env.sh \
    endcell/jobs/gemma2_standard_preflight.sh \
    endcell/jobs/gemma2_standard_smoke.sbatch \
    endcell/jobs/gemma2_standard_train.sbatch \
    endcell/jobs/gemma2_standard_eval.sbatch
"$PY" -m py_compile \
    endcell/train/train_c2s_tahoe_endcell.py \
    endcell/train/gemma_tokenizer_probe.py \
    endcell/analysis/freeze_nir_manifest.py \
    endcell/analysis/nir_benchmark.py \
    endcell/analysis/compare_backbones.py \
    endcell/analysis/residual_eval.py \
    endcell/eval/evaluate_endcell.py \
    endcell/jobs/gemma2_standard_checkpoint_fingerprint.py \
    endcell/jobs/gemma2_standard_provenance.py \
    endcell/jobs/gemma2_standard_cli_contract.py \
    endcell/jobs/gemma2_standard_preflight_contract.py \
    endcell/jobs/gemma2_standard_tests_contract.py \
    tests/test_gemma_phase1a_training_contract.py \
    tests/test_gemma_eval_contract.py
"$PY" endcell/jobs/gemma2_standard_cli_contract.py --repo "$REPO"
[[ -f "$TEST_DEPS/pytest/__init__.py" ]] || {
    echo "[FATAL] isolated pytest 8.3.5 is missing at $TEST_DEPS" >&2
    exit 2
}
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$TEST_DEPS:$GEMMA_PROTOBUF_DIR" \
"$PY" -m pytest \
    tests/test_gemma_phase1a_training_contract.py \
    tests/test_gemma_eval_contract.py -q
"$PY" endcell/analysis/nir_benchmark.py --selftest \
    --out RESULTS/gemma2_nir_selftest.json
"$PY" endcell/analysis/residual_eval.py --selftest
"$PY" endcell/eval/evaluate_endcell.py --selftest \
    --out RESULTS/gemma2_validity_selftest.json

CREATE=("$PY" "$CONTRACT" create --certificate "$CERT" --command "$COMMAND")
for item in "${SOURCES[@]}"; do CREATE+=(--source "$item"); done
"${CREATE[@]}"
"$PY" "$CONTRACT" verify --certificate "$CERT" --verify-current-environment
echo "[PASS] all Section 1 tests passed and are certificate-bound"
