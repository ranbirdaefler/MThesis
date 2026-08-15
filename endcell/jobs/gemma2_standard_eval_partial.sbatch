#!/bin/bash
#SBATCH --job-name=gemma2_partial_eval
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemma2_standard_partial_eval_%j.out

# This additive launcher leaves every source bound into PREFLIGHT_PASSED.json unchanged.
# It admits exactly one immutable optimizer-boundary checkpoint, records that decision,
# and delegates all generation and scoring to the already certified evaluation script.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME

REPO="${REPO:-$HOME/tahoe}"
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
MODEL=/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell/checkpoint-41301-mb660816
MODEL_ID=vandijklab/C2S-Scale-Gemma-2-2B
REVISION=5ddf28b8f1c81b7ab7a9be192924da82b6c5d512
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
ORIGINAL="$REPO/endcell/jobs/gemma2_standard_eval.sbatch"
ADMISSION_TOOL="$REPO/endcell/jobs/gemma2_partial_checkpoint_admission.py"
FINGERPRINT_TOOL="$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
OUT="$REPO/RESULTS/gemma2_standard_eval/gemma"
MANIFEST="${TIER2_MANIFEST:?Set TIER2_MANIFEST to the immutable hash-named Tier-2 manifest}"

mkdir -p "$REPO/logs" "$OUT"
cd "$REPO"
source "$REPO/endcell/jobs/gemma2_standard_protobuf_env.sh"

RUNTIME="${SLURM_TMPDIR:-/tmp}/gemma2-partial-runtime-${SLURM_JOB_ID}.json"
"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL_ID" \
    --revision "$REVISION" --require-generation-cap 1800 \
    --verify-current-environment --runtime-contract-out "$RUNTIME"

PATCHED="${SLURM_TMPDIR:-/tmp}/gemma2_standard_eval.partial.${SLURM_JOB_ID}.sbatch"
"$PY" - "$ORIGINAL" "$PATCHED" <<'PY'
import os, sys
source, target = sys.argv[1:]
text = open(source, encoding="utf-8").read()
needle = "--require_complete_sft --expected_global_step 42198"
replacement = "--expected_global_step 42198"
if text.count(needle) != 1:
    raise SystemExit(f"[FATAL] expected exactly one certified terminal-state gate, found {text.count(needle)}")
patched = text.replace(needle, replacement)
with open(target, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(patched)
os.chmod(target, 0o700)
PY

PARTIAL_FP=$("$PY" "$ADMISSION_TOOL" --checkpoint "$MODEL" \
    --runtime-contract "$RUNTIME" --fingerprint-tool "$FINGERPRINT_TOOL" \
    --out "$OUT/gemma_partial_checkpoint_admission.json" --digest-only)
[[ "$PARTIAL_FP" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[FATAL] invalid admitted checkpoint fingerprint: $PARTIAL_FP" >&2
    exit 22
}

"$PY" - "$0" "$ORIGINAL" "$PATCHED" "$OUT/gemma_partial_launcher.json" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path

wrapper, original, patched, output = map(Path, sys.argv[1:])
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
document = {
    "schema_version": 1,
    "policy": "remove_only_the_terminal-completion_gate_after_exact_partial-state_admission",
    "wrapper_sha256": digest(wrapper.resolve()),
    "certified_eval_sha256": digest(original.resolve()),
    "runtime_patched_eval_sha256": digest(patched.resolve()),
    "substitution_count": 1,
}
payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
output = output.resolve()
if output.exists() and output.read_bytes() != payload:
    raise SystemExit(f"[FATAL] refusing to overwrite differing launcher record: {output}")
if not output.exists():
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, output)
PY

echo "[PASS] admitted immutable partial Gemma checkpoint: step 41301/42198 (97.9%)"
echo "[PASS] certified evaluator preserved; only its terminal-completion gate is removed in the recorded runtime copy"
export CHECKPOINT_ROLE=gemma_sft
export MODEL_PATH="$MODEL"
export TIER2_MANIFEST="$MANIFEST"
exec bash "$PATCHED"
