#!/bin/bash
# =============================================================================
# gemma2_residual_smoke.sbatch — Stage 4: the gate between preflight and the run
# =============================================================================
#
# WHAT THIS PROVES, and why each part of it is worth one H200 allocation
#
#   1. THE UNFORKED TRAINER ACCEPTS THE RESIDUAL DIRECTORY.
#      "train_c2s_tahoe_endcell.py takes arbitrary --train_file/--eval_file paths"
#      is a reading of the code. This project's stated failure mode is exactly the
#      confident reading. 150 optimizer steps settle it by execution.
#
#      NOTHING here forks or edits the trainer. That is not fastidiousness: the
#      standard Gemma arm is live at ~41,301 of 42,198 steps and
#      _validate_resume_fingerprints (train_c2s_tahoe_endcell.py:610) compares
#      trainer_sha256 out of every checkpoint's contract. Any edit to the shared
#      trainer strands the standard arm's remaining ~900 steps permanently.
#
#   2. THE LOSS IS FINITE AND DECREASING.
#      Finiteness is a hard, unambiguous gate -- a NaN/Inf is breakage, full stop.
#      The decrease gate is deliberately weak (see LOSS GATE below): it is here to
#      catch a flat or diverging run, not to certify convergence in 100 steps.
#
#   3. NO SEQUENCE IS TRUNCATED.
#      The trainer prints cumulative "Truncated response tokens" and "Sentinel
#      losses" on every log line. Both must stay 0. Stage 3 already proved this over
#      every row on CPU; this confirms the same arithmetic inside the real
#      C2SDataset on the real device, which is where --strict_token_contract
#      actually raises.
#
#   4. A RESUMABLE CHECKPOINT IS WRITTEN AND RELOADS.
#      Run to step 50, send USR1, require rc 99 and a published
#      checkpoint-N-mbM/training_state.pt; then resume with
#      --resume_from_checkpoint auto to step 100 and require rc 99 again. Both
#      checkpoint publishes must complete within MAX_CHECKPOINT_SECONDS (420 s),
#      inside the 600 s --signal margin the production job will run under.
#
#   5. A MEASURED SECONDS-PER-OPTIMIZER-STEP.
#      The Stage 5 cost estimate ("~20 GPU-hours, one allocation plus a tail") is an
#      inference from the standard arm's per-step cost over much longer sequences.
#      This replaces it with a number, printed before any 24-hour allocation is
#      committed. Startup (model load + 150k-row dataset construction) is excluded
#      by timing from the "Total optimization steps:" line, not from job start.
#
# LOSS GATE, stated honestly
#   With warmup_ratio 0.03 over the residual schedule, step 50 is early in warmup and
#   the learning rate is still ramping. A strict monotone-decrease requirement would
#   fail healthy runs and burn the allocation. So the gate is:
#       * every logged loss and window loss must be finite and positive  (HARD)
#       * min(last two window losses) < the first window loss            (HARD, weak)
#   The full trajectory is printed either way. Read it; do not just read the verdict.
#
# SEQUENCE BUDGET -- ONE SOURCE, SHARED WITH STAGE 5
#   --max_length is schedule.max_length as recorded in RESIDUAL_DATA_PASSED.json:
#   the budget the Stage 3 tokenizer probe actually validated
#   (gemma2_residual_preflight.sbatch passes --max_length 8192 to both the probe
#   and the contract's `create`). It is READ FROM THE CERTIFICATE here, not
#   written as a literal, because gemma2_residual_train.sbatch resolves its own
#   --max_length from that same field (its `max_length = certified_max_length`
#   branch). One source, two readers. Cross-file line numbers are deliberately
#   NOT cited below: Stage 5 is under active edit and they rot.
#
#   This job used to hardcode `SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-8192}"` while
#   Stage 5 derived ceil((max_total_sequence_tokens + 1) / 64) * 64 from the
#   measurement. Those agree only if the measured maximum lands in [8128, 8191];
#   the residual response is ~200 gene symbols against a one-sentence control
#   prompt, so it never would. Because Stage 5 gate (e) refuses any smoke
#   certificate recorded at a different --max_length, the default deadlocked: a PASSING
#   Stage 4 was rejected on allocation #1, after this H200 allocation had already
#   been spent. Two places deriving one number was the bug. Stage 5 now takes the
#   certified budget and so does this job.
#
#   Nothing is lost by holding the certified 8192: C2SDataset does not pad to
#   max_length, so a tighter budget saves neither memory nor time and only
#   diverges from the standard Gemma arm. Truncation is still impossible -- the
#   assertions below demand max_length > max_total_sequence_tokens, the longest
#   BOS+prompt+response+EOS sequence measured across BOTH residual jsonls.
#
#   To move the budget you must move it in BOTH jobs, and Stage 5's header says so
#   too: RESIDUAL_MAX_LENGTH=<int> there, SMOKE_MAX_LENGTH=<int> and a re-run
#   here. Exporting RESIDUAL_MAX_LENGTH before submitting both is also honoured by
#   this job, so one export moves both together.
#
# WHAT THIS DOES *NOT* PROVE
#   The evaluation loop almost certainly never executes in 100 steps -- eval runs at
#   the epoch boundary. What is proved is that the eval JSONL parses and builds a
#   C2SDataset at startup (train_c2s_tahoe_endcell.py:1501). That is the check that
#   matters, because that same line SILENTLY skips a mistyped --eval_file: no loader,
#   no warning, no error. The Stage 5 job must therefore gate the eval path with a
#   [[ -f ]] test AND a sha256 comparison, not rely on the trainer to complain.
#
# FREEZE RULE (applies from Stage 5 allocation #1, restated here because this job is
# where the digests first get bound)
#   Once the first production residual allocation starts, residual.jsonl,
#   residual_val.jsonl and train_c2s_tahoe_endcell.py are IMMUTABLE for the life of
#   the run. build_residual_targets.py does not refuse an existing out_dir -- it
#   makedirs(exist_ok=True) and os.replace's over both jsonls -- and
#   _training_contract hashes both files plus the trainer into every checkpoint's
#   resume contract. A rerun "just to regenerate report.json" changes train_sha256
#   and every published residual checkpoint refuses to resume, forever.
#       chmod a-w <TGT>/residual.jsonl <TGT>/residual_val.jsonl
#
# SUBMIT
#     cd "$HOME/tahoe"
#     sbatch endcell/jobs/gemma2_residual_smoke.sbatch
#
# PREREQUISITE (must have exited 0)
#     RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json   <- Stage 3
#
# WRITES
#     RESULTS/gemma2_residual_smoke/SMOKE_PASSED.json
#     RESULTS/gemma2_residual_smoke_<jobid>/{segment1,segment2}.log, timings,
#         nvidia_smi.csv, eval_smoke64.jsonl, cost_projection.json
#   Throwaway checkpoints go to
#     $ROOT/checkpoints/gemma2b_residual_smoke_<jobid>       (must not pre-exist)
#   and are removed on success only, under the guards in cleanup_smoke_output below.
#   The production tree gemma2b_sft_endcell_residual is NEVER created or opened here,
#   and the standard tree gemma2b_sft_endcell is never touched at all.
# =============================================================================

#SBATCH --job-name=gemma2_res_smoke
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gemma2_residual_smoke_%j.out

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
ROOT="${RESIDUAL_ROOT:-/data/BuffaF-Projetcs/florian_c2s}"
MODEL=vandijklab/C2S-Scale-Gemma-2-2B
REVISION=5ddf28b8f1c81b7ab7a9be192924da82b6c5d512
TRAINER="$REPO/endcell/train/train_c2s_tahoe_endcell.py"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
RESIDUAL_CONTRACT="$REPO/endcell/jobs/gemma2_residual_data_contract.py"
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
RESIDUAL_CERT="$REPO/RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json"
PRODUCTION_OUT="$ROOT/checkpoints/gemma2b_sft_endcell_residual"
OUT="$ROOT/checkpoints/gemma2b_residual_smoke_${SLURM_JOB_ID}"
RESULT="$REPO/RESULTS/gemma2_residual_smoke_${SLURM_JOB_ID}"
SMOKE_CERT_DIR="$REPO/RESULTS/gemma2_residual_smoke"
SMOKE_CERT="$SMOKE_CERT_DIR/SMOKE_PASSED.json"
MAX_CHECKPOINT_SECONDS="${MAX_CHECKPOINT_SECONDS:-420}"
MIN_HBM_MIB="${MIN_HBM_MIB:-135000}"
RESIDUAL_SMOKE_CLEANUP="${RESIDUAL_SMOKE_CLEANUP:-1}"
# The sequence budget this smoke exercises is NOT a constant here. It is READ from
# schedule.max_length in the residual data certificate a few blocks below -- the
# same field Stage 5 resolves its --max_length from (see "SEQUENCE BUDGET" in the
# header). Overrides, in precedence order:
#   SMOKE_MAX_LENGTH=<int>     the variable Stage 5's own header tells you to set
#                              here when you run Stage 5 with RESIDUAL_MAX_LENGTH.
#                              Set it alone and Stage 5 will refuse, correctly.
#   RESIDUAL_MAX_LENGTH=<int>  the variable Stage 5 itself reads. Export it once
#                              before submitting both jobs to move both budgets
#                              together.
# Neither override escapes the two bounds assertions below.
SMOKE_MAX_LENGTH_OVERRIDE="${SMOKE_MAX_LENGTH:-}"
RESIDUAL_MAX_LENGTH_OVERRIDE="${RESIDUAL_MAX_LENGTH:-}"
mkdir -p "$REPO/logs" "$RESULT" "$SMOKE_CERT_DIR"
cd "$REPO"

[[ "${SLURM_JOB_PARTITION:-}" == "gpuh200" ]] || {
    echo "[FATAL] submitted to '${SLURM_JOB_PARTITION:-unknown}', expected 'gpuh200'" >&2
    exit 2
}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || {
    echo "[FATAL] SLURM_JOB_ID is not numeric; refusing (the cleanup guard depends on it)" >&2
    exit 2
}

# ---- Invariant 4: new paths only, and never the production tree ---------------
[[ ! -e "$OUT" ]] || {
    echo "[FATAL] throwaway smoke output path already exists: $OUT" >&2; exit 2;
}
[[ "$OUT" != "$PRODUCTION_OUT" ]] || {
    echo "[FATAL] smoke output resolves to the production residual tree" >&2; exit 2;
}
mkdir -p "$OUT"

# A failed current smoke must never leave an older authorization at the canonical path.
if [[ -f "$SMOKE_CERT" ]]; then
    mv "$SMOKE_CERT" "$SMOKE_CERT_DIR/SMOKE_PASSED.previous.$(date -u +%Y%m%dT%H%M%SZ).json"
fi

# ---- Residual data certificate: existence, freshness, and the step total ------
[[ -f "$RESIDUAL_CERT" ]] || {
    echo "[FATAL] no residual data authorization: $RESIDUAL_CERT" >&2
    echo "        sbatch endcell/jobs/gemma2_residual_preflight.sbatch" >&2
    exit 2
}
# verify re-hashes both jsonls and refuses if either changed since Stage 3.
"$PY" "$RESIDUAL_CONTRACT" verify --certificate "$RESIDUAL_CERT" \
    --require-standard-certificate "$CERT"
mapfile -t RESIDUAL_VALUES < <("$PY" - "$RESIDUAL_CERT" \
    "$SMOKE_MAX_LENGTH_OVERRIDE" "$RESIDUAL_MAX_LENGTH_OVERRIDE" <<'PY'
import json, sys

cert_path, smoke_override, residual_override = sys.argv[1:]


def fail(message):
    raise SystemExit(f"[FATAL] residual data certificate: {message}")


document = json.load(open(cert_path, encoding="utf-8"))
if document.get("schema_version") != 1 or document.get("residual_data_passed") is not True:
    raise SystemExit("[FATAL] invalid residual data certificate")

max_total = int(document["max_total_sequence_tokens"])
certified_max_length = int(document["schedule"]["max_length"])

# ONE source for --max_length: schedule.max_length out of this certificate, the
# budget the Stage 3 tokenizer probe validated, resolved exactly as Stage 5
# (gemma2_residual_train.sbatch) resolves it. Hardcoding 8192 here while
# Stage 5 derived a tighter budget from max_total is what deadlocked Stage 4
# against Stage 5; the certificate is now the single place the number comes from.
# Override precedence mirrors the header. An override is still bounds-checked.
if smoke_override:
    try:
        max_length = int(smoke_override)
    except ValueError:
        fail(f"SMOKE_MAX_LENGTH is not an integer: {smoke_override!r}")
    derivation = (f"SMOKE_MAX_LENGTH override = {max_length}; Stage 5 must be run with "
                  f"RESIDUAL_MAX_LENGTH={max_length} or gate (e) will refuse this "
                  f"certificate")
elif residual_override:
    try:
        max_length = int(residual_override)
    except ValueError:
        fail(f"RESIDUAL_MAX_LENGTH is not an integer: {residual_override!r}")
    derivation = (f"RESIDUAL_MAX_LENGTH override = {max_length}, the same variable "
                  f"Stage 5 reads")
else:
    max_length = certified_max_length
    derivation = (f"schedule.max_length = {max_length} from the certificate, the budget "
                  f"the Stage 3 probe validated and the one Stage 5 resolves to")

# The same bounds Stage 5 asserts, so a budget that would be refused there is
# refused here, on CPU-cheap terms, before the H200 is spent rather than after.
if max_total >= certified_max_length:
    fail(f"max_total_sequence_tokens {max_total} does not fit under the certified "
         f"budget {certified_max_length}")
if certified_max_length > 8192:
    fail(f"certified schedule.max_length {certified_max_length} exceeds 8192")
if max_length <= max_total:
    fail(f"--max_length {max_length} is not strictly above the measured maximum "
         f"sequence {max_total}: truncation is guaranteed")
if max_length > certified_max_length:
    fail(f"--max_length {max_length} exceeds the certified budget "
         f"{certified_max_length}; the probe never validated that length")

print(document["target_dir"])
print(document["files"]["train"]["path"])
print(document["files"]["val"]["path"])
print(document["files"]["train"]["sha256"])
print(document["files"]["val"]["sha256"])
print(document["schedule"]["total_steps"])
print(document["residual_generation_cap"])
print(document["max_response_tokens"])
print(max_total)
print(certified_max_length)
print(max_length)
print(derivation)
PY
)
[[ "${#RESIDUAL_VALUES[@]}" -eq 12 ]] || {
    echo "[FATAL] residual data certificate rejected (see message above)" >&2; exit 2;
}
TGT="${RESIDUAL_VALUES[0]}"
TRAIN_JSONL="${RESIDUAL_VALUES[1]}"
VAL_JSONL="${RESIDUAL_VALUES[2]}"
TRAIN_SHA256="${RESIDUAL_VALUES[3]}"
VAL_SHA256="${RESIDUAL_VALUES[4]}"
RESIDUAL_TOTAL="${RESIDUAL_VALUES[5]}"
RESIDUAL_GENERATION_CAP="${RESIDUAL_VALUES[6]}"
MAX_RESPONSE_TOKENS="${RESIDUAL_VALUES[7]}"
MAX_TOTAL_TOKENS="${RESIDUAL_VALUES[8]}"
CERTIFIED_MAX_LENGTH="${RESIDUAL_VALUES[9]}"
SMOKE_MAX_LENGTH="${RESIDUAL_VALUES[10]}"
MAX_LENGTH_DERIVATION="${RESIDUAL_VALUES[11]}"
[[ "$RESIDUAL_TOTAL" =~ ^[0-9]+$ && "$RESIDUAL_TOTAL" -gt 100 ]] || {
    echo "[FATAL] residual total optimization steps is not a usable integer > 100: "\
         "$RESIDUAL_TOTAL" >&2
    exit 2
}
# Belt-and-braces: the python above already refused anything out of bounds, but the
# certificate field this smoke RECORDS must be a plain integer, because Stage 5
# compares it with int() equality.
[[ "$SMOKE_MAX_LENGTH" =~ ^[0-9]+$ && "$MAX_TOTAL_TOKENS" =~ ^[0-9]+$ ]] || {
    echo "[FATAL] non-integer sequence budget: max_length='$SMOKE_MAX_LENGTH' "\
         "max_total_sequence_tokens='$MAX_TOTAL_TOKENS'" >&2
    exit 2
}
[[ "$MAX_TOTAL_TOKENS" -lt "$SMOKE_MAX_LENGTH" ]] || {
    echo "[FATAL] measured max_total_sequence_tokens=$MAX_TOTAL_TOKENS does not fit "\
         "strictly inside --max_length $SMOKE_MAX_LENGTH" >&2
    exit 2
}
echo "[PASS] residual data authorized"
echo "       target_dir              $TGT"
echo "       RESIDUAL_TOTAL          $RESIDUAL_TOTAL optimizer steps"
echo "       max_response_tokens     $MAX_RESPONSE_TOKENS  (NOT the standard 1699)"
echo "       residual_generation_cap $RESIDUAL_GENERATION_CAP  (NOT 1800; feeds Stage 6)"
echo "       max_total_sequence      $MAX_TOTAL_TOKENS < $SMOKE_MAX_LENGTH (--max_length)"
echo "       certified budget        $CERTIFIED_MAX_LENGTH (schedule.max_length, what the probe validated)"
echo "       --max_length            $SMOKE_MAX_LENGTH   [$MAX_LENGTH_DERIVATION]"
echo "       Stage 5 derives --max_length the same way and refuses this certificate"
echo "       on any difference; re-run this smoke if you change either override."

# ---- MODEL provenance only: no --require-generation-cap ----------------------
RUNTIME_CONTRACT="${SLURM_TMPDIR:-/tmp}/gemma2-residual-smoke-runtime-${SLURM_JOB_ID}.json"
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
PREFLIGHT_SHA256="${RUNTIME_VALUES[1]}"
SNAPSHOT_INVENTORY_SHA256="${RUNTIME_VALUES[2]}"
AUTHORITATIVE_FILES_SHA256="${RUNTIME_VALUES[3]}"
[[ -d "$SNAPSHOT_PATH" ]] || {
    echo "[FATAL] certificate returned no exact Gemma snapshot: $SNAPSHOT_PATH" >&2; exit 2;
}
echo "[PASS] smoke cold-start source is certified snapshot: $SNAPSHOT_PATH"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_HBM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d " ")"
[[ "$GPU_NAME" == *H200* ]] || { echo "[FATAL] smoke requires NVIDIA H200, got $GPU_NAME" >&2; exit 2; }
[[ "$GPU_HBM" =~ ^[0-9]+$ && "$GPU_HBM" -ge "$MIN_HBM_MIB" ]] || {
    echo "[FATAL] smoke requires at least $MIN_HBM_MIB MiB HBM, got ${GPU_HBM:-unknown}" >&2
    exit 2
}

HELP="$($PY "$TRAINER" --help 2>&1)"
has_flag() { grep -Fq -- "$1" <<<"$HELP"; }
for flag in --prepend_bos --strict_token_contract --resumable \
    --resume_from_checkpoint --model_revision --model_load_path --attn_implementation \
    --preflight_certificate_sha256 --parent_snapshot_inventory_sha256 \
    --parent_authoritative_files_sha256; do
    has_flag "$flag" || { echo "[FATAL] trainer interface missing $flag" >&2; exit 3; }
done

# ---- Eval shard. The trainer SILENTLY skips a missing --eval_file, so build it
# ---- here and assert it, rather than discovering the omission from a log with no
# ---- eval lines. Same path and same bytes across both segments, because
# ---- _training_contract hashes eval_file into the resume fingerprint.
EVAL_SMOKE="$RESULT/eval_smoke64.jsonl"
head -n 64 "$VAL_JSONL" > "$EVAL_SMOKE"
EVAL_ROWS="$(wc -l < "$EVAL_SMOKE")"
[[ "$EVAL_ROWS" -eq 64 ]] || {
    echo "[FATAL] eval shard has $EVAL_ROWS rows, expected 64" >&2; exit 3;
}
[[ -f "$EVAL_SMOKE" && -s "$EVAL_SMOKE" ]] || {
    echo "[FATAL] eval shard is missing or empty: $EVAL_SMOKE" >&2; exit 3;
}
echo "[PASS] eval shard: 64 rows carved from $VAL_JSONL"

# ---- Byte-identical to the standard smoke CLI except for the two data paths ---
COMMON=(
    --mode full --model_name "$MODEL" --model_revision "$REVISION" \
    --model_load_path "$SNAPSHOT_PATH"
    --preflight_certificate_sha256 "$PREFLIGHT_SHA256"
    --parent_snapshot_inventory_sha256 "$SNAPSHOT_INVENTORY_SHA256"
    --parent_authoritative_files_sha256 "$AUTHORITATIVE_FILES_SHA256"
    --train_file "$TRAIN_JSONL" --eval_file "$EVAL_SMOKE"
    --output_dir "$OUT" --num_epochs 1 --batch_size 1 --grad_accum 16
    --bf16 --gradient_checkpointing --max_length "$SMOKE_MAX_LENGTH"
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 --de_weight 1.0
    --log_every 10 --save_every 1000 --keep_checkpoints 2 --seed 42
    --prepend_bos --strict_token_contract --resumable --attn_implementation eager
)

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv -l 5 > "$RESULT/nvidia_smi.csv" &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

# run_until_step_then_signal TARGET LOG ELAPSED CKPT_SECONDS STARTUP_SECONDS [extra args...]
run_until_step_then_signal() {
    local target_step="$1" log_file="$2" elapsed_file="$3" checkpoint_file="$4" startup_file="$5"
    shift 5
    local fifo="${SLURM_TMPDIR:-/tmp}/gemma2-res-smoke-${SLURM_JOB_ID}-${target_step}.fifo"
    [[ ! -e "$fifo" ]] || { echo "[FATAL] temporary FIFO already exists: $fifo" >&2; return 6; }
    mkfifo "$fifo"
    tee "$log_file" < "$fifo" &
    local tee_pid=$!
    SECONDS=0
    "$PY" "$TRAINER" "${COMMON[@]}" "$@" > "$fifo" 2>&1 &
    local train_pid=$!
    local observed=0 abort_reason="" signal_second=0 startup_second=-1
    local segment_timeout="${SMOKE_SEGMENT_TIMEOUT_SECONDS:-4800}"
    while kill -0 "$train_pid" 2>/dev/null; do
        local reported_total
        reported_total=$(sed -nE "s/.*Total optimization steps:[[:space:]]*([0-9]+).*/\1/p" "$log_file" | tail -1)
        if [[ -n "$reported_total" ]]; then
            # This catches a wrong row count or a stale constant LOUDLY, before any
            # gradient is spent, and is the reason the number is read from the
            # certificate rather than hardcoded.
            if [[ "$reported_total" != "$RESIDUAL_TOTAL" ]]; then
                abort_reason="trainer reported $reported_total optimization steps, expected $RESIDUAL_TOTAL"
                kill -TERM "$train_pid"
                break
            fi
            if [[ "$startup_second" -lt 0 ]]; then
                # Model load + 150k-row dataset construction are complete by this
                # line; everything after it is optimizer work.
                startup_second=$SECONDS
                echo "[smoke] startup (model + dataset) completed at ${startup_second}s"
            fi
        fi
        if grep -Eq "Step[[:space:]]+${target_step}/${RESIDUAL_TOTAL}" "$log_file" 2>/dev/null; then
            observed=1
            signal_second=$SECONDS
            echo "[smoke] observed step $target_step; sending USR1 to trainer pid $train_pid"
            kill -USR1 "$train_pid"
            break
        fi
        if [[ "$SECONDS" -gt "$segment_timeout" ]]; then
            abort_reason="trainer did not reach logged step $target_step within ${segment_timeout}s"
            kill -TERM "$train_pid"
            break
        fi
        sleep 2
    done
    set +e
    wait "$train_pid"; local rc=$?
    wait "$tee_pid"; local tee_rc=$?
    set -e
    echo "$SECONDS" > "$elapsed_file"
    local checkpoint_seconds=$((SECONDS - signal_second))
    echo "$checkpoint_seconds" > "$checkpoint_file"
    echo "$startup_second" > "$startup_file"
    [[ -z "$abort_reason" ]] || { echo "[FATAL] $abort_reason" >&2; return 6; }
    [[ "$observed" -eq 1 ]] || { echo "[FATAL] trainer exited before step $target_step (rc=$rc)" >&2; return 7; }
    [[ "$rc" -eq 99 ]] || { echo "[FATAL] signalled trainer exited $rc, expected 99" >&2; return 8; }
    [[ "$tee_rc" -eq 0 ]] || { echo "[FATAL] tee failed with status $tee_rc" >&2; return 9; }
    [[ "$checkpoint_seconds" -le "$MAX_CHECKPOINT_SECONDS" ]] || {
        echo "[FATAL] checkpoint took ${checkpoint_seconds}s; 600s signal margin is unsafe" >&2
        return 10
    }
    [[ "$startup_second" -ge 0 ]] || {
        echo "[FATAL] never observed the 'Total optimization steps' line" >&2; return 11;
    }
}

run_until_step_then_signal 50 "$RESULT/segment1.log" "$RESULT/segment1_seconds.txt" \
    "$RESULT/segment1_checkpoint_seconds.txt" "$RESULT/segment1_startup_seconds.txt"

mapfile -t STATES < <(find "$OUT" -maxdepth 2 -type f \
    -path "$OUT/checkpoint-[0-9]*/training_state.pt" | sort)
[[ "${#STATES[@]}" -gt 0 ]] || {
    echo "[FATAL] first segment produced no published checkpoint-N/training_state.pt" >&2
    exit 4
}
echo "[PASS] published resumable checkpoint(s): ${STATES[*]}"

run_until_step_then_signal 100 "$RESULT/segment2.log" "$RESULT/segment2_seconds.txt" \
    "$RESULT/segment2_checkpoint_seconds.txt" "$RESULT/segment2_startup_seconds.txt" \
    --resume_from_checkpoint auto

grep -Eiq "Restored optimizer/scheduler/RNG.*global_step=(5[0-9]|[6-9][0-9])" "$RESULT/segment2.log" || {
    echo "[FATAL] no evidence of restoring the post-step-50 state" >&2; exit 5;
}
grep -Eiq "Step[[:space:]]+100/${RESIDUAL_TOTAL}" "$RESULT/segment2.log" || {
    echo "[FATAL] resumed segment contains no evidence of reaching step 100" >&2; exit 5;
}
echo "[PASS] auto-resume restored an optimizer boundary and reached step 100"

# ---- Loss finiteness / trajectory, truncation counters, cost projection -------
"$PY" - "$RESULT/segment1.log" "$RESULT/segment2.log" "$RESULT/cost_projection.json" \
    "$RESIDUAL_TOTAL" \
    "$RESULT/segment1_startup_seconds.txt" "$RESULT/segment2_startup_seconds.txt" \
    "$RESULT/segment1_seconds.txt" "$RESULT/segment2_seconds.txt" \
    "$RESULT/segment1_checkpoint_seconds.txt" "$RESULT/segment2_checkpoint_seconds.txt" <<'PY'
import json, math, re, sys
from pathlib import Path

(seg1, seg2, out_path, total_steps,
 s1_startup, s2_startup, s1_elapsed, s2_elapsed, s1_ckpt, s2_ckpt) = sys.argv[1:]
total_steps = int(total_steps)

LINE = re.compile(
    r"Step\s+(?P<step>\d+)/(?P<total>\d+)\s*\|\s*"
    r"Loss:\s*(?P<loss>[-+0-9.eEnaNIif]+)\s*\|\s*"
    r"Window loss:\s*(?P<window>[-+0-9.eEnaNIif]+)\s*\|\s*"
    r"LR:\s*(?P<lr>\S+)\s*\|\s*"
    r"Tokens:\s*(?P<tokens>[0-9,]+)\s*\|\s*"
    r"Truncated response tokens:\s*(?P<trunc>[0-9,]+)\s*\|\s*"
    r"Sentinel losses:\s*(?P<sentinel>[0-9,]+)")

def parse(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            rows.append({
                "step": int(match.group("step")),
                "loss": match.group("loss"),
                "window": match.group("window"),
                "lr": match.group("lr"),
                "truncated_response_tokens": int(match.group("trunc").replace(",", "")),
                "sentinel_losses": int(match.group("sentinel").replace(",", "")),
            })
    return rows

rows = parse(seg1) + parse(seg2)
rows.sort(key=lambda r: r["step"])
failures = []
if len(rows) < 8:
    failures.append(f"only {len(rows)} parseable log lines across both segments; "
                    "expected ~10 at --log_every 10 over 100 steps. The trainer's log "
                    "format may have changed, in which case this gate is blind.")

# 3. NO SEQUENCE IS TRUNCATED. Cumulative counters; any non-zero at any point fails.
for row in rows:
    if row["truncated_response_tokens"] != 0:
        failures.append(f"step {row['step']}: truncated_response_tokens="
                        f"{row['truncated_response_tokens']} (must be 0)")
        break
for row in rows:
    if row["sentinel_losses"] != 0:
        failures.append(f"step {row['step']}: sentinel_losses={row['sentinel_losses']} "
                        "(must be 0; residual responses always end in [END_CELL], so a "
                        "sentinel loss means truncation)")
        break

# 2. LOSS IS FINITE AND DECREASING.
def as_float(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None

losses, windows = [], []
for row in rows:
    loss, window = as_float(row["loss"]), as_float(row["window"])
    if loss is None or loss <= 0.0:
        failures.append(f"step {row['step']}: non-finite or non-positive loss {row['loss']!r}")
    else:
        losses.append((row["step"], loss))
    if window is None or window <= 0.0:
        failures.append(f"step {row['step']}: non-finite or non-positive window loss "
                        f"{row['window']!r}")
    else:
        windows.append((row["step"], window))

decreasing = None
if len(windows) >= 3:
    first_step, first = windows[0]
    tail = windows[-2:]
    best_step, best = min(tail, key=lambda pair: pair[1])
    decreasing = best < first
    if not decreasing:
        failures.append(
            f"window loss did not decrease: step {first_step}={first:.4f} -> "
            f"best of last two (step {best_step})={best:.4f}. Over 100 steps early in "
            "warmup this is weak evidence, but a flat or rising curve here means the "
            "model is not learning the residual target at all. Inspect before Stage 5.")
elif not failures:
    failures.append("too few window-loss samples to judge the trajectory")

# 5. MEASURED SECONDS PER OPTIMIZER STEP.
def read_int(path):
    return int(Path(path).read_text().strip())

s1_startup_s, s2_startup_s = read_int(s1_startup), read_int(s2_startup)
s1_signal_s = read_int(s1_elapsed) - read_int(s1_ckpt)
s2_signal_s = read_int(s2_elapsed) - read_int(s2_ckpt)
seg1_step_seconds = (s1_signal_s - s1_startup_s) / 50.0        # steps 1..50
seg2_step_seconds = (s2_signal_s - s2_startup_s) / 50.0        # steps 51..100
measured = max(seg1_step_seconds, seg2_step_seconds)           # conservative
projected_hours = total_steps * measured / 3600.0

projection = {
    "schema_version": 1,
    "residual_total_steps": total_steps,
    "segment1": {"startup_seconds": s1_startup_s, "steps_seconds": s1_signal_s - s1_startup_s,
                 "seconds_per_step": round(seg1_step_seconds, 3),
                 "checkpoint_seconds": read_int(s1_ckpt)},
    "segment2": {"startup_seconds": s2_startup_s, "steps_seconds": s2_signal_s - s2_startup_s,
                 "seconds_per_step": round(seg2_step_seconds, 3),
                 "checkpoint_seconds": read_int(s2_ckpt)},
    "measured_seconds_per_step": round(measured, 3),
    "projected_total_gpu_hours": round(projected_hours, 2),
    "projected_allocations_at_23h50m": math.ceil(projected_hours / 23.0),
    "loss_trajectory": [{"step": s, "window_loss": round(v, 5)} for s, v in windows],
    "cumulative_loss_trajectory": [{"step": s, "loss": round(v, 5)} for s, v in losses],
    "window_loss_decreasing": decreasing,
    "truncated_response_tokens_max": max((r["truncated_response_tokens"] for r in rows),
                                         default=None),
    "sentinel_losses_max": max((r["sentinel_losses"] for r in rows), default=None),
    "log_lines_parsed": len(rows),
    "failures": failures,
}
Path(out_path).write_text(
    json.dumps(projection, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

print("\n================ LOSS TRAJECTORY (window) ================")
for step, value in windows:
    print(f"  step {step:4d}   window_loss {value:.4f}")
print("\n================ MEASURED COST ================")
print(f"  startup (model+dataset)   segment1 {s1_startup_s}s   segment2 {s2_startup_s}s")
print(f"  measured_seconds_per_step {measured:.3f}   "
      f"(seg1 {seg1_step_seconds:.3f}, seg2 {seg2_step_seconds:.3f})")
print(f"  residual_total_steps      {total_steps}")
print(f"  projected_total_gpu_hours {projected_hours:.2f}")
print(f"  projected_allocations     {projection['projected_allocations_at_23h50m']} "
      f"at --time=23:50:00")
print("\n  This projection REPLACES the ~20 GPU-hour inference in the plan. Decide")
print("  whether to commit Stage 5 on this number, not on that one. Note the")
print("  contention: the standard arm still needs at least one more allocation for")
print("  its last ~900 steps, and both trees want >= 150 GiB on the same filesystem.")

if failures:
    print("\n================ SMOKE FAILURES ================", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(12)
print("\n[PASS] loss finite and decreasing; zero truncated response tokens; "
      "zero sentinel losses")
PY

# ---- Certificate --------------------------------------------------------------
"$PY" - "$SMOKE_CERT" "$CERT" "$RESIDUAL_CERT" "$GPU_NAME" "$GPU_HBM" \
    "$RESULT/segment1_checkpoint_seconds.txt" "$RESULT/segment2_checkpoint_seconds.txt" \
    "$SNAPSHOT_PATH" "$SNAPSHOT_INVENTORY_SHA256" "$AUTHORITATIVE_FILES_SHA256" "$TRAINER" \
    "$REPO/endcell/jobs/gemma2_residual_smoke.sbatch" "$RESIDUAL_TOTAL" \
    "$TGT" "$TRAIN_SHA256" "$VAL_SHA256" "$RESULT/cost_projection.json" \
    "$SMOKE_MAX_LENGTH" "$MAX_LENGTH_DERIVATION" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

(out, cert, residual_cert, gpu, hbm, first, second, parent_snapshot, snapshot_digest,
 authoritative_digest, trainer, job, residual_total, target_dir, train_sha256,
 val_sha256, projection_path, max_length, max_length_derivation) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

projection = json.load(open(projection_path, encoding="utf-8"))
document = {
    "schema_version": 4,
    "smoke_passed": True,
    "arm": "residual",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu_name": gpu,
    "gpu_hbm_mib": int(hbm),
    "preflight_certificate": {"path": str(Path(cert).resolve()), "sha256": sha256(cert)},
    "residual_data_certificate": {"path": str(Path(residual_cert).resolve()),
                                  "sha256": sha256(residual_cert)},
    "residual_target_dir": target_dir,
    "residual_train_sha256": train_sha256,
    "residual_val_sha256": val_sha256,
    "parent_snapshot_path": str(Path(parent_snapshot).absolute()),
    "parent_snapshot_inventory_sha256": snapshot_digest,
    "parent_authoritative_files_sha256": authoritative_digest,
    "checkpoint_seconds": [int(Path(first).read_text().strip()),
                           int(Path(second).read_text().strip())],
    "sources": {
        "trainer": {"path": str(Path(trainer).resolve()), "sha256": sha256(trainer)},
        "smoke_job": {"path": str(Path(job).resolve()), "sha256": sha256(job)},
    },
    "steps": {"first": 50, "resumed": 100, "production_total": int(residual_total)},
    # Recorded at BOTH the top level and under "schedule" so a downstream job can
    # find it either way. max_length is part of _training_contract, so a production
    # run at a different value has not been smoke-tested at this sequence budget.
    # This is the value RESOLVED from the certificate (see SEQUENCE BUDGET in the
    # header), out of the same schedule.max_length field Stage 5 resolves from, so
    # Stage 5's gate-(e) equality check on this field holds by construction rather
    # than by coincidence. max_length_derivation records which branch produced it,
    # so a certificate that came from an override says so.
    "max_length": int(max_length),
    "max_length_derivation": max_length_derivation,
    "schedule": {"max_length": int(max_length),
                 "batch_size": 1, "grad_accum": 16, "num_epochs": 1},
    "measured_seconds_per_step": projection["measured_seconds_per_step"],
    "projected_total_gpu_hours": projection["projected_total_gpu_hours"],
    "window_loss_decreasing": projection["window_loss_decreasing"],
    "truncated_response_tokens_max": projection["truncated_response_tokens_max"],
    "sentinel_losses_max": projection["sentinel_losses_max"],
}
target = Path(out).resolve()
fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(f"[PASS] atomic residual smoke authorization: {target}")
PY

# ---- Guarded removal of the throwaway checkpoint tree -------------------------
# Only reached on full success. Four independent guards, because a mis-set variable
# here is the one thing in this file that could destroy something.
cleanup_smoke_output() {
    [[ "$RESIDUAL_SMOKE_CLEANUP" == "1" ]] || {
        echo "[keep] RESIDUAL_SMOKE_CLEANUP != 1; leaving $OUT in place"; return 0;
    }
    [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo "[keep] non-numeric job id"; return 0; }
    [[ "$(basename "$OUT")" == "gemma2b_residual_smoke_${SLURM_JOB_ID}" ]] || {
        echo "[keep] $OUT does not carry this job's throwaway name"; return 0;
    }
    [[ "$(dirname "$OUT")" == "$ROOT/checkpoints" ]] || {
        echo "[keep] $OUT is not directly under $ROOT/checkpoints"; return 0;
    }
    [[ "$OUT" != "$PRODUCTION_OUT" && "$OUT" != "$ROOT/checkpoints/gemma2b_sft_endcell" ]] || {
        echo "[keep] refusing to touch a production tree"; return 0;
    }
    echo "[cleanup] removing throwaway smoke checkpoints: $OUT"
    rm -rf "$OUT"
}
cleanup_smoke_output

echo
echo "[PASS] residual smoke complete."
echo "       The unforked trainer accepted $TGT."
echo "       Total optimization steps reported by the trainer: $RESIDUAL_TOTAL."
echo "       Both checkpoint publishes fit inside the 600s signal margin."
echo "       Logs and cost projection: $RESULT"
echo "       Authorization: $SMOKE_CERT"
echo
echo "[next] Stage 5 (gemma2_residual_train.sbatch) is NOT authored by this change."
echo "       Before writing it, re-read the freeze rule in this file's header and"
echo "       chmod a-w the two residual jsonls."
