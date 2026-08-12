#!/bin/bash
#SBATCH --job-name=gemma2_std_smoke
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gemma2_standard_smoke_%j.out

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
MODEL=vandijklab/C2S-Scale-Gemma-2-2B
REVISION=5ddf28b8f1c81b7ab7a9be192924da82b6c5d512
TRAINER="$REPO/endcell/train/train_c2s_tahoe_endcell.py"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
OUT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_standard_smoke_${SLURM_JOB_ID}
RESULT="$REPO/RESULTS/gemma2_standard_smoke_${SLURM_JOB_ID}"
SMOKE_CERT_DIR="$REPO/RESULTS/gemma2_standard_smoke"
SMOKE_CERT="$SMOKE_CERT_DIR/SMOKE_PASSED.json"
MAX_CHECKPOINT_SECONDS="${MAX_CHECKPOINT_SECONDS:-420}"
MIN_HBM_MIB="${MIN_HBM_MIB:-135000}"
mkdir -p "$REPO/logs" "$RESULT" "$OUT" "$SMOKE_CERT_DIR"
cd "$REPO"

# A failed current smoke must never leave an older authorization at the canonical path.
if [[ -f "$SMOKE_CERT" ]]; then
    mv "$SMOKE_CERT" "$SMOKE_CERT_DIR/SMOKE_PASSED.previous.$(date -u +%Y%m%dT%H%M%SZ).json"
fi

"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL" \
    --revision "$REVISION" --require-generation-cap 1600 --verify-current-environment

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
    --resume_from_checkpoint --model_revision --attn_implementation; do
    has_flag "$flag" || { echo "[FATAL] trainer interface missing $flag" >&2; exit 3; }
done

head -n 64 "$DATA/eval_tier1_seen_conditions.jsonl" > "$RESULT/eval_smoke64.jsonl"
COMMON=(
    --mode full --model_name "$MODEL" --model_revision "$REVISION"
    --train_file "$DATA/train.jsonl" --eval_file "$RESULT/eval_smoke64.jsonl"
    --output_dir "$OUT" --num_epochs 1 --batch_size 1 --grad_accum 16
    --bf16 --gradient_checkpointing --max_length 8192
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 --de_weight 1.0
    --log_every 10 --save_every 1000 --keep_checkpoints 2 --seed 42
    --prepend_bos --strict_token_contract --resumable --attn_implementation eager
)

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv -l 5 > "$RESULT/nvidia_smi.csv" &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT

run_until_step_then_signal() {
    local target_step="$1" log_file="$2" elapsed_file="$3" checkpoint_file="$4"
    shift 4
    local fifo="${SLURM_TMPDIR:-/tmp}/gemma2-smoke-${SLURM_JOB_ID}-${target_step}.fifo"
    [[ ! -e "$fifo" ]] || { echo "[FATAL] temporary FIFO already exists: $fifo" >&2; return 6; }
    mkfifo "$fifo"
    tee "$log_file" < "$fifo" &
    local tee_pid=$!
    SECONDS=0
    "$PY" "$TRAINER" "${COMMON[@]}" "$@" > "$fifo" 2>&1 &
    local train_pid=$!
    local observed=0 abort_reason="" signal_second=0
    local segment_timeout="${SMOKE_SEGMENT_TIMEOUT_SECONDS:-4800}"
    while kill -0 "$train_pid" 2>/dev/null; do
        local reported_total
        reported_total=$(sed -nE "s/.*Total optimization steps:[[:space:]]*([0-9]+).*/\1/p" "$log_file" | tail -1)
        if [[ -n "$reported_total" && "$reported_total" != 42198 ]]; then
            abort_reason="trainer reported $reported_total optimization steps, expected 42198"
            kill -TERM "$train_pid"
            break
        fi
        if grep -Eq "Step[[:space:]]+${target_step}/42198" "$log_file" 2>/dev/null; then
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
    [[ -z "$abort_reason" ]] || { echo "[FATAL] $abort_reason" >&2; return 6; }
    [[ "$observed" -eq 1 ]] || { echo "[FATAL] trainer exited before step $target_step (rc=$rc)" >&2; return 7; }
    [[ "$rc" -eq 99 ]] || { echo "[FATAL] signalled trainer exited $rc, expected 99" >&2; return 8; }
    [[ "$tee_rc" -eq 0 ]] || { echo "[FATAL] tee failed with status $tee_rc" >&2; return 9; }
    [[ "$checkpoint_seconds" -le "$MAX_CHECKPOINT_SECONDS" ]] || {
        echo "[FATAL] checkpoint took ${checkpoint_seconds}s; 600s signal margin is unsafe" >&2
        return 10
    }
}

run_until_step_then_signal 50 "$RESULT/segment1.log" "$RESULT/segment1_seconds.txt" \
    "$RESULT/segment1_checkpoint_seconds.txt"

mapfile -t STATES < <(find "$OUT" -maxdepth 2 -type f \
    -path "$OUT/checkpoint-[0-9]*/training_state.pt" | sort)
[[ "${#STATES[@]}" -gt 0 ]] || {
    echo "[FATAL] first segment produced no published checkpoint-N/training_state.pt" >&2
    exit 4
}

run_until_step_then_signal 100 "$RESULT/segment2.log" "$RESULT/segment2_seconds.txt" \
    "$RESULT/segment2_checkpoint_seconds.txt" --resume_from_checkpoint auto

grep -Eiq "Restored optimizer/scheduler/RNG.*global_step=(5[0-9]|[6-9][0-9])" "$RESULT/segment2.log" || {
    echo "[FATAL] no evidence of restoring the post-step-50 state" >&2; exit 5;
}
grep -Eiq "Step[[:space:]]+100/42198" "$RESULT/segment2.log" || {
    echo "[FATAL] resumed segment contains no evidence of reaching step 100" >&2; exit 5;
}

"$PY" - "$SMOKE_CERT" "$CERT" "$GPU_NAME" "$GPU_HBM" \
    "$RESULT/segment1_checkpoint_seconds.txt" "$RESULT/segment2_checkpoint_seconds.txt" \
    "$TRAINER" "$REPO/endcell/jobs/gemma2_standard_smoke.sbatch" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

out, cert, gpu, hbm, first, second, trainer, job = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

document = {
    "schema_version": 1,
    "smoke_passed": True,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu_name": gpu,
    "gpu_hbm_mib": int(hbm),
    "preflight_certificate": {"path": str(Path(cert).resolve()), "sha256": sha256(cert)},
    "checkpoint_seconds": [int(Path(first).read_text().strip()), int(Path(second).read_text().strip())],
    "sources": {
        "trainer": {"path": str(Path(trainer).resolve()), "sha256": sha256(trainer)},
        "smoke_job": {"path": str(Path(job).resolve()), "sha256": sha256(job)},
    },
    "steps": {"first": 50, "resumed": 100, "production_total": 42198},
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
print(f"[PASS] atomic smoke authorization: {target}")
PY

echo "[PASS] production schedule resumed through step 100; each measured checkpoint fit safely inside 600s"
