#!/bin/bash
#SBATCH --job-name=gemma2_std_sft
#SBATCH --account=3180408
#SBATCH --partition=long_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=47:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/gemma2_standard_train_%j.out

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
FINGERPRINT_TOOL="$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
SMOKE_CERT="$REPO/RESULTS/gemma2_standard_smoke/SMOKE_PASSED.json"
OUT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell
RUN_META="$REPO/RESULTS/gemma2_standard_train_${SLURM_JOB_ID}"
MIN_HBM_MIB="${MIN_HBM_MIB:-135000}"
MIN_FREE_GIB=150
mkdir -p "$REPO/logs" "$REPO/RESULTS" "$RUN_META" "$OUT"
cd "$REPO"

[[ "${SLURM_JOB_PARTITION:-}" == "long_gpuh200" ]] || {
    echo "[FATAL] submitted to '${SLURM_JOB_PARTITION:-unknown}', expected 'long_gpuh200'" >&2
    exit 2
}
"$PY" "$CONTRACT" verify --certificate "$CERT" --model-id "$MODEL" \
    --revision "$REVISION" --require-generation-cap 1600 --verify-current-environment
[[ -f "$SMOKE_CERT" ]] || {
    echo "[FATAL] no canonical interruption/resume smoke authorization: $SMOKE_CERT" >&2; exit 2;
}
"$PY" - "$SMOKE_CERT" "$CERT" "$TRAINER" "$REPO/endcell/jobs/gemma2_standard_smoke.sbatch" <<'PY'
import hashlib, json, sys
from pathlib import Path

smoke_path, cert_path, trainer_path, smoke_job_path = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

document = json.load(open(smoke_path, encoding="utf-8"))
if document.get("schema_version") != 1 or document.get("smoke_passed") is not True:
    raise SystemExit("[FATAL] invalid smoke authorization")
if document.get("preflight_certificate", {}).get("sha256") != sha256(cert_path):
    raise SystemExit("[FATAL] smoke was authorized by a different preflight certificate")
if document.get("sources", {}).get("trainer", {}).get("sha256") != sha256(trainer_path):
    raise SystemExit("[FATAL] trainer changed since interruption/resume smoke")
if document.get("sources", {}).get("smoke_job", {}).get("sha256") != sha256(smoke_job_path):
    raise SystemExit("[FATAL] smoke job changed since the passing smoke")
times = document.get("checkpoint_seconds")
if not isinstance(times, list) or len(times) != 2 or max(times) > 420:
    raise SystemExit(f"[FATAL] unsafe or missing checkpoint timings: {times}")
if "H200" not in str(document.get("gpu_name", "")) or int(document.get("gpu_hbm_mib", 0)) < 135000:
    raise SystemExit("[FATAL] passing smoke was not measured on the required H200/HBM class")
if document.get("steps") != {"first": 50, "resumed": 100, "production_total": 42198}:
    raise SystemExit("[FATAL] smoke did not exercise the production schedule and resume boundary")
print("[PASS] exact preflight/trainer passed interruption-resume smoke")
PY
[[ -f "$FINGERPRINT_TOOL" ]] || {
    echo "[FATAL] checkpoint validator missing: $FINGERPRINT_TOOL" >&2; exit 2;
}
AVAILABLE_BYTES="$(df -B1 --output=avail "$OUT" | tail -1 | tr -d ' ')"
REQUIRED_BYTES="$((MIN_FREE_GIB * 1024 * 1024 * 1024))"
[[ "$AVAILABLE_BYTES" =~ ^[0-9]+$ && "$AVAILABLE_BYTES" -ge "$REQUIRED_BYTES" ]] || {
    echo "[FATAL] production requires at least $MIN_FREE_GIB GiB free after download/smoke; " \
         "available_bytes=${AVAILABLE_BYTES:-unknown}" >&2
    exit 2
}
echo "[PASS] production disk gate: available_bytes=$AVAILABLE_BYTES >= $REQUIRED_BYTES ($MIN_FREE_GIB GiB)"
[[ "$(wc -l < "$DATA/train.jsonl")" -eq 675183 ]] || {
    echo "[FATAL] train row count changed" >&2; exit 2;
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_HBM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d " ")"
[[ "$GPU_NAME" == *H200* ]] || {
    echo "[FATAL] canonical job requires NVIDIA H200, got $GPU_NAME" >&2; exit 2;
}
[[ "$GPU_HBM" =~ ^[0-9]+$ && "$GPU_HBM" -ge "$MIN_HBM_MIB" ]] || {
    echo "[FATAL] require at least $MIN_HBM_MIB MiB HBM, got ${GPU_HBM:-unknown}" >&2
    exit 2
}

# Hold this advisory lock for the complete batch job. Its content records the owner.
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock unavailable" >&2; exit 3; }
exec 9>"$OUT/.gemma2_standard_train.lock"
flock -n 9 || {
    echo "[FATAL] another job owns $OUT: $(cat "$OUT/.gemma2_standard_train.lock" 2>/dev/null || echo unknown)" >&2
    exit 3
}
printf "job_id=%s host=%s started_utc=%s\n" "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" >&9

HELP="$($PY "$TRAINER" --help 2>&1)"
for flag in --prepend_bos --strict_token_contract --resumable \
    --resume_from_checkpoint --model_revision --attn_implementation; do
    grep -Fq -- "$flag" <<<"$HELP" || { echo "[FATAL] trainer interface missing $flag" >&2; exit 3; }
done

RESUME_ARGS=()
mapfile -t PUBLISHED_STATES < <(find "$OUT" -maxdepth 2 -type f \
    -path "$OUT/checkpoint-[0-9]*/training_state.pt" | sort -V)
mapfile -t PUBLISHED_DIRS < <(find "$OUT" -mindepth 1 -maxdepth 1 -type d \
    -name "checkpoint-[0-9]*" | sort -V)
for directory in "${PUBLISHED_DIRS[@]}"; do
    [[ -f "$directory/training_state.pt" ]] || {
        echo "[FATAL] published checkpoint lacks training_state.pt: $directory" >&2; exit 4;
    }
done
if find "$OUT" -mindepth 1 -maxdepth 1 -type d -name ".checkpoint-*.tmp-*" | grep -q .; then
    echo "[FATAL] unpublished temporary checkpoint exists; manual audit required" >&2
    exit 4
fi

if [[ -f "$OUT/final/config.json" ]]; then
    "$PY" "$FINGERPRINT_TOOL" --checkpoint "$OUT/final" --require_complete_sft \
        --expected_global_step 42198 --expected_epoch 1 \
        --expected_microbatch_position 0 --expected_accumulation_position 0 \
        --digest_only >/dev/null
    echo "[PASS] canonical final checkpoint manifest, content, and terminal state validated; no rerun"
    exit 0
elif [[ "${#PUBLISHED_STATES[@]}" -gt 0 ]]; then
    RESUME_ARGS=(--resume_from_checkpoint auto)
    echo "[resume] ${#PUBLISHED_STATES[@]} published checkpoint state(s); strict auto-resume required"
else
    mapfile -t UNEXPECTED < <(find "$OUT" -mindepth 1 -maxdepth 1 \
        ! -name ".gemma2_standard_train.lock")
    [[ "${#UNEXPECTED[@]}" -eq 0 ]] || {
        printf "[FATAL] no published resume state, but output is not fresh:\n%s\n" "${UNEXPECTED[*]}" >&2
        exit 4
    }
    echo "[fresh] no prior published state; cold-starting pinned cached Gemma"
fi

CMD=(
    "$PY" "$TRAINER" --mode full --model_name "$MODEL" --model_revision "$REVISION"
    --train_file "$DATA/train.jsonl" --eval_file "$DATA/eval_tier1_seen_conditions.jsonl"
    --output_dir "$OUT" --num_epochs 1 --batch_size 1 --grad_accum 16
    --bf16 --gradient_checkpointing --max_length 8192
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 --de_weight 1.0
    --log_every 50 --save_every 1000 --keep_checkpoints 2 --seed 42
    --prepend_bos --strict_token_contract --resumable --attn_implementation eager
    "${RESUME_ARGS[@]}"
)

printf "%q " "${CMD[@]}" > "$RUN_META/command.txt"; printf "\n" >> "$RUN_META/command.txt"
if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$REPO" rev-parse HEAD > "$RUN_META/git_commit.txt"
else
    printf '%s\n' "unavailable: source-bundle run; see certificate-bound source SHA-256" \
        > "$RUN_META/git_commit.txt"
fi
cp "$CERT" "$RUN_META/PREFLIGHT_PASSED.json"
nvidia-smi -q > "$RUN_META/nvidia_smi_before.txt"

TRAIN_PID=""
on_timeout_signal() {
    echo "[signal] USR1 received with 600s warning; forwarding one checkpoint request"
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID"
        set +e
        wait "$TRAIN_PID"; child_rc=$?
        set -e
        [[ "$child_rc" -eq 99 ]] || {
            echo "[FATAL] trainer returned $child_rc after checkpoint signal, expected 99" >&2
            exit 5
        }
    fi
    echo "[signal] resumable checkpoint published; submit the identical job to continue"
    exit 99
}
trap on_timeout_signal USR1

"${CMD[@]}" &
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"; TRAIN_RC=$?
set -e
TRAIN_PID=""
[[ "$TRAIN_RC" -eq 0 ]] || exit "$TRAIN_RC"

[[ -f "$OUT/final/config.json" && -f "$OUT/final/training_state.pt" && \
   -f "$OUT/final/checkpoint_manifest.json" ]] || {
    echo "[FATAL] trainer returned success without complete final artifact" >&2; exit 5;
}
"$PY" "$FINGERPRINT_TOOL" --checkpoint "$OUT/final" --require_complete_sft \
    --expected_global_step 42198 --expected_epoch 1 \
    --expected_microbatch_position 0 --expected_accumulation_position 0 \
    --digest_only >/dev/null
echo "[PASS] final manifest/content and terminal state: completed=true, step=42198, epoch=1, microbatch=0, accumulation=0"
echo "[PASS] canonical one-epoch standard-target SFT completed -> $OUT/final"
