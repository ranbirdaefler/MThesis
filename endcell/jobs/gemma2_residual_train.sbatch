#!/bin/bash
#SBATCH --job-name=gemma2_res_sft
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=23:50:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/gemma2_residual_train_%j.out
#
# =============================================================================
# RESIDUAL-TARGET Gemma-2-2B SFT -- segmented, resumable, one epoch.
#
# This is gemma2_standard_train.sbatch with four retargetings and nothing else:
#   (1) --train_file / --eval_file point at the RESIDUAL build resolved by the
#       Stage 1 inventory and frozen by RESIDUAL_DATA_PASSED.json;
#   (2) --output_dir is a NEW tree, gemma2b_sft_endcell_residual, which cannot
#       collide with the live standard arm's gemma2b_sft_endcell and which gives
#       this job a distinct flock file for free;
#   (3) --max_length is the budget the residual certificate itself records in
#       schedule.max_length -- the one the Stage 3 tokenizer probe validated and
#       the one Stage 4 smoke-tests by default -- with the measured-maximum
#       derivation available as an explicit opt-in (see below);
#   (4) the CPU preflight gates run to completion, and must pass, before any
#       GPU work, any lock, any output tree, and any trainer launch.
# The shared trainer endcell/train/train_c2s_tahoe_endcell.py is NOT forked and
# NOT edited. The standard arm is mid-flight and hashes trainer_sha256 into its
# resume fingerprint; an edit there permanently strands its remaining steps.
#
# -----------------------------------------------------------------------------
# RUNBOOK
# -----------------------------------------------------------------------------
#   Prerequisites, in order, each of which must already have exited 0:
#     Stage 1  srun ... endcell/jobs/gemma2_residual_inventory.sh
#              -> RESULTS/gemma2_residual/RESOLVED_TGT.txt   (REQUIRED HERE: this
#                 job refuses without it, and always binds the residual
#                 certificate's target_dir to it -- see gate (b))
#     Stage 2  srun ... endcell/jobs/gemma2_residual_data_audit.py
#     Stage 3  sbatch endcell/jobs/gemma2_residual_preflight.sbatch
#              -> RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json
#     Stage 4  sbatch endcell/jobs/gemma2_residual_smoke.sbatch
#              -> RESULTS/gemma2_residual_smoke/SMOKE_PASSED.json
#
#   This job creates NO directory under /data until every gate has passed, so a
#   refused submission leaves the output path untouched and Stage 1 stays
#   repeatable.
#
#   Allocation #1 (cold start, refuses if the output tree already holds state):
#     cd ~/tahoe && sbatch endcell/jobs/gemma2_residual_train.sbatch
#
#   Allocations #2..N (after each rc-99 exit), resubmitted IDENTICALLY until the
#   job exits 0:
#     cd ~/tahoe && sbatch endcell/jobs/gemma2_residual_train.sbatch --resume
#
#   --resume is PERMISSION TO CONTINUE, not a hint. Without it the job refuses
#   to start when the output tree contains a published checkpoint (exit 4).
#   With it the job refuses to start when the output tree contains NO published
#   checkpoint (exit 4) -- there is no branch in which the flag is ignored, and
#   no branch in which existing state is overwritten. RESIDUAL_RESUME=1 in the
#   environment is equivalent to the flag.
#
#   Exit codes: 0 done; 99 checkpointed at the signal, resubmit with --resume;
#   2 gate/provenance refusal; 3 lock or interface refusal; 4 output-tree
#   refusal (clobber, resume mismatch, unpublished tmp checkpoint); 5 trainer
#   post-condition failure.
#
#   GEMMA_RESIDUAL_OUT=<new absolute path> relocates the output tree. It is the
#   same variable gemma2_residual_inventory.sh reads, so an exit-4 "start over in
#   a new tree" is followed by setting it on BOTH Stage 1 and this job -- never by
#   deleting or moving the tree that already exists (invariant 4). The guard that
#   refuses to land on the live standard arm applies to the override too.
#
# -----------------------------------------------------------------------------
# FREEZE RULE -- read before submitting allocation #1
# -----------------------------------------------------------------------------
#   From the moment allocation #1 starts, these are IMMUTABLE for the life of
#   the run:
#       <TGT>/residual.jsonl
#       <TGT>/residual_val.jsonl
#       endcell/train/train_c2s_tahoe_endcell.py
#       the resolved --max_length, --seed, --batch_size, --grad_accum
#   The trainer hashes train_sha256, eval_sha256, trainer_sha256, max_length and
#   seed into every checkpoint's contract and _validate_resume_fingerprints
#   refuses any mismatch, so a rebuild or an edit mid-run strands the run
#   permanently -- there is no recovery short of restarting from step 0.
#   build_residual_targets.py does NOT refuse an existing out_dir: it
#   makedirs(exist_ok=True) and os.replace's over residual.jsonl and
#   residual_val.jsonl. Re-running it "just to regenerate report.json" is the
#   destructive act. After Stage 3 passes, run:  chmod -R a-w <TGT>
#   This job enforces the freeze on every allocation by re-hashing both jsonls
#   against RESIDUAL_DATA_PASSED.json and by pinning the resolved run
#   parameters into $OUT/.residual_run_pin.json on allocation #1.
#
# -----------------------------------------------------------------------------
# WHAT THIS JOB CONSUMES from RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json
# -----------------------------------------------------------------------------
#   Written by gemma2_residual_data_contract.py `create` at Stage 3, schema 1:
#     files.train.{path,sha256,rows}      files.val.{path,sha256,rows}
#     schedule.{batch_size,grad_accum,num_epochs,max_length,total_steps}
#     residual_generation_cap  max_response_tokens  max_total_sequence_tokens
#     target_dir  prompt_order_in_data  standard_preflight_certificate.sha256
#
#   THE FREEZE CHECK IS NOT REIMPLEMENTED HERE. This job invokes that tool's own
#   `verify` subcommand, which re-hashes both jsonls against the digests recorded
#   at Stage 3 and refuses on any difference. One implementation of the freeze,
#   invoked, beats two implementations that can drift apart. What this job adds
#   on top are the assertions `verify` does not make: that the certified schedule
#   shape is the 1/16/1 this job actually runs, that total_steps recomputes from
#   rows_train exactly as the trainer derives it, that cap_derivation ties the
#   recorded generation cap to THIS certificate's measured max response (so the
#   standard arm's 1800 cannot have been pasted in), that both files sit in
#   target_dir and neither is a standard-data file, and the derivation of
#   --max_length below.
#
#   This job RE-VERIFIES the certificate; it deliberately does not re-run the
#   20-50 minute tokenizer probe inside an H200 allocation. Stage 3 is where the
#   probe is measured, on CPU, where a refusal costs minutes instead of a GPU
#   allocation.
#
# -----------------------------------------------------------------------------
# WHERE --max_length COMES FROM -- ONE SOURCE, SHARED WITH THE SMOKE
# -----------------------------------------------------------------------------
#   --max_length is schedule.max_length as recorded in RESIDUAL_DATA_PASSED.json,
#   which Stage 3 writes from the --max_length it passed to the tokenizer probe
#   (gemma2_residual_preflight.sbatch: `--max_length 8192` to the probe and
#   `--max_length 8192` to the contract's `create`). That is the budget the probe
#   actually validated, and it is the same 8192 that
#   gemma2_residual_smoke.sbatch certifies by default
#   (`SMOKE_MAX_LENGTH="${SMOKE_MAX_LENGTH:-8192}"`).
#
#   This job DOES NOT default to the tighter ceil((max_total + 1) / 64) * 64
#   derivation any more. That derivation and the smoke's 8192 default could never
#   agree, and because gate (e) below refuses any smoke recorded at a different
#   --max_length, the default configuration deadlocked: a passing Stage 4 was
#   rejected by Stage 5 on allocation #1, every time. Two places deriving one
#   number is the bug; the certificate is now the single place it comes from.
#
#   Nothing is lost by holding 8192. C2SDataset does not pad to max_length, so a
#   tighter budget saves no memory and no time; it only diverges from the Pythia
#   arm and from the standard Gemma arm, which is the opposite of what a matched
#   replication wants. Truncation is still impossible: the assertions below
#   demand max_length > max_total_sequence_tokens, i.e. strictly above the
#   longest BOS+prompt+response+EOS sequence measured across BOTH residual jsonls
#   under the run's own 256002-length tokenizer. That matters because
#   residual_to_sentence always ends in [END_CELL]: any response truncation drops
#   the sentinel and --strict_token_contract raises inside
#   C2SDataset.__getitem__, in the training loop, hours into an allocation rather
#   than at startup.
#
#   RESIDUAL_MAX_LENGTH=<int> overrides the certified budget (e.g. to run the
#   tight measured-maximum budget deliberately). It is still checked against both
#   bounds. IF YOU SET IT, YOU MUST ALSO RE-RUN STAGE 4 WITH THE SAME VALUE:
#       SMOKE_MAX_LENGTH=<int> sbatch endcell/jobs/gemma2_residual_smoke.sbatch
#   otherwise gate (e) refuses, and it is right to: max_length is in the
#   trainer's contract dict, so a production run at a budget the smoke never
#   exercised is untested. The run pin freezes whichever value allocation #1
#   used, so it cannot change mid-run either.
#
# -----------------------------------------------------------------------------
# CERTIFICATE SCOPING -- why the standard contract is called without a cap
# -----------------------------------------------------------------------------
#   `gemma2_standard_preflight_contract.py verify` is invoked here WITHOUT
#   --require-generation-cap. That argument is optional
#   (contract.py: `if args.require_generation_cap is not None`), so verify still
#   checks snapshot inventory, authoritative revision bytes, the protobuf tree,
#   pip freeze, tokenizer length 256000/256002 and sentinels 256000/256001 --
#   everything about the MODEL -- while the 1800 cap and the five hardcoded
#   CANONICAL_TAHOE_DATA_SHA256 entries stay bound to the standard DATA, where
#   they belong. Residual DATA is gated by RESIDUAL_DATA_PASSED.json instead.
#   No constant in the standard contract changes and no standard gate weakens.
# =============================================================================

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
TRAINER="$REPO/endcell/train/train_c2s_tahoe_endcell.py"
CONTRACT="$REPO/endcell/jobs/gemma2_standard_preflight_contract.py"
FINGERPRINT_TOOL="$REPO/endcell/jobs/gemma2_standard_checkpoint_fingerprint.py"
CERT="$REPO/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json"
RESIDUAL_CERT="$REPO/RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json"
RESIDUAL_CONTRACT="$REPO/endcell/jobs/gemma2_residual_data_contract.py"
# Stage 1's single output: the ONE residual build whose residual.jsonl sha256
# matches the Pythia residual checkpoint's recorded train_sha256. Six
# residual_targets* directories exist and every one of them trains cleanly and
# produces a plausible number, so the certificate's target_dir is checked
# against this file on every allocation -- see gate (b).
RESOLVED_TGT_FILE="$REPO/RESULTS/gemma2_residual/RESOLVED_TGT.txt"
SMOKE_CERT="$REPO/RESULTS/gemma2_residual_smoke/SMOKE_PASSED.json"
SMOKE_JOB="$REPO/endcell/jobs/gemma2_residual_smoke.sbatch"
STANDARD_DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
STANDARD_OUT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell
# Overridable, and named exactly as gemma2_residual_inventory.sh names it, so
# that the exit-4 advice "choose a different --output_dir" can actually be
# followed without editing this file: re-run Stage 1 and this job with the same
# GEMMA_RESIDUAL_OUT=<new path>. The collision guard below still refuses to land
# on the live standard arm whatever the override says.
OUT="${GEMMA_RESIDUAL_OUT:-/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell_residual}"
RUN_META="$REPO/RESULTS/gemma2_residual_train_${SLURM_JOB_ID}"
PIN="$OUT/.residual_run_pin.json"
LOCK_NAME=".gemma2_residual_train.lock"
PIN_NAME=".residual_run_pin.json"
MIN_HBM_MIB="${MIN_HBM_MIB:-135000}"
MIN_FREE_GIB=150

# --- resume permission -------------------------------------------------------
RESUME_REQUESTED=0
case "${1:-}" in
    --resume) RESUME_REQUESTED=1 ;;
    "")       ;;
    *) echo "[FATAL] unknown argument '${1}'; the only accepted argument is --resume" >&2; exit 2 ;;
esac
[[ "${RESIDUAL_RESUME:-0}" != "1" ]] || RESUME_REQUESTED=1

[[ "$OUT" != "$STANDARD_OUT" ]] || {
    echo "[FATAL] residual output path collides with the live standard arm" >&2; exit 2;
}
[[ "$OUT" == /* ]] || {
    echo "[FATAL] GEMMA_RESIDUAL_OUT must be an absolute path, got '$OUT'" >&2; exit 2;
}
[[ "$OUT" != "$STANDARD_OUT"/* ]] || {
    echo "[FATAL] residual output path is inside the live standard arm: $OUT" >&2; exit 2;
}

# $OUT is deliberately NOT created here. Every CPU gate below can refuse, and a
# refusal that leaves an empty output tree behind permanently trips
# gemma2_residual_inventory.sh's "[FATAL] Gemma residual output path already
# exists" (exit 3) -- which would make Stage 1, the ONLY producer of
# RESOLVED_TGT.txt and the only place the Pythia checkpoint's train_sha256 is
# compared, unrepeatable after a single failed submission. $OUT is created just
# before the lock is taken, once every gate has passed.
mkdir -p "$REPO/logs" "$REPO/RESULTS" "$RUN_META"
cd "$REPO"

[[ "${SLURM_JOB_PARTITION:-}" == "gpuh200" ]] || {
    echo "[FATAL] submitted to '${SLURM_JOB_PARTITION:-unknown}', expected 'gpuh200'" >&2
    exit 2
}

# Published-state scan, defined once so the fast-fail below and the authoritative
# post-lock decision can never drift apart.
# The `-d` guard makes this a no-op on a genuinely fresh run, where $OUT does not
# exist yet because it is not created until every CPU gate has passed. Without it
# `find` would exit non-zero under `set -e`.
published_states() {
    [[ -d "$OUT" ]] || return 0
    find "$OUT" -maxdepth 2 -type f -path "$OUT/checkpoint-[0-9]*/training_state.pt" | sort -V
}

# --- fast-fail clobber guard, before any expensive gate ----------------------
mapfile -t EARLY_STATES < <(published_states)
if [[ "${#EARLY_STATES[@]}" -gt 0 && "$RESUME_REQUESTED" -eq 0 ]]; then
    printf "[FATAL] %s already holds %d published checkpoint state(s).\n" \
        "$OUT" "${#EARLY_STATES[@]}" >&2
    printf "        Refusing to start without explicit permission to continue.\n" >&2
    printf "        To continue this run: sbatch endcell/jobs/gemma2_residual_train.sbatch --resume\n" >&2
    printf "        To start a different run: GEMMA_RESIDUAL_OUT=<new absolute path>\n" >&2
    printf "        on both Stage 1 and this job. Never clear this one.\n" >&2
    exit 4
fi
if [[ "${#EARLY_STATES[@]}" -eq 0 && "$RESUME_REQUESTED" -eq 1 && ! -f "$OUT/final/config.json" ]]; then
    printf "[FATAL] --resume was given but %s holds no published checkpoint state.\n" "$OUT" >&2
    printf "        Refusing to silently cold-start a run that was asked to continue.\n" >&2
    printf "        If this really is allocation #1, submit without --resume.\n" >&2
    printf "        If a previous allocation died before its FIRST publish, the tree holds\n" >&2
    printf "        at most the trainer's startup run_provenance.json, which carries no\n" >&2
    printf "        progress: submitting WITHOUT --resume cold-starts cleanly and is the\n" >&2
    printf "        supported recovery.\n" >&2
    printf "        If it holds anything else (best/, partial artifacts), that is an audit,\n" >&2
    printf "        not a retry: inspect it, then archive it or start over with\n" >&2
    printf "        GEMMA_RESIDUAL_OUT=<new absolute path> on both Stage 1 and this job.\n" >&2
    printf "        Do not clear this tree while any other job may still hold the lock.\n" >&2
    exit 4
fi

# =============================================================================
# CPU PREFLIGHT GATES -- all of these run and must pass before any GPU work.
# =============================================================================

# --- (a) MODEL provenance, via the standard certificate, no generation cap ---
RUNTIME_CONTRACT="${SLURM_TMPDIR:-/tmp}/gemma2-preflight-runtime-${SLURM_JOB_ID}.json"
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
echo "[PASS] model provenance verified (generation cap deliberately not asserted here:"
echo "       1800 is a property of the STANDARD target; residual data is gated separately)"

# --- (b) RESIDUAL DATA certificate: the freeze, via its own verifier --------
# gemma2_residual_data_contract.py `verify` re-hashes both jsonls against the
# digests recorded at Stage 3 and refuses on any difference. That check is not
# duplicated here: one implementation of the freeze, invoked, is safer than two
# implementations that can drift. --require-standard-certificate additionally
# binds the residual data to the same MODEL provenance verified in (a).
[[ -f "$RESIDUAL_CERT" ]] || {
    echo "[FATAL] no residual data authorization: $RESIDUAL_CERT" >&2
    echo "        Run the CPU preflight first: sbatch endcell/jobs/gemma2_residual_preflight.sbatch" >&2
    exit 2
}
[[ -f "$RESIDUAL_CONTRACT" ]] || {
    echo "[FATAL] residual data contract tool missing: $RESIDUAL_CONTRACT" >&2; exit 2;
}
#
# --require-target-dir is ALWAYS passed, and its value comes from Stage 1's
# RESOLVED_TGT.txt rather than from an environment variable that is unset by
# default. This is the check the whole inventory exists to make. Six
# residual_targets* directories are on disk; exactly one has a residual.jsonl
# whose sha256 matches the Pythia residual checkpoint's recorded train_sha256,
# and that is the only one whose Gemma arm mirrors the arm the thesis reports.
# The other five all tokenize, all pass the freeze check against a certificate
# issued FOR THEM, all train to completion, and all produce a plausible number.
# Nothing downstream would ever say otherwise. So the binding is made here,
# unconditionally, or the run does not start.
[[ -f "$RESOLVED_TGT_FILE" ]] || {
    echo "[FATAL] Stage 1 has not resolved a residual build: $RESOLVED_TGT_FILE" >&2
    echo "        Without it, nothing ties this run to the build whose residual.jsonl" >&2
    echo "        matches the Pythia checkpoint's train_sha256, and a certificate issued" >&2
    echo "        for any of the other residual_targets* directories would train to" >&2
    echo "        completion with no error." >&2
    echo "        Run Stage 1 first:" >&2
    echo "          srun -p defq -c 2 --mem 8G --time 01:00:00 \\" >&2
    echo "               bash endcell/jobs/gemma2_residual_inventory.sh" >&2
    exit 2
}
TGT_RESOLVED="$(head -n 1 "$RESOLVED_TGT_FILE" | tr -d '\r')"
[[ -n "$TGT_RESOLVED" ]] || {
    echo "[FATAL] $RESOLVED_TGT_FILE is empty; re-run Stage 1" >&2; exit 2;
}
[[ -d "$TGT_RESOLVED" ]] || {
    echo "[FATAL] Stage 1 resolved a target dir that no longer exists: $TGT_RESOLVED" >&2
    exit 2
}
# RESIDUAL_TGT remains ONLY an explicit override, never the switch that decides
# whether the check happens at all. Overriding it is a deliberate act and it is
# announced loudly, because it detaches this run from the Pythia checkpoint.
TGT_REQUIRED="$TGT_RESOLVED"
if [[ -n "${RESIDUAL_TGT:-}" && "$RESIDUAL_TGT" != "$TGT_RESOLVED" ]]; then
    echo "[warn] RESIDUAL_TGT overrides the Stage 1 resolution:" >&2
    echo "[warn]   Stage 1 resolved : $TGT_RESOLVED" >&2
    echo "[warn]   RESIDUAL_TGT     : $RESIDUAL_TGT" >&2
    echo "[warn] This run will NOT be the build matching the Pythia checkpoint's" >&2
    echo "[warn] train_sha256. Do not report it as a matched replication." >&2
    TGT_REQUIRED="$RESIDUAL_TGT"
fi
VERIFY_ARGS=(verify --certificate "$RESIDUAL_CERT" --require-standard-certificate "$CERT"
             --require-target-dir "$TGT_REQUIRED")
"$PY" "$RESIDUAL_CONTRACT" "${VERIFY_ARGS[@]}" || {
    echo "[FATAL] residual data certificate refused; not starting the run" >&2
    exit 2
}
echo "[PASS] residual certificate is bound to the Stage 1 resolution: $TGT_REQUIRED"

# Residual-scoped assertions the certificate's own verifier does not make,
# plus the derivation of --max_length from the MEASURED maximum sequence.
mapfile -t RESIDUAL_VALUES < <("$PY" - "$RESIDUAL_CERT" "$STANDARD_DATA" \
    "${RESIDUAL_MAX_LENGTH:-}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

cert_path, standard_data_dir, max_length_override = sys.argv[1:]


def fail(message):
    raise SystemExit(f"[FATAL] residual data certificate: {message}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(document, dotted, kind):
    value = document
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            fail(f"missing field {dotted}; is this a schema-1 certificate written by "
                 f"gemma2_residual_data_contract.py?")
        value = value[part]
    if kind is int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            fail(f"{dotted} is not a positive int: {value!r}")
    elif kind is str:
        if not isinstance(value, str) or not value:
            fail(f"{dotted} is not a non-empty string: {value!r}")
    return value


document = json.load(open(cert_path, encoding="utf-8"))
if document.get("schema_version") != 1:
    fail(f"schema_version {document.get('schema_version')!r} != 1")
if document.get("residual_data_passed") is not True:
    fail("residual_data_passed is not exactly true")

target_dir = Path(require(document, "target_dir", str))
train_file = Path(require(document, "files.train.path", str))
eval_file = Path(require(document, "files.val.path", str))
train_digest = require(document, "files.train.sha256", str)
eval_digest = require(document, "files.val.sha256", str)
rows_train = require(document, "files.train.rows", int)
rows_val = require(document, "files.val.rows", int)

batch_size = require(document, "schedule.batch_size", int)
grad_accum = require(document, "schedule.grad_accum", int)
num_epochs = require(document, "schedule.num_epochs", int)
certified_max_length = require(document, "schedule.max_length", int)
total_steps = require(document, "schedule.total_steps", int)

cap = require(document, "residual_generation_cap", int)
max_response = require(document, "max_response_tokens", int)
max_total = require(document, "max_total_sequence_tokens", int)

# The run parameters this job actually passes must be the ones that were
# measured: a certificate written for a different accumulation shape describes a
# different step schedule and therefore a different LR curve.
if (batch_size, grad_accum, num_epochs) != (1, 16, 1):
    fail(f"certificate was measured for batch_size/grad_accum/num_epochs = "
         f"{batch_size}/{grad_accum}/{num_epochs}, this job runs 1/16/1")

# Recompute the schedule exactly as the trainer does on the resumable path:
#   microbatches_per_epoch = ceil(rows / batch_size)
#   total_steps = (microbatches_per_epoch // grad_accum) * num_epochs
derived_steps = (math.ceil(rows_train / batch_size) // grad_accum) * num_epochs
if derived_steps != total_steps:
    fail(f"recorded total_steps {total_steps} != {derived_steps} derived from "
         f"rows_train={rows_train} at 1/16/1")

# --- the two numbers that must never be carried over from the standard arm ---
# The check that matters is provenance, not the literal values: cap_derivation is
# written by `create` as the arithmetic that produced the cap FROM this
# certificate's own measured max response, so requiring both numbers to appear in
# it proves the cap was derived here rather than pasted from the standard arm.
# A numeric blocklist on 1800/1699 would instead deadlock the arm in the (remote)
# case where the residual target legitimately measures those values.
if max_response < 20:
    fail(f"max_response_tokens {max_response} < 20: collapsed targets")
if cap <= max_response:
    fail(f"residual_generation_cap {cap} does not strictly exceed max_response_tokens "
         f"{max_response}")
derivation_record = document.get("cap_derivation")
if not isinstance(derivation_record, str) or not derivation_record:
    fail("certificate records no cap_derivation; the generation cap cannot be shown "
         "to have been derived from the residual measurement")
if str(max_response) not in derivation_record or str(cap) not in derivation_record:
    fail(f"cap_derivation {derivation_record!r} does not relate the recorded cap {cap} "
         f"to the recorded max_response_tokens {max_response}: the cap was not derived "
         f"from this measurement")
if cap == 1800 or max_response == 1699:
    print(f"[warn] residual cap/{max_response} coincides with the STANDARD arm's "
          f"1800/1699. cap_derivation shows it was derived from the residual "
          f"measurement, so this run continues -- but confirm it before Stage 6.",
          file=sys.stderr)

# --- sequence budget, and the MEASURED --max_length --------------------------
if max_total >= certified_max_length:
    fail(f"max_total_sequence_tokens {max_total} does not fit under the certified "
         f"budget {certified_max_length}")
if certified_max_length > 8192:
    fail(f"certified schedule.max_length {certified_max_length} exceeds 8192")

# ONE source for --max_length: schedule.max_length, as recorded in this
# certificate by Stage 3 from the budget the tokenizer probe actually validated,
# and the same value gemma2_residual_smoke.sbatch certifies by default. Deriving
# a tighter budget here instead is what deadlocked Stage 4 against Stage 5: the
# smoke recorded 8192, this job computed ~1300-1800 from max_total, and the smoke
# gate below refused the mismatch on every allocation #1. C2SDataset does not pad
# to max_length, so the tighter budget bought nothing and diverged from both the
# Pythia arm and the standard Gemma arm.
if max_length_override:
    try:
        max_length = int(max_length_override)
    except ValueError:
        fail(f"RESIDUAL_MAX_LENGTH is not an integer: {max_length_override!r}")
    derivation = (f"RESIDUAL_MAX_LENGTH override = {max_length}; Stage 4 must have been "
                  f"re-run with SMOKE_MAX_LENGTH={max_length}")
else:
    max_length = certified_max_length
    derivation = (f"schedule.max_length = {max_length} from the certificate, the budget "
                  f"the Stage 3 probe validated and the Stage 4 smoke exercised")
# Strict inequality both ways: above the longest measured row, so no row of
# either file can be truncated (residual_to_sentence always ends in [END_CELL],
# so ANY response truncation loses the sentinel and --strict_token_contract
# raises inside C2SDataset.__getitem__ mid-allocation); and at or below the
# budget the tokenizer probe actually validated.
if max_length <= max_total:
    fail(f"--max_length {max_length} is not strictly above the measured maximum "
         f"sequence {max_total}: truncation is guaranteed")
if max_length > certified_max_length:
    fail(f"--max_length {max_length} exceeds the certified budget "
         f"{certified_max_length}; the probe never validated that length")

# --- file identity -----------------------------------------------------------
standard_data = Path(standard_data_dir)
for label, path in (("train", train_file), ("eval", eval_file)):
    if not path.is_absolute():
        fail(f"{label} file path is not absolute: {path}")
    if not path.is_file():
        fail(f"{label} file does not exist: {path}")
    if path.parent != target_dir:
        fail(f"{label} file is not inside target_dir {target_dir}: {path}")
    try:
        path.relative_to(standard_data)
    except ValueError:
        pass
    else:
        fail(f"{label} file points into the STANDARD data directory: {path}")
if train_file == eval_file:
    fail("train and eval file are the same path")
if train_file.name != "residual.jsonl":
    fail(f"train file is not residual.jsonl: {train_file.name}")
if eval_file.name != "residual_val.jsonl":
    fail(f"eval file is not residual_val.jsonl: {eval_file.name}")
if rows_val >= rows_train:
    fail(f"rows_val {rows_val} is not smaller than rows_train {rows_train}")

prompt_order = document.get("prompt_order_in_data")
for line in (str(train_file), str(eval_file), train_digest, eval_digest,
             str(rows_train), str(rows_val), str(total_steps), str(max_length),
             str(cap), str(max_response), str(max_total), str(target_dir),
             sha256(cert_path), str(prompt_order), derivation):
    print(line)
PY
)
[[ "${#RESIDUAL_VALUES[@]}" -eq 15 ]] || {
    echo "[FATAL] residual data certificate rejected (see message above)" >&2; exit 2;
}
TRAIN_FILE="${RESIDUAL_VALUES[0]}"
EVAL_FILE="${RESIDUAL_VALUES[1]}"
TRAIN_SHA256="${RESIDUAL_VALUES[2]}"
EVAL_SHA256="${RESIDUAL_VALUES[3]}"
ROWS_TRAIN="${RESIDUAL_VALUES[4]}"
ROWS_VAL="${RESIDUAL_VALUES[5]}"
RESIDUAL_TOTAL="${RESIDUAL_VALUES[6]}"
MAX_LENGTH="${RESIDUAL_VALUES[7]}"
RESIDUAL_GENERATION_CAP="${RESIDUAL_VALUES[8]}"
MAX_RESPONSE_TOKENS="${RESIDUAL_VALUES[9]}"
MAX_TOTAL_SEQUENCE_TOKENS="${RESIDUAL_VALUES[10]}"
TGT="${RESIDUAL_VALUES[11]}"
RESIDUAL_CERT_SHA256="${RESIDUAL_VALUES[12]}"
PROMPT_ORDER_IN_DATA="${RESIDUAL_VALUES[13]}"
MAX_LENGTH_DERIVATION="${RESIDUAL_VALUES[14]}"

echo "[PASS] residual data authorization: $RESIDUAL_CERT"
echo "       target_dir              = $TGT"
echo "       train_file              = $TRAIN_FILE ($ROWS_TRAIN rows, sha256 $TRAIN_SHA256)"
echo "       eval_file               = $EVAL_FILE ($ROWS_VAL rows, sha256 $EVAL_SHA256)"
echo "       total_optimizer_steps   = $RESIDUAL_TOTAL"
echo "       max_response_tokens     = $MAX_RESPONSE_TOKENS   (MEASURED; the standard arm's 1699 does NOT apply)"
echo "       max_total_sequence      = $MAX_TOTAL_SEQUENCE_TOKENS"
echo "       --max_length            = $MAX_LENGTH   [$MAX_LENGTH_DERIVATION]"
echo "       residual_generation_cap = $RESIDUAL_GENERATION_CAP   (for Stage 6 eval; 1800 does NOT apply)"
echo "       prompt_order_in_data    = $PROMPT_ORDER_IN_DATA   (Stage 6 eval must use this)"

# --- (c) explicit file tests -------------------------------------------------
# train_c2s_tahoe_endcell.py:1501 gates the eval loader on
#   `if args.eval_file and os.path.exists(args.eval_file)`
# so a mistyped eval path produces no loader, no warning and no error. Worse,
# the STANDARD eval file exists and loads and yields a smooth eval loss over a
# completely different output distribution. Refuse here rather than shrug there.
[[ -f "$TRAIN_FILE" ]] || { echo "[FATAL] train file missing: $TRAIN_FILE" >&2; exit 2; }
[[ -f "$EVAL_FILE" ]]  || { echo "[FATAL] eval file missing: $EVAL_FILE" >&2; exit 2; }
[[ "$EVAL_FILE" != "$STANDARD_DATA/eval_tier1_seen_conditions.jsonl" ]] || {
    echo "[FATAL] eval file is the STANDARD tier1 eval; that is a different distribution" >&2
    exit 2
}
[[ "$TRAIN_FILE" != "$STANDARD_DATA/train.jsonl" ]] || {
    echo "[FATAL] train file is the STANDARD training file" >&2; exit 2;
}

# --- (d) row-count gate, from the Stage 2 measurement -----------------------
OBSERVED_ROWS="$(wc -l < "$TRAIN_FILE")"
[[ "$OBSERVED_ROWS" -eq "$ROWS_TRAIN" ]] || {
    echo "[FATAL] residual train row count is $OBSERVED_ROWS, certificate says $ROWS_TRAIN" >&2
    exit 2
}
OBSERVED_VAL_ROWS="$(wc -l < "$EVAL_FILE")"
[[ "$OBSERVED_VAL_ROWS" -eq "$ROWS_VAL" ]] || {
    echo "[FATAL] residual val row count is $OBSERVED_VAL_ROWS, certificate says $ROWS_VAL" >&2
    exit 2
}
echo "[PASS] row-count gate: train=$ROWS_TRAIN val=$ROWS_VAL -> $RESIDUAL_TOTAL optimizer steps"
echo "[NOTE] residual_val.jsonl is carved from TRAIN wells and covers very few of them."
echo "       Its eval loss is a same-distribution sanity trace, not an arm-level result."

# --- (e) interruption/resume smoke authorization ----------------------------
[[ -f "$SMOKE_CERT" ]] || {
    echo "[FATAL] no residual interruption/resume smoke authorization: $SMOKE_CERT" >&2
    echo "        Run: sbatch endcell/jobs/gemma2_residual_smoke.sbatch" >&2
    exit 2
}
"$PY" - "$SMOKE_CERT" "$CERT" "$SNAPSHOT_PATH" "$SNAPSHOT_INVENTORY_SHA256" \
    "$AUTHORITATIVE_FILES_SHA256" "$TRAINER" "$SMOKE_JOB" "$RESIDUAL_TOTAL" \
    "$RESIDUAL_CERT_SHA256" "$MAX_LENGTH" <<'PY'
import hashlib, json, sys
from pathlib import Path

(smoke_path, cert_path, parent_snapshot, expected_snapshot_digest,
 expected_authoritative_digest, trainer_path, smoke_job_path, residual_total,
 residual_cert_digest, max_length) = sys.argv[1:]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


document = json.load(open(smoke_path, encoding="utf-8"))
if document.get("schema_version") != 4 or document.get("smoke_passed") is not True:
    raise SystemExit("[FATAL] invalid smoke authorization")
if document.get("preflight_certificate", {}).get("sha256") != sha256(cert_path):
    raise SystemExit("[FATAL] smoke was authorized by a different preflight certificate")
if document.get("parent_snapshot_path") != str(Path(parent_snapshot).absolute()):
    raise SystemExit("[FATAL] smoke used a different parent snapshot load source")
if document.get("parent_snapshot_inventory_sha256") != expected_snapshot_digest:
    raise SystemExit("[FATAL] smoke used a different parent snapshot inventory")
if document.get("parent_authoritative_files_sha256") != expected_authoritative_digest:
    raise SystemExit("[FATAL] smoke used a different official revision manifest")
# The unforked shared trainer is the whole point of this design: if it changed,
# the standard arm is already stranded and this arm is not the same experiment.
if document.get("sources", {}).get("trainer", {}).get("sha256") != sha256(trainer_path):
    raise SystemExit("[FATAL] trainer changed since interruption/resume smoke")
if document.get("sources", {}).get("smoke_job", {}).get("sha256") != sha256(smoke_job_path):
    raise SystemExit("[FATAL] smoke job changed since the passing smoke")
times = document.get("checkpoint_seconds")
if not isinstance(times, list) or len(times) != 2 or max(times) > 420:
    raise SystemExit(f"[FATAL] unsafe or missing checkpoint timings: {times}")
if "H200" not in str(document.get("gpu_name", "")) or int(document.get("gpu_hbm_mib", 0)) < 135000:
    raise SystemExit("[FATAL] passing smoke was not measured on the required H200/HBM class")
expected_steps = {"first": 50, "resumed": 100, "production_total": int(residual_total)}
if document.get("steps") != expected_steps:
    raise SystemExit(
        f"[FATAL] smoke exercised {document.get('steps')!r}, expected {expected_steps!r}: "
        f"the smoke ran against a different residual schedule")
# Optional, but verified whenever the residual smoke records them: the smoke
# should have exercised the same residual data certificate and the same
# --max_length this job is about to train under. These keys do not exist in the
# standard schema-4 certificate, so they are checked only if present rather than
# required -- an absent key means the smoke job did not record it, not that it
# disagreed.
recorded_certificate = document.get("residual_data_certificate_sha256")
if recorded_certificate is not None and recorded_certificate != residual_cert_digest:
    raise SystemExit("[FATAL] smoke was run against a different residual data certificate")
recorded_length = document.get("max_length")
if recorded_length is None:
    recorded_length = document.get("schedule", {}).get("max_length")
if recorded_length is not None and int(recorded_length) != int(max_length):
    raise SystemExit(
        f"[FATAL] smoke ran at --max_length {recorded_length}, this job runs at "
        f"{max_length}: the smoke did not exercise this sequence budget.\n"
        f"        Both numbers now come from one place -- schedule.max_length in "
        f"RESIDUAL_DATA_PASSED.json --\n"
        f"        so this can only differ if RESIDUAL_MAX_LENGTH was set here, or "
        f"SMOKE_MAX_LENGTH there.\n"
        f"        Re-run Stage 4 at the budget this job will use:\n"
        f"          SMOKE_MAX_LENGTH={max_length} sbatch "
        f"endcell/jobs/gemma2_residual_smoke.sbatch\n"
        f"        or unset RESIDUAL_MAX_LENGTH and resubmit.")
print("[PASS] exact preflight/trainer passed the residual interruption-resume smoke")
PY

[[ -f "$FINGERPRINT_TOOL" ]] || {
    echo "[FATAL] checkpoint validator missing: $FINGERPRINT_TOOL" >&2; exit 2;
}

# --- (f) disk, on the filesystem the residual tree will grow on -------------
# $OUT does not exist yet on a fresh run -- it is not created until every gate
# has passed -- so measure the filesystem via the nearest existing ancestor,
# which is the same filesystem the tree will grow on.
DF_TARGET="$OUT"
while [[ ! -d "$DF_TARGET" ]]; do
    PARENT="$(dirname "$DF_TARGET")"
    [[ "$PARENT" != "$DF_TARGET" ]] || {
        echo "[FATAL] no existing ancestor of $OUT to measure free space on" >&2; exit 2;
    }
    DF_TARGET="$PARENT"
done
AVAILABLE_BYTES="$(df -B1 --output=avail "$DF_TARGET" | tail -1 | tr -d ' ')"
REQUIRED_BYTES="$((MIN_FREE_GIB * 1024 * 1024 * 1024))"
[[ "$AVAILABLE_BYTES" =~ ^[0-9]+$ && "$AVAILABLE_BYTES" -ge "$REQUIRED_BYTES" ]] || {
    echo "[FATAL] production requires at least $MIN_FREE_GIB GiB free on the residual tree's " \
         "filesystem; the standard arm is still publishing to the same /data. " \
         "available_bytes=${AVAILABLE_BYTES:-unknown}" >&2
    exit 2
}
echo "[PASS] production disk gate: available_bytes=$AVAILABLE_BYTES >= $REQUIRED_BYTES ($MIN_FREE_GIB GiB) on $DF_TARGET"
echo "[PASS] CPU preflight complete; proceeding to GPU gates"

# =============================================================================
# GPU GATES
# =============================================================================
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_HBM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d " ")"
[[ "$GPU_NAME" == *H200* ]] || {
    echo "[FATAL] canonical job requires NVIDIA H200, got $GPU_NAME" >&2; exit 2;
}
[[ "$GPU_HBM" =~ ^[0-9]+$ && "$GPU_HBM" -ge "$MIN_HBM_MIB" ]] || {
    echo "[FATAL] require at least $MIN_HBM_MIB MiB HBM, got ${GPU_HBM:-unknown}" >&2
    exit 2
}

# Hold this advisory lock for the complete batch job. Its content records the
# owner. The new output path gives the residual arm a lock file distinct from
# the standard arm's, so the two can hold allocations concurrently.
command -v flock >/dev/null 2>&1 || { echo "[FATAL] flock unavailable" >&2; exit 3; }
# FIRST creation of the output tree, and the last moment before it: every CPU
# gate and both GPU gates are behind us, so no refusal can leave an empty $OUT
# behind for gemma2_residual_inventory.sh to trip over on the next Stage 1.
mkdir -p "$OUT"
exec 9>"$OUT/$LOCK_NAME"
flock -n 9 || {
    echo "[FATAL] another job owns $OUT: $(cat "$OUT/$LOCK_NAME" 2>/dev/null || echo unknown)" >&2
    exit 3
}
printf "job_id=%s host=%s started_utc=%s\n" "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)" >&9

HELP="$($PY "$TRAINER" --help 2>&1)"
for flag in --prepend_bos --strict_token_contract --resumable \
    --resume_from_checkpoint --model_revision --model_load_path --attn_implementation \
    --preflight_certificate_sha256 --parent_snapshot_inventory_sha256 \
    --parent_authoritative_files_sha256; do
    grep -Fq -- "$flag" <<<"$HELP" || { echo "[FATAL] trainer interface missing $flag" >&2; exit 3; }
done

# =============================================================================
# OUTPUT-TREE STATE: resume, cold start, or refuse. Never clobber.
# =============================================================================
RESUME_ARGS=()
ANCESTRY_ARGS=(
    --require_gemma_ancestry
    --expected_model_id "$MODEL"
    --expected_revision "$REVISION"
    --expected_parent_snapshot "$SNAPSHOT_PATH"
    --expected_preflight_certificate_sha256 "$PREFLIGHT_SHA256"
    --expected_snapshot_inventory_sha256 "$SNAPSHOT_INVENTORY_SHA256"
    --expected_authoritative_files_sha256 "$AUTHORITATIVE_FILES_SHA256"
)
mapfile -t PUBLISHED_STATES < <(published_states)
mapfile -t PUBLISHED_DIRS < <(find "$OUT" -mindepth 1 -maxdepth 1 -type d \
    -name "checkpoint-[0-9]*" | sort -V)
for directory in "${PUBLISHED_DIRS[@]}"; do
    [[ -f "$directory/training_state.pt" ]] || {
        echo "[FATAL] published checkpoint lacks training_state.pt: $directory" >&2; exit 4;
    }
done
if find "$OUT" -mindepth 1 -maxdepth 1 -type d -name ".checkpoint-*.tmp-*" | grep -q .; then
    echo "[FATAL] unpublished temporary checkpoint exists; manual audit required" >&2
    echo "        A publish that ran out of space leaves these behind. Check df before retrying." >&2
    exit 4
fi

# Re-assert the resume contract under the lock, in case state appeared between
# the fast-fail scan and here.
if [[ "${#PUBLISHED_STATES[@]}" -gt 0 && "$RESUME_REQUESTED" -eq 0 ]]; then
    echo "[FATAL] published checkpoint state appeared in $OUT; refusing without --resume" >&2
    exit 4
fi

if [[ -f "$OUT/final/config.json" ]]; then
    "$PY" "$FINGERPRINT_TOOL" --checkpoint "$OUT/final" --require_complete_sft \
        --expected_global_step "$RESIDUAL_TOTAL" --expected_epoch 1 \
        --expected_microbatch_position 0 --expected_accumulation_position 0 \
        "${ANCESTRY_ARGS[@]}" --digest_only >/dev/null
    echo "[PASS] residual final checkpoint manifest, content, and terminal state validated; no rerun"
    exit 0
elif [[ "${#PUBLISHED_STATES[@]}" -gt 0 ]]; then
    RESUME_ARGS=(--resume_from_checkpoint auto)
    echo "[resume] ${#PUBLISHED_STATES[@]} published checkpoint state(s); strict auto-resume required"
else
    # run_provenance.json is EXEMPT. The trainer writes it immediately after the
    # model and dataset load and before the first optimizer step
    # (train_c2s_tahoe_endcell.py: os.makedirs(args.output_dir) then
    # _atomic_write_json(.../PROVENANCE_FILE)), while the first publish is
    # --save_every 1000 steps -- potentially hours -- later. So any cold-start
    # death in between (OOM, node failure, walltime kill) left a tree holding
    # exactly one file, which the cold-start path refused here and the --resume
    # path refused at the fast-fail: a run that could be neither continued nor
    # restarted, with invariant 4 forbidding the only advice offered. It is a
    # startup artifact, not evidence of progress, and it is rewritten on the
    # next start.
    #
    # best/ is deliberately NOT exempt. It only exists if an eval improved, i.e.
    # if the dead allocation really did train, and a cold start would overwrite
    # it. That tree wants the audit this refusal demands -- and now that
    # GEMMA_RESIDUAL_OUT exists, the refusal's advice can actually be taken.
    mapfile -t UNEXPECTED < <(find "$OUT" -mindepth 1 -maxdepth 1 \
        ! -name "$LOCK_NAME" ! -name "$PIN_NAME" ! -name "run_provenance.json")
    [[ "${#UNEXPECTED[@]}" -eq 0 ]] || {
        printf "[FATAL] no published resume state, but output is not fresh:\n%s\n" "${UNEXPECTED[*]}" >&2
        printf "        Refusing to write into an occupied tree. Never clear this one.\n" >&2
        printf "        Start over in a NEW tree, without editing this job:\n" >&2
        printf "          GEMMA_RESIDUAL_OUT=<new absolute path> \\\\\n" >&2
        printf "            srun -p defq -c 2 --mem 8G bash endcell/jobs/gemma2_residual_inventory.sh\n" >&2
        printf "          GEMMA_RESIDUAL_OUT=<same new path> \\\\\n" >&2
        printf "            sbatch endcell/jobs/gemma2_residual_train.sbatch\n" >&2
        exit 4
    }
    echo "[fresh] no prior published state; cold-starting pinned cached Gemma on residual targets"
fi

# --- run pin: the parameters of this run, fixed at allocation #1 ------------
# max_length, seed, batch_size, grad_accum, the two data digests and the trainer
# digest all enter the trainer's contract dict, which _validate_resume_fingerprints
# compares on every resume. Pinning them here turns a mid-run drift into a
# refusal that costs seconds, instead of one that costs a model load first.
"$PY" - "$PIN" "$RESIDUAL_CERT_SHA256" "$MAX_LENGTH" "$RESIDUAL_TOTAL" \
    "$TRAIN_SHA256" "$EVAL_SHA256" "$TRAINER" "$TRAIN_FILE" "$EVAL_FILE" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

(pin_path, residual_cert_digest, max_length, total_steps, train_digest, eval_digest,
 trainer_path, train_file, eval_file) = sys.argv[1:]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


pinned = {
    "schema_version": 1,
    "max_length": int(max_length),
    "total_steps": int(total_steps),
    "batch_size": 1,
    "grad_accum": 16,
    "num_epochs": 1,
    "seed": 42,
    "train_file": train_file,
    "eval_file": eval_file,
    "train_sha256": train_digest,
    "eval_sha256": eval_digest,
    "trainer_sha256": sha256(trainer_path),
}
target = Path(pin_path)
if target.is_file():
    existing = json.load(open(target, encoding="utf-8"))
    # Informational only. Every value the trainer's contract actually compares is
    # pinned above in its own right, so a benign re-issue of the certificate --
    # same data, new created_at_utc -- must not strand a live run.
    created = existing.pop("created_at_utc", None)
    previous_certificate = existing.pop("residual_data_certificate_sha256", None)
    if existing != pinned:
        differences = sorted(
            set(existing.items()) ^ set(pinned.items()), key=lambda item: item[0])
        raise SystemExit(
            "[FATAL] run parameters changed since this run started"
            f" (pin written {created}): {differences}. The trainer would refuse to resume;"
            " this run cannot be continued with the current inputs.")
    if previous_certificate is not None and previous_certificate != residual_cert_digest:
        print(f"[warn] RESIDUAL_DATA_PASSED.json was re-issued since allocation #1 "
              f"({previous_certificate} -> {residual_cert_digest}); every pinned run "
              f"parameter still matches, so this run continues.")
    print(f"[PASS] run pin unchanged since allocation #1: {target}")
else:
    pinned["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    pinned["residual_data_certificate_sha256"] = residual_cert_digest
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                     dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(pinned, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"[PASS] run parameters pinned for the life of this run: {target}")
PY

# =============================================================================
# TRAIN
# =============================================================================
# Byte-identical to the standard CLI apart from --train_file, --eval_file,
# --output_dir and --max_length. --seed 42 in particular is inherited and must
# be: DeterministicEpochSampler rebuilds the order from torch.randperm(seed +
# epoch), so a changed seed silently reorders the remaining data on resume.
CMD=(
    "$PY" "$TRAINER" --mode full --model_name "$MODEL" --model_revision "$REVISION" \
    --model_load_path "$SNAPSHOT_PATH"
    --preflight_certificate_sha256 "$PREFLIGHT_SHA256"
    --parent_snapshot_inventory_sha256 "$SNAPSHOT_INVENTORY_SHA256"
    --parent_authoritative_files_sha256 "$AUTHORITATIVE_FILES_SHA256"
    --train_file "$TRAIN_FILE" --eval_file "$EVAL_FILE"
    --output_dir "$OUT" --num_epochs 1 --batch_size 1 --grad_accum 16
    --bf16 --gradient_checkpointing --max_length "$MAX_LENGTH"
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
cp "$RESIDUAL_CERT" "$RUN_META/RESIDUAL_DATA_PASSED.json"
cp "$SMOKE_CERT" "$RUN_META/SMOKE_PASSED.json"
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
    echo "[signal] resumable checkpoint published; continue with:"
    echo "[signal]   sbatch endcell/jobs/gemma2_residual_train.sbatch --resume"
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
    --expected_global_step "$RESIDUAL_TOTAL" --expected_epoch 1 \
    --expected_microbatch_position 0 --expected_accumulation_position 0 \
    "${ANCESTRY_ARGS[@]}" --digest_only >/dev/null
echo "[PASS] final manifest/content and terminal state: completed=true, step=$RESIDUAL_TOTAL, epoch=1, microbatch=0, accumulation=0"
echo "[PASS] one-epoch RESIDUAL-target SFT completed -> $OUT/final"
echo "[next] Stage 6 eval must use residual_generation_cap=$RESIDUAL_GENERATION_CAP (NOT 1800)"
echo "[next] and --prompt_order from ${TGT}/report.json (build default: drug_first)"
