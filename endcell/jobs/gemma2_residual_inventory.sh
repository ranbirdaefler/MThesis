#!/usr/bin/env bash
# =============================================================================
# gemma2_residual_inventory.sh — Stage 1: WHICH residual build, resolved by DIGEST
# =============================================================================
#
# WHY THIS EXISTS
#   Six residual_targets* directories are referenced in this project, built with
#   different --split_unit, --repro_thr and --generic_scope. Every one of them
#   trains cleanly and produces a plausible number. Only one of them is the build
#   the thesis reports for Pythia. The authority on that is not HPC_QUEUE.md prose
#   and not a "GATE PASSED" line in an old log: it is the contract dict inside the
#   Pythia residual checkpoint's training_state.pt, which records the sha256 of the
#   exact bytes it was trained on (train_c2s_tahoe_endcell.py:1062, _training_contract).
#
#   This script resolves the Gemma residual arm's data directory by comparing
#   sha256(residual.jsonl) against that recorded train_sha256. If nothing matches,
#   it STOPS. That is an author decision (rebuild to mirror the reported arm, or
#   re-scope the claim), not something a script should paper over.
#
# WHAT IT ALSO SETTLES, while the Pythia checkpoint is already open
#   The Pythia residual arm's lr / warmup_ratio / weight_decay / num_epochs /
#   grad_accum / max_length / seed. The Gemma arm is a matched replication only if
#   these agree; the script prints them side by side with the Gemma arm's intended
#   values and flags any difference. It does not fail on a difference — a deliberate
#   backbone-driven change is legitimate — but it makes it impossible to ship one
#   by accident.
#
# HOW TO RUN  (login node runs srun; the work runs on a compute node)
#
#     cd "$HOME/tahoe"
#     srun -p defq -c 2 --mem 8G --time 00:30:00 --pty \
#         bash endcell/jobs/gemma2_residual_inventory.sh
#
#   Non-interactive equivalent (no --pty), which is what you want if you are
#   piping the output to a file:
#
#     srun -p defq -c 2 --mem 8G --time 00:30:00 \
#         bash endcell/jobs/gemma2_residual_inventory.sh \
#         2>&1 | tee logs/gemma2_residual_inventory.log
#
# WRITES (new paths only; never deletes or overwrites data, caches or checkpoints)
#     RESULTS/gemma2_residual/inventory.json
#     RESULTS/gemma2_residual/RESOLVED_TGT.txt   <- single line, the resolved dir
#
# EXIT CODES
#     0  exactly one candidate matched; RESOLVED_TGT.txt written
#     2  environment/precondition failure (not on a compute node, missing python)
#     3  Gemma residual output dir already exists, or insufficient free space
#     4  zero or more than one candidate matched the Pythia checkpoint digest
#     5  no Pythia residual checkpoint could be read at all
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME

# ---- Invariant 3: never run computation on the login node -------------------
# torch.load of a Pythia optimizer state is real work and real memory. Refuse
# unless we are inside a Slurm allocation step.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    cat >&2 <<'USAGE'
[FATAL] this script must not run on the login node.

Re-run it inside an allocation:

    srun -p defq -c 2 --mem 8G --time 00:30:00 \
        bash endcell/jobs/gemma2_residual_inventory.sh
USAGE
    exit 2
fi

PY="${PY:-/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python}"
REPO="${REPO:-$HOME/tahoe}"
ROOT="${RESIDUAL_ROOT:-/data/BuffaF-Projetcs/florian_c2s}"
GEMMA_OUT="${GEMMA_RESIDUAL_OUT:-$ROOT/checkpoints/gemma2b_sft_endcell_residual}"
RESULT="$REPO/RESULTS/gemma2_residual"
MIN_FREE_GIB="${MIN_FREE_GIB:-150}"

[[ -x "$PY" ]] || { echo "[FATAL] canonical python not executable: $PY" >&2; exit 2; }
[[ -d "$ROOT" ]] || { echo "[FATAL] project root not found: $ROOT" >&2; exit 2; }
mkdir -p "$RESULT"
cd "$REPO"

echo "[info] node=$(hostname) job=$SLURM_JOB_ID root=$ROOT"

# ---- Invariant 4: the Gemma residual tree must be NEW ------------------------
# Checked before anything expensive runs, and checked again by the train job.
if [[ -e "$GEMMA_OUT" ]]; then
    echo "[FATAL] Gemma residual output path already exists: $GEMMA_OUT" >&2
    echo "        Refusing to plan a run that would clobber it. Move it aside or" >&2
    echo "        set GEMMA_RESIDUAL_OUT to a different NEW path." >&2
    exit 3
fi

# df on the parent, since GEMMA_OUT itself must not exist yet.
DF_TARGET="$(dirname "$GEMMA_OUT")"
[[ -d "$DF_TARGET" ]] || { echo "[FATAL] checkpoints parent missing: $DF_TARGET" >&2; exit 3; }
AVAIL_BYTES="$(df -B1 --output=avail "$DF_TARGET" | tail -1 | tr -d ' ')"
REQUIRED_BYTES="$((MIN_FREE_GIB * 1024 * 1024 * 1024))"
if ! [[ "$AVAIL_BYTES" =~ ^[0-9]+$ ]] || [[ "$AVAIL_BYTES" -lt "$REQUIRED_BYTES" ]]; then
    echo "[FATAL] need >= ${MIN_FREE_GIB} GiB free on $DF_TARGET; available_bytes=${AVAIL_BYTES:-unknown}" >&2
    echo "        NOTE: the standard arm still needs allocations to finish and wants" >&2
    echo "        the same 150 GiB on this filesystem. Both trees do not fit twice." >&2
    exit 3
fi
echo "[PASS] disk gate on $DF_TARGET: available_bytes=$AVAIL_BYTES >= $REQUIRED_BYTES"

"$PY" - "$ROOT" "$GEMMA_OUT" "$RESULT" "$AVAIL_BYTES" <<'PYEOF'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

root, gemma_out, result_dir, avail_bytes = sys.argv[1:]
root = Path(root)
result_dir = Path(result_dir)

# The full referenced set. Any that do not exist are reported as absent rather
# than skipped, because "the directory the plan named is not on disk" is itself
# a finding the author needs to see.
CANDIDATE_NAMES = [
    "residual_targets_v3",
    "residual_targets_repaired",
    "residual_targets",
    "residual_targets_holdout",
    "residual_targets_holdout2",
    "residual_targets_reorder",
]
REQUIRED_MEMBERS = [
    "residual.jsonl",
    "residual_val.jsonl",
    "report.json",
    "holdout.json",
    "reconstruction.npz",
]
# Build parameters worth surfacing. NOTE: report.json stores the generic scope
# under the key "scope", not "generic_scope" (build_residual_targets.py:1168), and
# it does NOT record prompt_order or val_frac at all. Both of those are therefore
# derived below from the data itself rather than asserted from the report.
REPORT_FIELDS = [
    "repro_thr", "scope", "frame", "split_unit", "k_up", "k_down", "shrink_k",
    "min_plate_drugs", "fit_digest", "leave_one_drug_out", "drug_weighted_generic",
    "split_before_fit", "eval_repro_filter", "n_conditions_inventoried",
    "n_conditions_kept", "n_examples", "n_validation_examples",
    "n_validation_wells", "down_token", "end_token",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path):
    total = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def first_json_line(path):
    with open(path, encoding="utf-8") as handle:
        line = handle.readline()
    return json.loads(line) if line.strip() else None


def derive_prompt_order(row):
    """report.json does not record --prompt_order, so recover it from the prompt.

    format_prompt (build_residual_targets.py:130) emits, for drug_first:
        "Predict the response of ...\nControl cell: ...\n\nResponse cell:"
    and for drug_last:
        "Control cell: ...\nPredict the response of ...\n\nResponse cell:"
    The two are distinguishable by which marker appears first.
    """
    if not row:
        return None
    prompt = row.get("prompt", "")
    i_instr = prompt.find("Predict the response of")
    i_ctrl = prompt.find("Control cell:")
    if i_instr < 0 or i_ctrl < 0:
        return "UNRECOGNIZED"
    return "drug_first" if i_instr < i_ctrl else "drug_last"


# ---------------------------------------------------------------- candidates
candidates = {}
for name in CANDIDATE_NAMES:
    path = root / name
    entry = {"path": str(path), "exists": path.is_dir()}
    if not entry["exists"]:
        entry["complete"] = False
        entry["missing_members"] = REQUIRED_MEMBERS
        candidates[name] = entry
        continue
    present = {m: (path / m).is_file() for m in REQUIRED_MEMBERS}
    entry["members_present"] = present
    entry["missing_members"] = sorted(m for m, ok in present.items() if not ok)
    entry["complete"] = not entry["missing_members"]
    entry["files"] = {}
    for member in ("residual.jsonl", "residual_val.jsonl"):
        member_path = path / member
        if not member_path.is_file():
            continue
        print(f"[scan] hashing {member_path} ...", flush=True)
        entry["files"][member] = {
            "sha256": sha256_file(member_path),
            "bytes": member_path.stat().st_size,
            "rows": count_lines(member_path),
        }
    if (path / "report.json").is_file():
        try:
            report = json.load(open(path / "report.json", encoding="utf-8"))
        except Exception as exc:  # a corrupt report is a finding, not a crash
            entry["report"] = {"UNREADABLE": repr(exc)}
        else:
            entry["report"] = {k: report.get(k) for k in REPORT_FIELDS}
            # val_frac is not recorded; derive the realised value.
            n_val_wells = report.get("n_validation_wells")
            entry["report"]["derived_val_examples"] = report.get("n_validation_examples")
            entry["report"]["recorded_val_wells"] = n_val_wells
    if (path / "residual.jsonl").is_file():
        entry["derived_prompt_order"] = derive_prompt_order(
            first_json_line(path / "residual.jsonl"))
    candidates[name] = entry

# ------------------------------------------------- Pythia residual checkpoints
# The checkpoint is the authority on the bytes it was trained on.
pythia_checkpoints = {}
checkpoint_root = root / "checkpoints"
pythia_dirs = []
if checkpoint_root.is_dir():
    pythia_dirs = sorted(
        d for d in checkpoint_root.iterdir()
        if d.is_dir() and d.name.startswith("pythia_sft_residual"))

if pythia_dirs:
    import torch  # imported late: this is the only step that needs it

for directory in pythia_dirs:
    state_path = directory / "final" / "training_state.pt"
    record = {"path": str(state_path), "exists": state_path.is_file()}
    if record["exists"]:
        print(f"[scan] torch.load {state_path} ...", flush=True)
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            record["UNREADABLE"] = repr(exc)
        else:
            contract = state.get("contract", {}) or {}
            record["global_step"] = state.get("global_step")
            record["epoch"] = state.get("epoch")
            record["completed"] = bool(state.get("completed", False))
            record["train_sha256"] = contract.get("train_sha256")
            record["eval_sha256"] = contract.get("eval_sha256")
            # Hyperparameters, so the Gemma arm can be declared matched or not.
            record["hyperparameters"] = {
                k: contract.get(k) for k in (
                    "learning_rate", "weight_decay", "warmup_ratio", "num_epochs",
                    "batch_size", "grad_accum", "max_length", "seed", "de_weight",
                    "de_share", "bf16", "gradient_checkpointing", "prepend_bos",
                    "strict_token_contract", "model_name", "model_revision",
                    "data_order_contract",
                )
            }
            del state
    pythia_checkpoints[directory.name] = record

# ------------------------------------------------------------------ matching
readable = {n: r for n, r in pythia_checkpoints.items() if r.get("train_sha256")}
reported_train_sha256 = {r["train_sha256"] for r in readable.values()}
reported_eval_sha256 = {r.get("eval_sha256") for r in readable.values() if r.get("eval_sha256")}

for name, entry in candidates.items():
    train_digest = entry.get("files", {}).get("residual.jsonl", {}).get("sha256")
    val_digest = entry.get("files", {}).get("residual_val.jsonl", {}).get("sha256")
    entry["PYTHIA_TRAIN_SHA256_MATCH"] = bool(
        train_digest and train_digest in reported_train_sha256)
    entry["PYTHIA_EVAL_SHA256_MATCH"] = bool(
        val_digest and val_digest in reported_eval_sha256)
    entry["matched_pythia_checkpoints"] = sorted(
        n for n, r in readable.items() if r["train_sha256"] == train_digest)

resolved = sorted(
    n for n, e in candidates.items()
    if e.get("complete") and e.get("PYTHIA_TRAIN_SHA256_MATCH"))

# Hyperparameter comparison against the Gemma arm's intended CLI.
GEMMA_INTENDED = {
    "learning_rate": 1e-5, "weight_decay": 0.01, "warmup_ratio": 0.03,
    "num_epochs": 1, "batch_size": 1, "grad_accum": 16, "max_length": 8192,
    "seed": 42, "de_weight": 1.0, "de_share": None, "bf16": True,
    "gradient_checkpointing": True, "prepend_bos": True,
    "strict_token_contract": True,
}
hyperparameter_diff = {}
matched_checkpoint_names = (
    candidates[resolved[0]].get("matched_pythia_checkpoints", []) if resolved else [])
for name in matched_checkpoint_names:
    saved = readable[name]["hyperparameters"]
    differences = {
        key: {"pythia_residual": saved.get(key), "gemma_residual_intended": value}
        for key, value in GEMMA_INTENDED.items()
        if key in saved and saved.get(key) != value
    }
    hyperparameter_diff[name] = differences

document = {
    "schema_version": 1,
    "root": str(root),
    "gemma_residual_output_dir": {"path": gemma_out, "exists": Path(gemma_out).exists()},
    "free_bytes_on_output_filesystem": int(avail_bytes),
    "candidates": candidates,
    "pythia_residual_checkpoints": pythia_checkpoints,
    "resolved": resolved,
    "hyperparameter_diff_vs_gemma_intended": hyperparameter_diff,
}

result_dir.mkdir(parents=True, exist_ok=True)
target = result_dir / "inventory.json"
fd, temporary = tempfile.mkstemp(prefix=".inventory.", suffix=".tmp", dir=str(result_dir))
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
print(f"\n[write] {target}")

# ------------------------------------------------------------------- reporting
print("\n================ CANDIDATE SUMMARY ================")
for name, entry in candidates.items():
    if not entry["exists"]:
        print(f"  {name:32s} ABSENT")
        continue
    files = entry.get("files", {})
    train = files.get("residual.jsonl", {})
    val = files.get("residual_val.jsonl", {})
    flag = "MATCH" if entry["PYTHIA_TRAIN_SHA256_MATCH"] else "no-match"
    print(f"  {name:32s} complete={str(entry['complete']):5s} {flag}")
    print(f"      residual.jsonl      rows={train.get('rows')} "
          f"bytes={train.get('bytes')} sha256={train.get('sha256')}")
    print(f"      residual_val.jsonl  rows={val.get('rows')} "
          f"bytes={val.get('bytes')} sha256={val.get('sha256')}")
    print(f"      derived_prompt_order={entry.get('derived_prompt_order')}")
    report = entry.get("report", {})
    if report:
        print(f"      report: repro_thr={report.get('repro_thr')} "
              f"scope={report.get('scope')} split_unit={report.get('split_unit')} "
              f"k_up={report.get('k_up')} k_down={report.get('k_down')} "
              f"shrink_k={report.get('shrink_k')} "
              f"min_plate_drugs={report.get('min_plate_drugs')}")
        print(f"              fit_digest={report.get('fit_digest')}")

print("\n============ PYTHIA RESIDUAL CHECKPOINTS ============")
if not pythia_checkpoints:
    print("  (none found under checkpoints/pythia_sft_residual*)")
for name, record in pythia_checkpoints.items():
    print(f"  {name}")
    print(f"      exists={record.get('exists')} completed={record.get('completed')} "
          f"global_step={record.get('global_step')} epoch={record.get('epoch')}")
    print(f"      train_sha256={record.get('train_sha256')}")
    print(f"      eval_sha256={record.get('eval_sha256')}")
    hyper = record.get("hyperparameters") or {}
    if hyper:
        print(f"      lr={hyper.get('learning_rate')} wd={hyper.get('weight_decay')} "
              f"warmup_ratio={hyper.get('warmup_ratio')} epochs={hyper.get('num_epochs')} "
              f"grad_accum={hyper.get('grad_accum')} max_length={hyper.get('max_length')} "
              f"seed={hyper.get('seed')}")

if hyperparameter_diff:
    print("\n===== HYPERPARAMETER DIFF vs the Gemma arm's intended CLI =====")
    for name, differences in hyperparameter_diff.items():
        if not differences:
            print(f"  {name}: identical on every compared field "
                  f"-> matched replication")
        else:
            print(f"  {name}: DIFFERS on {sorted(differences)}")
            for key, pair in sorted(differences.items()):
                print(f"      {key}: pythia={pair['pythia_residual']!r} "
                      f"gemma_intended={pair['gemma_residual_intended']!r}")
            print("      ^ NOT a matched replication on these fields. Decide "
                  "deliberately before Stage 5.")

# ------------------------------------------------------------------- verdict
if not readable:
    print("\n[FATAL] no Pythia residual checkpoint could be read, so there is no "
          "authority on which build the thesis reports.", file=sys.stderr)
    print("        Looked under: " + str(checkpoint_root / "pythia_sft_residual*"),
          file=sys.stderr)
    raise SystemExit(5)

if len(resolved) != 1:
    print(f"\n[FATAL] {len(resolved)} candidate directories match the Pythia residual "
          f"checkpoint's recorded train_sha256; exactly 1 is required.", file=sys.stderr)
    print(f"        matched={resolved}", file=sys.stderr)
    print("        STOP. If zero matched, the Gemma arm would not be mirroring the arm",
          file=sys.stderr)
    print("        the thesis reports. That is an author decision (rebuild to mirror it,",
          file=sys.stderr)
    print("        or re-scope the claim), not something this script should paper over.",
          file=sys.stderr)
    print("        If you rebuild: fit_digest reporting SAME is NOT evidence of",
          file=sys.stderr)
    print("        equivalence -- it is computed before --repro_thr resolves. Only the",
          file=sys.stderr)
    print("        residual.jsonl sha256 is evidence.", file=sys.stderr)
    raise SystemExit(4)

target_path = candidates[resolved[0]]["path"]
tgt_file = result_dir / "RESOLVED_TGT.txt"
with open(tgt_file, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(target_path + "\n")
print(f"\n[PASS] resolved residual target directory: {target_path}")
print(f"[write] {tgt_file}")
print("[next] Stage 2:  srun -p defq -c 2 --mem 16G --time 01:00:00 \\")
print("           $PY endcell/jobs/gemma2_residual_data_audit.py \\")
print(f"               --target_dir {target_path}")
PYEOF
