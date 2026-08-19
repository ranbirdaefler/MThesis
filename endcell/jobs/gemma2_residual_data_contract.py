#!/usr/bin/env python3
r"""Residual-scoped data certificate: create and verify RESIDUAL_DATA_PASSED.json.

WHY A SEPARATE CERTIFICATE INSTEAD OF EXTENDING THE STANDARD ONE
  gemma2_standard_preflight_contract.py pins EXPECTED_KEYS['data'] to exactly
  {train, tier1..tier4} against five hardcoded CANONICAL_TAHOE_DATA_SHA256 entries,
  and its validate_probe requires exactly five files with response_contains_down == 0.
  Residual data violates all three by construction: two files, a different gene
  universe, and [DOWN] in every response.

  The routing that makes this work without touching a single constant, verified at
  gemma2_standard_preflight_contract.py:488 (`if args.require_generation_cap is not
  None`): the residual jobs call the EXISTING `verify` subcommand WITHOUT
  --require-generation-cap. verify then still checks snapshot inventory,
  authoritative revision bytes, protobuf tree, pip freeze, tokenizer 256000/256002
  and sentinels 256000/256001 -- everything about the MODEL -- while the 1800 cap and
  the five canonical data digests stay bound to the standard data, where they belong.
  Residual DATA is gated here instead.

  Do NOT "fix" the standard constants to accommodate the residual arm. That breaks
  the live standard arm's certificate and tests/test_gemma_phase1a_training_contract.py.

THE THING THIS FILE EXISTS TO MEASURE
  The 1800 generation cap was derived from a maximum truth response of 1699 tokens
  under the Gemma tokenizer FOR THE STANDARD TARGET (~946-1500 gene symbols). The
  residual response is a different string: ~200 gene symbols split by [DOWN]. Reusing
  1800 would almost certainly be non-binding and therefore harmless-looking, which is
  exactly why it would survive review -- but the arm would have shipped a cap nobody
  measured. This derives the cap from the measured maximum instead.

THE ONLY UNK DETECTOR IN THE PIPELINE
  The residual prompt embeds an ot_cache panel control sentence -- a gene universe
  that has never been passed through the Gemma tokenizer in this project. The trainer
  does NOT check for unknown tokens; --strict_token_contract only checks sentinels and
  truncation. The base Gemma tokenizer's UNK path is demonstrably live: before
  sentinel registration it maps [END_CELL] to id 3. An UNK-riddled residual target
  would train to completion and quietly degrade the arm. gemma_tokenizer_probe.py
  raises on a non-zero UNK count for Gemma, and this re-checks the recorded counters
  so the certificate itself carries the proof.

SUBCOMMANDS
  create   validate a probe report + row audit, derive the cap, write the certificate
  verify   re-hash the residual jsonls and confirm they still match the certificate
           (this is the FREEZE gate: build_residual_targets.py does not refuse an
           existing out_dir -- it os.replace's over residual.jsonl -- and every
           published checkpoint hashes both files into its resume contract via
           _training_contract, so a mid-run rebuild strands the run permanently)

EXIT CODES: 0 pass, non-zero on any violation. Job scripts gate on this.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

CERTIFICATE_SCHEMA_VERSION = 1
PROBE_SCHEMA_VERSION = 3
PARITY_ROWS = 512
GEMMA_MODEL_ID = "vandijklab/C2S-Scale-Gemma-2-2B"

# Same counter set the standard contract demands, so a probe that silently stopped
# emitting one is caught here too.
MANDATORY_COUNTERS = {
    "rows", "semantic_truncations", "prompt_truncations", "sentinel_losses",
    "response_missing_end_cell", "response_contains_down", "prompt_unk_tokens",
    "response_unk_tokens", "prompt_token_count", "response_token_count",
}
# Rounded up to this multiple, then required to be strictly greater than the
# measured maximum so the cap is provably non-binding for truth.
CAP_ROUNDING = 50
MIN_SEMANTIC_TOKENS = 20


def fail(message):
    raise SystemExit(f"[FATAL] {message}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, document):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
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
    return target


def derive_generation_cap(max_response_tokens):
    """Next multiple of CAP_ROUNDING strictly greater than the measured maximum.

    Strict inequality, not >=: the response already contains [END_CELL], so a cap
    equal to the maximum truth length leaves no room for the terminator to be
    emitted and the cap would be binding on the longest true target.
    """
    cap = int(math.ceil((max_response_tokens + 1) / CAP_ROUNDING) * CAP_ROUNDING)
    if cap <= max_response_tokens:
        cap += CAP_ROUNDING
    return cap


def validate_residual_probe(probe_path, train_path, val_path, snapshot_path,
                            model_id, revision):
    """Residual-scoped mirror of the standard validate_probe. Two files, not five;
    response_contains_down required rather than forbidden."""
    report = json.load(open(probe_path, encoding="utf-8"))
    if report.get("schema_version") != PROBE_SCHEMA_VERSION:
        fail(f"unsupported tokenizer-probe schema: {report.get('schema_version')}")
    if report.get("max_examples_per_file") != 0:
        fail("tokenizer audit was not run over every row (--max_examples must be 0)")

    expected_by_path = {
        str(Path(train_path).resolve()): sha256_file(train_path),
        str(Path(val_path).resolve()): sha256_file(val_path),
    }
    if len(expected_by_path) != 2:
        fail("residual train and val resolve to the same path")
    input_hashes = report.get("input_hashes")
    if not isinstance(input_hashes, dict):
        fail("tokenizer audit has no input_hashes mapping")
    normalized = {str(Path(p).resolve()): d for p, d in input_hashes.items()}
    if normalized != expected_by_path:
        fail("tokenizer audit input_hashes do not match the residual files on disk; "
             f"probe={normalized} disk={expected_by_path}")

    models = report.get("models", [])
    if len(models) != 1:
        fail(f"tokenizer audit must contain exactly one model, got {len(models)}")
    model = models[0]
    if (model.get("model_name"), model.get("revision")) != (model_id, revision):
        fail("tokenizer audit used a different model or revision")

    expected_load_source = {
        "kind": "exact_local_snapshot",
        "path": os.path.abspath(snapshot_path),
        "local_files_only": True,
        "revision_argument": None,
    }
    if model.get("load_source") != expected_load_source:
        fail("tokenizer audit did not load the exact certified snapshot. Loading the "
             "tokenizer by HuggingFace model id on this cluster once resolved a broken "
             "5-token vocabulary; only the pinned snapshot path is safe.")
    if model.get("snapshot_load_guard") != {
            "before_and_after_each_load": True,
            "operations": ["config", "fast_tokenizer", "slow_tokenizer"]}:
        fail("tokenizer audit did not guard every exact-snapshot load")

    if model.get("model_config_vocab_size") != 256_000:
        fail("pinned Gemma model config vocab_size is not 256000")
    base = model.get("base_tokenizer_contract") or {}
    if (base.get("vocab_size"), base.get("length")) != (256_000, 256_000):
        fail("Gemma base tokenizer is collapsed or mismatched; expected 256000/256000 "
             "before sentinel registration")
    tokenizer = model.get("tokenizer", {})
    if tokenizer.get("is_fast") is not True:
        fail("tokenizer audit did not use the required fast tokenizer")
    if (tokenizer.get("vocab_size"), tokenizer.get("length_after_sentinels")) != (
            256_000, 256_002):
        fail("Gemma tokenizer contract is not 256000 base / 256002 with sentinels")
    if tokenizer.get("sentinels") != {"[END_CELL]": 256_000, "[DOWN]": 256_001}:
        fail(f"Gemma sentinel ids are not 256000/256001: {tokenizer.get('sentinels')}")
    expected_vocab_file = os.path.join(expected_load_source["path"], "tokenizer.model")
    if tokenizer.get("vocab_file") != expected_vocab_file:
        fail("fast Gemma tokenizer did not open the certified snapshot tokenizer.model")
    if (tokenizer.get("pad_token_id"), tokenizer.get("eos_token_id"),
            tokenizer.get("bos_token_id")) != (0, 1, 2):
        fail("Gemma PAD/EOS/BOS contract is not 0/1/2")
    unk = tokenizer.get("unk_token_id")
    if unk is None:
        fail("tokenizer reports no unk_token_id, so the UNK counters below are vacuous")

    parity = model.get("slow_fast_parity", {})
    if (parity.get("available") is not True or parity.get("mismatch_count") != 0 or
            parity.get("rows_checked") != PARITY_ROWS or
            parity.get("slow_vocab_file") != expected_vocab_file):
        fail(f"Gemma slow/fast tokenizer parity was not proved: {parity}")

    files = model.get("files", [])
    if len(files) != 2:
        fail(f"residual tokenizer audit must contain exactly 2 files, got {len(files)}")

    per_file = {}
    max_response_tokens = 0
    max_total_tokens = 0
    observed_paths = []
    for file_report in files:
        if not isinstance(file_report, dict):
            fail("malformed tokenizer file report")
        path = str(Path(file_report.get("path", "")).resolve())
        observed_paths.append(path)
        if path not in expected_by_path:
            fail(f"tokenizer audit contains an unexpected file: {path}")
        if file_report.get("sha256") != expected_by_path[path]:
            fail(f"tokenizer audit hash differs from the file on disk: {path}")

        counts = file_report.get("counts")
        if not isinstance(counts, dict):
            fail(f"tokenizer audit has no counters: {path}")
        missing = MANDATORY_COUNTERS - set(counts)
        if missing:
            fail(f"tokenizer audit omitted counters for {path}: {sorted(missing)}")
        rows = counts["rows"]
        if rows <= 0:
            fail(f"empty tokenizer audit: {path}")

        # Truncation and sentinel integrity. Under --strict_token_contract any
        # response truncation loses the trailing [END_CELL] and raises ValueError
        # inside C2SDataset.__getitem__ -- hours into an H200 allocation, at the
        # offending row, not at startup. This is that failure moved to a CPU job.
        for key in ("semantic_truncations", "prompt_truncations", "sentinel_losses",
                    "response_missing_end_cell"):
            if counts[key] != 0:
                fail(f"{path}: {key}={counts[key]}, expected 0")

        # UNK. The only place in the whole pipeline this is checked.
        for key in ("prompt_unk_tokens", "response_unk_tokens"):
            value = counts[key]
            if type(value) is not int or value < 0:
                fail(f"{path}: invalid {key}={value!r}")
            if value != 0:
                fail(f"{path}: {key}={value}. The ot_cache panel gene universe has "
                     "never been passed through the Gemma tokenizer in this project. "
                     "The trainer does not check UNK, so this is invisible for the "
                     "entire run if it is not caught here. STOP the arm.")

        # [DOWN] is REQUIRED here, where the standard contract forbids it. This
        # re-scopes the check for residual data instead of silencing it.
        down = counts["response_contains_down"]
        if down <= 0:
            fail(f"{path}: response_contains_down={down}. residual_to_sentence always "
                 "emits [DOWN]; zero means this is not residual data.")
        if down != rows:
            fail(f"{path}: [DOWN] present in {down} of {rows} rows. "
                 "residual_to_sentence emits the literal unconditionally, so anything "
                 "but every row means the build is not what the reader thinks it is.")

        distributions = file_report.get("distributions", {})
        prompt_dist = distributions.get("prompt_tokens", {})
        response_dist = distributions.get("response_tokens", {})
        total_dist = distributions.get("total_tokens", {})
        for name, dist in (("prompt_tokens", prompt_dist),
                           ("response_tokens", response_dist),
                           ("total_tokens", total_dist)):
            if "min" not in dist or "max" not in dist:
                fail(f"{path}: tokenizer audit has no {name} distribution")
        if prompt_dist["min"] < MIN_SEMANTIC_TOKENS or \
                response_dist["min"] < MIN_SEMANTIC_TOKENS:
            fail(f"{path}: collapsed tokenizer lengths, prompt_min="
                 f"{prompt_dist['min']} response_min={response_dist['min']} "
                 f"(both must be >= {MIN_SEMANTIC_TOKENS})")

        max_response_tokens = max(max_response_tokens, int(response_dist["max"]))
        max_total_tokens = max(max_total_tokens, int(total_dist["max"]))
        per_file[os.path.basename(path)] = {
            "path": path,
            "sha256": file_report["sha256"],
            "rows": rows,
            "prompt_tokens": prompt_dist,
            "response_tokens": response_dist,
            "total_tokens": total_dist,
            "response_contains_down": down,
            "prompt_unk_tokens": counts["prompt_unk_tokens"],
            "response_unk_tokens": counts["response_unk_tokens"],
        }

    if len(set(observed_paths)) != 2:
        fail("tokenizer audit file reports are duplicated")
    return per_file, max_response_tokens, max_total_tokens, report


def create(args):
    target_dir = Path(args.target_dir)
    train_path = target_dir / args.train_name
    val_path = target_dir / args.val_name
    for path in (train_path, val_path, Path(args.probe), Path(args.row_audit),
                 Path(args.standard_certificate)):
        if not path.is_file():
            fail(f"missing input: {path}")

    per_file, max_response, max_total, probe_report = validate_residual_probe(
        args.probe, train_path, val_path, args.snapshot_path,
        args.model_id, args.revision)

    # total_tokens already includes the prepended BOS (counted inside prompt_tokens)
    # and the single appended EOS; see gemma_tokenizer_probe._semantic_lengths_from_ids.
    if max_total >= args.max_length:
        fail(f"max_total_sequence_tokens={max_total} does not fit under "
             f"--max_length {args.max_length}. Changing --max_length changes the "
             "training contract and therefore the whole run; do not do it silently.")

    audit = json.load(open(args.row_audit, encoding="utf-8"))
    if audit.get("audit_passed") is not True:
        fail("row audit did not pass; run gemma2_residual_data_audit.py first")
    if str(Path(audit.get("target_dir", "")).resolve()) != str(target_dir.resolve()):
        fail("row audit describes a different target directory")
    if audit.get("val_train_sample_id_overlap") != 0:
        fail("row audit recorded val/train well overlap")

    rows_train = per_file[args.train_name]["rows"]
    rows_val = per_file[args.val_name]["rows"]
    if rows_train != audit.get("rows_train") or rows_val != audit.get("rows_val"):
        fail(f"row counts disagree between probe ({rows_train}/{rows_val}) and audit "
             f"({audit.get('rows_train')}/{audit.get('rows_val')})")
    if per_file[args.train_name]["sha256"] != audit["train"]["sha256"] or \
            per_file[args.val_name]["sha256"] != audit["val"]["sha256"]:
        fail("file digests disagree between the probe and the row audit; the residual "
             "files changed between Stage 2 and Stage 3")

    microbatches = -(-rows_train // args.batch_size)
    total_steps = (microbatches // args.grad_accum) * args.num_epochs
    if total_steps != audit.get("derived_total_steps"):
        fail(f"derived total steps disagree: contract={total_steps} "
             f"audit={audit.get('derived_total_steps')}")
    if total_steps <= 0:
        fail("derived total optimization steps is not positive")

    cap = derive_generation_cap(max_response)
    if not cap > max_response:
        fail(f"derived cap {cap} is not strictly greater than {max_response}")

    document = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "residual_data_passed": True,
        "target_dir": str(target_dir.resolve()),
        "model_id": args.model_id,
        "revision": args.revision,
        "snapshot_path": os.path.abspath(args.snapshot_path),
        # The standard certificate supplied MODEL provenance only; it was verified
        # WITHOUT --require-generation-cap, so its 1800/1699 never touched this arm.
        "standard_preflight_certificate": {
            "path": str(Path(args.standard_certificate).resolve()),
            "sha256": sha256_file(args.standard_certificate),
            "generation_cap_requirement_applied": False,
        },
        "tokenizer_probe": {
            "path": str(Path(args.probe).resolve()),
            "sha256": sha256_file(args.probe),
        },
        "row_audit": {
            "path": str(Path(args.row_audit).resolve()),
            "sha256": sha256_file(args.row_audit),
        },
        "files": {
            "train": {"name": args.train_name,
                      "path": str(train_path.resolve()),
                      "sha256": per_file[args.train_name]["sha256"],
                      "rows": rows_train},
            "val": {"name": args.val_name,
                    "path": str(val_path.resolve()),
                    "sha256": per_file[args.val_name]["sha256"],
                    "rows": rows_val},
        },
        "schedule": {
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "total_steps": total_steps,
            "warmup_steps_at_ratio_0.03": int(total_steps * 0.03),
        },
        # THE MEASURED REPLACEMENTS for 1800 and 1699. Neither was carried over.
        "residual_generation_cap": cap,
        "max_response_tokens": max_response,
        "max_total_sequence_tokens": max_total,
        "cap_derivation": (
            f"ceil(({max_response} + 1) / {CAP_ROUNDING}) * {CAP_ROUNDING} = {cap}, "
            f"strictly greater than the measured maximum truth response"),
        "prompt_order_in_data": audit.get("derived_prompt_order"),
        "val_wells": audit.get("val_wells"),
        "val_drugs": audit.get("val_drugs"),
        "down_rate_train": audit.get("down_rate_train"),
        "per_file_token_distributions": per_file,
    }
    target = atomic_json(args.output, document)

    print("\n============ RESIDUAL DATA CERTIFICATE ============")
    print(f"  target_dir                 {document['target_dir']}")
    print(f"  rows_train / rows_val      {rows_train} / {rows_val}")
    print(f"  RESIDUAL_TOTAL_STEPS       {total_steps}")
    print(f"  max_response_tokens        {max_response}   "
          f"(the replacement for the standard target's 1699)")
    print(f"  residual_generation_cap    {cap}   "
          f"(the replacement for 1800 -- MEASURED, feeds Stage 6)")
    print(f"  max_total_sequence_tokens  {max_total} < {args.max_length}")
    print(f"  prompt_order_in_data       {document['prompt_order_in_data']}")
    print(f"  train sha256               {document['files']['train']['sha256']}")
    print(f"  val   sha256               {document['files']['val']['sha256']}")
    print(f"\n[PASS] wrote {target}")
    print("\n  FREEZE STARTS NOW. Both jsonls are immutable for the life of this arm.")
    print("  build_residual_targets.py does NOT refuse an existing out_dir: it")
    print("  makedirs(exist_ok=True) and os.replace's over residual.jsonl. A rerun")
    print("  'just to regenerate the report' changes train_sha256 and every published")
    print("  residual checkpoint refuses to resume forever. Consider:")
    print(f"      chmod a-w {document['target_dir']}/residual.jsonl "
          f"{document['target_dir']}/residual_val.jsonl")
    return 0


def verify(args):
    document = json.load(open(args.certificate, encoding="utf-8"))
    if (document.get("schema_version") != CERTIFICATE_SCHEMA_VERSION or
            document.get("residual_data_passed") is not True):
        fail("invalid or unsuccessful residual data certificate")

    for role in ("train", "val"):
        entry = document.get("files", {}).get(role, {})
        path = entry.get("path")
        if not path or not os.path.isfile(path):
            fail(f"certified residual {role} file is missing: {path}")
        observed = sha256_file(path)
        if observed != entry.get("sha256"):
            fail(f"residual {role} file changed since the certificate was issued.\n"
                 f"          path     {path}\n"
                 f"          expected {entry.get('sha256')}\n"
                 f"          observed {observed}\n"
                 "          _validate_resume_fingerprints hashes both files into every "
                 "checkpoint's\n          contract, so continuing would strand this run "
                 "permanently. REFUSING.")

    if args.require_target_dir is not None:
        if str(Path(args.require_target_dir).resolve()) != document.get("target_dir"):
            fail(f"certificate describes {document.get('target_dir')}, "
                 f"job expects {args.require_target_dir}")
    if args.require_total_steps is not None:
        if document.get("schedule", {}).get("total_steps") != args.require_total_steps:
            fail(f"certificate total_steps="
                 f"{document.get('schedule', {}).get('total_steps')} != "
                 f"{args.require_total_steps}")
    if args.require_standard_certificate is not None:
        expected = sha256_file(args.require_standard_certificate)
        recorded = document.get("standard_preflight_certificate", {}).get("sha256")
        if recorded != expected:
            fail("residual certificate was issued against a different standard "
                 "preflight certificate")

    if args.print_field:
        value = document
        for part in args.print_field.split("."):
            if not isinstance(value, dict) or part not in value:
                fail(f"certificate has no field {args.print_field}")
            value = value[part]
        print(value)
    else:
        print(f"[PASS] residual data certificate verified; both jsonls unchanged: "
              f"{Path(args.certificate).resolve()}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--target_dir", required=True)
    create_parser.add_argument("--train_name", default="residual.jsonl")
    create_parser.add_argument("--val_name", default="residual_val.jsonl")
    create_parser.add_argument("--probe", required=True,
                               help="gemma_tokenizer_probe.py report over the two files")
    create_parser.add_argument("--row_audit", required=True,
                               help="gemma2_residual_data_audit.py output")
    create_parser.add_argument("--standard-certificate", dest="standard_certificate",
                               required=True,
                               help="the standard PREFLIGHT_PASSED.json that supplied "
                                    "MODEL provenance (verified WITHOUT "
                                    "--require-generation-cap)")
    create_parser.add_argument("--snapshot-path", dest="snapshot_path", required=True)
    create_parser.add_argument("--model-id", dest="model_id", default=GEMMA_MODEL_ID)
    create_parser.add_argument("--revision", required=True)
    create_parser.add_argument("--batch_size", type=int, default=1)
    create_parser.add_argument("--grad_accum", type=int, default=16)
    create_parser.add_argument("--num_epochs", type=int, default=1)
    create_parser.add_argument("--max_length", type=int, default=8192)
    create_parser.add_argument("--output", required=True)
    create_parser.set_defaults(func=create)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--certificate", required=True)
    verify_parser.add_argument("--require-target-dir", dest="require_target_dir")
    verify_parser.add_argument("--require-total-steps", dest="require_total_steps",
                               type=int)
    verify_parser.add_argument("--require-standard-certificate",
                               dest="require_standard_certificate")
    verify_parser.add_argument("--print-field", dest="print_field",
                               help="dotted path, e.g. schedule.total_steps")
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
