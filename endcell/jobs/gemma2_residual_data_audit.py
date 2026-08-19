#!/usr/bin/env python3
r"""Stage 2: tokenizer-free row-schema audit of the resolved residual target directory.

WHY THIS RUNS BEFORE THE TOKENIZER STAGE
  It is roughly ten times cheaper (no torch, no transformers, no protobuf runtime)
  and it catches the coarse failures that would otherwise only surface as a
  confusing tokenizer report, or worse, hours into an H200 allocation:

    * a stray 'de_genes' key. train_c2s_tahoe_endcell.py:1484 raises SystemExit if
      DE weighting is requested and no example carries one. The Gemma residual CLI
      passes --de_weight 1.0, which makes `args.de_weight != 1.0` false, so that
      branch is never entered and DE weighting is a mathematical no-op. That is only
      true while no row carries de_genes; this audit is what keeps it true.
    * a leading or trailing space on 'response'. The trainer encodes ' ' + response
      (C2SDataset / gemma_tokenizer_probe.audit_file both do), so a leading space
      double-spaces every target and shifts the whole token stream.
    * a missing trailing [END_CELL]. Under --strict_token_contract any truncation
      that drops the sentinel raises ValueError inside C2SDataset.__getitem__ --
      i.e. mid-training-loop, not at startup.
    * a val shard that shares sample_ids with train. residual_val.jsonl is carved by
      WELL from TRAINING wells, so an overlap means the eval loss is scored on rows
      the model is also training on.
    * an ambiguous prompt order. derived_prompt_order is the ONLY record of
      --prompt_order (report.json does not store it); it is copied verbatim into the
      certificate as prompt_order_in_data and gemma2_residual_train.sbatch:471 tells
      the operator Stage 6 eval must use it. If the train rows do not agree on
      exactly one recognized order, or if val disagrees with train, there is no order
      for Stage 6 to use and this audit refuses rather than emitting an ambiguous
      value that prints just as calmly as a real one.

WHAT IT MEASURES RATHER THAN ASSERTS  (invariant 6)
  Row counts, the derived optimizer-step total, the [DOWN] emission rate, the number
  of distinct wells and drugs in the validation shard, and the prompt order actually
  present in the data. build_residual_targets.py's report.json records NEITHER
  --prompt_order NOR --val_frac, so both have to come from the rows themselves.

  The [DOWN] rate is REPORTED, not silently asserted: FINDINGS.md flags '[DOWN]
  emission rate' as an open Pythia-side defect. residual_to_sentence
  (build_residual_targets.py:113) always emits the literal, so the rate should be
  exactly 1.0; anything else is a real signal about the build, not a lint error.

HOW TO RUN

    cd "$HOME/tahoe"
    srun -p defq -c 2 --mem 16G --time 01:00:00 \
        /data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python \
        endcell/jobs/gemma2_residual_data_audit.py \
        --target_dir "$(cat RESULTS/gemma2_residual/RESOLVED_TGT.txt)"

WRITES
    RESULTS/gemma2_residual/row_audit.json

EXIT CODES
    0  every row conforms; row_audit.json written
    2  usage / missing input
    3  a schema violation (the audit's reason for existing)
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REQUIRED_ROW_KEYS = {"prompt", "response", "metadata"}
# build_residual_targets.py:1380 restricts --prompt_order to exactly these two, so a
# single one of them is always the correct expectation for a whole file.
VALID_PROMPT_ORDERS = ("drug_first", "drug_last")
UNRECOGNIZED_PROMPT_ORDER = "UNRECOGNIZED"
END_CELL = "[END_CELL]"
DOWN = "[DOWN]"
PROMPT_TAIL = "Response cell:"
# Reported per file so a violation names its file, and capped so a systematically
# broken build prints a readable diagnosis instead of 150,000 lines.
MAX_REPORTED_VIOLATIONS = 20


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Violations:
    """Collect violations without letting a broken file produce unbounded output."""

    def __init__(self):
        self.total = 0
        self.by_kind = {}
        self.examples = []

    def add(self, kind, path, row_number, detail):
        self.total += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if len(self.examples) < MAX_REPORTED_VIOLATIONS:
            self.examples.append({
                "kind": kind,
                "file": os.path.basename(path),
                "row": row_number,
                "detail": detail,
            })


def derive_prompt_order(prompt):
    """Recover --prompt_order from the prompt text; report.json does not store it."""
    instruction = prompt.find("Predict the response of")
    control = prompt.find("Control cell:")
    if instruction < 0 or control < 0:
        return UNRECOGNIZED_PROMPT_ORDER
    return "drug_first" if instruction < control else "drug_last"


def audit_file(path, violations):
    stats = {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "rows": 0,
        "rows_with_down": 0,
        "rows_with_de_genes": 0,
        "response_chars_min": None,
        "response_chars_max": None,
        "prompt_chars_min": None,
        "prompt_chars_max": None,
        "response_gene_count_min": None,
        "response_gene_count_max": None,
    }
    prompt_orders = {}
    sample_ids = set()
    drugs = set()
    plates = set()
    cell_lines = set()
    targets = {}

    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                violations.add("blank_line", path, index, "empty line in jsonl")
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                violations.add("unparseable_json", path, index, repr(exc))
                continue
            if not isinstance(row, dict):
                violations.add("row_not_object", path, index, type(row).__name__)
                continue

            keys = set(row)
            if keys != REQUIRED_ROW_KEYS:
                violations.add(
                    "unexpected_row_keys", path, index,
                    f"keys={sorted(keys)} expected={sorted(REQUIRED_ROW_KEYS)}")
            # Checked independently of the key-set test so it is unambiguous in the
            # report which failure occurred, and so a nested de_genes is also caught.
            if "de_genes" in keys or "de_genes" in (row.get("metadata") or {}):
                stats["rows_with_de_genes"] += 1
                violations.add(
                    "de_genes_present", path, index,
                    "a de_genes key makes --de_weight 1.0 no longer a no-op")

            prompt = row.get("prompt")
            response = row.get("response")
            metadata = row.get("metadata")

            if not isinstance(prompt, str) or not prompt:
                violations.add("prompt_not_nonempty_str", path, index, repr(type(prompt)))
                continue
            if not isinstance(response, str) or not response:
                violations.add("response_not_nonempty_str", path, index, repr(type(response)))
                continue
            if not isinstance(metadata, dict):
                violations.add("metadata_not_object", path, index, repr(type(metadata)))
                continue

            if response != response.strip():
                violations.add(
                    "response_has_edge_whitespace", path, index,
                    "the trainer encodes ' ' + response; a leading/trailing space "
                    "shifts every supervised token")
            if not response.endswith(END_CELL):
                violations.add(
                    "response_missing_trailing_end_cell", path, index,
                    f"response tail={response[-40:]!r}")
            if DOWN in response:
                stats["rows_with_down"] += 1
            if not prompt.endswith(PROMPT_TAIL):
                violations.add(
                    "prompt_missing_response_cell_tail", path, index,
                    f"prompt tail={prompt[-40:]!r}")
            if END_CELL not in prompt:
                violations.add(
                    "prompt_missing_control_end_cell", path, index,
                    "the control sentence should terminate with [END_CELL]")
            if metadata.get("target") != "residual":
                violations.add(
                    "metadata_target_not_residual", path, index,
                    f"target={metadata.get('target')!r}")

            order = derive_prompt_order(prompt)
            prompt_orders[order] = prompt_orders.get(order, 0) + 1
            if metadata.get("sample_id") is not None:
                sample_ids.add(metadata["sample_id"])
            if metadata.get("drug") is not None:
                drugs.add(metadata["drug"])
            if metadata.get("plate") is not None:
                plates.add(metadata["plate"])
            if metadata.get("cell_line_id") is not None:
                cell_lines.add(metadata["cell_line_id"])
            targets[metadata.get("target")] = targets.get(metadata.get("target"), 0) + 1

            gene_count = sum(
                1 for token in response.split() if token not in (END_CELL, DOWN))
            for key, value in (
                    ("response_chars", len(response)),
                    ("prompt_chars", len(prompt)),
                    ("response_gene_count", gene_count)):
                low, high = stats[f"{key}_min"], stats[f"{key}_max"]
                stats[f"{key}_min"] = value if low is None else min(low, value)
                stats[f"{key}_max"] = value if high is None else max(high, value)
            stats["rows"] += 1

    rows = max(stats["rows"], 1)
    stats["down_rate"] = stats["rows_with_down"] / rows
    stats["prompt_order_counts"] = prompt_orders
    stats["distinct_sample_ids"] = len(sample_ids)
    stats["distinct_drugs"] = len(drugs)
    stats["distinct_plates"] = len(plates)
    stats["distinct_cell_lines"] = len(cell_lines)
    stats["metadata_target_counts"] = targets
    return stats, sample_ids, drugs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target_dir", required=True,
                        help="the residual build resolved by Stage 1 "
                             "(RESULTS/gemma2_residual/RESOLVED_TGT.txt)")
    parser.add_argument("--train_name", default="residual.jsonl")
    parser.add_argument("--val_name", default="residual_val.jsonl")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="must equal the trainer's --batch_size; used to derive steps")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="must equal the trainer's --grad_accum; used to derive steps")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output", default="RESULTS/gemma2_residual/row_audit.json")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    train_path = target_dir / args.train_name
    val_path = target_dir / args.val_name
    for path in (train_path, val_path):
        if not path.is_file():
            print(f"[FATAL] missing residual file: {path}", file=sys.stderr)
            return 2

    violations = Violations()
    print(f"[audit] {train_path}", flush=True)
    train_stats, train_ids, train_drugs = audit_file(train_path, violations)
    print(f"[audit] {val_path}", flush=True)
    val_stats, val_ids, val_drugs = audit_file(val_path, violations)

    overlap = sorted(train_ids & val_ids)
    if overlap:
        violations.add(
            "val_train_sample_id_overlap", str(val_path), -1,
            f"{len(overlap)} shared wells, e.g. {overlap[:5]}")

    # --- prompt order: one value, or a refusal -------------------------------
    # This is the single field Stage 6 is told to trust, and it is recoverable from
    # nowhere else. Anything other than one recognized order across BOTH files is a
    # violation, and derived_prompt_order stays null so no downstream consumer can
    # copy an ambiguous value into the certificate and print it as authoritative.
    train_orders = set(train_stats["prompt_order_counts"])
    val_orders = set(val_stats["prompt_order_counts"])
    derived_prompt_order = None
    if len(train_orders) == 1 and train_orders.issubset(VALID_PROMPT_ORDERS):
        derived_prompt_order = next(iter(train_orders))
    else:
        violations.add(
            "ambiguous_prompt_order", str(train_path), -1,
            f"train prompt orders={train_stats['prompt_order_counts']}; expected "
            f"exactly one of {list(VALID_PROMPT_ORDERS)}. Stage 6 eval has no order "
            "to use, so there is nothing to certify")
    if val_orders != train_orders:
        violations.add(
            "prompt_order_train_val_mismatch", str(val_path), -1,
            f"val prompt orders={val_stats['prompt_order_counts']} != train "
            f"{train_stats['prompt_order_counts']}; the eval shard would be scored "
            "under a different prompt order than the model is trained on")

    rows_train = train_stats["rows"]
    # Mirrors train_c2s_tahoe_endcell.py:1553-1557 under --resumable:
    #   microbatches_per_epoch = ceil(len(train_dataset) / batch_size)
    #   total_steps = (microbatches_per_epoch // grad_accum) * num_epochs
    microbatches_per_epoch = -(-rows_train // args.batch_size)
    derived_total_steps = (microbatches_per_epoch // args.grad_accum) * args.num_epochs
    derived_warmup_steps = int(derived_total_steps * 0.03)

    document = {
        "schema_version": 1,
        "target_dir": str(target_dir.resolve()),
        "train": train_stats,
        "val": val_stats,
        "rows_train": rows_train,
        "rows_val": val_stats["rows"],
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "num_epochs": args.num_epochs,
        "derived_total_steps": derived_total_steps,
        "derived_warmup_steps_at_ratio_0.03": derived_warmup_steps,
        "down_rate_train": train_stats["down_rate"],
        "down_rate_val": val_stats["down_rate"],
        "val_wells": val_stats["distinct_sample_ids"],
        "val_drugs": val_stats["distinct_drugs"],
        "val_plates": val_stats["distinct_plates"],
        "train_wells": train_stats["distinct_sample_ids"],
        "train_drugs": train_stats["distinct_drugs"],
        "val_train_sample_id_overlap": len(overlap),
        "val_drugs_also_in_train": len(val_drugs & train_drugs),
        # A single string or null -- never a list. gemma2_residual_data_contract.py:388
        # copies this verbatim into prompt_order_in_data.
        "derived_prompt_order": derived_prompt_order,
        "violations": {
            "total": violations.total,
            "by_kind": violations.by_kind,
            "examples": violations.examples,
        },
        "audit_passed": violations.total == 0,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print("\n================ ROW AUDIT ================")
    print(f"  target_dir              {document['target_dir']}")
    print(f"  rows_train              {rows_train}")
    print(f"  rows_val                {val_stats['rows']}")
    print(f"  derived_total_steps     {derived_total_steps}   "
          f"(batch_size {args.batch_size} x grad_accum {args.grad_accum})")
    print(f"  warmup @0.03            {derived_warmup_steps}")
    print(f"  train sha256            {train_stats['sha256']}")
    print(f"  val   sha256            {val_stats['sha256']}")
    print(f"  down_rate train/val     {train_stats['down_rate']:.6f} / "
          f"{val_stats['down_rate']:.6f}")
    print(f"  prompt_order (train)    {train_stats['prompt_order_counts']}")
    print(f"  prompt_order (val)      {val_stats['prompt_order_counts']}")
    print(f"  derived_prompt_order    {derived_prompt_order}"
          f"{'' if derived_prompt_order else '   [REFUSED: see violations]'}")
    print(f"  response genes min/max  {train_stats['response_gene_count_min']} / "
          f"{train_stats['response_gene_count_max']}")
    print(f"  train wells / drugs     {train_stats['distinct_sample_ids']} / "
          f"{train_stats['distinct_drugs']}")
    print(f"  VAL wells / drugs       {val_stats['distinct_sample_ids']} / "
          f"{val_stats['distinct_drugs']}")
    print(f"  val/train well overlap  {len(overlap)}")

    if val_stats["distinct_sample_ids"] <= 8:
        print("\n  [READ THIS] the validation shard spans "
              f"{val_stats['distinct_sample_ids']} wells and "
              f"{val_stats['distinct_drugs']} drugs, carved from TRAIN wells. It is")
        print("  same-distribution and same-drug. It will produce a smooth, believable")
        print("  eval-loss curve that is a near-noise estimate over that many biological")
        print("  replicates. Use it to detect divergence, never as an arm-level result.")
        print("  Rebuilding with a larger --val_frac would change residual.jsonl too and")
        print("  destroy comparability with the Pythia arm, so the right move is to")
        print("  accept it and report it as what it is.")

    if document["down_rate_train"] != 1.0:
        print(f"\n  [NOTE] [DOWN] appears in {document['down_rate_train']:.6f} of train rows, "
              "not 1.0.")
        print("  residual_to_sentence always emits the literal, so a rate below 1.0 means")
        print("  the build differs from the reader's model of it. FINDINGS.md tracks")
        print("  '[DOWN] emission rate' as an open Pythia-side defect; this is the number.")

    if violations.total:
        print("\n================ VIOLATIONS ================", file=sys.stderr)
        for kind, count in sorted(violations.by_kind.items()):
            print(f"  {kind}: {count}", file=sys.stderr)
        for example in violations.examples:
            print(f"    {example['file']}:{example['row']} {example['kind']}: "
                  f"{example['detail']}", file=sys.stderr)
        print(f"\n[FATAL] {violations.total} schema violation(s); wrote {output}",
              file=sys.stderr)
        return 3

    print(f"\n[PASS] every row conforms; wrote {output}")
    print("\n  COPY THESE TWO LITERALS into the residual sbatch gates. Do NOT use the")
    print("  repository's provisional 147710 / 9231.")
    print(f"      EXPECTED_TRAIN_ROWS={rows_train}")
    print(f"      RESIDUAL_TOTAL_STEPS={derived_total_steps}")
    print("\n[next] Stage 3: sbatch endcell/jobs/gemma2_residual_preflight.sbatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
