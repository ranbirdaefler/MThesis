#!/usr/bin/env python
"""Build and verify the committed residual-training adequacy artifact.

The raw cluster logs remain intentionally untracked.  This script is the refresh path from those
logs to a deterministic, committed JSON artifact containing (i) strict provenance and epoch
summaries for the canonical and ten-epoch residual runs and (ii) the structured curve points used
by ``fig_training_curves.py``.  A clean clone can validate and plot the committed artifact without
the logs; when the logs are present, ``--verify`` also checks their SHA256 hashes and reproduces the
artifact byte-for-byte at the JSON-object level.

Usage
-----
  python endcell/analysis/training_adequacy_long.py --selftest
  python endcell/analysis/training_adequacy_long.py --refresh-from-logs
  python endcell/analysis/training_adequacy_long.py --verify-production
  python endcell/analysis/training_adequacy_long.py --verify
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, "training_logs")
DEFAULT_LONG = os.path.join(LOGS, "retrain_long_618456.out")
DEFAULT_CANONICAL = os.path.join(LOGS, "retrain_614269.out")
DEFAULT_OUT = os.path.join(ROOT, "RESULTS_cluster", "training_adequacy_long.json")

CURVE_SPECS = {
    "retrain_614269": {
        "path": os.path.join(LOGS, "retrain_614269.out"),
        "label": "residual (the arm the results use)",
        "colour": "#1a1a1a",
        "canonical": True,
        "completed": True,
        "nominal_total_steps": 9231,
        "minimum_points": 180,
        "expected_points": 184,
        "expected_last_step": 9200,
        "sha256": "a961e264c0a941b815adf918c4a6d98831e0609f7b369817a24fa0f17d4d6694",
    },
    "ot_train_600856": {
        "path": os.path.join(LOGS, "ot_train_600856.out"),
        "label": "optimal transport, epsilon = 0.5",
        "colour": "#4c72b0",
        "canonical": False,
        "completed": True,
        "nominal_total_steps": 23842,
        "minimum_points": 470,
        "expected_points": 476,
        "expected_last_step": 23800,
        "sha256": "b9143aa582152fcc24393f91611f098cd5a2f46d40e8ac0548a5b924d8cdcd55",
    },
    "arm1a_consensus_592483": {
        "path": os.path.join(LOGS, "arm1a_consensus_592483.out"),
        "label": "consensus (wall-clock stop at 61.5%)",
        "colour": "#c44e52",
        "canonical": False,
        "completed": False,
        "nominal_total_steps": 42198,
        "minimum_points": 515,
        "expected_points": 519,
        "expected_last_step": 25950,
        "sha256": "e11f9d5b08107c28625aec9eb17211708f652481470184072b1cb62609797250",
    },
}

SCHEMA_VERSION = 2
MODEL_SOURCE = "vandijklab/C2S-Scale-Pythia-1b-pt"
TRAIN_BASENAME = "residual.jsonl"
VALID_BASENAME = "residual_val.jsonl"
EXPECTED_RUN_SOURCES = {
    "long_run": {"basename": "retrain_long_618456.out",
                 "sha256": "9e13b9e6435603f50a196b60c74c6d90d51fb5becb2d52aae66e40523affec34"},
    "canonical_one_epoch": {"basename": "retrain_614269.out",
                            "sha256": "a961e264c0a941b815adf918c4a6d98831e0609f7b369817a24fa0f17d4d6694"},
}

SEED_RE = re.compile(r"Seed:\s*(\d+)")
TOKENIZER_RE = re.compile(r"Loading tokenizer from\s+(.+?)\.\.\.")
MODEL_RE = re.compile(r"Loading model from\s+(.+?)\.\.\.")
DATA_RE = re.compile(r"Loading data from\s+(.+?)\.\.\.")
LOADED_RE = re.compile(r"Loaded\s+(\d+)\s+examples")
TOTAL_RE = re.compile(r"Total optimization steps:\s*(\d+)")
WARMUP_RE = re.compile(r"Warmup steps:\s*(\d+)")
BATCH_RE = re.compile(r"Effective batch size:\s*(\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+complete\s*\|\s*Train loss:\s*([0-9.]+)")
EVAL_RE = re.compile(r"Eval loss:\s*([0-9.]+)")
CHECKPOINT_RE = re.compile(r"Saved checkpoint to\s+.*[\\/]checkpoint-(\d+)")
FINAL_RE = re.compile(r"Saved final model to\s+(.+?)[\r\n]")
STEP_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Step\s+(\d+)/(\d+)\s*\|\s*"
    r"Loss:\s*([0-9.]+)\s*\|\s*LR:\s*([0-9.eE+-]+)"
)


def _strict_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw.decode("utf-8", errors="strict"), hashlib.sha256(raw).hexdigest()


def _one(pattern, text, name):
    hits = pattern.findall(text)
    if len(hits) != 1:
        raise ValueError("expected exactly one %s; found %d" % (name, len(hits)))
    return hits[0]


def _parse_data_sources(lines):
    rows = []
    pending = None
    for line in lines:
        m = DATA_RE.search(line)
        if m:
            if pending is not None:
                raise ValueError("data path was not followed by its loaded-example count")
            pending = m.group(1)
            continue
        m = LOADED_RE.search(line)
        if m and pending is not None:
            rows.append({"path": pending, "basename": os.path.basename(pending),
                         "n_examples": int(m.group(1))})
            pending = None
    if pending is not None or len(rows) != 2:
        raise ValueError("expected exactly two data sources with counts; found %d" % len(rows))
    return rows


def parse_log_text(text, expected_epochs, run_name="run"):
    """Parse one complete training log and reject incomplete or ambiguous structure."""
    lines = text.splitlines()
    seed = int(_one(SEED_RE, text, "seed line"))
    tokenizer = _one(TOKENIZER_RE, text, "tokenizer source")
    model = _one(MODEL_RE, text, "model source")
    nominal = int(_one(TOTAL_RE, text, "nominal optimization-step horizon"))
    warmup = int(_one(WARMUP_RE, text, "warmup-step count"))
    effective_batch = int(_one(BATCH_RE, text, "effective batch size"))
    data = _parse_data_sources(lines)

    if text.count("Training complete!") != 1:
        raise ValueError("%s: missing or repeated Training complete marker" % run_name)
    final_hits = FINAL_RE.findall(text + "\n")
    if len(final_hits) != 1:
        raise ValueError("%s: expected exactly one final-model save" % run_name)

    rows = []
    for i, line in enumerate(lines):
        m = EPOCH_RE.search(line)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not EVAL_RE.search(lines[j]):
            raise ValueError("%s: epoch %s evaluation is not the next non-empty log line"
                             % (run_name, m.group(1)))
        rows.append({"epoch": int(m.group(1)), "train_loss": float(m.group(2)),
                     "eval_loss": float(EVAL_RE.search(lines[j]).group(1))})

    observed = [row["epoch"] for row in rows]
    wanted = list(range(1, expected_epochs + 1))
    if observed != wanted:
        raise ValueError("%s: expected epochs %r, observed %r" % (run_name, wanted, observed))

    checkpoints = [int(x) for x in CHECKPOINT_RE.findall(text)]
    if not checkpoints or any(b <= a for a, b in zip(checkpoints, checkpoints[1:])):
        raise ValueError("%s: saved checkpoints are absent or not strictly increasing" % run_name)

    train_n = data[0]["n_examples"]
    actual_per_epoch = train_n // effective_batch
    actual_updates = actual_per_epoch * expected_epochs
    expected_nominal = (train_n * expected_epochs) // effective_batch
    if nominal != expected_nominal:
        raise ValueError("%s: nominal horizon %d != floor(n*epochs/batch) %d"
                         % (run_name, nominal, expected_nominal))
    if checkpoints[-1] > actual_updates:
        raise ValueError("%s: checkpoint exceeds inferred optimizer updates" % run_name)
    if expected_epochs > 1 and checkpoints[-1] != actual_updates:
        raise ValueError("%s: final checkpoint %d != inferred actual updates %d"
                         % (run_name, checkpoints[-1], actual_updates))

    points = []
    for m in STEP_RE.finditer(text):
        points.append({"epoch": int(m.group(1)), "step": int(m.group(2)),
                       "nominal_total_steps": int(m.group(3)),
                       "cumulative_train_loss": float(m.group(4)),
                       "learning_rate": float(m.group(5))})
    if not points:
        raise ValueError("%s: no periodic training points" % run_name)
    if any(b["step"] <= a["step"] for a, b in zip(points, points[1:])):
        raise ValueError("%s: periodic steps are not strictly increasing" % run_name)
    if {p["nominal_total_steps"] for p in points} != {nominal}:
        raise ValueError("%s: inconsistent periodic-step denominators" % run_name)
    if points[-1]["step"] > actual_updates:
        raise ValueError("%s: final logged step exceeds actual optimizer updates" % run_name)

    return {
        "run_name": run_name,
        "seed": seed,
        "tokenizer_source": tokenizer,
        "model_source": model,
        "training_data": data[0],
        "validation_data": data[1],
        "effective_batch_size": effective_batch,
        "nominal_scheduler_horizon": nominal,
        "warmup_steps": warmup,
        "peak_logged_learning_rate": max(point["learning_rate"] for point in points),
        "actual_updates_per_epoch": actual_per_epoch,
        "actual_optimizer_updates": actual_updates,
        "highest_saved_checkpoint": checkpoints[-1],
        "epochs": rows,
        "final_model_path": final_hits[0].strip(),
        "completed": True,
    }


def parse_log(path, expected_epochs, run_name):
    text, sha = _strict_text(path)
    parsed = parse_log_text(text, expected_epochs=expected_epochs, run_name=run_name)
    parsed["source"] = {"basename": os.path.basename(path), "sha256": sha}
    return parsed


def parse_curve(path, spec):
    text, sha = _strict_text(path)
    points = []
    for m in STEP_RE.finditer(text):
        points.append({"epoch": int(m.group(1)), "step": int(m.group(2)),
                       "cumulative_train_loss": float(m.group(4)),
                       "learning_rate": float(m.group(5))})
        if int(m.group(3)) != spec["nominal_total_steps"]:
            raise ValueError("%s: curve denominator changed" % os.path.basename(path))
    if len(points) < spec["minimum_points"]:
        raise ValueError("%s: only %d curve points (minimum %d)"
                         % (os.path.basename(path), len(points), spec["minimum_points"]))
    if any(b["step"] <= a["step"] for a, b in zip(points, points[1:])):
        raise ValueError("%s: curve steps are not strictly increasing" % os.path.basename(path))
    complete = text.count("Training complete!") == 1 and len(FINAL_RE.findall(text + "\n")) == 1
    signal_terminated = "CANCELLED" in text and "SIGNAL Terminated" in text
    if complete != spec["completed"]:
        raise ValueError("%s: completion state %r != expected %r"
                         % (os.path.basename(path), complete, spec["completed"]))
    if complete and points[-1]["step"] < spec["nominal_total_steps"] - 50:
        raise ValueError("%s: completed curve ends too early" % os.path.basename(path))
    if not complete and points[-1]["step"] >= spec["nominal_total_steps"]:
        raise ValueError("%s: incomplete curve reaches nominal horizon" % os.path.basename(path))
    if not complete and not signal_terminated:
        raise ValueError("%s: incomplete curve has no termination marker" % os.path.basename(path))
    return {
        "source": {"basename": os.path.basename(path), "sha256": sha},
        "label": spec["label"], "colour": spec["colour"],
        "canonical": spec["canonical"], "completed": complete,
        "nominal_total_steps": spec["nominal_total_steps"],
        "signal_terminated_before_completion": bool(signal_terminated and not complete),
        "points": points,
    }


def _same(a, b, field):
    if a[field] != b[field]:
        raise ValueError("canonical and long runs differ on directly logged field %s" % field)
    return a[field]


def build_artifact(long_path=DEFAULT_LONG, canonical_path=DEFAULT_CANONICAL,
                   curve_specs=None):
    curve_specs = curve_specs or CURVE_SPECS
    long = parse_log(long_path, 10, "ten_epoch_residual")
    canonical = parse_log(canonical_path, 1, "canonical_one_epoch_residual")
    for field in ("seed", "tokenizer_source", "model_source", "training_data",
                  "validation_data", "effective_batch_size"):
        _same(long, canonical, field)

    best = min(long["epochs"], key=lambda row: (row["eval_loss"], row["epoch"]))
    curves = {stem: parse_curve(spec["path"], spec) for stem, spec in curve_specs.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": {"long_run": long.pop("source"),
                    "canonical_one_epoch": canonical.pop("source")},
        "directly_logged_match": {
            "model_source": _same(long, canonical, "model_source"),
            "tokenizer_source": _same(long, canonical, "tokenizer_source"),
            "training_data": _same(long, canonical, "training_data"),
            "validation_data": _same(long, canonical, "validation_data"),
            "seed": _same(long, canonical, "seed"),
            "effective_batch_size": _same(long, canonical, "effective_batch_size"),
            "peak_logged_learning_rate": _same(
                long, canonical, "peak_logged_learning_rate"
            ),
        },
        "comparison_design": {
            "changed": "num_epochs: 1 to 10",
            "schedule_caveat": (
                "Changing num_epochs stretches the cosine schedule across ten epochs; the long "
                "run is not a literal continuation of the canonical one-epoch schedule."
            ),
            "optimizer_provenance_limit": (
                "The retained logs do not print every optimizer argument, so equality of unlogged "
                "optimizer hyperparameters is not asserted by this artifact."
            ),
        },
        "long_run": {
            **long,
            "full_epochs": 10,
            "best_eval_epoch": best["epoch"],
            "best_eval_loss": best["eval_loss"],
            "final_train_loss": long["epochs"][-1]["train_loss"],
            "final_eval_loss": long["epochs"][-1]["eval_loss"],
        },
        "canonical_one_epoch": {
            **canonical,
            "full_epochs": 1,
            "epoch": canonical["epochs"][0]["epoch"],
            "train_loss": canonical["epochs"][0]["train_loss"],
            "eval_loss": canonical["epochs"][0]["eval_loss"],
        },
        "curve_series": curves,
        "interpretation_bounds": {
            "supported": (
                "Ordinary longer optimization under this ten-epoch schedule does not rescue "
                "held-out cross-entropy: validation loss is best at epoch 1 and worsens while "
                "training loss continues to fall."
            ),
            "not_supported": [
                "all possible learning-rate schedules are ruled out",
                "NIR worsens with additional epochs",
            ],
        },
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_artifact(data, production=True):
    """Validate schema and, by default, the production values quoted by the thesis."""
    _require(isinstance(data, dict), "artifact root must be an object")
    _require(data.get("schema_version") == SCHEMA_VERSION, "wrong schema_version")
    for key in ("sources", "directly_logged_match", "comparison_design", "long_run",
                "canonical_one_epoch", "curve_series", "interpretation_bounds"):
        _require(key in data, "missing key %s" % key)
    long = data["long_run"]
    canon = data["canonical_one_epoch"]
    _require(long.get("full_epochs") == 10 and len(long.get("epochs", [])) == 10,
             "long run must contain ten full epochs")
    _require(canon.get("full_epochs") == 1 and len(canon.get("epochs", [])) == 1,
             "canonical run must contain one full epoch")
    _require(long.get("completed") is True and canon.get("completed") is True,
             "both runs must be complete")
    _require(long.get("actual_optimizer_updates") == 92310,
             "long actual optimizer updates must be 92310")
    _require(long.get("nominal_scheduler_horizon") == 92318,
             "long nominal scheduler horizon must be 92318")
    _require(long.get("highest_saved_checkpoint") == 92310,
             "long highest checkpoint must be 92310")
    _require(canon.get("actual_optimizer_updates") == 9231,
             "canonical actual optimizer updates must be 9231")
    _require(canon.get("nominal_scheduler_horizon") == 9231,
             "canonical nominal horizon must be 9231")
    expected_losses = [
        (1, 1.9456, 1.9199), (2, 1.7176, 2.0559), (3, 1.5810, 2.1708),
        (4, 1.5008, 2.2294), (5, 1.4634, 2.2535), (6, 1.4526, 2.2577),
        (7, 1.4496, 2.2558), (8, 1.4488, 2.2577), (9, 1.4486, 2.2564),
        (10, 1.4485, 2.2568),
    ]
    observed = [(r.get("epoch"), r.get("train_loss"), r.get("eval_loss"))
                for r in long["epochs"]]
    _require(observed == expected_losses, "long epoch losses differ from production values")
    _require(canon.get("train_loss") == 1.8854 and canon.get("eval_loss") == 1.8986,
             "canonical losses differ from production values")
    _require(long.get("best_eval_epoch") == 1 and long.get("best_eval_loss") == 1.9199,
             "wrong best long-run epoch/loss")
    curves = data["curve_series"]
    _require(set(curves) == set(CURVE_SPECS), "wrong curve-series set")
    for stem, spec in CURVE_SPECS.items():
        row = curves[stem]
        _require(row.get("nominal_total_steps") == spec["nominal_total_steps"],
                 "%s has wrong denominator" % stem)
        _require(row.get("completed") == spec["completed"],
                 "%s has wrong completion state" % stem)
        points = row.get("points")
        _require(isinstance(points, list) and len(points) >= spec["minimum_points"],
                 "%s has too few points" % stem)
        steps = [p.get("step") for p in points]
        _require(all(isinstance(s, int) for s in steps) and
                 all(b > a for a, b in zip(steps, steps[1:])),
                 "%s steps are not strictly increasing" % stem)
        _require(all(isinstance(p.get("cumulative_train_loss"), (int, float)) and
                     isinstance(p.get("learning_rate"), (int, float)) for p in points),
                 "%s has malformed point values" % stem)
    if production:
        _require(data["directly_logged_match"].get("model_source") == MODEL_SOURCE,
                 "wrong model source")
        _require(data["directly_logged_match"]["training_data"].get("n_examples") == 147710,
                 "wrong training count")
        _require(data["directly_logged_match"]["validation_data"].get("n_examples") == 3600,
                 "wrong validation count")
        _require(data["directly_logged_match"].get("peak_logged_learning_rate") == 1e-5,
                 "wrong or unmatched peak logged learning rate")
        _require(data["sources"] == EXPECTED_RUN_SOURCES,
                 "canonical/long source filename or SHA256 mismatch")
        for stem, spec in CURVE_SPECS.items():
            row = curves[stem]
            _require(row["source"].get("basename") == os.path.basename(spec["path"]),
                     "%s source basename mismatch" % stem)
            _require(row["source"].get("sha256") == spec["sha256"],
                     "%s source SHA256 mismatch" % stem)
            _require(len(row["points"]) == spec["expected_points"],
                     "%s production point count mismatch" % stem)
            _require(row["points"][-1]["step"] == spec["expected_last_step"],
                     "%s production final logged step mismatch" % stem)
            if not spec["completed"]:
                _require(row.get("signal_terminated_before_completion") is True,
                         "%s lacks termination evidence" % stem)
    return True


def load_artifact(path=DEFAULT_OUT):
    with open(path, "r", encoding="utf-8", errors="strict") as fh:
        data = json.load(fh)
    validate_artifact(data)
    return data


def selftest():
    def synthetic(epochs, complete=True, misplaced=False):
        lines = [
            "Seed: 42",
            "Loading tokenizer from %s..." % MODEL_SOURCE,
            "Loading model from %s..." % MODEL_SOURCE,
            "Loading data from /x/residual.jsonl...", "Loaded 147710 examples",
            "Loading data from /x/residual_val.jsonl...", "Loaded 3600 examples",
            "Total optimization steps: %d" % ((147710 * epochs) // 16),
            "Warmup steps: %d" % int(((147710 * epochs) // 16) * 0.03),
            "Effective batch size: 16",
            "Epoch 1 | Step 50/%d | Loss: 2.0 | LR: 1e-5" % ((147710 * epochs) // 16),
        ]
        per_epoch = 147710 // 16
        for epoch in range(1, epochs + 1):
            lines += ["Saved checkpoint to /x/checkpoint-%d" % (per_epoch * epoch),
                      "Epoch %d complete | Train loss: %.4f" % (epoch, 2.0 - epoch / 100.0)]
            if misplaced and epoch == 1:
                lines.append("an unrelated record")
            lines.append("Eval loss: %.4f" % (1.0 + epoch / 100.0))
        lines.append("Saved final model to /x/final")
        if complete:
            lines.append("Training complete!")
        return "\n".join(lines) + "\n"

    parse_log_text(synthetic(10), 10, "good_long")
    parse_log_text(synthetic(1), 1, "good_canonical")
    cases = [
        (synthetic(6), 10, "truncated six-epoch run"),
        (synthetic(10, misplaced=True), 10, "misplaced evaluation"),
        (synthetic(10, complete=False), 10, "missing completion"),
        (synthetic(10), 1, "wrong expected epoch count"),
    ]
    for text, expected, name in cases:
        try:
            parse_log_text(text, expected, name)
        except ValueError:
            pass
        else:
            raise AssertionError("adversarial case accepted: %s" % name)
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        bad_path = fh.name
        fh.write(b"\xff\xfe")
    try:
        try:
            _strict_text(bad_path)
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("invalid UTF-8 was silently accepted")
    finally:
        os.unlink(bad_path)
    print("SELFTEST PASSED (including truncated, misplaced-eval, completion, epoch and UTF-8 gates)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-log", default=DEFAULT_LONG)
    ap.add_argument("--canonical-log", default=DEFAULT_CANONICAL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refresh-from-logs", action="store_true",
                    help="explicitly refresh the committed artifact from ignored raw logs")
    ap.add_argument("--verify-artifact", "--verify-production", dest="verify_artifact",
                    action="store_true",
                    help="validate the committed structured artifact without raw logs")
    ap.add_argument("--verify", action="store_true",
                    help="validate artifact and reproduce it when every raw log is available")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.verify_artifact or args.verify:
        committed = load_artifact(args.out)
        print("ARTIFACT VALIDATION PASSED")
        if args.verify:
            raw_paths = ([args.long_log, args.canonical_log] +
                         [spec["path"] for spec in CURVE_SPECS.values()])
            if all(os.path.isfile(path) for path in raw_paths):
                rebuilt = build_artifact(args.long_log, args.canonical_log)
                if rebuilt != committed:
                    raise SystemExit("PRODUCTION VERIFY FAILED: artifact differs from raw logs")
                print("PRODUCTION VERIFY PASSED (schema, values and source SHA256)")
            else:
                print("RAW LOGS ABSENT: clean-clone artifact validation only")
        return

    result = build_artifact(args.long_log, args.canonical_log)
    validate_artifact(result)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("->", args.out)


if __name__ == "__main__":
    main()
