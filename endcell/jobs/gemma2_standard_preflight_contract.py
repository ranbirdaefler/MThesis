#!/usr/bin/env python3
"""Create or verify the fail-closed Gemma launch certificate.

This helper is intentionally standard-library only so every Slurm job can verify
the exact data, code, environment and pinned model snapshot approved by preflight.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from gemma2_standard_provenance import (
    AUTHORITATIVE_REVISION_FILES,
    CANONICAL_TAHOE_DATA_SHA256,
    OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES,
    ProvenanceError,
    authoritative_revision_files_sha256,
    snapshot_inventory,
    stable_json_sha256,
    verify_authoritative_snapshot,
)


SCHEMA_VERSION = 8
PROBE_SCHEMA_VERSION = 3
RUNTIME_CONTRACT_SCHEMA_VERSION = 2
PARITY_ROWS = 512
PROTOBUF_VERSION = "5.29.5"
PROTOBUF_ROOT = Path(
    "/data/BuffaF-Projetcs/florian_c2s/test_deps/protobuf-5.29.5")
EXPECTED_KEYS = {
    "data": {"train", "tier1", "tier2", "tier3", "tier4"},
    "sources": {
        "trainer", "tokenizer_probe", "protobuf_env", "provenance", "evaluate_endcell",
        "nir_benchmark", "freeze_manifest", "compare_backbones", "residual_eval",
        "preflight", "contract", "cli_contract", "tests_contract", "tests_command",
        "fingerprint", "smoke", "train_job", "eval_job", "phase1a_test", "eval_test",
        "runbook",
    },
    "environment": {"environment", "pip_freeze"},
    "tests": {"section1"},
}
MANDATORY_COUNTERS = {
    "rows", "semantic_truncations", "prompt_truncations", "sentinel_losses",
    "response_missing_end_cell", "response_contains_down", "prompt_unk_tokens",
    "response_unk_tokens", "prompt_token_count", "response_token_count",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protobuf_tree_digest(root=PROTOBUF_ROOT, version=PROTOBUF_VERSION):
    """Hash every installed Protobuf payload named by dist-info/RECORD."""
    root = Path(root).resolve()
    record = root / f"protobuf-{version}.dist-info" / "RECORD"
    if not record.is_file():
        raise SystemExit(f"[FATAL] protobuf installation RECORD is missing: {record}")
    inventory = []
    seen = set()
    with record.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if not row or not row[0]:
            raise SystemExit("[FATAL] protobuf RECORD contains an empty path")
        raw = row[0]
        relative = PurePosixPath(raw)
        if (relative.is_absolute() or "\\" in raw or
                relative.as_posix() != raw or ".." in relative.parts):
            raise SystemExit(f"[FATAL] unsafe protobuf RECORD path: {raw!r}")
        candidate = root / Path(*relative.parts)
        if candidate.is_symlink():
            raise SystemExit(f"[FATAL] protobuf RECORD payload is a symlink: {raw!r}")
        try:
            path = candidate.resolve(strict=True)
        except OSError as exc:
            raise SystemExit(f"[FATAL] protobuf RECORD payload is missing: {raw!r}: {exc}")
        try:
            path.relative_to(root)
        except ValueError:
            raise SystemExit(f"[FATAL] protobuf RECORD path escapes isolated root: {raw!r}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise SystemExit(f"[FATAL] duplicate protobuf RECORD payload: {normalized}")
        seen.add(normalized)
        if not path.is_file():
            raise SystemExit(f"[FATAL] protobuf RECORD payload is not a file: {path}")
        inventory.append({
            "path": normalized,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not inventory:
        raise SystemExit("[FATAL] protobuf digest inventory is empty")
    if "google/_upb/_message.abi3.so" not in seen:
        raise SystemExit("[FATAL] protobuf RECORD omits google/_upb/_message.abi3.so")
    payload = json.dumps(
        inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(inventory)


def require_exact_keys(group_name, entries):
    observed = set(entries)
    expected = EXPECTED_KEYS[group_name]
    if observed != expected:
        raise SystemExit(
            f"[FATAL] {group_name} keys differ from the certified contract: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")


def require_canonical_data_hashes(entries):
    """Reject both stale data and a freshly regenerated certificate over altered data."""
    require_exact_keys("data", entries)
    observed = {name: entry.get("sha256") for name, entry in entries.items()}
    if observed != CANONICAL_TAHOE_DATA_SHA256:
        mismatches = {
            name: {"observed": observed.get(name), "expected": expected}
            for name, expected in CANONICAL_TAHOE_DATA_SHA256.items()
            if observed.get(name) != expected
        }
        message = "[FATAL] canonical Tahoe data hashes changed; a new certificate cannot bless different data"
        raise SystemExit(f"{message}: {mismatches}")


def parse_named_paths(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        if not name or name in result:
            raise SystemExit(f"invalid or duplicate name in {item!r}")
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_file():
            raise SystemExit(f"missing certificate input: {resolved}")
        result[name] = {"path": resolved, "sha256": sha256_file(resolved)}
    return result


def certified_snapshot(model_id, revision, snapshot_path, hub_cache):
    cache = Path(os.path.abspath(hub_cache))
    snapshot = Path(os.path.abspath(snapshot_path))
    expected = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if snapshot != expected:
        raise SystemExit(
            f"[FATAL] snapshot path is not the exact pinned revision directory: "
            f"{snapshot} != {expected}")
    try:
        authoritative = verify_authoritative_snapshot(snapshot)
        inventory = snapshot_inventory(snapshot)
    except ProvenanceError as exc:
        raise SystemExit(f"[FATAL] {exc}") from exc
    return {
        "hub_cache": str(cache),
        "snapshot_path": str(snapshot),
        "snapshot_relative_to_hub_cache": snapshot.relative_to(cache).as_posix(),
        "inventory": inventory,
        "inventory_sha256": stable_json_sha256(inventory),
        "authoritative_revision_files": authoritative,
        "authoritative_revision_files_sha256": authoritative_revision_files_sha256(),
        "official_snapshot_top_level_files": sorted(OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES),
    }


def atomic_json(path, document):
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                     dir=str(target.parent))
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


def validate_probe(path, generation_cap, expected_model_id, expected_revision,
                   expected_snapshot_path, expected_snapshot_inventory_sha256,
                   expected_authoritative_files_sha256, expected_data):
    report = json.load(open(path, encoding="utf-8"))
    if report.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise SystemExit("[FATAL] unsupported tokenizer-probe schema")
    if report.get("max_examples_per_file") != 0:
        raise SystemExit("[FATAL] tokenizer audit was not run over every row")
    require_exact_keys("data", expected_data)
    certified_by_path = {
        str(Path(entry["path"]).resolve()): entry["sha256"]
        for entry in expected_data.values()
    }
    if len(certified_by_path) != 5:
        raise SystemExit("[FATAL] certified data entries are not five unique paths")
    input_hashes = report.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise SystemExit("[FATAL] tokenizer audit has no input_hashes mapping")
    normalized_input_hashes = {
        str(Path(path).resolve()): digest for path, digest in input_hashes.items()
    }
    if normalized_input_hashes != certified_by_path:
        raise SystemExit("[FATAL] tokenizer audit input_hashes do not match certified data")
    models = report.get("models", [])
    if len(models) != 1:
        raise SystemExit("[FATAL] tokenizer audit must contain exactly one model")
    model = models[0]
    if (model.get("model_name"), model.get("revision")) != (
            expected_model_id, expected_revision):
        raise SystemExit("[FATAL] tokenizer audit used a different model or revision")
    expected_load_source = {
        "kind": "exact_local_snapshot",
        "path": os.path.abspath(expected_snapshot_path),
        "local_files_only": True,
        "revision_argument": None,
    }
    if model.get("load_source") != expected_load_source:
        raise SystemExit(
            "[FATAL] tokenizer audit did not load the exact certified snapshot while retaining "
            "logical model/revision provenance")
    if model.get("parent_snapshot_inventory_sha256") != expected_snapshot_inventory_sha256:
        raise SystemExit("[FATAL] tokenizer audit used a different snapshot inventory")
    if model.get("authoritative_revision_files_sha256") != \
            expected_authoritative_files_sha256:
        raise SystemExit("[FATAL] tokenizer audit lacks the official revision manifest binding")
    expected_guard = {
        "before_and_after_each_load": True,
        "operations": ["config", "fast_tokenizer", "slow_tokenizer"],
    }
    if model.get("snapshot_load_guard") != expected_guard:
        raise SystemExit("[FATAL] tokenizer audit did not guard every exact-snapshot load")
    expected_vocab_file = os.path.join(expected_load_source["path"], "tokenizer.model")
    tokenizer = model.get("tokenizer", {})
    if tokenizer.get("is_fast") is not True:
        raise SystemExit("[FATAL] tokenizer audit did not use the required fast tokenizer")
    base = model.get("base_tokenizer_contract", {})
    if model.get("model_config_vocab_size") != 256_000:
        raise SystemExit("[FATAL] pinned Gemma model config vocab_size is not 256000")
    if (base.get("vocab_size"), base.get("length")) != (256_000, 256_000):
        raise SystemExit(
            "[FATAL] Gemma base tokenizer is collapsed or mismatched; expected "
            "vocab_size=len(tokenizer)=256000 before sentinels")
    if (tokenizer.get("vocab_size"), tokenizer.get("length_after_sentinels")) != (
            256_000, 256_002):
        raise SystemExit(
            "[FATAL] Gemma tokenizer contract is not 256000 base / 256002 with sentinels")
    if tokenizer.get("sentinels") != {"[END_CELL]": 256_000, "[DOWN]": 256_001}:
        raise SystemExit("[FATAL] Gemma sentinel ids are not 256000/256001")
    if tokenizer.get("vocab_file") != expected_vocab_file:
        raise SystemExit(
            "[FATAL] fast Gemma tokenizer did not open the certified snapshot tokenizer.model")
    if (tokenizer.get("pad_token_id"), tokenizer.get("eos_token_id"),
            tokenizer.get("bos_token_id")) != (0, 1, 2):
        raise SystemExit("[FATAL] Gemma PAD/EOS/BOS contract is not 0/1/2")
    parity = model.get("slow_fast_parity", {})
    if (parity.get("available") is not True or parity.get("mismatch_count") != 0 or
            parity.get("rows_checked") != PARITY_ROWS or
            parity.get("slow_vocab_file") != expected_vocab_file):
        raise SystemExit("[FATAL] Gemma slow/fast tokenizer parity was not proved")
    maximum_response = 0
    files = model.get("files", [])
    if len(files) != 5:
        raise SystemExit("[FATAL] tokenizer audit must contain all five canonical JSONLs")
    observed_paths = []
    for file_report in files:
        if not isinstance(file_report, dict):
            raise SystemExit("[FATAL] malformed tokenizer file report")
        report_path = str(Path(file_report.get("path", "")).resolve())
        observed_paths.append(report_path)
        if report_path not in certified_by_path:
            raise SystemExit(f"[FATAL] tokenizer audit contains uncertified data: {report_path}")
        if file_report.get("sha256") != certified_by_path[report_path]:
            raise SystemExit(
                f"[FATAL] tokenizer audit hash differs from certified data: {report_path}")
        counts = file_report.get("counts")
        if not isinstance(counts, dict):
            raise SystemExit(f"[FATAL] tokenizer audit has no counters: {report_path}")
        missing_counters = MANDATORY_COUNTERS - set(counts)
        if missing_counters:
            raise SystemExit(
                f"[FATAL] tokenizer audit omitted counters for {report_path}: "
                f"{sorted(missing_counters)}")
        for key in ("semantic_truncations", "prompt_truncations", "sentinel_losses",
                    "response_missing_end_cell", "response_contains_down"):
            if counts[key] != 0:
                raise SystemExit(
                    f"[FATAL] tokenizer audit failed for {file_report.get('path')}: "
                    f"{key}={counts[key]}")
        if counts["rows"] <= 0:
            raise SystemExit(f"[FATAL] empty tokenizer audit: {file_report.get('path')}")
        for key in ("prompt_unk_tokens", "response_unk_tokens"):
            value = counts[key]
            if type(value) is not int or value < 0:
                raise SystemExit(
                    f"[FATAL] tokenizer audit has invalid {key} for "
                    f"{file_report.get('path')}: {value!r}")
            if value != 0:
                raise SystemExit(
                    f"[FATAL] tokenizer audit contains unknown tokens for "
                    f"{file_report.get('path')}: {key}={value}")
        distributions = file_report.get("distributions", {})
        prompt_min = distributions.get("prompt_tokens", {}).get("min", 0)
        response_min = distributions.get("response_tokens", {}).get("min", 0)
        if prompt_min < 20 or response_min < 20:
            raise SystemExit(
                f"[FATAL] collapsed tokenizer lengths for {file_report.get('path')}: "
                f"prompt_min={prompt_min}, response_min={response_min}")
        maximum_response = max(
            maximum_response,
            int(distributions.get("response_tokens", {}).get("max", 0)))
    if len(set(observed_paths)) != 5 or set(observed_paths) != set(certified_by_path):
        raise SystemExit("[FATAL] tokenizer audit file reports are duplicated or incomplete")
    # The response already includes END_CELL. A strict inequality leaves at least
    # one token of headroom and proves the generation cap is nonbinding for truth.
    authorized = maximum_response < generation_cap
    if not authorized:
        raise SystemExit(
            f"[FATAL] generation cap {generation_cap} is binding: "
            f"maximum truth response is {maximum_response} tokens")
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "generation_cap": generation_cap,
        "maximum_truth_response_tokens": maximum_response,
        "generation_cap_authorized": True,
        "logical_model_id": expected_model_id,
        "logical_revision": expected_revision,
        "certified_load_source": expected_load_source,
        "snapshot_load_guard": expected_guard,
    }


def create(args):
    snapshot = certified_snapshot(
        args.model_id, args.revision, args.snapshot_path, args.hub_cache)
    data = parse_named_paths(args.data)
    require_canonical_data_hashes(data)
    sources = parse_named_paths(args.source)
    environment = parse_named_paths(args.environment)
    tests = parse_named_paths([f"section1={args.tests_certificate}"])
    for name, entries in (("data", data), ("sources", sources),
                          ("environment", environment), ("tests", tests)):
        require_exact_keys(name, entries)
    observed_protobuf_digest, protobuf_file_count = protobuf_tree_digest(args.protobuf_root)
    if observed_protobuf_digest != args.protobuf_tree_sha256:
        raise SystemExit("[FATAL] helper and certificate computed different Protobuf digests")
    certificate = {
        "schema_version": SCHEMA_VERSION,
        "preflight_passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "repo_id": args.model_id,
            "revision": args.revision,
            **snapshot,
        },
        "data": data,
        "sources": sources,
        "environment": environment,
        "tests": tests,
        "protobuf_runtime": {
            "root": str(Path(args.protobuf_root).resolve()),
            "version": PROTOBUF_VERSION,
            "tree_sha256": observed_protobuf_digest,
            "file_count": protobuf_file_count,
        },
        "tokenizer_probe": validate_probe(
            args.tokenizer_probe, args.generation_cap, args.model_id, args.revision,
            snapshot["snapshot_path"], snapshot["inventory_sha256"],
            snapshot["authoritative_revision_files_sha256"], data),
    }
    atomic_json(args.certificate, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))


def verify_entries(group_name, entries):
    if not entries:
        raise SystemExit(f"[FATAL] certificate has no {group_name} entries")
    for name, entry in entries.items():
        path = entry.get("path")
        expected = entry.get("sha256")
        if not path or not Path(path).is_file():
            raise SystemExit(f"[FATAL] certified {group_name}.{name} is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise SystemExit(
                f"[FATAL] certified {group_name}.{name} changed: {observed} != {expected}")


def runtime_contract_document(certificate, *, model_id, revision, snapshot_path,
                              snapshot_inventory_sha256,
                              authoritative_revision_files_sha256):
    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "preflight_certificate_path": str(Path(certificate).resolve()),
        "preflight_certificate_sha256": sha256_file(certificate),
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(Path(snapshot_path).absolute()),
        "snapshot_inventory_sha256": snapshot_inventory_sha256,
        "authoritative_revision_files_sha256": authoritative_revision_files_sha256,
    }


def verify(args):
    document = json.load(open(args.certificate, encoding="utf-8"))
    if (document.get("schema_version") != SCHEMA_VERSION or
            document.get("preflight_passed") is not True):
        raise SystemExit("[FATAL] invalid or unsuccessful preflight certificate")
    model = document.get("model", {})
    if model.get("repo_id") != args.model_id or model.get("revision") != args.revision:
        raise SystemExit("[FATAL] preflight certificate authorizes a different model/revision")
    snapshot = Path(model.get("snapshot_path", ""))
    expected_snapshot = certified_snapshot(
        args.model_id, args.revision, snapshot, model.get("hub_cache", ""))
    if model.get("snapshot_relative_to_hub_cache") != expected_snapshot.get(
            "snapshot_relative_to_hub_cache"):
        raise SystemExit("[FATAL] certified snapshot-relative path changed")
    if model.get("inventory") != expected_snapshot.get("inventory"):
        raise SystemExit("[FATAL] pinned snapshot inventory, symlinks or contents changed")
    if model.get("inventory_sha256") != expected_snapshot.get("inventory_sha256"):
        raise SystemExit("[FATAL] pinned snapshot inventory digest changed")
    if model.get("authoritative_revision_files") != AUTHORITATIVE_REVISION_FILES:
        raise SystemExit("[FATAL] certificate does not contain the pinned official revision files")
    if model.get("official_snapshot_top_level_files") != sorted(
            OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES):
        raise SystemExit("[FATAL] certificate does not contain the official snapshot filename set")
    if model.get("official_snapshot_top_level_files") != expected_snapshot.get(
            "official_snapshot_top_level_files"):
        raise SystemExit("[FATAL] official Gemma snapshot filename set changed")
    if model.get("authoritative_revision_files") != expected_snapshot.get(
            "authoritative_revision_files"):
        raise SystemExit("[FATAL] official Gemma revision bytes changed")
    expected_authoritative_digest = authoritative_revision_files_sha256()
    if (model.get("authoritative_revision_files_sha256") != expected_authoritative_digest or
            expected_snapshot.get("authoritative_revision_files_sha256") !=
            expected_authoritative_digest):
        raise SystemExit("[FATAL] official Gemma revision manifest digest changed")
    for group in ("data", "sources", "environment", "tests"):
        entries = document.get(group, {})
        require_exact_keys(group, entries)
        if group == "data":
            require_canonical_data_hashes(entries)
        verify_entries(group, entries)
    protobuf = document.get("protobuf_runtime", {})
    expected_protobuf = {
        "root": str(PROTOBUF_ROOT.resolve()),
        "version": PROTOBUF_VERSION,
    }
    if {key: protobuf.get(key) for key in expected_protobuf} != expected_protobuf:
        raise SystemExit("[FATAL] certificate authorizes a different Protobuf runtime")
    observed_digest, observed_count = protobuf_tree_digest(PROTOBUF_ROOT)
    if (protobuf.get("tree_sha256"), protobuf.get("file_count")) != (
            observed_digest, observed_count):
        raise SystemExit("[FATAL] isolated Protobuf package contents changed")
    if os.environ.get("PYTHONPATH") != str(PROTOBUF_ROOT):
        raise SystemExit("[FATAL] production PYTHONPATH is not exactly the Protobuf directory")
    if "PYTHONHOME" in os.environ:
        raise SystemExit("[FATAL] PYTHONHOME must be unset for the certified Gemma runtime")
    if os.environ.get("GEMMA_PROTOBUF_TREE_SHA256") != observed_digest:
        raise SystemExit("[FATAL] helper Protobuf digest is absent or differs from certificate")
    if args.verify_current_environment:
        pip_entry = document.get("environment", {}).get("pip_freeze")
        if not pip_entry:
            raise SystemExit("[FATAL] certificate has no pip_freeze environment record")
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")
        normalized = "\n".join(sorted(line.rstrip() for line in output.splitlines())) + "\n"
        observed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if observed != pip_entry.get("sha256"):
            raise SystemExit(
                f"[FATAL] active Python environment changed: {observed} != "
                f"{pip_entry.get('sha256')}")
    probe = document.get("tokenizer_probe", {})
    verify_entries("tokenizer_probe", {"report": probe})
    if args.require_generation_cap is not None:
        if (probe.get("generation_cap_authorized") is not True or
                probe.get("generation_cap") != args.require_generation_cap):
            raise SystemExit(
                f"[FATAL] generation cap {args.require_generation_cap} was not authorized")
    runtime_contract = runtime_contract_document(
        args.certificate, model_id=args.model_id, revision=args.revision,
        snapshot_path=snapshot,
        snapshot_inventory_sha256=model["inventory_sha256"],
        authoritative_revision_files_sha256=model[
            "authoritative_revision_files_sha256"])
    if args.runtime_contract_out:
        atomic_json(args.runtime_contract_out, runtime_contract)
    if args.print_runtime_contract:
        print(json.dumps(runtime_contract, sort_keys=True, separators=(",", ":")))
    elif args.print_snapshot:
        print(str(snapshot))
    else:
        print(f"[PASS] verified preflight certificate {Path(args.certificate).resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--certificate", required=True)
    common.add_argument("--model-id", required=True)
    common.add_argument("--revision", required=True)

    create_parser = subparsers.add_parser("create", parents=[common])
    create_parser.add_argument("--snapshot-path", required=True)
    create_parser.add_argument("--hub-cache", required=True)
    create_parser.add_argument("--tests-certificate", required=True)
    create_parser.add_argument("--tokenizer-probe", required=True)
    create_parser.add_argument("--generation-cap", required=True, type=int)
    create_parser.add_argument("--protobuf-root", required=True)
    create_parser.add_argument("--protobuf-tree-sha256", required=True)
    create_parser.add_argument("--data", action="append", default=[])
    create_parser.add_argument("--source", action="append", default=[])
    create_parser.add_argument("--environment", action="append", default=[])
    create_parser.set_defaults(func=create)

    verify_parser = subparsers.add_parser("verify", parents=[common])
    verify_parser.add_argument("--require-generation-cap", type=int)
    verify_parser.add_argument("--print-snapshot", action="store_true")
    verify_parser.add_argument("--print-runtime-contract", action="store_true")
    verify_parser.add_argument("--runtime-contract-out")
    verify_parser.add_argument("--verify-current-environment", action="store_true")
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
