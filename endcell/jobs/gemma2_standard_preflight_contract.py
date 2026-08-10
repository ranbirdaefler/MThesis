#!/usr/bin/env python3
"""Create or verify the fail-closed Gemma launch certificate.

This helper is intentionally standard-library only so every Slurm job can verify
the exact data, code, environment and pinned model snapshot approved by preflight.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def snapshot_inventory(snapshot_path):
    root = Path(snapshot_path)
    if not root.is_dir():
        raise SystemExit(f"[FATAL] model snapshot is missing: {root}")
    inventory = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            raise SystemExit(f"[FATAL] snapshot entry is not a readable file: {relative}")
        inventory[relative] = {
            "relative_path": relative,
            "symlink_target": os.readlink(path) if path.is_symlink() else None,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not inventory:
        raise SystemExit("[FATAL] model snapshot inventory is empty")
    return inventory


def certified_snapshot(model_id, revision, snapshot_path, hub_cache):
    cache = Path(os.path.abspath(hub_cache))
    snapshot = Path(os.path.abspath(snapshot_path))
    expected = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if snapshot != expected:
        raise SystemExit(
            f"[FATAL] snapshot path is not the exact pinned revision directory: "
            f"{snapshot} != {expected}")
    return {
        "hub_cache": str(cache),
        "snapshot_path": str(snapshot),
        "snapshot_relative_to_hub_cache": snapshot.relative_to(cache).as_posix(),
        "inventory": snapshot_inventory(snapshot),
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


def validate_probe(path, generation_cap):
    report = json.load(open(path, encoding="utf-8"))
    if report.get("max_examples_per_file") != 0:
        raise SystemExit("[FATAL] tokenizer audit was not run over every row")
    models = report.get("models", [])
    if len(models) != 1:
        raise SystemExit("[FATAL] tokenizer audit must contain exactly one model")
    model = models[0]
    tokenizer = model.get("tokenizer", {})
    if (tokenizer.get("pad_token_id"), tokenizer.get("eos_token_id"),
            tokenizer.get("bos_token_id")) != (0, 1, 2):
        raise SystemExit("[FATAL] Gemma PAD/EOS/BOS contract is not 0/1/2")
    maximum_response = 0
    for file_report in model.get("files", []):
        counts = file_report.get("counts", {})
        for key in ("semantic_truncations", "prompt_truncations", "sentinel_losses",
                    "response_missing_end_cell", "response_contains_down"):
            if counts.get(key, 0):
                raise SystemExit(
                    f"[FATAL] tokenizer audit failed for {file_report.get('path')}: "
                    f"{key}={counts[key]}")
        maximum_response = max(
            maximum_response,
            int(file_report.get("distributions", {}).get("response_tokens", {}).get("max", 0)))
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
    }


def create(args):
    snapshot = certified_snapshot(
        args.model_id, args.revision, args.snapshot_path, args.hub_cache)
    certificate = {
        "schema_version": 2,
        "preflight_passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "repo_id": args.model_id,
            "revision": args.revision,
            **snapshot,
        },
        "data": parse_named_paths(args.data),
        "sources": parse_named_paths(args.source),
        "environment": parse_named_paths(args.environment),
        "tests": parse_named_paths([f"section1={args.tests_certificate}"]),
        "tokenizer_probe": validate_probe(args.tokenizer_probe, args.generation_cap),
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


def verify(args):
    document = json.load(open(args.certificate, encoding="utf-8"))
    if document.get("schema_version") != 2 or document.get("preflight_passed") is not True:
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
    for group in ("data", "sources", "environment", "tests"):
        verify_entries(group, document.get(group, {}))
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
    if args.print_snapshot:
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
    create_parser.add_argument("--data", action="append", default=[])
    create_parser.add_argument("--source", action="append", default=[])
    create_parser.add_argument("--environment", action="append", default=[])
    create_parser.set_defaults(func=create)

    verify_parser = subparsers.add_parser("verify", parents=[common])
    verify_parser.add_argument("--require-generation-cap", type=int)
    verify_parser.add_argument("--print-snapshot", action="store_true")
    verify_parser.add_argument("--verify-current-environment", action="store_true")
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
