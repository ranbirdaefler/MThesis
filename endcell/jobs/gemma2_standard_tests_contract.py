#!/usr/bin/env python3
"""Create and verify the fail-closed Gemma cluster-test certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_files(items: list[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"[FATAL] expected NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in result or not path.is_file():
            raise SystemExit(f"[FATAL] invalid test-certificate input: {item!r}")
        result[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def normalized_pip_freeze() -> str:
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")
    return "\n".join(sorted(line.rstrip() for line in output.splitlines())) + "\n"


def atomic_json(path: str, document: dict[str, object]) -> None:
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


def verify_file_group(group: str, entries: dict[str, dict[str, object]]) -> None:
    if not entries:
        raise SystemExit(f"[FATAL] tests certificate has no {group}")
    for name, entry in entries.items():
        path = Path(str(entry.get("path", "")))
        if not path.is_file():
            raise SystemExit(f"[FATAL] certified {group}.{name} is missing: {path}")
        observed = sha256_file(path)
        if observed != entry.get("sha256"):
            raise SystemExit(
                f"[FATAL] certified {group}.{name} changed: "
                f"{observed} != {entry.get('sha256')}")


def create(args: argparse.Namespace) -> None:
    command = Path(args.command).resolve()
    if not command.is_file():
        raise SystemExit(f"[FATAL] test command is missing: {command}")
    pip_freeze = normalized_pip_freeze()
    document = {
        "schema_version": 1,
        "tests_passed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "path": str(command),
            "size": command.stat().st_size,
            "sha256": sha256_file(command),
        },
        "sources": named_files(args.source),
        "environment": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pip_freeze_sha256": hashlib.sha256(pip_freeze.encode("utf-8")).hexdigest(),
        },
    }
    atomic_json(args.certificate, document)
    print(f"[PASS] atomic tests certificate: {Path(args.certificate).resolve()}")


def verify(args: argparse.Namespace) -> None:
    path = Path(args.certificate).resolve()
    document = json.load(open(path, encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("tests_passed") is not True:
        raise SystemExit("[FATAL] invalid or unsuccessful tests certificate")
    verify_file_group("command", {"section1": document.get("command", {})})
    verify_file_group("sources", document.get("sources", {}))
    environment = document.get("environment", {})
    if args.verify_current_environment:
        if str(Path(sys.executable).resolve()) != environment.get("python_executable"):
            raise SystemExit("[FATAL] Python executable changed since cluster tests")
        if platform.python_version() != environment.get("python_version"):
            raise SystemExit("[FATAL] Python version changed since cluster tests")
        if platform.platform() != environment.get("platform"):
            raise SystemExit("[FATAL] platform changed since cluster tests")
        observed = hashlib.sha256(normalized_pip_freeze().encode("utf-8")).hexdigest()
        if observed != environment.get("pip_freeze_sha256"):
            raise SystemExit("[FATAL] Python packages changed since cluster tests")
    print(f"[PASS] verified cluster-test certificate {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--certificate", required=True)
    create_parser.add_argument("--command", required=True)
    create_parser.add_argument("--source", action="append", default=[])
    create_parser.set_defaults(func=create)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--certificate", required=True)
    verify_parser.add_argument("--verify-current-environment", action="store_true")
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
