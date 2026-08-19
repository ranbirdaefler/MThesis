#!/usr/bin/env python3
"""Admit one immutable partial Gemma SFT checkpoint without changing preflight sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gemma2_standard_checkpoint_fingerprint as fingerprint


CANONICAL_STEP = 41301
CANONICAL_EPOCH = 0
CANONICAL_MICROBATCH = 660816
PLANNED_STEPS = 42198


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, document: dict) -> None:
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing admission record: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument("--fingerprint-tool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--digest-only", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    expected_name = f"checkpoint-{CANONICAL_STEP}-mb{CANONICAL_MICROBATCH}"
    if checkpoint.name != expected_name:
        raise SystemExit(f"partial checkpoint must be {expected_name}, got {checkpoint.name}")

    validation = fingerprint.validate_checkpoint_manifest(
        checkpoint, required=True, verify_state_identity=True)
    state = validation["state"]
    expected_state = {
        "completed": False,
        "global_step": CANONICAL_STEP,
        "epoch": CANONICAL_EPOCH,
        "microbatch_position": CANONICAL_MICROBATCH,
        "accumulation_position": 0,
    }
    observed_state = {key: state.get(key) for key in expected_state}
    if observed_state != expected_state:
        raise SystemExit(
            f"partial checkpoint state mismatch: observed={observed_state}, expected={expected_state}")

    runtime_path = Path(args.runtime_contract).resolve()
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("schema_version") != 2:
        raise SystemExit("runtime contract schema is not 2")
    ancestry = {
        "model_id": "vandijklab/C2S-Scale-Gemma-2-2B",
        "revision": "5ddf28b8f1c81b7ab7a9be192924da82b6c5d512",
        "parent_snapshot": runtime["snapshot_path"],
        "preflight_certificate_sha256": runtime["preflight_certificate_sha256"],
        "snapshot_inventory_sha256": runtime["snapshot_inventory_sha256"],
        "authoritative_files_sha256": runtime["authoritative_revision_files_sha256"],
    }
    fingerprint.validate_gemma_checkpoint_ancestry(checkpoint, **ancestry)

    tool = Path(args.fingerprint_tool).resolve()
    command = [
        sys.executable, str(tool), "--checkpoint", str(checkpoint),
        "--require_gemma_ancestry",
        "--expected_model_id", ancestry["model_id"],
        "--expected_revision", ancestry["revision"],
        "--expected_parent_snapshot", ancestry["parent_snapshot"],
        "--expected_preflight_certificate_sha256", ancestry["preflight_certificate_sha256"],
        "--expected_snapshot_inventory_sha256", ancestry["snapshot_inventory_sha256"],
        "--expected_authoritative_files_sha256", ancestry["authoritative_files_sha256"],
        "--digest_only",
    ]
    model_fingerprint = subprocess.check_output(command, text=True).strip()
    if len(model_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in model_fingerprint):
        raise SystemExit(f"invalid model fingerprint: {model_fingerprint!r}")

    report = {
        "schema_version": 1,
        "decision": "admit_partial_sft_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": validation["manifest_sha256"],
        "model_fingerprint": model_fingerprint,
        "training_state": observed_state,
        "planned_global_steps": PLANNED_STEPS,
        "training_fraction": CANONICAL_STEP / PLANNED_STEPS,
        "optimizer_boundary_verified": True,
        "certified_ancestry_verified": True,
        "preflight_certificate_sha256": runtime["preflight_certificate_sha256"],
        "admission_script_sha256": sha256_file(Path(__file__).resolve()),
        "fingerprint_tool_sha256": sha256_file(tool),
    }
    write_immutable(Path(args.out).resolve(), report)
    print(model_fingerprint if args.digest_only else json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
