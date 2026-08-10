#!/usr/bin/env python3
"""Validate and fingerprint local Hugging Face checkpoints.

Trainer-produced checkpoints are verified against their immutable schema-2
``checkpoint_manifest.json`` before any inference fingerprint is emitted.
Legacy and upstream checkpoints may omit that manifest unless the caller
explicitly requires the canonical completed-SFT contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_SCHEMA = 2
TRAINING_STATE = "training_state.pt"
PROVENANCE_FILE = "run_provenance.json"
MANIFEST_KEYS = {"schema_version", "identity", "files"}
IDENTITY_KEYS = {
    "schema_version", "global_step", "epoch", "microbatch_position",
    "accumulation_position", "completed", "contract_fingerprint",
    "provenance_contract_fingerprint",
}
EXACT_NAMES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
}


class CheckpointValidationError(RuntimeError):
    """The published checkpoint no longer matches its trainer manifest."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    name = path.name
    return (
        name in EXACT_NAMES
        or name.endswith(".safetensors")
        or (name.startswith("pytorch_model") and name.endswith(".bin"))
    )


def checkpoint_file_records(checkpoint: Path) -> dict[str, dict[str, object]]:
    """Mirror the trainer's recursive inventory, excluding only the manifest itself."""
    records: dict[str, dict[str, object]] = {}
    for root, dirs, names in os.walk(checkpoint):
        dirs.sort()
        for name in sorted(names):
            path = Path(root) / name
            relative = path.relative_to(checkpoint).as_posix()
            if relative != CHECKPOINT_MANIFEST:
                records[relative] = {
                    "size": path.stat().st_size,
                    "sha256": file_hash(path),
                }
    return records


def _torch_load(path: Path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - cluster environment supplies Torch
        raise CheckpointValidationError(
            "Torch is required to validate trainer state identity") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location="cpu")


def _checkpoint_identity(state: dict, provenance: dict) -> dict:
    return {
        "schema_version": int(state.get("schema_version", -1)),
        "global_step": int(state.get("global_step", -1)),
        "epoch": int(state.get("epoch", -1)),
        "microbatch_position": int(state.get("microbatch_position", -1)),
        "accumulation_position": int(state.get("accumulation_position", -1)),
        "completed": bool(state.get("completed", False)),
        "contract_fingerprint": state.get("contract_fingerprint"),
        "provenance_contract_fingerprint": provenance.get("contract_fingerprint"),
    }


def validate_checkpoint_manifest(
    checkpoint: Path,
    *,
    required: bool,
    verify_state_identity: bool = True,
) -> dict | None:
    """Fail closed on schema, inventory, size, content, or state-identity mismatch."""
    manifest_path = checkpoint / CHECKPOINT_MANIFEST
    if not manifest_path.is_file():
        if required:
            raise CheckpointValidationError(
                f"checkpoint lacks required {CHECKPOINT_MANIFEST}: {checkpoint}")
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"invalid checkpoint manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise CheckpointValidationError(
            f"checkpoint manifest has unexpected top-level schema: {manifest_path}")
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
        raise CheckpointValidationError(
            f"unsupported checkpoint manifest schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise CheckpointValidationError(f"checkpoint manifest identity is malformed: {manifest_path}")
    recorded_files = manifest.get("files")
    if not isinstance(recorded_files, dict):
        raise CheckpointValidationError(f"checkpoint manifest files are malformed: {manifest_path}")
    for relative, record in recorded_files.items():
        candidate = Path(relative)
        if (not isinstance(relative, str) or candidate.is_absolute() or ".." in candidate.parts
                or not isinstance(record, dict) or set(record) != {"size", "sha256"}
                or not isinstance(record.get("size"), int) or record["size"] < 0
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64):
            raise CheckpointValidationError(
                f"checkpoint manifest contains an invalid file record: {relative!r}")
    actual_files = checkpoint_file_records(checkpoint)
    if actual_files != recorded_files:
        raise CheckpointValidationError(
            f"checkpoint file inventory/content mismatch: {checkpoint}")
    required_files = {TRAINING_STATE, PROVENANCE_FILE, "config.json"}
    missing = sorted(required_files - set(actual_files))
    if missing:
        raise CheckpointValidationError(
            f"checkpoint is incomplete ({missing}): {checkpoint}")
    if not any(name.endswith((".safetensors", ".bin")) for name in actual_files):
        raise CheckpointValidationError(f"checkpoint has no model weight file: {checkpoint}")

    state = None
    if verify_state_identity:
        state = _torch_load(checkpoint / TRAINING_STATE)
        try:
            provenance = json.loads((checkpoint / PROVENANCE_FILE).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(
                f"invalid checkpoint provenance: {checkpoint / PROVENANCE_FILE}") from exc
        if not isinstance(state, dict) or not isinstance(provenance, dict):
            raise CheckpointValidationError(f"checkpoint state/provenance is malformed: {checkpoint}")
        if _checkpoint_identity(state, provenance) != identity:
            raise CheckpointValidationError(
                f"checkpoint state disagrees with its manifest: {checkpoint}")
    return {"manifest": manifest, "state": state, "manifest_sha256": file_hash(manifest_path)}


def validate_terminal_state(state: dict, args: argparse.Namespace) -> None:
    expected = {
        "completed": True,
        "global_step": args.expected_global_step,
        "epoch": args.expected_epoch,
        "microbatch_position": args.expected_microbatch_position,
        "accumulation_position": args.expected_accumulation_position,
    }
    observed = {
        "completed": state.get("completed"),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "microbatch_position": state.get("microbatch_position"),
        "accumulation_position": state.get("accumulation_position"),
    }
    if observed != expected:
        raise CheckpointValidationError(
            f"terminal training state mismatch: observed={observed}, expected={expected}")


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def corruption_selftest() -> None:
    """Prove same-size corruption and inventory drift are rejected without requiring Torch."""
    with tempfile.TemporaryDirectory(prefix="gemma-checkpoint-selftest-") as temp:
        checkpoint = Path(temp) / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"model-bytes")
        (checkpoint / TRAINING_STATE).write_bytes(b"synthetic-state")
        (checkpoint / PROVENANCE_FILE).write_text("{}\n", encoding="utf-8")
        identity = {
            "schema_version": 1,
            "global_step": 42198,
            "epoch": 1,
            "microbatch_position": 0,
            "accumulation_position": 0,
            "completed": True,
            "contract_fingerprint": "a" * 64,
            "provenance_contract_fingerprint": "a" * 64,
        }
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "identity": identity,
            "files": checkpoint_file_records(checkpoint),
        }
        (checkpoint / CHECKPOINT_MANIFEST).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validate_checkpoint_manifest(checkpoint, required=True, verify_state_identity=False)
        terminal_args = argparse.Namespace(
            expected_global_step=42198,
            expected_epoch=1,
            expected_microbatch_position=0,
            expected_accumulation_position=0,
        )
        validate_terminal_state(identity, terminal_args)
        wrong_terminal = dict(identity, microbatch_position=675183)
        try:
            validate_terminal_state(wrong_terminal, terminal_args)
        except CheckpointValidationError:
            pass
        else:
            raise AssertionError("nonterminal microbatch position was accepted")

        weight_path = checkpoint / "model.safetensors"
        original = weight_path.read_bytes()
        weight_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        try:
            validate_checkpoint_manifest(checkpoint, required=True, verify_state_identity=False)
        except CheckpointValidationError:
            pass
        else:
            raise AssertionError("same-size weight corruption was accepted")
        weight_path.write_bytes(original)
        validate_checkpoint_manifest(checkpoint, required=True, verify_state_identity=False)

        (checkpoint / "unexpected.bin").write_bytes(b"unexpected")
        try:
            validate_checkpoint_manifest(checkpoint, required=True, verify_state_identity=False)
        except CheckpointValidationError:
            pass
        else:
            raise AssertionError("unmanifested file was accepted")
    print("CHECKPOINT_CORRUPTION_SELFTEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--out")
    parser.add_argument("--digest_only", action="store_true")
    parser.add_argument("--require_complete_sft", action="store_true")
    parser.add_argument("--expected_global_step", type=int, default=42198)
    parser.add_argument("--expected_epoch", type=int, default=1)
    parser.add_argument("--expected_microbatch_position", type=int, default=0)
    parser.add_argument("--expected_accumulation_position", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        corruption_selftest()
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --selftest is used")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {checkpoint}")

    validation = validate_checkpoint_manifest(
        checkpoint, required=args.require_complete_sft, verify_state_identity=True)
    if args.require_complete_sft:
        validate_terminal_state(validation["state"], args)

    files = sorted((path for path in checkpoint.iterdir() if path.is_file() and included(path)),
                   key=lambda path: path.name)
    weights = [path for path in files if path.suffix in {".safetensors", ".bin"}]
    if not weights or not (checkpoint / "config.json").is_file():
        raise SystemExit("checkpoint lacks config.json or model weight files")

    rows = [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in files]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report = {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "fingerprint": hashlib.sha256(payload).hexdigest(),
        "files": rows,
        "excludes_training_state": True,
        "trainer_manifest_validated": validation is not None,
        "trainer_manifest_sha256": None if validation is None else validation["manifest_sha256"],
        "terminal_training_state_validated": bool(args.require_complete_sft),
    }
    if args.out:
        _write_json_atomic(Path(args.out), report)
    print(report["fingerprint"] if args.digest_only else json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
