#!/usr/bin/env python3
"""Shared, dependency-free provenance primitives for the canonical Gemma arm.

The preflight certificate, trainer, checkpoint validator and evaluation launcher must
agree byte-for-byte on what constitutes the pinned parent snapshot.  Keeping the
inventory and digest implementation here prevents those trust boundaries from
silently drifting apart.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, TypeVar


GEMMA_MODEL_ID = "vandijklab/C2S-Scale-Gemma-2-2B"
GEMMA_MODEL_REVISION = "5ddf28b8f1c81b7ab7a9be192924da82b6c5d512"
SHA256_LENGTH = 64

# Independent upstream pins for every file Transformers needs to construct the
# canonical parent.  The three LFS hashes are the official ``lfs.sha256`` values
# returned by the Hugging Face revision API with ``blobs=true``.  The five small
# Git-managed files were fetched from the immutable ``resolve/<revision>`` URLs
# and hashed independently on 2026-08-12.  Preflight must match these constants;
# it is not allowed to generate a new truth from whatever happens to be cached.
#
# API: https://huggingface.co/api/models/vandijklab/C2S-Scale-Gemma-2-2B/
#      revision/5ddf28b8f1c81b7ab7a9be192924da82b6c5d512?blobs=true
AUTHORITATIVE_REVISION_FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "size": 805,
        "sha256": "7f55dee0fe58a424b42205e6f091e68693f25ea9f9b76c44d9bbbd98fec13c70",
        "upstream_hash_kind": "raw-revision-sha256",
    },
    "generation_config.json": {
        "size": 168,
        "sha256": "784ca26f45a937725a3056440711d6f09c4693ed38cfb40ee0b8dc9237d65fce",
        "upstream_hash_kind": "raw-revision-sha256",
    },
    "model.safetensors.index.json": {
        "size": 24_223,
        "sha256": "974f980f459399b2fda03e556f9917f5d03228f4c07dfbe7a6b146be02d4e91c",
        "upstream_hash_kind": "raw-revision-sha256",
    },
    "model-00001-of-00002.safetensors": {
        "size": 4_988_025_760,
        "sha256": "2ab9af99ef0535f7ba784acce88f7123433010bc632a3ae47daabc940b41b9dc",
        "upstream_hash_kind": "huggingface-lfs-sha256",
    },
    "model-00002-of-00002.safetensors": {
        "size": 240_691_728,
        "sha256": "0b95695d3c23e9a549606eba8874b60172247b39fbe6241f870df68d2bdc35db",
        "upstream_hash_kind": "huggingface-lfs-sha256",
    },
    "special_tokens_map.json": {
        "size": 555,
        "sha256": "db82f8bd9b25d14f9c788e6bde64de84d42f1c2538f1c245ba6cb3e872d14b18",
        "upstream_hash_kind": "raw-revision-sha256",
    },
    "tokenizer.model": {
        "size": 4_945_541,
        "sha256": "a343c33e9f5cd740f55625b7fc556d71599c239ada13f6308387d6ce79b3a9d6",
        "upstream_hash_kind": "huggingface-lfs-sha256",
    },
    "tokenizer_config.json": {
        "size": 1_120,
        "sha256": "a1625569b41792112ff0626e1e2c2f82aa0091dc64457dc332ee95ae82e8583f",
        "upstream_hash_kind": "raw-revision-sha256",
    },
}

# Exact top-level tree returned by the immutable Hugging Face revision above.
# The three non-load-bearing repository files are allowed because they are part
# of that official revision; no other file is permitted.  This is deliberately
# stricter than pinning only the eight files above: Transformers gives several
# conventional filenames (for example ``model.safetensors`` and
# ``tokenizer.json``) precedence over the sharded model/tokenizer files, so an
# extra cache entry must never be able to redirect a load.
OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES = frozenset({
    ".gitattributes",
    "LICENSE",
    "README.md",
    *AUTHORITATIVE_REVISION_FILES,
})

CANONICAL_TAHOE_DATA_SHA256: dict[str, str] = {
    "train": "4bed186da4c5348dbb899182f2ab0f635c0f73ce7994be8738a23f8e6c61d000",
    "tier1": "bb2d8f45c32b7f64f0e874c3e29d3e4e086194c3cc18d97fea5016c494f8bb21",
    "tier2": "054dc5370103fb9da381046b13cc7235b556bb39d7ffced4976fd45768727c39",
    "tier3": "29c8cb31e82ce39d988983457ebea34b78a15369b5dbd69cb2db76185ca1d2d8",
    "tier4": "0015f09ce47bc45a574ef7d9dbd54e8342929d56fc965b50869a61fb75722d75",
}

_T = TypeVar("_T")


class ProvenanceError(RuntimeError):
    """A certificate, snapshot or checkpoint violates the pinned provenance contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != SHA256_LENGTH or
            any(character not in "0123456789abcdef" for character in value)):
        raise ProvenanceError(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def snapshot_inventory(snapshot_path: str | Path) -> dict[str, dict[str, object]]:
    """Return the complete deterministic inventory certified by preflight.

    Symlink identity and target contents are both bound: ``symlink_target`` records
    the link while ``size`` and ``sha256`` follow it, matching the bytes Transformers
    will open.  Directories are excluded; every other entry must resolve to a file.
    """
    root = Path(os.path.abspath(os.fspath(snapshot_path)))
    if not root.is_dir():
        raise ProvenanceError(f"model snapshot is missing: {root}")
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            raise ProvenanceError(f"snapshot entry is not a readable file: {relative}")
        inventory[relative] = {
            "relative_path": relative,
            "symlink_target": os.readlink(path) if path.is_symlink() else None,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if not inventory:
        raise ProvenanceError("model snapshot inventory is empty")
    return inventory


def snapshot_inventory_sha256(snapshot_path: str | Path) -> str:
    return stable_json_sha256(snapshot_inventory(snapshot_path))


def authoritative_revision_files_sha256() -> str:
    """Digest the independently pinned upstream manifest itself."""
    return stable_json_sha256(AUTHORITATIVE_REVISION_FILES)


def verify_authoritative_snapshot(snapshot_path: str | Path) -> dict[str, dict[str, object]]:
    """Reject a cache that differs from the official immutable revision bytes.

    Hugging Face snapshots normally contain symlinks into the blob cache.  ``stat``
    and ``sha256_file`` intentionally follow those links because those are the bytes
    opened by Transformers.  The top-level filename set must exactly match the
    immutable revision, so an additional conventional model/tokenizer/adapter file
    cannot override one of the independently pinned files.
    """
    root = Path(os.path.abspath(os.fspath(snapshot_path)))
    if not root.is_dir():
        raise ProvenanceError(f"model snapshot is missing: {root}")
    observed_names = {path.name for path in root.iterdir()}
    missing = OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES - observed_names
    unexpected = observed_names - OFFICIAL_SNAPSHOT_TOP_LEVEL_FILES
    if missing or unexpected:
        raise ProvenanceError(
            "official Gemma snapshot filename set changed: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    nonfiles = sorted(
        name for name in observed_names if not (root / name).is_file())
    if nonfiles:
        raise ProvenanceError(
            f"official Gemma snapshot entries are not readable files: {nonfiles}")
    observed: dict[str, dict[str, object]] = {}
    for relative, expected in AUTHORITATIVE_REVISION_FILES.items():
        path = root / relative
        if not path.is_file():
            raise ProvenanceError(f"authoritative Gemma revision file is missing: {relative}")
        record = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "upstream_hash_kind": expected["upstream_hash_kind"],
        }
        if record != expected:
            raise ProvenanceError(
                f"authoritative Gemma revision mismatch for {relative}: "
                f"observed={record}, expected={expected}")
        observed[relative] = record
    return observed


def paths_refer_to_same_location(first: str | Path, second: str | Path) -> bool:
    """Compare paths across lexical BeeGFS aliases without changing stored provenance."""
    first_abs = os.path.abspath(os.fspath(first))
    second_abs = os.path.abspath(os.fspath(second))
    if first_abs == second_abs:
        return True
    try:
        return os.path.samefile(first_abs, second_abs)
    except OSError:
        return os.path.realpath(first_abs) == os.path.realpath(second_abs)


def require_snapshot_digest(snapshot_path: str | Path, expected_sha256: str,
                            *, stage: str) -> str:
    expected = require_sha256(expected_sha256, "expected snapshot inventory")
    observed = snapshot_inventory_sha256(snapshot_path)
    if observed != expected:
        raise ProvenanceError(
            f"pinned Gemma snapshot changed {stage}: observed={observed}, expected={expected}")
    return observed


def guarded_snapshot_load(load: Callable[[], _T], snapshot_path: str | Path,
                          expected_sha256: str, *, stage: str) -> _T:
    """Check a shared snapshot immediately before and after one load operation.

    This detects persistent cache mutation during config/tokenizer/model loading.
    It does not claim protection against an adversarial same-account process that
    can alter and restore bytes between checks; the runbook states that explicit
    non-adversarial shared-cache trust boundary.
    """
    require_snapshot_digest(snapshot_path, expected_sha256, stage=f"before {stage}")
    result = load()
    require_snapshot_digest(snapshot_path, expected_sha256, stage=f"after {stage}")
    return result
