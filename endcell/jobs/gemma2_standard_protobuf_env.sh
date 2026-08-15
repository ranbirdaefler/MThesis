#!/usr/bin/env bash
# Bind Gemma jobs to the isolated SentencePiece-conversion dependency without modifying c2s.
# Source only after PY has been set to the canonical c2s Python executable.

: "${PY:?Set PY to the canonical c2s Python executable before sourcing this file}"

CANONICAL_GEMMA_PROTOBUF_DIR=/data/BuffaF-Projetcs/florian_c2s/test_deps/protobuf-5.29.5
if [[ -n "${GEMMA_PROTOBUF_DIR:-}" && "$GEMMA_PROTOBUF_DIR" != "$CANONICAL_GEMMA_PROTOBUF_DIR" ]]; then
    echo "[FATAL] GEMMA_PROTOBUF_DIR override is not permitted: $GEMMA_PROTOBUF_DIR" >&2
    return 2 2>/dev/null || exit 2
fi
GEMMA_PROTOBUF_DIR="$CANONICAL_GEMMA_PROTOBUF_DIR"
GEMMA_PROTOBUF_VERSION=5.29.5

[[ -f "$GEMMA_PROTOBUF_DIR/google/protobuf/__init__.py" ]] || {
    echo "[FATAL] isolated protobuf is missing at $GEMMA_PROTOBUF_DIR" >&2
    return 2 2>/dev/null || exit 2
}

export GEMMA_PROTOBUF_DIR GEMMA_PROTOBUF_VERSION
unset PYTHONHOME
export PYTHONPATH="$GEMMA_PROTOBUF_DIR"

GEMMA_PROTOBUF_TREE_SHA256="$("$PY" - "$GEMMA_PROTOBUF_DIR" "$GEMMA_PROTOBUF_VERSION" <<'PY'
import hashlib
import csv
import json
import pathlib
import sys

try:
    import google.protobuf
except Exception as exc:
    raise SystemExit(f"[FATAL] isolated protobuf cannot be imported: {exc!r}")

expected_root = pathlib.Path(sys.argv[1]).resolve()
observed_file = pathlib.Path(google.protobuf.__file__).resolve()
try:
    observed_file.relative_to(expected_root)
except ValueError:
    raise SystemExit(
        f"[FATAL] protobuf resolved outside the isolated directory: "
        f"{observed_file} not under {expected_root}")

expected_version = sys.argv[2]
observed_version = getattr(google.protobuf, "__version__", None)
if observed_version != expected_version:
    raise SystemExit(
        f"[FATAL] protobuf version {observed_version!r} != {expected_version!r}")

dist_info = expected_root / f"protobuf-{expected_version}.dist-info"
record = dist_info / "RECORD"
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
    relative = pathlib.PurePosixPath(raw)
    if (relative.is_absolute() or "\\" in raw or
            relative.as_posix() != raw or ".." in relative.parts):
        raise SystemExit(f"[FATAL] unsafe protobuf RECORD path: {raw!r}")
    candidate = expected_root / pathlib.Path(*relative.parts)
    if candidate.is_symlink():
        raise SystemExit(f"[FATAL] protobuf RECORD payload is a symlink: {raw!r}")
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"[FATAL] protobuf RECORD payload is missing: {raw!r}: {exc}")
    try:
        path.relative_to(expected_root)
    except ValueError:
        raise SystemExit(f"[FATAL] protobuf RECORD path escapes isolated root: {raw!r}")
    normalized = relative.as_posix()
    if normalized in seen:
        raise SystemExit(f"[FATAL] duplicate protobuf RECORD payload: {normalized}")
    seen.add(normalized)
    if not path.is_file():
        raise SystemExit(f"[FATAL] protobuf RECORD payload is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as payload_handle:
        for chunk in iter(lambda: payload_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    inventory.append({
        "path": normalized,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    })
if not inventory:
    raise SystemExit("[FATAL] protobuf digest inventory is empty")
if "google/_upb/_message.abi3.so" not in seen:
    raise SystemExit("[FATAL] protobuf RECORD omits google/_upb/_message.abi3.so")
payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
 )" || { return 2 2>/dev/null || exit 2; }
[[ "$GEMMA_PROTOBUF_TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "[FATAL] invalid protobuf tree digest: $GEMMA_PROTOBUF_TREE_SHA256" >&2
    return 2 2>/dev/null || exit 2
}
export GEMMA_PROTOBUF_TREE_SHA256
echo "[PASS] isolated protobuf $GEMMA_PROTOBUF_VERSION tree $GEMMA_PROTOBUF_TREE_SHA256"
