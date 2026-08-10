#!/usr/bin/env python
"""Freeze an ordered, model-independent NIR evaluation manifest.

The manifest fixes biological support, truth halves, prompt identities, the existing
wrong-condition sensitivity partner, and deterministic generation seed inputs before
any model output is inspected.  It can be imported by evaluators or run directly.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 2
DEFAULT_CONTRACT_VERSION = "nir-generation-v1"
END_CELL = "[END_CELL]"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: str, value: Any, *, default=None,
                      ensure_ascii: bool = False, overwrite: bool = True) -> None:
    """Durably publish one complete JSON document under an exclusive writer lock."""
    target = os.path.abspath(path)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    lock_path = target + ".lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"output is locked by another writer: {lock_path}") from exc
    tmp = None
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_fd)
        if not overwrite and os.path.exists(target):
            raise FileExistsError(f"refusing to overwrite immutable artifact: {target}")
        fd, tmp = tempfile.mkstemp(prefix=f".{Path(target).name}.", suffix=".tmp", dir=parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=ensure_ascii, default=default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def stable_output_seed(model_fingerprint: str, row_id: str, prompt_id: str,
                       draw_id: int, contract_version: str) -> int:
    """Return a positive 63-bit seed, independent of process/batch/order state."""
    payload = "\x1f".join((model_fingerprint, row_id, prompt_id, str(draw_id), contract_version))
    seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return seed & ((1 << 63) - 1) or 1


def control_from_prompt(prompt: str) -> Optional[str]:
    marker = "Control cell:"
    start = prompt.find(marker)
    if start < 0:
        return None
    start += len(marker)
    tail = prompt[start:].lstrip()
    line = tail.splitlines()[0].strip() if tail else ""
    return line or None


def treatment_well(meta: Mapping[str, Any]) -> Optional[str]:
    # A dose is not an experimental well.  Missing well identity is a hard provenance failure.
    for key in ("sample_id", "sample", "treatment_sample", "well", "well_id"):
        value = meta.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _train_drugs(path: str) -> Tuple[set, int]:
    drugs = set()
    n = 0
    with open(path, "rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_no}")
            try:
                ex = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid training JSONL at {path}:{line_no}: {exc}") from exc
            metadata = ex.get("metadata")
            if (not isinstance(metadata, dict) or "drug" not in metadata
                    or metadata["drug"] is None or not str(metadata["drug"]).strip()):
                raise ValueError(
                    f"training JSONL row lacks nonempty metadata.drug at {path}:{line_no}")
            drugs.add(str(metadata["drug"]))
            n += 1
    return drugs, n


def _read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    hashes: List[str] = []
    with open(path, "rb") as handle:
        for line_no, raw in enumerate(handle):
            if not raw.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_no + 1}; source indices ambiguous")
            try:
                records.append(json.loads(raw.decode("utf-8")))
            except Exception as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no + 1}: {exc}") from exc
            hashes.append(sha256_bytes(raw.rstrip(b"\r\n")))
    return records, hashes


def _source_identity(ex: Mapping[str, Any]) -> Tuple[Any, ...]:
    meta = ex.get("metadata", {}) or {}
    return (
        meta.get("drug"), meta.get("cell_line_id"), meta.get("plate"), treatment_well(meta),
        str(meta.get("dose", meta.get("dose_float"))), sha256_text(ex.get("response", "")),
        sha256_text(control_from_prompt(ex.get("prompt", "")) or ""),
    )


def _masked_condition_prompt(prompt: str, drug: str) -> str:
    lines = prompt.splitlines()
    instruction = next((i for i, line in enumerate(lines)
                        if line.startswith("Predict the response of")), None)
    if instruction is None or drug not in lines[instruction]:
        raise ValueError("instruction line or exact drug span missing")
    line = lines[instruction].replace(drug, "<DRUG>", 1)
    line, n = re.subn(r"(Mechanism:\s*).*$", r"\1<MOA>.", line, count=1)
    if n != 1:
        raise ValueError("Mechanism field missing from instruction line")
    lines[instruction] = line
    return "\n".join(lines)


def _scramble_index(records: Sequence[Mapping[str, Any]], hashes: Sequence[str]):
    out: Dict[Tuple[Any, ...], deque] = defaultdict(deque)
    for line_index, (ex, line_hash) in enumerate(zip(records, hashes)):
        out[_source_identity(ex)].append((line_index, ex, line_hash))
    return out


def _scramble_record(original: Mapping[str, Any], original_line_index: int,
                     index: Mapping[Tuple[Any, ...], deque]) -> Optional[Dict[str, Any]]:
    queue = index.get(_source_identity(original))
    if not queue:
        return None
    line_index, wrong, line_hash = queue.popleft()
    om = original.get("metadata", {}) or {}
    wm = wrong.get("metadata", {}) or {}
    partner = wm.get("scrambled_to_drug")
    if not partner or wm.get("scrambled_from_drug") not in (None, om.get("drug")):
        raise ValueError(f"scramble metadata mismatch for source line {original_line_index}")
    if wrong.get("response") != original.get("response"):
        raise ValueError(f"scramble truth changed for source line {original_line_index}")
    if control_from_prompt(wrong.get("prompt", "")) != control_from_prompt(original.get("prompt", "")):
        raise ValueError(f"scramble control changed for source line {original_line_index}")
    if _masked_condition_prompt(original["prompt"], str(om.get("drug"))) != \
            _masked_condition_prompt(wrong["prompt"], str(partner)):
        raise ValueError(f"scramble changed non-condition prompt bytes at source line {original_line_index}")
    return {
        "source_line_index": line_index,
        "source_line_sha256": line_hash,
        "prompt": wrong["prompt"],
        "prompt_id": sha256_text(wrong["prompt"]),
        "prompt_sha256": sha256_text(wrong["prompt"]),
        "partner_drug": partner,
        "provenance": "existing_scramble_file",
        "label": "wrong-condition sensitivity",
        "replaced_fields": ["drug", "moa"],
        "non_condition_bytes_verified": True,
    }


def build_manifest(eval_file: str, *, tier: str = "tier2_unseen_drugs",
                   train_file: str, scramble_file: Optional[str] = None, k_samples: int = 8,
                   min_cells: int = 8, min_drugs_per_group: int = 3,
                   max_groups: int = 80, same_plate_only: bool = True, seed: int = 42,
                   contract_version: str = DEFAULT_CONTRACT_VERSION,
                   model_fingerprints: Sequence[str] = (),
                   model_declarations: Sequence[Mapping[str, str]] = (),
                   expected_support: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
    if not train_file:
        raise ValueError("train_file is required to prove Tier-2 drug disjointness")
    examples, line_hashes = _read_jsonl(eval_file)
    train_drugs, train_line_count = _train_drugs(train_file)
    scramble_idx = None
    scramble_hash = None
    if scramble_file:
        scram, scram_hashes = _read_jsonl(scramble_file)
        scramble_idx = _scramble_index(scram, scram_hashes)
        scramble_hash = file_sha256(scramble_file)

    groups: Dict[Tuple[str, Optional[str]], Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list))
    for line_index, ex in enumerate(examples):
        meta = ex.get("metadata", {}) or {}
        cell_line, drug, plate = meta.get("cell_line_id"), meta.get("drug"), meta.get("plate")
        if cell_line is None or drug is None or not control_from_prompt(ex.get("prompt", "")):
            continue
        if same_plate_only and plate is None:
            continue
        groups[(str(cell_line), str(plate) if same_plate_only else None)][str(drug)].append(line_index)

    eval_drugs = {drug for by_drug in groups.values() for drug in by_drug}
    overlap = sorted(eval_drugs & train_drugs)
    if tier == "tier2_unseen_drugs" and overlap:
        raise ValueError(f"Tier-2 leakage: {len(overlap)} evaluation drugs occur in training: "
                         f"{overlap[:10]}")

    rng = np.random.RandomState(seed)
    rows: List[Dict[str, Any]] = []
    used_groups = 0
    for (cell_line, plate), by_drug in groups.items():
        eligible = {drug: indices for drug, indices in by_drug.items()
                    if len(indices) >= min_cells}
        if len(eligible) < min_drugs_per_group:
            continue
        group_id = sha256_json({"tier": tier, "cell_line": cell_line, "plate": plate})
        for drug, indices in eligible.items():
            if len(indices) < 4:
                continue
            shuffled = list(indices)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            a_indices, b_indices = shuffled[:half], shuffled[half:]
            selected = indices[:k_samples]
            prompt_specs = []
            for draw_id, line_index in enumerate(selected):
                ex = examples[line_index]
                prompt = ex["prompt"]
                prompt_id = sha256_text(prompt)
                spec = {
                    "draw_id": draw_id,
                    "source_line_index": line_index,
                    "source_line_sha256": line_hashes[line_index],
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_id,
                    "control_sha256": sha256_text(control_from_prompt(prompt) or ""),
                    "seeds_by_model": {},
                }
                if scramble_idx is not None:
                    wrong = _scramble_record(ex, line_index, scramble_idx)
                    if wrong is None:
                        raise ValueError(f"no existing scramble partner for source line {line_index}")
                    spec["wrong_condition"] = wrong
                prompt_specs.append(spec)
            meta_values = [examples[i].get("metadata", {}) or {} for i in indices]
            missing_well = [indices[j] for j, m in enumerate(meta_values) if treatment_well(m) is None]
            if missing_well:
                raise ValueError(f"{tier}/{cell_line}/{plate}/{drug} lacks a genuine sample/well ID "
                                 f"on source lines {missing_well[:10]}")
            wells = sorted({treatment_well(m) for m in meta_values})
            doses = sorted({str(m.get("dose", m.get("dose_float"))) for m in meta_values})
            if len(wells) != 1:
                raise ValueError(f"{tier}/{cell_line}/{plate}/{drug} maps to {len(wells)} treatment "
                                 f"wells ({wells}); a unique inference cluster cannot be frozen")
            if len(doses) != 1:
                raise ValueError(f"{tier}/{cell_line}/{plate}/{drug} maps to multiple doses {doses}")
            row_core = {
                "tier": tier, "group_id": group_id, "cell_line": cell_line, "plate": plate,
                "drug": drug, "dose_values": doses, "treatment_well_ids": wells,
                "source_line_indices": indices, "truth_half_a_indices": a_indices,
                "truth_half_b_indices": b_indices,
            }
            row_id = sha256_json(row_core)
            for spec in prompt_specs:
                for fingerprint in model_fingerprints:
                    spec["seeds_by_model"][fingerprint] = stable_output_seed(
                        fingerprint, row_id, spec["prompt_id"], spec["draw_id"], contract_version)
                    wrong = spec.get("wrong_condition")
                    if wrong:
                        wrong.setdefault("seeds_by_model", {})[fingerprint] = stable_output_seed(
                            fingerprint, row_id, wrong["prompt_id"], spec["draw_id"], contract_version)
            rows.append({
                **row_core,
                "row_id": row_id,
                "source_lines": [{"index": i, "sha256": line_hashes[i]} for i in indices],
                "truth_half_a": [{"index": i, "response_sha256": sha256_text(examples[i]["response"])}
                                 for i in a_indices],
                "truth_half_b": [{"index": i, "response_sha256": sha256_text(examples[i]["response"])}
                                 for i in b_indices],
                "prompt_specs": prompt_specs,
            })
        used_groups += 1
        if used_groups >= max_groups:
            break

    support = {
        "rows": len(rows),
        "drugs": len({r["drug"] for r in rows}),
        "cell_lines": len({r["cell_line"] for r in rows}),
        "groups": len({r["group_id"] for r in rows}),
    }
    if expected_support:
        mismatches = {key: (expected_support[key], support.get(key)) for key in expected_support
                      if expected_support[key] is not None and support.get(key) != expected_support[key]}
        if mismatches:
            raise ValueError(f"frozen support differs from preregistration: {mismatches}")

    declarations = [dict(x) for x in model_declarations]
    declared_fingerprints = {x.get("fingerprint") for x in declarations}
    for fingerprint in model_fingerprints:
        existing = next((x for x in declarations if x.get("fingerprint") == fingerprint), None)
        if existing is not None and existing.get("role") != "scored":
            raise ValueError(f"fingerprint {fingerprint} cannot be both scored and validity-only")
        if fingerprint not in declared_fingerprints:
            declarations.append({"model_id": fingerprint, "fingerprint": fingerprint,
                                 "role": "scored"})
            declared_fingerprints.add(fingerprint)
    if len({x.get("fingerprint") for x in declarations}) != len(declarations):
        raise ValueError("model declaration fingerprints must be unique")
    for declaration in declarations:
        if not declaration.get("model_id") or not declaration.get("fingerprint"):
            raise ValueError("every model declaration requires model_id and fingerprint")
        if declaration.get("role") not in ("scored", "validity_only_parent"):
            raise ValueError("model role must be scored or validity_only_parent")

    body = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "creator": {"path": os.path.abspath(__file__), "sha256": file_sha256(__file__)},
        "generation_contract_version": contract_version,
        "seed_derivation": "SHA256(model fingerprint, row id, prompt id, draw id, contract version)",
        "config": {
            "tier": tier, "k_samples": k_samples, "min_cells": min_cells,
            "min_drugs_per_group": min_drugs_per_group, "max_groups": max_groups,
            "same_plate_only": same_plate_only, "seed": seed,
            "wrong_condition_label": "wrong-condition sensitivity",
        },
        "sources": {
            "training": {"path": os.path.abspath(train_file), "sha256": file_sha256(train_file),
                         "line_count": train_line_count,
                         "drug_count": len(train_drugs)},
            "evaluation": {"path": os.path.abspath(eval_file), "sha256": file_sha256(eval_file),
                           "line_count": len(examples)},
            "scramble": ({"path": os.path.abspath(scramble_file), "sha256": scramble_hash}
                         if scramble_file else None),
        },
        "tier2_disjointness": {"checked": tier == "tier2_unseen_drugs",
                               "evaluation_drugs_in_training": overlap},
        "model_fingerprints": list(model_fingerprints),
        "model_declarations": declarations,
        "support": support,
        "expected_support": dict(expected_support or {}),
        "rows": rows,
    }
    body["manifest_sha256"] = sha256_json(body)
    return body


def verify_manifest(manifest: Mapping[str, Any], eval_file: str,
                    scramble_file: Optional[str] = None,
                    train_file: Optional[str] = None) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema {manifest.get('schema_version')}")
    expected = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    if expected != sha256_json(body):
        raise ValueError("manifest content hash mismatch")
    frozen_train = manifest["sources"].get("training")
    if frozen_train:
        if not train_file:
            raise ValueError("train_file is required to verify frozen Tier-2 disjointness")
        if frozen_train["sha256"] != file_sha256(train_file):
            raise ValueError("training JSONL hash differs from frozen manifest")
        if manifest.get("config", {}).get("tier") == "tier2_unseen_drugs":
            train_drugs, _ = _train_drugs(train_file)
            eval_records, _ = _read_jsonl(eval_file)
            eval_drugs = {str((row.get("metadata") or {}).get("drug")) for row in eval_records
                          if (row.get("metadata") or {}).get("drug") is not None}
            overlap = sorted(train_drugs & eval_drugs)
            if overlap:
                raise ValueError(f"Tier-2 leakage recheck failed: {overlap[:10]}")
    if manifest["sources"]["evaluation"]["sha256"] != file_sha256(eval_file):
        raise ValueError("evaluation JSONL hash differs from frozen manifest")
    frozen_scramble = manifest["sources"].get("scramble")
    if bool(frozen_scramble) != bool(scramble_file):
        raise ValueError("scramble source presence differs from frozen manifest")
    if scramble_file and frozen_scramble["sha256"] != file_sha256(scramble_file):
        raise ValueError("scramble JSONL hash differs from frozen manifest")
    rows = manifest.get("rows", [])
    support = {"rows": len(rows), "drugs": len({r["drug"] for r in rows}),
               "cell_lines": len({r["cell_line"] for r in rows}),
               "groups": len({r["group_id"] for r in rows})}
    if support != manifest.get("support"):
        raise ValueError(f"manifest support summary mismatch: {support} != {manifest.get('support')}")
    for key, expected_value in manifest.get("expected_support", {}).items():
        if expected_value is not None and support.get(key) != expected_value:
            raise ValueError(f"manifest expected support {key}={expected_value}, observed {support.get(key)}")


class PredictionCache:
    """Append-only JSONL cache with schema validation and a lifetime writer lock."""

    def __init__(self, path: str, contract: Mapping[str, Any]):
        self.path = os.path.abspath(path)
        self.lock_path = self.path + ".lock"
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"prediction cache already has a writer: {self.lock_path}") from exc
        os.write(self._lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(self._lock_fd)
        self._closed = False
        atexit.register(self.close)
        self.contract = dict(contract)
        self.contract_hash = sha256_json(self.contract)
        self.records: Dict[str, Dict[str, Any]] = {}
        try:
            self._load_or_create()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_record(item: Mapping[str, Any], line_no: Optional[int] = None) -> None:
        required = {
            "type": str, "output_id": str, "row_id": str, "prompt_id": str,
            "prompt_sha256": str, "draw_id": int, "seed": int,
            "raw_token_ids": list, "decoded_text": str,
            "termination_reason": str, "validity": dict,
        }
        where = f" at line {line_no}" if line_no is not None else ""
        for key, typ in required.items():
            if key not in item or not isinstance(item[key], typ):
                raise ValueError(f"prediction-cache field {key!r} must be {typ.__name__}{where}")
        if item["type"] != "prediction" or not all(isinstance(x, int) for x in item["raw_token_ids"]):
            raise ValueError(f"invalid prediction-cache record schema{where}")

    def _load_or_create(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8", newline="") as handle:
                header = None
                lines = handle.readlines()
                for line_no, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"corrupt prediction cache at line {line_no}; refusing append") from exc
                    if header is None:
                        header = item
                        if (item.get("type") != "header" or item.get("schema_version") != 2 or
                                item.get("contract_hash") != self.contract_hash):
                            raise ValueError("prediction-cache contract mismatch")
                        continue
                    output_id = item.get("output_id")
                    self._validate_record(item, line_no)
                    if output_id in self.records and self.records[output_id] != item:
                        raise ValueError(f"conflicting duplicate cache output {output_id}")
                    self.records[output_id] = item
                if header is None:
                    raise ValueError("empty prediction cache has no contract header")
        else:
            with open(self.path, "x", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json({"type": "header", "schema_version": 2,
                                             "contract": self.contract,
                                             "contract_hash": self.contract_hash}) + "\n")

    def get(self, output_id: str) -> Optional[Dict[str, Any]]:
        return self.records.get(output_id)

    def put(self, record: Mapping[str, Any]) -> None:
        item = dict(record)
        self._validate_record(item)
        output_id = item.get("output_id")
        if not output_id:
            raise ValueError("prediction-cache record requires output_id")
        old = self.records.get(output_id)
        if old is not None:
            if old != item:
                raise ValueError(f"attempted to overwrite cached output {output_id}")
            return
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(item) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[output_id] = item

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            os.close(self._lock_fd)
        finally:
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--scram_dir")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_samples", type=int, default=8)
    ap.add_argument("--min_cells", type=int, default=8)
    ap.add_argument("--min_drugs_per_group", type=int, default=3)
    ap.add_argument("--max_groups", type=int, default=80)
    ap.add_argument("--same_plate_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generation_contract_version", default=DEFAULT_CONTRACT_VERSION)
    ap.add_argument("--model_fingerprint", action="append", default=[])
    ap.add_argument("--validity_only_parent", action="append", default=[],
                    metavar="MODEL_ID=FINGERPRINT")
    ap.add_argument("--expected_rows", type=int, default=606)
    ap.add_argument("--expected_drugs", type=int, default=35)
    ap.add_argument("--expected_cell_lines", type=int, default=40)
    ap.add_argument("--expected_groups", type=int, default=80)
    ap.add_argument("--hash_named", action="store_true",
                    help="publish to <stem>-<manifest hash>.json; never overwrite")
    args = ap.parse_args()
    eval_file = os.path.join(args.eval_dir, f"eval_{args.tier}.jsonl")
    scramble_file = (os.path.join(args.scram_dir, f"eval_{args.tier}.jsonl")
                     if args.scram_dir else None)
    parent_declarations = []
    for value in args.validity_only_parent:
        if "=" not in value:
            ap.error("--validity_only_parent requires MODEL_ID=FINGERPRINT")
        model_id, fingerprint = value.split("=", 1)
        parent_declarations.append({"model_id": model_id, "fingerprint": fingerprint,
                                    "role": "validity_only_parent"})
    expected_support = {"rows": args.expected_rows, "drugs": args.expected_drugs,
                        "cell_lines": args.expected_cell_lines, "groups": args.expected_groups}
    manifest = build_manifest(
        eval_file, tier=args.tier, train_file=args.train_file,
        scramble_file=scramble_file, k_samples=args.k_samples,
        min_cells=args.min_cells, min_drugs_per_group=args.min_drugs_per_group,
        max_groups=args.max_groups, same_plate_only=args.same_plate_only, seed=args.seed,
        contract_version=args.generation_contract_version,
        model_fingerprints=args.model_fingerprint,
        model_declarations=parent_declarations, expected_support=expected_support)
    out = args.out
    if args.hash_named:
        p = Path(out)
        out = str(p.with_name(f"{p.stem}-{manifest['manifest_sha256']}{p.suffix or '.json'}"))
    atomic_write_json(out, manifest, overwrite=False)
    print(f"wrote {out}: {len(manifest['rows'])} rows, hash {manifest['manifest_sha256']}")


if __name__ == "__main__":
    main()
