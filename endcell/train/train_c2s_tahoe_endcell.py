"""
C2S-Scale SFT Training on Tahoe-100M Perturbation Data

Fine-tunes the pretrained C2S-Scale-Pythia-1b-pt model on drug perturbation
prediction pairs constructed by tahoe_c2s_preprocess.py.

Designed for:
  - 40GB MIG slice (A100-80GB with MIG enabled)
  - ~3100 token sequences (1500 gene control + 1500 gene response + prompt)
  - bf16 + gradient checkpointing

Usage:
    # Local test (CPU, tiny data, 2 steps)
    python train_c2s_tahoe.py --mode test \
        --train_file ./tahoe_c2s_data/train.jsonl \
        --eval_file ./tahoe_c2s_data/eval_tier1_seen_conditions.jsonl

    # HPC full run
    python train_c2s_tahoe.py --mode full \
        --train_file ./data/train.jsonl \
        --eval_file ./data/eval_tier1_seen_conditions.jsonl \
        --output_dir ./checkpoints \
        --num_epochs 1 --batch_size 1 --grad_accum 16 \
        --bf16 --gradient_checkpointing --max_length 4096 \
        --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import logging
import math
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINELS = ("[END_CELL]", "[DOWN]")
GEMMA_BASE_VOCAB_SIZE = 256_000
GEMMA_TOKENIZER_LENGTH = GEMMA_BASE_VOCAB_SIZE + len(SENTINELS)
GEMMA_SENTINEL_IDS = dict(zip(SENTINELS, range(GEMMA_BASE_VOCAB_SIZE,
                                                GEMMA_TOKENIZER_LENGTH)))
GEMMA_PROTOBUF_VERSION = "5.29.5"
GEMMA_PROTOBUF_ROOT = Path(
    "/data/BuffaF-Projetcs/florian_c2s/test_deps/protobuf-5.29.5")
TRAINING_STATE = "training_state.pt"
PROVENANCE_FILE = "run_provenance.json"
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_SCHEMA = 2


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _stable_json_hash(value):
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=str).encode("utf-8"))


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _tokenizer_contract(tokenizer, include_implementation=True):
    """Return a stable tokenizer contract.

    The semantic contract deliberately excludes the Python slow/fast implementation class:
    Hugging Face may reload an equivalent serialization through the other implementation.  The
    full training fingerprint retains those details so an actual resumed run cannot silently
    change tokenizer implementation.
    """
    added = getattr(tokenizer, "get_added_vocab", lambda: {})()
    contract = {
        "length": len(tokenizer),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_map": tokenizer.special_tokens_map,
        "added_vocab": dict(sorted(added.items())),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }
    if include_implementation:
        contract["class"] = tokenizer.__class__.__name__
        contract["is_fast"] = getattr(tokenizer, "is_fast", None)
    return contract


def tokenizer_fingerprint(tokenizer):
    """Hash the exact runtime tokenizer contract used for resumable training."""
    return _stable_json_hash(_tokenizer_contract(tokenizer, include_implementation=True))


def semantic_tokenizer_fingerprint(tokenizer):
    """Hash token semantics while allowing an equivalent slow/fast implementation reload."""
    contract = _tokenizer_contract(tokenizer, include_implementation=False)
    return _stable_json_hash(contract)


def protobuf_tree_digest(root=GEMMA_PROTOBUF_ROOT, version=GEMMA_PROTOBUF_VERSION):
    """Digest every installed Protobuf payload named by dist-info/RECORD."""
    root = Path(root).resolve()
    record = root / f"protobuf-{version}.dist-info" / "RECORD"
    if not record.is_file():
        raise RuntimeError(f"protobuf installation RECORD is missing: {record}")
    inventory = []
    seen = set()
    with record.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if not row or not row[0]:
            raise RuntimeError("protobuf RECORD contains an empty path")
        raw = row[0]
        relative = PurePosixPath(raw)
        if (relative.is_absolute() or "\\" in raw or
                relative.as_posix() != raw or ".." in relative.parts):
            raise RuntimeError(f"unsafe protobuf RECORD path: {raw!r}")
        candidate = root / Path(*relative.parts)
        if candidate.is_symlink():
            raise RuntimeError(f"protobuf RECORD payload is a symlink: {raw!r}")
        try:
            path = candidate.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"protobuf RECORD payload is missing: {raw!r}: {exc}") from exc
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"protobuf RECORD path escapes isolated root: {raw!r}") from exc
        normalized = relative.as_posix()
        if normalized in seen:
            raise RuntimeError(f"duplicate protobuf RECORD payload: {normalized}")
        seen.add(normalized)
        if not path.is_file():
            raise RuntimeError(f"protobuf RECORD payload is not a file: {path}")
        inventory.append({
            "path": normalized,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not inventory:
        raise RuntimeError("protobuf digest inventory is empty")
    if "google/_upb/_message.abi3.so" not in seen:
        raise RuntimeError("protobuf RECORD omits google/_upb/_message.abi3.so")
    return _stable_json_hash(inventory)


def require_gemma_protobuf_runtime():
    """Fail before AutoTokenizer can silently fall back from SentencePiece.

    Transformers 5.12.1 can construct a five-token tokenizer when the Python protobuf package is
    absent even though Gemma's ``tokenizer.model`` contains 256,000 pieces.  Cluster launchers also
    pin and verify the isolated package path; this import check protects direct trainer/probe use.
    """
    try:
        module = importlib.import_module("google.protobuf")
    except Exception as exc:
        raise RuntimeError(
            "Gemma requires Python protobuf to load tokenizer.model; refusing any tiktoken "
            "fallback or collapsed tokenizer") from exc
    observed = getattr(module, "__version__", None)
    if observed != GEMMA_PROTOBUF_VERSION:
        raise RuntimeError(
            f"Gemma requires Python protobuf {GEMMA_PROTOBUF_VERSION}; observed {observed!r}")
    # PYTHONPATH is intentionally bound to the canonical lexical path exported by
    # gemma2_standard_protobuf_env.sh.  On the cluster that path can resolve through
    # a BeeGFS alias (for example /data -> /mnt/beegfsnew), so resolving it before
    # the string comparison would reject the helper's own valid environment.
    expected_pythonpath = str(GEMMA_PROTOBUF_ROOT)
    resolved_root = GEMMA_PROTOBUF_ROOT.resolve()
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Gemma protobuf resolved outside isolated root: {module_path}") from exc
    observed_pythonpath = os.environ.get("PYTHONPATH")
    if observed_pythonpath != expected_pythonpath:
        raise RuntimeError(
            "Gemma requires PYTHONPATH to equal the isolated Protobuf directory: "
            f"observed={observed_pythonpath!r}, expected={expected_pythonpath!r}")
    if "PYTHONHOME" in os.environ:
        raise RuntimeError("Gemma requires PYTHONHOME to be unset")
    observed_digest = protobuf_tree_digest(resolved_root)
    expected_digest = os.environ.get("GEMMA_PROTOBUF_TREE_SHA256")
    if expected_digest != observed_digest:
        raise RuntimeError(
            "Gemma Protobuf payload is not helper-bound or its content digest changed")
    return {"version": observed, "path": str(module_path),
            "tree_sha256": observed_digest}


def gemma_mode_from_config(config, *, prepend_bos, strict_token_contract):
    """Select Gemma safeguards from model identity, never from user-provided flags."""
    is_gemma = getattr(config, "model_type", None) == "gemma2"
    if is_gemma and not (prepend_bos and strict_token_contract):
        raise ValueError(
            "Gemma-2 requires both --prepend_bos and --strict_token_contract")
    return is_gemma


def validate_gemma_base_tokenizer_contract(tokenizer, model_config_vocab_size):
    """Validate the pinned Gemma tokenizer before Tahoe sentinels are registered."""
    observed_vocab = int(getattr(tokenizer, "vocab_size", -1))
    observed_length = len(tokenizer)
    config_vocab = int(model_config_vocab_size)
    if config_vocab != GEMMA_BASE_VOCAB_SIZE:
        raise ValueError(
            f"pinned Gemma config vocab_size={config_vocab}, expected "
            f"{GEMMA_BASE_VOCAB_SIZE}")
    if observed_vocab != config_vocab or observed_length != config_vocab:
        raise ValueError(
            "collapsed or mismatched Gemma tokenizer before sentinels: "
            f"tokenizer.vocab_size={observed_vocab}, len(tokenizer)={observed_length}, "
            f"model config vocab_size={config_vocab}")
    sanity = tokenizer.encode("TP53 GAPDH EGFR BRCA1", add_special_tokens=False)
    unknowns = (sum(token_id == tokenizer.unk_token_id for token_id in sanity)
                if tokenizer.unk_token_id is not None else 0)
    if len(sanity) < 4 or unknowns:
        raise ValueError(
            "Gemma tokenizer failed the noncollapsed biological-text sanity check: "
            f"tokens={len(sanity)}, unknowns={unknowns}")
    return {
        "vocab_size": observed_vocab,
        "length": observed_length,
        "sanity_tokens": len(sanity),
        "sanity_unknowns": unknowns,
    }


def validate_gemma_registered_tokenizer_contract(tokenizer, sentinel_result):
    """Validate the exact post-registration Gemma/Tahoe vocabulary contract."""
    observed_vocab = int(getattr(tokenizer, "vocab_size", -1))
    observed_length = len(tokenizer)
    observed_ids = {token: int(token_id)
                    for token, token_id in sentinel_result["ids"].items()}
    if observed_vocab != GEMMA_BASE_VOCAB_SIZE:
        raise ValueError(
            f"Gemma base vocabulary changed after sentinels: {observed_vocab} != "
            f"{GEMMA_BASE_VOCAB_SIZE}")
    if observed_length != GEMMA_TOKENIZER_LENGTH:
        raise ValueError(
            f"Gemma tokenizer length {observed_length} != {GEMMA_TOKENIZER_LENGTH}")
    if observed_ids != GEMMA_SENTINEL_IDS:
        raise ValueError(
            f"Gemma sentinel ids {observed_ids} != {GEMMA_SENTINEL_IDS}")
    return {"vocab_size": observed_vocab, "length": observed_length,
            "sentinel_ids": observed_ids}


def validate_model_embedding_rows_before_resize(model, tokenizer, *, strict_gemma,
                                                resume, base_tokenizer_length):
    """Prove a resize cannot shrink the pinned Gemma embedding table.

    Cold starts must present the untouched 256,000-row parent before adding two rows. Resumes must
    already contain the fully resized 256,002-row table. Legacy Pythia retains its historical
    resize behavior because ``strict_gemma`` is false there.
    """
    rows = int(model.get_input_embeddings().weight.shape[0])
    if not strict_gemma:
        return {"embedding_rows": rows, "strict_gemma": False}
    config_vocab = int(getattr(model.config, "vocab_size", -1))
    if resume:
        if rows != GEMMA_TOKENIZER_LENGTH or config_vocab != GEMMA_TOKENIZER_LENGTH:
            raise ValueError(
                "Gemma resume checkpoint is not already resized: "
                f"embedding_rows={rows}, config.vocab_size={config_vocab}, "
                f"expected={GEMMA_TOKENIZER_LENGTH}")
        if len(tokenizer) != rows:
            raise ValueError(
                f"Gemma resume tokenizer/model mismatch: len(tokenizer)={len(tokenizer)}, "
                f"embedding_rows={rows}")
    else:
        if base_tokenizer_length != GEMMA_BASE_VOCAB_SIZE:
            raise ValueError(
                "refusing Gemma resize from a collapsed/mismatched tokenizer: "
                f"base tokenizer length={base_tokenizer_length}, expected "
                f"{GEMMA_BASE_VOCAB_SIZE}")
        if rows != GEMMA_BASE_VOCAB_SIZE or config_vocab != GEMMA_BASE_VOCAB_SIZE:
            raise ValueError(
                "pinned Gemma parent embedding/config mismatch before resize: "
                f"embedding_rows={rows}, config.vocab_size={config_vocab}, expected "
                f"{GEMMA_BASE_VOCAB_SIZE}")
    return {"embedding_rows": rows, "config_vocab_size": config_vocab,
            "strict_gemma": True, "resume": bool(resume)}


def _canonical_model_config(config):
    value = dict(config.to_dict())
    for volatile in ("_name_or_path", "_commit_hash", "transformers_version"):
        value.pop(volatile, None)
    return value


def model_fingerprint(model, requested_source, requested_revision):
    """Fingerprint model provenance and parameter structure without re-reading multi-GB weights."""
    structure = [(n, tuple(p.shape), str(p.dtype), bool(p.requires_grad))
                 for n, p in model.named_parameters()]
    return _stable_json_hash({
        "requested_source": requested_source,
        "requested_revision": requested_revision,
        "architecture": model.__class__.__name__,
        "config": _canonical_model_config(model.config),
        "parameters": structure,
    })


def register_sentinels(tokenizer, sentinels=SENTINELS, strict=False,
                       reload_use_fast=None):
    """Register atomic sentinels and optionally prove the contract survives serialization.

    Hugging Face documents that added special tokens are never split and that model embeddings
    must subsequently be resized to ``len(tokenizer)``.  This helper validates the stronger
    contract required by the Tahoe formats; resizing remains the caller's responsibility because
    the tokenizer is loaded before the model.
    """
    preexisting_special_ids = set(getattr(tokenizer, "all_special_ids", []))
    # A resumed tokenizer legitimately already carries Tahoe's sentinels. Exclude those exact
    # registered tokens from the set of *other* special ids against which collisions are checked.
    registered = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    for token in sentinels:
        if token in registered:
            token_id = tokenizer.convert_tokens_to_ids(token)
            preexisting_special_ids.discard(token_id)
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    requested = existing + [token for token in sentinels if token not in existing]
    added = tokenizer.add_special_tokens({"additional_special_tokens": requested})

    def inspect(tok):
        ids = {}
        for token in sentinels:
            encoded = tok.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"{token} is not atomic with add_special_tokens=False: {encoded}")
            token_id = encoded[0]
            if tok.unk_token_id is not None and token_id == tok.unk_token_id:
                raise ValueError(f"{token} maps to unk_token_id={tok.unk_token_id}")
            declared_id = tok.convert_tokens_to_ids(token)
            if declared_id != token_id:
                raise ValueError(f"{token} encode id {token_id} != declared id {declared_id}")
            ids[token] = token_id
        if len(set(ids.values())) != len(ids):
            raise ValueError(f"sentinels do not have distinct ids: {ids}")
        if strict and set(ids.values()) & preexisting_special_ids:
            raise ValueError("a sentinel reused a pre-existing special-token id: "
                             f"{set(ids.values()) & preexisting_special_ids}")
        return ids

    ids = inspect(tokenizer)
    if strict:
        with tempfile.TemporaryDirectory(prefix="tahoe-tokenizer-") as tmp:
            tokenizer.save_pretrained(tmp)
            if reload_use_fast is None and hasattr(tokenizer, "is_fast"):
                reload_use_fast = bool(tokenizer.is_fast)
            reload_kwargs = ({"use_fast": reload_use_fast}
                             if reload_use_fast is not None else {})
            reloaded = AutoTokenizer.from_pretrained(tmp, **reload_kwargs)
            reload_ids = inspect(reloaded)
            if reload_ids != ids:
                raise ValueError(f"sentinel ids changed after save/reload: {ids} -> {reload_ids}")
            before = semantic_tokenizer_fingerprint(tokenizer)
            after = semantic_tokenizer_fingerprint(reloaded)
            if after != before:
                raise ValueError("semantic tokenizer contract changed after save/reload")
    return {"added": added, "ids": ids}


def validate_training_token_contract(tokenizer, prepend_bos=False,
                                     require_distinct_pad_eos=False):
    if prepend_bos and tokenizer.bos_token_id is None:
        raise ValueError("exactly-one-BOS construction requires bos_token_id")
    if require_distinct_pad_eos:
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("strict Gemma contract requires native PAD and EOS ids")
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise ValueError("strict Gemma contract requires PAD distinct from EOS")


class DeterministicEpochSampler(Sampler):
    """Epoch-local shuffle whose order can be reconstructed without global RNG state."""
    def __init__(self, data_source, seed, epoch=0, start_index=0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(len(data_source), generator=generator).tolist()
        if not 0 <= self.start_index <= len(self.order):
            raise ValueError(f"invalid sampler start_index={self.start_index}")

    def __iter__(self):
        return iter(self.order[self.start_index:])

    def __len__(self):
        return len(self.order) - self.start_index


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        import numpy as np
        np.random.set_state(state["numpy"])


def _validate_resume_fingerprints(saved, current):
    mismatches = {k: (saved.get(k), current.get(k)) for k in sorted(set(saved) | set(current))
                  if saved.get(k) != current.get(k)}
    if mismatches:
        detail = "; ".join(f"{k}: saved={a!r}, current={b!r}"
                           for k, (a, b) in mismatches.items())
        raise RuntimeError(f"REFUSING TO RESUME: provenance fingerprint mismatch: {detail}")


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location=map_location)


def _checkpoint_identity(training_state, provenance):
    """Small immutable identity for a published checkpoint directory."""
    return {
        "schema_version": int(training_state.get("schema_version", -1)),
        "global_step": int(training_state.get("global_step", -1)),
        "epoch": int(training_state.get("epoch", -1)),
        "microbatch_position": int(training_state.get("microbatch_position", -1)),
        "accumulation_position": int(training_state.get("accumulation_position", -1)),
        "completed": bool(training_state.get("completed", False)),
        "contract_fingerprint": training_state.get("contract_fingerprint"),
        "provenance_contract_fingerprint": provenance.get("contract_fingerprint"),
    }


def _published_file_records(path):
    """Return content-addressed evidence for every published file except the manifest."""
    started = time.monotonic()
    files = {}
    for root, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            if rel != CHECKPOINT_MANIFEST:
                files[rel] = {
                    "size": os.path.getsize(full),
                    "sha256": sha256_file(full),
                }
    return files, time.monotonic() - started


_CHECKPOINT_DIR_RE = re.compile(r"checkpoint-(\d+)(?:-mb(\d+))?")


def _checkpoint_dir_name(global_step, microbatch_position):
    """Name resumable checkpoints by both optimizer and consumed-microbatch position."""
    if global_step < 0 or microbatch_position < 0:
        raise ValueError("checkpoint positions must be non-negative")
    return f"checkpoint-{int(global_step)}-mb{int(microbatch_position)}"


def _parse_checkpoint_dir_name(name):
    match = _CHECKPOINT_DIR_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1)), (None if match.group(2) is None else int(match.group(2)))


def _checkpoint_dir(output_dir, global_step, microbatch_position):
    return os.path.join(output_dir, _checkpoint_dir_name(global_step, microbatch_position))


def _validate_published_checkpoint(path, expected_identity=None):
    """Validate a fully published immutable checkpoint without accepting temp directories."""
    if not os.path.isdir(path):
        raise RuntimeError(f"checkpoint is not a published directory: {path}")
    manifest_path = os.path.join(path, CHECKPOINT_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"checkpoint lacks {CHECKPOINT_MANIFEST}: {path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported checkpoint manifest in {path}")
    identity = manifest.get("identity")
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError(
            f"REFUSING TO OVERWRITE immutable checkpoint {path}: "
            f"existing identity={identity!r}, requested={expected_identity!r}")
    actual_files, hash_duration = _published_file_records(path)
    if actual_files != manifest.get("files"):
        raise RuntimeError(f"checkpoint file inventory/content mismatch: {path}")
    required = {TRAINING_STATE, PROVENANCE_FILE, "config.json"}
    missing = sorted(required - set(actual_files))
    if missing:
        raise RuntimeError(f"checkpoint is incomplete ({missing}): {path}")
    if not any(name.endswith((".safetensors", ".bin")) for name in actual_files):
        raise RuntimeError(f"checkpoint has no model weight file: {path}")
    state = _torch_load(os.path.join(path, TRAINING_STATE))
    with open(os.path.join(path, PROVENANCE_FILE), encoding="utf-8") as f:
        provenance = json.load(f)
    if _checkpoint_identity(state, provenance) != identity:
        raise RuntimeError(f"checkpoint state disagrees with its manifest: {path}")
    return {"identity": identity, "state": state, "manifest": manifest,
            "hash_duration_seconds": hash_duration}


def _latest_resumable_checkpoint(output_dir):
    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        published_final = _validate_published_checkpoint(final_dir)
        if published_final["state"].get("completed"):
            raise RuntimeError(f"training is already complete: {final_dir}")
        raise RuntimeError(f"published final checkpoint is not marked complete: {final_dir}")
    candidates = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            parsed = _parse_checkpoint_dir_name(name)
            state_path = os.path.join(output_dir, name, TRAINING_STATE)
            if parsed is not None and os.path.isfile(state_path):
                step_from_name, mb_from_name = parsed
                manifest_path = os.path.join(output_dir, name, CHECKPOINT_MANIFEST)
                if not os.path.isfile(manifest_path):
                    raise RuntimeError(f"checkpoint lacks {CHECKPOINT_MANIFEST}: "
                                       f"{os.path.join(output_dir, name)}")
                with open(manifest_path, encoding="utf-8") as f:
                    identity = json.load(f).get("identity", {})
                step = int(identity.get("global_step", -1))
                epoch = int(identity.get("epoch", -1))
                microbatch = int(identity.get("microbatch_position", -1))
                if step != step_from_name or (mb_from_name is not None and
                                               microbatch != mb_from_name):
                    raise RuntimeError(f"checkpoint directory position disagrees with manifest: "
                                       f"{os.path.join(output_dir, name)}")
                candidates.append((step, epoch, microbatch, os.path.join(output_dir, name)))
    if not candidates:
        return None
    latest_step, _, latest_microbatch, latest = max(candidates)
    published = _validate_published_checkpoint(latest)
    if int(published["state"].get("global_step", -1)) != latest_step:
        raise RuntimeError(f"checkpoint directory step disagrees with state: {latest}")
    if int(published["state"].get("microbatch_position", -1)) != latest_microbatch:
        raise RuntimeError(f"checkpoint directory microbatch disagrees with state: {latest}")
    if published["state"].get("completed"):
        raise RuntimeError(f"checkpoint unexpectedly marked complete: {latest}")
    if int(published["state"].get("accumulation_position", -1)) != 0:
        raise RuntimeError(f"checkpoint is not at an optimizer boundary: {latest}")
    return latest


def resolve_resume_checkpoint(value, output_dir):
    if value is None:
        return None
    if value == "auto":
        path = _latest_resumable_checkpoint(output_dir)
        if path is None:
            raise FileNotFoundError(
                f"--resume_from_checkpoint auto requested, but no valid published "
                f"checkpoint-N[-mbM] directory exists in {output_dir}")
        return path
    path = os.path.abspath(value)
    published = _validate_published_checkpoint(path)
    if published["state"].get("completed"):
        raise RuntimeError(f"refusing to resume a completed checkpoint: {path}")
    if int(published["state"].get("accumulation_position", -1)) != 0:
        raise RuntimeError(f"checkpoint is not at an optimizer boundary: {path}")
    return path


@dataclass
class StopRequest:
    requested: bool = False
    signum: int = 0


def install_stop_handlers(stop_request):
    def request_stop(signum, _frame):
        stop_request.requested = True
        stop_request.signum = int(signum)
    for name in ("SIGUSR1", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, request_stop)


def _atomic_write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def _prune_checkpoints(output_dir, keep):
    """Keep only the most recent `keep` resumable checkpoint dirs (best/ and final/
    are never matched, so always preserved). Protects a shared/full filesystem."""
    cks = []
    for d in os.listdir(output_dir):
        parsed = _parse_checkpoint_dir_name(d)
        if parsed is not None:
            step, microbatch = parsed
            cks.append((step, -1 if microbatch is None else microbatch,
                        os.path.join(output_dir, d)))
    cks.sort()
    for _, _, path in cks[:-keep]:
        shutil.rmtree(path, ignore_errors=True)
        logger.info(f"  Pruned old checkpoint {path}")


# =============================================================================
# Dataset
# =============================================================================

class C2SDataset(Dataset):
    """
    Loads JSONL examples with {"prompt": ..., "response": ...} and tokenizes them
    for causal LM training with loss only on response tokens.
    """
    def __init__(self, filepath, tokenizer, max_length=4096, de_weight=1.0,
                 prepend_bos=False, strict_accounting=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prepend_bos = bool(prepend_bos)
        self.strict_accounting = bool(strict_accounting)
        # DE-WEIGHTED SFT: up-weight the token loss on the genes this drug moves differently from the
        # average drug (the "de_genes" field written by build_de_weights.py). Attacks Q15's token-dilution
        # diagnosis without touching the target or the output format, so every prior tier number stays
        # comparable. de_weight == 1.0 is a mathematical no-op -- the weighted mean over supervised
        # tokens with all weights 1 IS the unweighted mean -- which makes it a true control arm.
        self.de_weight = de_weight
        self.examples = []

        logger.info(f"Loading data from {filepath}...")
        with open(filepath) as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
        logger.info(f"  Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        response = ex["response"]

        # Tokenize prompt and response separately to know where to mask
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(" " + response, add_special_tokens=False)
        original_response_ids = list(response_ids)

        if self.prepend_bos:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                raise ValueError("prepend_bos=True but tokenizer has no bos_token_id")
            if bos_id in prompt_ids or bos_id in response_ids:
                raise ValueError(f"row {idx} already contains bos_token_id={bos_id}; "
                                 "refusing to construct more than one BOS")
            prompt_ids = [bos_id] + prompt_ids

        # Combine: prompt + response + eos
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]

        # Truncate from the end if too long
        original_length = len(input_ids)
        prompt_truncated = False
        if len(input_ids) > self.max_length:
            # Keep full prompt, truncate response
            max_response = self.max_length - len(prompt_ids) - 1  # -1 for eos
            if max_response < 50:
                # Prompt itself is too long, truncate prompt too
                input_ids = input_ids[:self.max_length]
                prompt_len = min(len(prompt_ids), self.max_length // 2)
                prompt_truncated = True
            else:
                response_ids = response_ids[:max_response]
                input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
                prompt_len = len(prompt_ids)
        else:
            prompt_len = len(prompt_ids)

        final_response_ids = input_ids[prompt_len:-1] if input_ids else []
        sentinel_ids = {self.tokenizer.convert_tokens_to_ids(t) for t in SENTINELS}
        sentinel_ids.discard(None)
        sentinel_ids.discard(self.tokenizer.unk_token_id)
        original_sentinels = sentinel_ids.intersection(original_response_ids)
        final_sentinels = sentinel_ids.intersection(final_response_ids)
        lost_sentinels = sorted(original_sentinels - final_sentinels)
        response_tokens_truncated = max(0, len(original_response_ids) - len(final_response_ids))
        end_cell_id = self.tokenizer.convert_tokens_to_ids("[END_CELL]")
        if self.strict_accounting and end_cell_id not in original_response_ids:
            raise ValueError(f"row {idx} response does not contain atomic [END_CELL]")
        if self.strict_accounting and (prompt_truncated or lost_sentinels):
            raise ValueError(
                f"semantic truncation at row {idx}: prompt_truncated={prompt_truncated}, "
                f"response_tokens_truncated={response_tokens_truncated}, "
                f"lost_sentinel_ids={lost_sentinels}")

        # Labels: -100 for prompt tokens (no loss), actual ids for response tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        assert len(input_ids) == len(labels), (
            f"Length mismatch: {len(input_ids)} vs {len(labels)}"
        )

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "row_index": idx,
            "accounting": {
                "original_tokens": original_length,
                "final_tokens": len(input_ids),
                "supervised_tokens": sum(x != -100 for x in labels),
                "truncated_tokens": max(0, original_length - len(input_ids)),
                "response_tokens_truncated": response_tokens_truncated,
                "prompt_truncated": prompt_truncated,
                "sentinel_loss": bool(lost_sentinels),
            },
        }
        # Weights are indexed as prompt_len + (response token offset), which is only valid when the
        # prompt was NOT truncated. In the pathological branch above (prompt alone exceeds max_length-50)
        # prompt_len is set to max_length//2, which is not where the response starts -- weighting there
        # would land the up-weight on arbitrary tokens. Fall back to uniform instead.
        if self.de_weight != 1.0 and prompt_len == len(prompt_ids):
            out["weights"] = torch.tensor(
                self._token_weights(response, prompt_len, len(input_ids), ex.get("de_genes") or []),
                dtype=torch.float,
            )
        elif self.de_weight != 1.0:
            out["weights"] = torch.ones(len(input_ids), dtype=torch.float)
        return out

    def _token_weights(self, response, prompt_len, total_len, de_genes):
        """Per-token weights aligned to `labels`. Prompt tokens and the trailing EOS get weight 1.

        Uses the fast tokenizer's OFFSET MAPPING rather than tokenizing gene-by-gene: gene symbols are
        multi-subword under BPE ("TNFAIP3" is several tokens) and every subword of a DE gene must be
        up-weighted, while re-tokenizing pieces separately is not guaranteed to reproduce the whole-string
        segmentation. Offsets are exact by construction.
        """
        w = [1.0] * total_len
        if not de_genes:
            return w
        text = " " + response
        try:
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offs = enc["offset_mapping"]
        except (TypeError, KeyError, NotImplementedError):
            return w                                    # slow tokenizer: degrade to uniform, never crash
        de = set(de_genes)
        mark = bytearray(len(text))                      # char-level mask of DE gene spans; O(len(text))
        pos = 0
        for tok in text.split(" "):
            if tok and tok in de:
                mark[pos:pos + len(tok)] = b"\x01" * len(tok)
            pos += len(tok) + 1
        for t, (a, b) in enumerate(offs):
            i = prompt_len + t
            if i >= total_len:
                break                                   # response was truncated to fit max_length
            # ANY marked char in the span, not just the first: GPT-NeoX BPE folds the leading space into
            # the token, so " TNFAIP3" spans (space, ..., 3) and testing mark[a] alone would test the
            # SPACE -- silently skipping the first subword of every gene and leaving the intervention a
            # partial no-op.
            if b > a and any(mark[a:b]):
                w[i] = self.de_weight
        return w


def measure_de_token_share(dataset, n_sample=400, seed=0):
    """Fraction of SUPERVISED tokens that belong to a DE gene, measured on the real tokenizer and the
    real data rather than estimated from gene counts. Genes differ in subword length, sentences differ in
    how many panel genes they contain, and the fraction is what determines whether a given weight is a
    no-op or a sledgehammer -- so it is measured, on a sample, at startup.
    """
    import random
    prev = dataset.de_weight
    dataset.de_weight = 2.0                     # any marker > 1; we only count which tokens get it
    rng = random.Random(seed)
    idx = rng.sample(range(len(dataset)), min(n_sample, len(dataset)))
    hi = tot = 0
    for i in idx:
        b = dataset[i]
        w, lab = b.get("weights"), b["labels"]
        if w is None:
            continue
        sup = lab != -100
        tot += int(sup.sum())
        hi += int(((w > 1.0) & sup).sum())
    dataset.de_weight = prev
    return hi / tot if tot else 0.0


def forward_loss(model, input_ids, attention_mask, labels, weights=None):
    """Causal-LM loss, optionally per-token weighted.

    NORMALISED so the mean weight over supervised tokens is 1. Without that normalisation, raising
    --de_weight would silently raise the effective learning rate and any observed effect would be
    uninterpretable -- indistinguishable from "we trained harder". With it, the intervention redistributes
    a fixed gradient budget toward the drug-discriminative genes and nothing else changes.
    """
    if weights is None:
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    import torch.nn.functional as F
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    lg, tg, w = logits[:, :-1, :], labels[:, 1:], weights[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1),
                         reduction="none", ignore_index=-100).view(tg.shape)
    ww = w * (tg != -100)
    return (ce * ww).sum() / ww.sum().clamp(min=1.0)


def collate_fn(batch, pad_token_id):
    """Pad batch to max length in batch, left-pad for causal LM."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                       b["input_ids"]])
        )
        labels.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long),
                       b["labels"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long),
                       torch.ones(len(b["input_ids"]), dtype=torch.long)])
        )

    out = {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "row_indices": [b["row_index"] for b in batch],
        "accounting": {
            key: sum(int(b["accounting"][key]) for b in batch)
            for key in batch[0]["accounting"]
        },
    }
    if "weights" in batch[0]:
        # pad with 0, not 1: padding is masked by labels == -100 anyway, but a 0 keeps the weight sum
        # honest if the mask and the weights ever disagree.
        out["weights"] = torch.stack([
            torch.cat([torch.zeros(max_len - len(b["weights"])), b["weights"]]) for b in batch
        ])
    return out


def _training_contract(args, tokenizer, model):
    data = {"train_sha256": sha256_file(args.train_file)}
    if args.eval_file and os.path.exists(args.eval_file):
        data["eval_sha256"] = sha256_file(args.eval_file)
    contract = {
        "trainer_sha256": sha256_file(os.path.abspath(__file__)),
        "data_order_contract": "torch.randperm(seed + epoch), then resume by row offset; v1",
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "attn_implementation": args.attn_implementation,
        "model_fingerprint": model_fingerprint(model, args.model_name, args.model_revision),
        "config_fingerprint": _stable_json_hash(_canonical_model_config(model.config)),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        **data,
        "max_length": args.max_length,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "de_weight": args.de_weight,
        "de_share": args.de_share,
        "bf16": bool(args.bf16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "prepend_bos": bool(args.prepend_bos),
        "strict_token_contract": bool(args.strict_token_contract),
        "seed": args.seed,
    }
    return contract


def _build_provenance(args, tokenizer, model, contract, sentinel_result):
    return {
        "schema_version": 1,
        "scientific_checkpoint": "final",
        "validation_role": "development loss only; never used to select the scientific checkpoint",
        "model": {
            "requested_source": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_commit": getattr(model.config, "_commit_hash", None) or args.model_revision,
            "architecture": model.__class__.__name__,
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "attention_implementation": args.attn_implementation,
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "length": len(tokenizer),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "prepend_exactly_one_bos": bool(args.prepend_bos),
            "chat_template_used": False,
            "sentinels": sentinel_result["ids"],
        },
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "contract": contract,
        "contract_fingerprint": _stable_json_hash(contract),
    }


def _save_checkpoint_atomic(target_dir, model, tokenizer, training_state, provenance):
    """Publish one immutable checkpoint directory and return save timing/evidence.

    A per-target exclusive lock makes cooperating writers fail closed.  Existing checkpoints are
    never removed or replaced: an identical one is reused after full validation, while a mismatch
    is an error.  Only this invocation's uniquely named temporary directory is ever cleaned up.
    """
    started = time.monotonic()
    parent = os.path.dirname(target_dir)
    os.makedirs(parent, exist_ok=True)
    expected_identity = _checkpoint_identity(training_state, provenance)
    lock_path = target_dir + ".publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"checkpoint publication lock already exists: {lock_path}") from exc
    os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
    os.close(lock_fd)
    tmp = None
    try:
        if os.path.exists(target_dir):
            verified = _validate_published_checkpoint(
                target_dir, expected_identity=expected_identity)
            elapsed = time.monotonic() - started
            logger.info(f"  Reused verified immutable checkpoint {target_dir} "
                        f"(save_duration_seconds={elapsed:.3f}, "
                        f"hash_duration_seconds="
                        f"{verified['hash_duration_seconds']:.3f})")
            return {"path": target_dir, "reused": True,
                    "duration_seconds": elapsed,
                    "hash_duration_seconds": verified["hash_duration_seconds"],
                    "identity": expected_identity}
        tmp = tempfile.mkdtemp(prefix=f".{os.path.basename(target_dir)}.tmp-", dir=parent)
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        torch.save(training_state, os.path.join(tmp, TRAINING_STATE))
        _atomic_write_json(os.path.join(tmp, PROVENANCE_FILE), provenance)
        file_records, manifest_hash_duration = _published_file_records(tmp)
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "identity": expected_identity,
            "files": file_records,
        }
        _atomic_write_json(os.path.join(tmp, CHECKPOINT_MANIFEST), manifest)
        os.replace(tmp, target_dir)
        tmp = None
        verified = _validate_published_checkpoint(
            target_dir, expected_identity=expected_identity)
        hash_duration = manifest_hash_duration + verified["hash_duration_seconds"]
        elapsed = time.monotonic() - started
        logger.info(f"  Published immutable checkpoint {target_dir} "
                    f"(save_duration_seconds={elapsed:.3f}, "
                    f"hash_duration_seconds={hash_duration:.3f})")
        return {"path": target_dir, "reused": False,
                "duration_seconds": elapsed,
                "hash_duration_seconds": hash_duration,
                "identity": expected_identity}
    except Exception:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _resume_state_payload(optimizer, scheduler, *, epoch, microbatch_position, global_step,
                          best_eval_loss, contract, epoch_loss_sum, epoch_tokens,
                          window_loss_sum, window_tokens, accounting=None, rng_state=None,
                          completed=False):
    if microbatch_position < 0:
        raise ValueError("microbatch_position must be non-negative")
    return {
        "schema_version": 1,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "microbatch_position": int(microbatch_position),
        "accumulation_position": 0,
        "global_step": int(global_step),
        "best_eval_loss": float(best_eval_loss),
        "rng": rng_state if rng_state is not None else _capture_rng_state(),
        "contract": contract,
        "contract_fingerprint": _stable_json_hash(contract),
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_tokens": int(epoch_tokens),
        "window_loss_sum": float(window_loss_sum),
        "window_tokens": int(window_tokens),
        "accounting": dict(accounting or {}),
        "completed": bool(completed),
    }


def _saved_checkpoint_matches(last_saved, *, global_step, epoch, microbatch_position):
    return (last_saved is not None and
            last_saved["global_step"] == global_step and
            last_saved["epoch"] == epoch and
            last_saved["microbatch_position"] == microbatch_position)


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    # --- Reproducibility ---
    # Without this the shuffle order, the dropout masks and any weight left uninitialised by the
    # checkpoint all differ between runs, so two runs of "the same" recipe are not comparable and a
    # difference between arms cannot be separated from run-to-run variation. Seeding does not make
    # the run bit-exact on GPU (cuBLAS reductions are non-deterministic), but it removes every
    # source of variation that is ours to control.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except ImportError:
        pass
    logger.info(f"Seed: {args.seed}")

    # --- Device setup ---
    if args.mode == "test":
        device = torch.device("cpu")
        args.bf16 = False
        logger.info("Test mode: using CPU, bf16 disabled")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.resume_from_checkpoint and not args.resumable:
        raise ValueError("--resume_from_checkpoint requires --resumable")
    resume_dir = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    load_source = resume_dir or args.model_name
    load_revision = None if resume_dir else args.model_revision
    if resume_dir:
        logger.info(f"Resuming model and tokenizer from {resume_dir}")

    identity_config = AutoConfig.from_pretrained(
        load_source, **({"revision": load_revision} if load_revision is not None else {}))
    strict_gemma = gemma_mode_from_config(
        identity_config, prepend_bos=args.prepend_bos,
        strict_token_contract=args.strict_token_contract)
    if strict_gemma:
        protobuf_runtime = require_gemma_protobuf_runtime()
        logger.info("  Helper-bound Python protobuf available for Gemma SentencePiece: "
                    f"{protobuf_runtime['version']} ({protobuf_runtime['tree_sha256']})")

    # --- Load tokenizer ---
    logger.info(f"Loading tokenizer from {load_source}...")
    tokenizer_kwargs = {"revision": load_revision} if load_revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(load_source, **tokenizer_kwargs)
    base_tokenizer_length = len(tokenizer)
    if strict_gemma and not resume_dir:
        base_contract = validate_gemma_base_tokenizer_contract(
            tokenizer, identity_config.vocab_size)
        logger.info("  Verified pinned Gemma base tokenizer: "
                    f"vocab={base_contract['vocab_size']}, "
                    f"length={base_contract['length']}, "
                    f"sanity_tokens={base_contract['sanity_tokens']}")
    if tokenizer.pad_token is None:
        if args.strict_token_contract:
            raise ValueError("strict token contract requires a native PAD token; refusing PAD=EOS")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"  Vocab size: {tokenizer.vocab_size}")

    # --- Register the sentinels as atomic special tokens ---
    # The [END_CELL] data format terminates every response with this marker. Without registering it,
    # the tokenizer splits it into subword pieces ('[', 'END', '_CELL', ']') and the model never sees
    # a clean end-of-cell signal. [DOWN] is the same story for the Arm 1b RESIDUAL targets, which
    # encode a signed DE signature as "<up genes> [DOWN] <down genes> [END_CELL]" -- if [DOWN] is split
    # the up/down boundary is not a clean symbol. Registering it is harmless for the ordinary
    # [END_CELL] datasets (the token simply never appears). add_special_tokens returns the number of
    # NEW tokens added (0 if already present); we resize embeddings only if >0.
    sentinel_result = register_sentinels(tokenizer, strict=args.strict_token_contract)
    if strict_gemma:
        validate_gemma_registered_tokenizer_contract(tokenizer, sentinel_result)
    if sentinel_result["added"]:
        logger.info(f"  Added {sentinel_result['added']} special token(s): "
                    + ", ".join(f"{t} -> id {i}" for t, i in sentinel_result["ids"].items()))
    else:
        logger.info(f"  {list(SENTINELS)} already in tokenizer vocab")
    for token, token_id in sentinel_result["ids"].items():
        logger.info(f"  {token} tokenizes atomically to id {token_id}")
    validate_training_token_contract(
        tokenizer, prepend_bos=args.prepend_bos,
        require_distinct_pad_eos=args.strict_token_contract)
    if args.strict_token_contract and args.prepend_bos and not tokenizer.is_fast:
        raise ValueError("strict Gemma contract requires the preflight-verified fast tokenizer")
    if args.prepend_bos:
        logger.info(f"  Exactly-one-BOS construction enabled (id {tokenizer.bos_token_id}); "
                    "no chat template is used")

    # --- Load model ---
    logger.info(f"Loading model from {load_source}...")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model_kwargs = {"revision": load_revision} if load_revision is not None else {}
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(
            load_source,
            dtype=dtype,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            load_source,
            torch_dtype=dtype,
            **model_kwargs,
        )

    embedding_contract = validate_model_embedding_rows_before_resize(
        model, tokenizer, strict_gemma=strict_gemma, resume=bool(resume_dir),
        base_tokenizer_length=base_tokenizer_length)
    if strict_gemma:
        logger.info("  Verified Gemma embeddings before resize: "
                    f"rows={embedding_contract['embedding_rows']}, "
                    f"resume={embedding_contract['resume']}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("  Gradient checkpointing enabled")
        if args.prepend_bos or args.strict_token_contract or args.resumable:
            model.config.use_cache = False

    # Resize token embeddings if we added the [END_CELL] special token above. This adds a
    # fresh (randomly-initialized) embedding row for the new token so the model can learn it.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"  Resized token embeddings to {len(tokenizer)} (for {list(SENTINELS)})")
    if strict_gemma:
        final_rows = int(model.get_input_embeddings().weight.shape[0])
        final_config_vocab = int(getattr(model.config, "vocab_size", -1))
        if (final_rows, final_config_vocab) != (GEMMA_TOKENIZER_LENGTH,
                                                GEMMA_TOKENIZER_LENGTH):
            raise ValueError(
                "Gemma resize did not produce the certified 256002-row state: "
                f"embedding_rows={final_rows}, config.vocab_size={final_config_vocab}")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,} total, {trainable:,} trainable")

    # --- Load data ---
    train_dataset = C2SDataset(
        args.train_file, tokenizer, max_length=args.max_length, de_weight=args.de_weight,
        prepend_bos=args.prepend_bos, strict_accounting=args.strict_token_contract)
    if args.de_share is not None or args.de_weight != 1.0:
        n_de = sum(1 for e in train_dataset.examples if e.get("de_genes"))
        if n_de == 0:
            raise SystemExit("DE weighting was requested but no example has a non-empty 'de_genes' "
                             "field. Run build_de_weights.py on the training file first.")
        if args.de_share is not None:
            f = measure_de_token_share(train_dataset)
            if not (0.0 < f < 1.0):
                raise SystemExit(f"measured DE token share f={f:.4f} is degenerate; cannot derive a "
                                 f"weight. Check that de_genes actually appear in the responses.")
            args.de_weight = args.de_share / (1.0 - args.de_share) * (1.0 - f) / f
            train_dataset.de_weight = args.de_weight
            logger.info(f"DE token share f = {100*f:.1f}% of supervised tokens -> "
                        f"--de_weight {args.de_weight:.2f} for a {100*args.de_share:.0f}% gradient share")
        logger.info(f"DE-weighted SFT: weight {args.de_weight:.2f} on DE gene tokens "
                    f"({n_de}/{len(train_dataset.examples)} examples carry a DE set)")
    eval_dataset = None
    if args.eval_file and os.path.exists(args.eval_file):
        eval_dataset = C2SDataset(
            args.eval_file, tokenizer, max_length=args.max_length,
            prepend_bos=args.prepend_bos, strict_accounting=args.strict_token_contract)

    # In test mode, use only a handful of examples
    if args.mode == "test":
        train_dataset.examples = train_dataset.examples[:20]
        if eval_dataset:
            eval_dataset.examples = eval_dataset.examples[:5]

    # Log token length stats
    sample_lens = []
    for i in range(min(50, len(train_dataset))):
        item = train_dataset[i]
        sample_lens.append(len(item["input_ids"]))
    logger.info(f"  Token length stats (sample of {len(sample_lens)}):")
    logger.info(f"    Mean: {sum(sample_lens)/len(sample_lens):.0f}")
    logger.info(f"    Min: {min(sample_lens)}, Max: {max(sample_lens)}")

    pad_id = tokenizer.pad_token_id
    legacy_train_loader = None
    if not args.resumable:
        # Preserve the published Pythia path exactly: global-RNG RandomSampler and one loader.
        legacy_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    eval_loader = None
    if eval_dataset and len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    microbatches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    if args.resumable:
        total_steps = (microbatches_per_epoch // args.grad_accum) * args.num_epochs
    else:
        total_steps = (len(legacy_train_loader) * args.num_epochs) // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"  Total optimization steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * args.grad_accum}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision scaler (only for fp16, not bf16)
    use_amp = args.bf16 and device.type == "cuda"

    # --- Training ---
    os.makedirs(args.output_dir, exist_ok=True)
    contract = _training_contract(args, tokenizer, model)
    provenance = _build_provenance(args, tokenizer, model, contract, sentinel_result)
    if args.resumable or args.strict_token_contract or args.prepend_bos:
        _atomic_write_json(os.path.join(args.output_dir, PROVENANCE_FILE), provenance)
        logger.info(f"  Contract fingerprint: {provenance['contract_fingerprint']}")

    global_step = 0
    best_eval_loss = float("inf")
    start_epoch = 0
    start_microbatch = 0
    resumed_epoch_loss = 0.0
    resumed_epoch_tokens = 0
    resumed_window_loss = 0.0
    resumed_window_tokens = 0
    accounting_total = {k: 0 for k in (
        "original_tokens", "final_tokens", "supervised_tokens", "truncated_tokens",
        "response_tokens_truncated", "prompt_truncated", "sentinel_loss")}

    if resume_dir:
        saved = _torch_load(os.path.join(resume_dir, TRAINING_STATE))
        _validate_resume_fingerprints(saved["contract"], contract)
        if saved.get("accumulation_position", 0) != 0:
            raise RuntimeError("REFUSING TO RESUME: checkpoint is not at an optimizer boundary")
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        global_step = int(saved["global_step"])
        best_eval_loss = float(saved["best_eval_loss"])
        start_epoch = int(saved["epoch"])
        start_microbatch = int(saved["microbatch_position"])
        resumed_epoch_loss = float(saved.get("epoch_loss_sum", 0.0))
        resumed_epoch_tokens = int(saved.get("epoch_tokens", 0))
        resumed_window_loss = float(saved.get("window_loss_sum", 0.0))
        resumed_window_tokens = int(saved.get("window_tokens", 0))
        accounting_total.update(saved.get("accounting") or {})
        _restore_rng_state(saved["rng"])
        logger.info(f"  Restored optimizer/scheduler/RNG at epoch={start_epoch + 1}, "
                    f"microbatch={start_microbatch}, global_step={global_step}")

    stop_request = StopRequest()
    if args.resumable:
        install_stop_handlers(stop_request)
    log_interval = args.log_every
    save_interval = args.save_every
    last_saved_checkpoint = None

    if args.mode == "test":
        log_interval = 1
        save_interval = 999999  # don't save in test mode
        args.num_epochs = 1

    logger.info(f"\n{'='*60}")
    logger.info("Starting training")
    logger.info(f"{'='*60}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = resumed_epoch_loss if epoch == start_epoch else 0.0
        epoch_tokens = resumed_epoch_tokens if epoch == start_epoch else 0
        window_loss = resumed_window_loss if epoch == start_epoch else 0.0
        window_tokens = resumed_window_tokens if epoch == start_epoch else 0
        microbatch_offset = start_microbatch if epoch == start_epoch else 0
        optimizer.zero_grad()

        sampler = None
        if args.resumable:
            sampler = DeterministicEpochSampler(
                train_dataset, args.seed, epoch,
                start_index=min(microbatch_offset * args.batch_size, len(train_dataset)))
            loader_generator = torch.Generator()
            loader_generator.manual_seed(args.seed + 1_000_000 + epoch)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                collate_fn=lambda b: collate_fn(b, pad_id), num_workers=0,
                pin_memory=(device.type == "cuda"), generator=loader_generator)
        else:
            train_loader = legacy_train_loader

        pending_row_indices = []
        last_boundary_microbatch = microbatch_offset
        boundary_epoch_loss = epoch_loss
        boundary_epoch_tokens = epoch_tokens
        boundary_window_loss = window_loss
        boundary_window_tokens = window_tokens
        boundary_accounting = dict(accounting_total)
        boundary_rng = _capture_rng_state() if args.resumable else None
        final_full_boundary = (microbatches_per_epoch // args.grad_accum) * args.grad_accum

        for step, batch in enumerate(train_loader, start=microbatch_offset):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            weights = batch["weights"].to(device) if "weights" in batch else None

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw = forward_loss(model, input_ids, attention_mask, labels, weights)
            else:
                raw = forward_loss(model, input_ids, attention_mask, labels, weights)
            loss = raw / args.grad_accum

            loss.backward()

            # Track tokens where loss is computed
            n_tokens = (labels != -100).sum().item()
            epoch_loss += raw.item() * n_tokens
            epoch_tokens += n_tokens
            window_loss += raw.item() * n_tokens
            window_tokens += n_tokens
            pending_row_indices.extend(batch["row_indices"])
            for key, value in batch["accounting"].items():
                accounting_total[key] += int(value)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                last_boundary_microbatch = step + 1
                pending_row_indices = []
                boundary_epoch_loss = epoch_loss
                boundary_epoch_tokens = epoch_tokens
                boundary_window_loss = window_loss
                boundary_window_tokens = window_tokens
                boundary_accounting = dict(accounting_total)
                if args.resumable:
                    boundary_rng = _capture_rng_state()

                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    avg_window_loss = window_loss / max(window_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | Window loss: {avg_window_loss:.4f} | "
                        f"LR: {lr:.2e} | Tokens: {epoch_tokens:,} | "
                        f"Truncated response tokens: "
                        f"{accounting_total['response_tokens_truncated']:,} | "
                        f"Sentinel losses: {accounting_total['sentinel_loss']:,}"
                    )
                    window_loss = 0.0
                    window_tokens = 0
                    boundary_window_loss = 0.0
                    boundary_window_tokens = 0

                # The epoch's final optimizer boundary is followed immediately by remainder
                # accounting/evaluation.  Skip the redundant periodic snapshot there; resumable
                # names still encode both optimizer step and microbatch position for any signal.
                if (global_step % save_interval == 0 and global_step > 0 and
                        step + 1 != final_full_boundary):
                    ckpt_dir = (_checkpoint_dir(args.output_dir, global_step, step + 1)
                                if args.resumable else
                                os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    if args.resumable:
                        state = _resume_state_payload(
                            optimizer, scheduler, epoch=epoch,
                            microbatch_position=step + 1, global_step=global_step,
                            best_eval_loss=best_eval_loss, contract=contract,
                            epoch_loss_sum=epoch_loss, epoch_tokens=epoch_tokens,
                            window_loss_sum=window_loss, window_tokens=window_tokens,
                            accounting=accounting_total)
                        save_result = _save_checkpoint_atomic(
                            ckpt_dir, model, tokenizer, state, provenance)
                        last_saved_checkpoint = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "microbatch_position": step + 1,
                            "result": save_result,
                        }
                    else:
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"  Saved checkpoint to {ckpt_dir}")
                    if getattr(args, "keep_checkpoints", 0):
                        _prune_checkpoints(args.output_dir, args.keep_checkpoints)

                if stop_request.requested:
                    ckpt_dir = _checkpoint_dir(
                        args.output_dir, global_step, step + 1)
                    same_boundary = _saved_checkpoint_matches(
                        last_saved_checkpoint, global_step=global_step, epoch=epoch,
                        microbatch_position=step + 1)
                    if same_boundary:
                        save_result = last_saved_checkpoint["result"]
                        logger.warning(
                            f"Signal {stop_request.signum} followed a completed checkpoint save; "
                            f"reusing {ckpt_dir} instead of writing it twice "
                            f"(save_duration_seconds="
                            f"{save_result['duration_seconds']:.3f})")
                    else:
                        state = _resume_state_payload(
                            optimizer, scheduler, epoch=epoch, microbatch_position=step + 1,
                            global_step=global_step, best_eval_loss=best_eval_loss,
                            contract=contract, epoch_loss_sum=epoch_loss,
                            epoch_tokens=epoch_tokens, window_loss_sum=window_loss,
                            window_tokens=window_tokens, accounting=accounting_total)
                        save_result = _save_checkpoint_atomic(
                            ckpt_dir, model, tokenizer, state, provenance)
                        last_saved_checkpoint = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "microbatch_position": step + 1,
                            "result": save_result,
                        }
                    logger.warning(f"Safe checkpoint available after signal "
                                   f"{stop_request.signum}: {ckpt_dir}; exiting with status 99")
                    raise SystemExit(99)

            # Test mode: stop after a few steps
            if args.mode == "test" and step >= 5:
                logger.info("  Test mode: stopping after 6 steps")
                break

        if pending_row_indices:
            if args.resumable or args.strict_token_contract:
                logger.info(f"  Dropped final incomplete accumulation: {len(pending_row_indices)} "
                            f"example(s), row indices={pending_row_indices}")
            optimizer.zero_grad()
            if stop_request.requested and args.resumable:
                ckpt_dir = _checkpoint_dir(
                    args.output_dir, global_step, last_boundary_microbatch)
                same_boundary = _saved_checkpoint_matches(
                    last_saved_checkpoint, global_step=global_step, epoch=epoch,
                    microbatch_position=last_boundary_microbatch)
                if same_boundary:
                    save_result = last_saved_checkpoint["result"]
                    logger.warning(
                        f"Reusing optimizer-boundary checkpoint after signal "
                        f"(save_duration_seconds={save_result['duration_seconds']:.3f})")
                else:
                    state = _resume_state_payload(
                        optimizer, scheduler, epoch=epoch,
                        microbatch_position=last_boundary_microbatch, global_step=global_step,
                        best_eval_loss=best_eval_loss, contract=contract,
                        epoch_loss_sum=boundary_epoch_loss, epoch_tokens=boundary_epoch_tokens,
                        window_loss_sum=boundary_window_loss, window_tokens=boundary_window_tokens,
                        accounting=boundary_accounting, rng_state=boundary_rng)
                    save_result = _save_checkpoint_atomic(
                        ckpt_dir, model, tokenizer, state, provenance)
                    last_saved_checkpoint = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "microbatch_position": last_boundary_microbatch,
                        "result": save_result,
                    }
                logger.warning(f"Signal {stop_request.signum} arrived inside an incomplete "
                               f"accumulation; checkpoint rewinds to optimizer boundary "
                               f"microbatch {last_boundary_microbatch}; exiting with status 99")
                raise SystemExit(99)

        # End of epoch: eval
        avg_train_loss = epoch_loss / max(epoch_tokens, 1)
        logger.info(f"\n  Epoch {epoch+1} complete | Train loss: {avg_train_loss:.4f}")

        if eval_loader is not None:
            model.eval()
            eval_loss = 0.0
            eval_tokens = 0
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                    n_tokens = (labels != -100).sum().item()
                    eval_loss += outputs.loss.item() * n_tokens
                    eval_tokens += n_tokens

                    if args.mode == "test":
                        break

                    if stop_request.requested and args.resumable:
                        ckpt_dir = _checkpoint_dir(
                            args.output_dir, global_step, microbatches_per_epoch)
                        state = _resume_state_payload(
                            optimizer, scheduler, epoch=epoch,
                            microbatch_position=microbatches_per_epoch,
                            global_step=global_step, best_eval_loss=best_eval_loss,
                            contract=contract, epoch_loss_sum=epoch_loss,
                            epoch_tokens=epoch_tokens, window_loss_sum=window_loss,
                            window_tokens=window_tokens, accounting=accounting_total)
                        save_result = _save_checkpoint_atomic(
                            ckpt_dir, model, tokenizer, state, provenance)
                        last_saved_checkpoint = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "microbatch_position": microbatches_per_epoch,
                            "result": save_result,
                        }
                        logger.warning(f"Safe epoch-boundary checkpoint written during evaluation "
                                       f"after signal {stop_request.signum}; evaluation will be "
                                       "restarted on resume; exiting with status 99")
                        raise SystemExit(99)

            avg_eval_loss = eval_loss / max(eval_tokens, 1)
            logger.info(f"  Eval loss: {avg_eval_loss:.4f}")

            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_dir = os.path.join(args.output_dir, "best")
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"  New best development loss — diagnostic snapshot saved to "
                            f"{best_dir}; the scientific checkpoint remains final")

        start_microbatch = 0
        resumed_epoch_loss = resumed_window_loss = 0.0
        resumed_epoch_tokens = resumed_window_tokens = 0

    logger.info("Final token accounting: " + json.dumps(accounting_total, sort_keys=True))

    # Save final model
    if args.mode != "test":
        final_dir = os.path.join(args.output_dir, "final")
        if args.resumable:
            state = _resume_state_payload(
                optimizer, scheduler, epoch=args.num_epochs, microbatch_position=0,
                global_step=global_step, best_eval_loss=best_eval_loss, contract=contract,
                epoch_loss_sum=0.0, epoch_tokens=0, window_loss_sum=0.0, window_tokens=0,
                accounting=accounting_total, completed=True)
            _save_checkpoint_atomic(final_dir, model, tokenizer, state, provenance)
        else:
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
        logger.info(f"\nSaved final model to {final_dir}")

    logger.info("Training complete!")
    return model, tokenizer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SFT C2S-Scale on Tahoe perturbation data")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model_name", type=str,
                        default="vandijklab/C2S-Scale-Pythia-1b-pt",
                        help="HuggingFace model name or local path")
    parser.add_argument("--model_revision", type=str, default=None,
                        help="Pinned Hugging Face revision/commit. Recorded in provenance and "
                             "passed to both model and tokenizer loaders.")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        help="Documented Transformers attention backend, e.g. eager, sdpa, or "
                             "flash_attention_2. Omitted preserves the model/library default.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--de_share", type=float, default=None,
                        help="Preferred over --de_weight. Target FRACTION OF THE GRADIENT the DE gene "
                             "tokens should carry (e.g. 0.5). The weight is then derived from the "
                             "token-level DE share f measured on this tokenizer and this data: "
                             "w = share/(1-share) * (1-f)/f. f depends on k_sig, on subword lengths, and "
                             "on how many panel genes a sentence actually contains, so deriving it beats "
                             "hardcoding a weight that may be a no-op or may swamp the loss entirely.")
    parser.add_argument("--de_weight", type=float, default=1.0,
                        help="DE-WEIGHTED SFT: multiply the token loss on this condition's "
                             "drug-specific DE genes (the 'de_genes' field from build_de_weights.py) by "
                             "this factor, then renormalise so the mean supervised-token weight stays 1. "
                             "1.0 = plain SFT and a mathematically exact control arm. Set it from the "
                             "measured DE token share f that build_de_weights.py prints, not by guessing. "
                             "Held-out eval loss is deliberately left UNWEIGHTED so it stays comparable "
                             "across settings.")
    parser.add_argument("--max_length", type=int, default=8192,
                        help="Max token length. A 946-gene control + 946-gene response is "
                             "~6,200 BPE tokens (~3.25 tok/gene), so keep this >= 8192 or the "
                             "response gets truncated and the target is corrupted.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="Gradient accumulation steps (effective_bs = batch_size * grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--prepend_bos", action="store_true",
                        help="Prepend exactly one masked BOS token. Required for the Gemma arm; "
                             "off by default to preserve the Pythia training contract.")
    parser.add_argument("--strict_token_contract", action="store_true",
                        help="Fail unless sentinels are atomic/non-UNK/distinct and survive "
                             "save/reload, PAD differs from EOS, and no semantic truncation occurs.")
    parser.add_argument("--resumable", action="store_true",
                        help="Write complete optimizer/scheduler/RNG/data-position checkpoints "
                             "with deterministic epoch order. Legacy model-only saving remains the "
                             "default when this flag is absent.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="With --resumable: checkpoint directory or 'auto' for the newest "
                             "complete checkpoint in output_dir.")
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N optimization steps")
    parser.add_argument("--save_every", type=int, default=200,
                        help="Save checkpoint every N optimization steps (frequent to survive preemption)")
    parser.add_argument("--keep_checkpoints", type=int, default=0,
                        help="If >0, keep only the most recent N checkpoint-{step} dirs "
                             "(best/ and final/ are always kept). Protects a shared/full disk.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds Python, NumPy and Torch, which fixes the shuffle order and the "
                             "dropout masks. Runs remain non-bit-exact on GPU, but every source of "
                             "variation under our control is removed, so a difference between two "
                             "arms is a difference between the arms.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
