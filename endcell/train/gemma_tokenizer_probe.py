"""Fail-closed tokenizer/data audit for the Pythia and Gemma standard-target arms.

Run this on a CPU compute node before allocating an H200.  It hashes the immutable JSONLs,
measures the complete token-length population under both tokenizers, checks slow/fast parity on a
deterministic prefix, and optionally loads each model to report the exact resized parameter count.
It never uses a chat template.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_c2s_tahoe_endcell import (
    SENTINELS,
    _atomic_write_json,
    register_sentinels,
    sha256_file,
    tokenizer_fingerprint,
)


GEMMA_REVISION = "5ddf28b8f1c81b7ab7a9be192924da82b6c5d512"
PYTHIA_REVISION = "830e4689d4238bbb5e2ec9a89b76e6a6d48061db"
DEFAULT_MODELS = (
    "vandijklab/C2S-Scale-Pythia-1b-pt",
    "vandijklab/C2S-Scale-Gemma-2-2B",
)
DEFAULT_DATA_DIR = "/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big"
DEFAULT_JSONLS = (
    "train.jsonl",
    "eval_tier1_seen_conditions.jsonl",
    "eval_tier2_unseen_drugs.jsonl",
    "eval_tier3_unseen_combos.jsonl",
    "eval_tier4_dose_interpolation.jsonl",
)


def _quantiles(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.int64)
    return {
        "n": int(a.size),
        "min": int(a.min()),
        "mean": float(a.mean()),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": int(a.max()),
        "sum": int(a.sum()),
    }


def _control_genes(prompt):
    marker = "Control cell:"
    if marker not in prompt:
        return []
    return prompt.split(marker, 1)[1].split("[END_CELL]", 1)[0].strip().split()


def _response_genes(response):
    return [x for x in response.strip().split() if x not in SENTINELS]


def _semantic_lengths_from_ids(tokenizer, prompt_ids, response_ids, prepend_bos, max_length):
    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    if prepend_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("Gemma audit requested BOS but tokenizer has no bos_token_id")
        if tokenizer.bos_token_id in prompt_ids or tokenizer.bos_token_id in response_ids:
            raise ValueError("input text already contains BOS under exactly-one-BOS contract")
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    original_total = len(prompt_ids) + len(response_ids) + 1
    max_response = max_length - len(prompt_ids) - 1
    prompt_truncated = False
    if original_total <= max_length:
        kept_response = response_ids
    elif max_response < 50:
        prompt_truncated = True
        # The legacy emergency branch slices the combined stream and no longer has a trustworthy
        # prompt/response boundary. Treat all response semantics as lost rather than allowing the
        # control cell's [END_CELL] to masquerade as the response terminator.
        kept_response = []
    else:
        kept_response = response_ids[:max_response]
    sentinel_ids = {tokenizer.convert_tokens_to_ids(x) for x in SENTINELS}
    original_sentinels = sentinel_ids.intersection(response_ids)
    kept_sentinels = sentinel_ids.intersection(kept_response)
    return {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "supervised_tokens": len(response_ids) + 1,
        "total_tokens": original_total,
        "truncated_response_tokens": max(0, len(response_ids) - len(kept_response)),
        "prompt_truncated": prompt_truncated,
        "sentinel_loss": bool(original_sentinels - kept_sentinels),
    }


def _semantic_lengths(tokenizer, prompt, response, prepend_bos, max_length):
    return _semantic_lengths_from_ids(
        tokenizer,
        tokenizer.encode(prompt, add_special_tokens=False),
        tokenizer.encode(" " + response, add_special_tokens=False),
        prepend_bos,
        max_length,
    )


def _iter_jsonl(path, limit=0):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            value = json.loads(line)
            if "prompt" not in value or "response" not in value:
                raise ValueError(f"{path}:{i + 1} lacks prompt/response")
            yield i, value


def audit_file(path, tokenizer, prepend_bos, max_length, limit=0, parity_sink=None,
               batch_size=256, file_hash=None):
    vectors = {k: [] for k in (
        "prompt_tokens", "response_tokens", "supervised_tokens", "total_tokens",
        "control_genes", "response_genes")}
    counts = Counter()
    batch = []

    def consume(rows):
        prompts = [ex["prompt"] for _, ex in rows]
        responses = [" " + ex["response"] for _, ex in rows]
        prompt_ids = tokenizer(
            prompts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        response_ids = tokenizer(
            responses, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        for (row_index, ex), pids, rids in zip(rows, prompt_ids, response_ids):
            stats = _semantic_lengths_from_ids(
                tokenizer, pids, rids, prepend_bos, max_length)
            for key in ("prompt_tokens", "response_tokens", "supervised_tokens", "total_tokens"):
                vectors[key].append(stats[key])
            control_n = len(_control_genes(ex["prompt"]))
            response_n = len(_response_genes(ex["response"]))
            vectors["control_genes"].append(control_n)
            vectors["response_genes"].append(response_n)
            counts["rows"] += 1
            counts["control_below_200"] += control_n < 200
            counts["response_below_200"] += response_n < 200
            counts["over_4096"] += stats["total_tokens"] > 4096
            counts["over_8192"] += stats["total_tokens"] > 8192
            counts["semantic_truncations"] += stats["truncated_response_tokens"] > 0
            counts["truncated_response_tokens"] += stats["truncated_response_tokens"]
            counts["prompt_truncations"] += stats["prompt_truncated"]
            counts["sentinel_losses"] += stats["sentinel_loss"]
            counts["response_missing_end_cell"] += "[END_CELL]" not in ex["response"]
            counts["response_contains_down"] += "[DOWN]" in ex["response"]
            if parity_sink is not None and len(parity_sink) < parity_sink.maxlen:
                parity_sink.append((path, row_index, ex["prompt"], ex["response"]))

    for row in _iter_jsonl(path, limit):
        batch.append(row)
        if len(batch) >= batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    n = max(counts["rows"], 1)
    return {
        "path": os.path.abspath(path),
        "sha256": file_hash or sha256_file(path),
        "counts": dict(counts),
        "fractions": {
            "control_below_200": counts["control_below_200"] / n,
            "response_below_200": counts["response_below_200"] / n,
            "over_4096": counts["over_4096"] / n,
            "over_8192": counts["over_8192"] / n,
            "semantic_truncation": counts["semantic_truncations"] / n,
            "sentinel_loss": counts["sentinel_losses"] / n,
        },
        "distributions": {k: _quantiles(v) for k, v in vectors.items()},
    }


class ParityRows(list):
    def __init__(self, maxlen):
        super().__init__()
        self.maxlen = maxlen


def check_slow_fast_parity(model_name, revision, fast_tokenizer, rows):
    try:
        slow = AutoTokenizer.from_pretrained(
            model_name, revision=revision, use_fast=False)
    except Exception as exc:
        return {"available": False, "error": repr(exc), "rows_checked": 0}
    # Preserve the explicitly requested slow implementation for its serialization check.  The
    # semantic contract validator would allow an equivalent fast reload, but parity should still
    # prove the actual slow tokenizer that will be compared below.
    register_sentinels(slow, strict=True, reload_use_fast=False)
    mismatches = []
    for path, row, prompt, response in rows:
        for field, text in (("prompt", prompt), ("response", " " + response)):
            a = fast_tokenizer.encode(text, add_special_tokens=False)
            b = slow.encode(text, add_special_tokens=False)
            if a != b:
                mismatches.append({"path": path, "row": row, "field": field,
                                   "fast_length": len(a), "slow_length": len(b)})
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    return {
        "available": True,
        "slow_class": slow.__class__.__name__,
        "rows_checked": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches_first_20": mismatches,
    }


def load_parameter_count(model_name, revision, tokenizer, attention_implementation):
    kwargs = {"revision": revision, "torch_dtype": torch.bfloat16}
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    before = sum(p.numel() for p in model.parameters())
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    after = sum(p.numel() for p in model.parameters())
    result = {
        "class": model.__class__.__name__,
        "parameters_before_sentinels": before,
        "parameters_after_sentinels": after,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config_max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
        "resolved_commit": getattr(model.config, "_commit_hash", None),
    }
    del model
    return result


def audit_model(model_name, revision, jsonls, file_hashes, args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, use_fast=True)
    sentinel = register_sentinels(tokenizer, strict=True, reload_use_fast=True)
    prepend_bos = "gemma" in model_name.lower()
    if prepend_bos and (tokenizer.pad_token_id is None or
                        tokenizer.pad_token_id == tokenizer.eos_token_id):
        raise ValueError("Gemma must retain native PAD distinct from EOS")
    if args.require_fast and not tokenizer.is_fast:
        raise ValueError(f"{model_name} did not load a fast tokenizer")

    parity_rows = ParityRows(args.parity_examples)
    files = [audit_file(path, tokenizer, prepend_bos, args.max_length,
                        limit=args.max_examples, parity_sink=parity_rows,
                        batch_size=args.batch_size, file_hash=file_hashes[path])
             for path in jsonls]
    parity = check_slow_fast_parity(model_name, revision, tokenizer, parity_rows)
    if parity.get("available") and parity["mismatch_count"]:
        raise ValueError(f"slow/fast tokenizer parity failed for {model_name}: {parity}")

    result = {
        "model_name": model_name,
        "revision": revision,
        "prepend_exactly_one_bos": prepend_bos,
        "chat_template_used": False,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "is_fast": bool(tokenizer.is_fast),
            "vocab_size": tokenizer.vocab_size,
            "length_after_sentinels": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "sentinels": sentinel["ids"],
            "fingerprint": tokenizer_fingerprint(tokenizer),
        },
        "slow_fast_parity": parity,
        "files": files,
    }
    if args.load_model:
        result["model"] = load_parameter_count(
            model_name, revision, tokenizer, args.attn_implementation)
    return result


def _parse_revisions(items):
    out = {
        "vandijklab/C2S-Scale-Pythia-1b-pt": PYTHIA_REVISION,
        "vandijklab/C2S-Scale-Gemma-2-2B": GEMMA_REVISION,
    }
    for item in items or []:
        if "=" not in item:
            raise ValueError("--revision must be MODEL=REVISION")
        model, revision = item.split("=", 1)
        out[model] = revision
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models",
                        help="Model to audit; repeat. Defaults to canonical Pythia and Gemma.")
    parser.add_argument("--revision", action="append",
                        help="Pinned MODEL=REVISION mapping; repeat as needed.")
    parser.add_argument("--jsonl", action="append", dest="jsonls",
                        help="Canonical JSONL; repeat. Defaults to all five cluster files.")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", default="RESULTS/gemma_tokenizer_probe.json")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Development-only row cap per file; 0 audits every row.")
    parser.add_argument("--parity_examples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Tokenizer batch size for the full JSONL audit.")
    parser.add_argument("--require_fast", action="store_true")
    parser.add_argument("--load_model", action="store_true",
                        help="Also load weights and report exact resized parameter counts.")
    parser.add_argument("--attn_implementation", default="eager")
    args = parser.parse_args()

    models = args.models or list(DEFAULT_MODELS)
    revisions = _parse_revisions(args.revision)
    jsonls = args.jsonls or [os.path.join(args.data_dir, x) for x in DEFAULT_JSONLS]
    missing = [x for x in jsonls if not os.path.isfile(x)]
    if missing:
        raise FileNotFoundError(f"canonical JSONLs missing: {missing}")
    file_hashes = {path: sha256_file(path) for path in jsonls}

    report = {
        "schema_version": 1,
        "max_length": args.max_length,
        "max_examples_per_file": args.max_examples,
        "input_hashes": file_hashes,
        "models": [audit_model(model, revisions.get(model), jsonls, file_hashes, args)
                   for model in models],
    }
    _atomic_write_json(os.path.abspath(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
