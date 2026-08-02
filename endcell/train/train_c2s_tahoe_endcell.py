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
import json
import os
import re
import shutil
import logging
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _prune_checkpoints(output_dir, keep):
    """Keep only the most recent `keep` checkpoint-{step} dirs (best/ and final/
    are never matched, so always preserved). Protects a shared/full filesystem."""
    cks = []
    for d in os.listdir(output_dir):
        m = re.match(r"checkpoint-(\d+)$", d)
        if m:
            cks.append((int(m.group(1)), os.path.join(output_dir, d)))
    cks.sort()
    for _, path in cks[:-keep]:
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
    def __init__(self, filepath, tokenizer, max_length=4096, de_weight=1.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
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

        # Combine: prompt + response + eos
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]

        # Truncate from the end if too long
        if len(input_ids) > self.max_length:
            # Keep full prompt, truncate response
            max_response = self.max_length - len(prompt_ids) - 1  # -1 for eos
            if max_response < 50:
                # Prompt itself is too long, truncate prompt too
                input_ids = input_ids[:self.max_length]
                prompt_len = min(len(prompt_ids), self.max_length // 2)
            else:
                response_ids = response_ids[:max_response]
                input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
                prompt_len = len(prompt_ids)
        else:
            prompt_len = len(prompt_ids)

        # Labels: -100 for prompt tokens (no loss), actual ids for response tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        assert len(input_ids) == len(labels), (
            f"Length mismatch: {len(input_ids)} vs {len(labels)}"
        )

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
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
    }
    if "weights" in batch[0]:
        # pad with 0, not 1: padding is masked by labels == -100 anyway, but a 0 keeps the weight sum
        # honest if the mask and the weights ever disagree.
        out["weights"] = torch.stack([
            torch.cat([torch.zeros(max_len - len(b["weights"])), b["weights"]]) for b in batch
        ])
    return out


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
    import random as _random
    _random.seed(args.seed)
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

    # --- Load tokenizer ---
    logger.info(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
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
    _sentinels = ["[END_CELL]", "[DOWN]"]
    added = tokenizer.add_special_tokens({"additional_special_tokens": _sentinels})
    if added:
        logger.info(f"  Added {added} special token(s): "
                    + ", ".join(f"{t} -> id {tokenizer.convert_tokens_to_ids(t)}" for t in _sentinels))
    else:
        logger.info(f"  {_sentinels} already in tokenizer vocab")
    # Verify each tokenizes atomically (single id, not split)
    for _t in _sentinels:
        _ids = tokenizer.encode(_t, add_special_tokens=False)
        if len(_ids) != 1:
            logger.warning(f"  {_t} does not tokenize to a single id: {_ids} "
                           "-- sentinel may be split; check tokenizer.")
        else:
            logger.info(f"  {_t} tokenizes atomically to id {_ids[0]}")

    # --- Load model ---
    logger.info(f"Loading model from {args.model_name}...")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    try:
        # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("  Gradient checkpointing enabled")

    # Resize token embeddings if we added the [END_CELL] special token above. This adds a
    # fresh (randomly-initialized) embedding row for the new token so the model can learn it.
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"  Resized token embeddings to {len(tokenizer)} (for {_sentinels})")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,} total, {trainable:,} trainable")

    # --- Load data ---
    train_dataset = C2SDataset(args.train_file, tokenizer, max_length=args.max_length,
                               de_weight=args.de_weight)
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
        eval_dataset = C2SDataset(args.eval_file, tokenizer, max_length=args.max_length)

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,  # safe default; increase on HPC if IO-bound
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

    total_steps = (len(train_loader) * args.num_epochs) // args.grad_accum
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
    global_step = 0
    best_eval_loss = float("inf")
    log_interval = args.log_every
    save_interval = args.save_every

    if args.mode == "test":
        log_interval = 1
        save_interval = 999999  # don't save in test mode
        args.num_epochs = 1

    logger.info(f"\n{'='*60}")
    logger.info("Starting training")
    logger.info(f"{'='*60}")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
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

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_interval == 0:
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"  Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Tokens: {epoch_tokens:,}"
                    )

                if global_step % save_interval == 0 and global_step > 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"  Saved checkpoint to {ckpt_dir}")
                    if getattr(args, "keep_checkpoints", 0):
                        _prune_checkpoints(args.output_dir, args.keep_checkpoints)

            # Test mode: stop after a few steps
            if args.mode == "test" and step >= 5:
                logger.info("  Test mode: stopping after 6 steps")
                break

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

            avg_eval_loss = eval_loss / max(eval_tokens, 1)
            logger.info(f"  Eval loss: {avg_eval_loss:.4f}")

            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                best_dir = os.path.join(args.output_dir, "best")
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"  New best eval loss — saved to {best_dir}")

    # Save final model
    if args.mode != "test":
        final_dir = os.path.join(args.output_dir, "final")
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
