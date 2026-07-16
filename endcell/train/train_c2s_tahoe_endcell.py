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
    def __init__(self, filepath, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
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

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


# =============================================================================
# Training loop
# =============================================================================

def train(args):
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

    # --- Register the [END_CELL] sentinel as an atomic special token ---
    # The [END_CELL] data format terminates every response with this marker. Without
    # registering it, the tokenizer splits it into subword pieces ('[', 'END', '_CELL', ']')
    # and the model never sees a clean end-of-cell signal. add_special_tokens returns the
    # number of NEW tokens added (0 if already present); we resize embeddings only if >0.
    added = tokenizer.add_special_tokens({"additional_special_tokens": ["[END_CELL]"]})
    if added:
        logger.info(f"  Added {added} special token(s): [END_CELL] -> id "
                    f"{tokenizer.convert_tokens_to_ids('[END_CELL]')}")
    else:
        logger.info("  [END_CELL] already in tokenizer vocab")
    # Verify it tokenizes atomically (single id, not split)
    _ec_ids = tokenizer.encode("[END_CELL]", add_special_tokens=False)
    if len(_ec_ids) != 1:
        logger.warning(f"  [END_CELL] does not tokenize to a single id: {_ec_ids} "
                       "-- sentinel may be split; check tokenizer.")
    else:
        logger.info(f"  [END_CELL] tokenizes atomically to id {_ec_ids[0]}")

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
        logger.info(f"  Resized token embeddings to {len(tokenizer)} (for [END_CELL])")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,} total, {trainable:,} trainable")

    # --- Load data ---
    train_dataset = C2SDataset(args.train_file, tokenizer, max_length=args.max_length)
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

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / args.grad_accum
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

            loss.backward()

            # Track tokens where loss is computed
            n_tokens = (labels != -100).sum().item()
            epoch_loss += outputs.loss.item() * n_tokens
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

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
