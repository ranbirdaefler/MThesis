"""
GRPO (Group Relative Policy Optimization) for C2S-Scale

Applies pathway-targeted reinforcement learning after SFT to improve
prediction accuracy on specific gene programs.

Following C2S-Scale paper Section 3.3:
  - For each prompt, generate K candidate cell sentences
  - Compute reward = Kendall τ on target pathway genes
  - Normalize rewards across group (zero-mean, unit-variance)
  - Policy gradient update with KL penalty against reference model

Usage:
    # Local test (CPU, synthetic data)
    python grpo_c2s_tahoe.py --mode test \
        --model_path ./checkpoints/final \
        --train_file ./tahoe_c2s_data/train.jsonl

    # HPC run (target weakest pathway from eval)
    python grpo_c2s_tahoe.py --mode full \
        --model_path ./checkpoints/final \
        --train_file ./data/train.jsonl \
        --output_dir ./checkpoints_grpo \
        --target_pathway apoptosis \
        --num_candidates 8 --num_epochs 2 \
        --bf16 --gradient_checkpointing --max_length 4096
"""

import argparse
import copy
import json
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats as scipy_stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Pathway gene sets (same as evaluation script)
# =============================================================================

PATHWAY_GENE_SETS = {
    "apoptosis": [
        "CASP3", "CASP8", "CASP9", "BAX", "BCL2", "BCL2L1", "BID", "CYCS",
        "FADD", "FAS", "TNFRSF10B", "BIRC5", "MCL1", "DIABLO", "APAF1",
        "BAK1", "BBC3", "PMAIP1", "BCL2L11", "CASP7", "CASP6", "CASP10",
        "XIAP", "BIRC2", "BIRC3", "CFLAR", "TNFSF10", "TRADD", "RIPK1",
        "CYLD", "DFFA", "DFFB", "ENDOG", "AIFM1", "HTRA2", "LMNA",
        "PARP1", "ACIN1", "SATB1", "ROCK1", "PAK2",
    ],
    "mapk_signaling": [
        "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "BRAF", "RAF1", "KRAS",
        "HRAS", "NRAS", "SOS1", "GRB2", "EGFR", "ERBB2", "FGFR1",
        "FOS", "JUN", "JUNB", "JUND", "ELK1", "MYC", "DUSP1", "DUSP6",
        "SPRY2", "SPRY4", "ETS1", "ETS2", "MAPK8", "MAPK9", "MAPK14",
        "MAP3K1", "MAP3K5", "MAP3K7", "MKNK1", "RPS6KA1", "RPS6KA3",
    ],
    "pi3k_akt": [
        "PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "AKT3", "MTOR",
        "PTEN", "TSC1", "TSC2", "RPTOR", "RICTOR", "RPS6KB1", "EIF4EBP1",
        "GSK3B", "FOXO1", "FOXO3", "BAD", "CDKN1A", "CDKN1B", "MDM2",
        "RPS6", "EIF4E", "PDK1", "SGK1", "INPP5D", "PIK3CD",
    ],
    "cell_cycle": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1",
        "CCNE1", "RB1", "E2F1", "E2F2", "TP53", "CDKN1A", "CDKN2A",
        "CDKN1B", "MYC", "CDC20", "CDC25A", "CDC25C", "PLK1", "AURKA",
        "AURKB", "BUB1", "MAD2L1", "CHEK1", "CHEK2", "ATM", "ATR",
        "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "ORC1",
        "PCNA", "TOP2A", "MKI67",
    ],
    "interferon_response": [
        "STAT1", "STAT2", "IRF1", "IRF7", "IRF9", "MX1", "MX2", "OAS1",
        "OAS2", "OAS3", "ISG15", "ISG20", "IFIT1", "IFIT2", "IFIT3",
        "IFI6", "IFI27", "IFI35", "IFI44", "IFI44L", "IFITM1", "IFITM2",
        "IFITM3", "BST2", "RSAD2", "DDX58", "HERC5", "USP18", "TRIM22",
        "GBP1", "GBP2", "CXCL10", "CXCL11",
    ],
}


# =============================================================================
# Reward: Kendall τ on pathway genes (C2S-Scale paper approach)
# =============================================================================

def compute_pathway_reward(predicted_sentence, true_sentence, pathway_genes):
    """
    Compute Kendall τ between predicted and true rank orderings,
    restricted to genes in the target pathway.

    Following C2S-Scale paper: genes not present in the output are assigned
    the worst possible rank (max_rank + 1). This ensures τ is always
    computable over ALL pathway genes, not just the intersection.

    Returns float in [-1, 1], or 0.0 if pathway has < 3 genes.
    """
    pred_genes = predicted_sentence.strip().split()
    true_genes = true_sentence.strip().split()

    # Build rank dicts (first occurrence = highest rank)
    pred_ranks = {}
    for rank, g in enumerate(pred_genes, 1):
        if g not in pred_ranks:
            pred_ranks[g] = rank
    true_ranks = {}
    for rank, g in enumerate(true_genes, 1):
        if g not in true_ranks:
            true_ranks[g] = rank

    # Worst rank for missing genes (following C2S-Scale paper)
    pred_worst = len(pred_genes) + 1
    true_worst = len(true_genes) + 1

    # Compute ranks for ALL pathway genes, assigning worst rank if missing
    pred_vals = []
    true_vals = []
    for g in pathway_genes:
        pred_vals.append(pred_ranks.get(g, pred_worst))
        true_vals.append(true_ranks.get(g, true_worst))

    if len(pathway_genes) < 3:
        return 0.0

    tau, _ = scipy_stats.kendalltau(pred_vals, true_vals)
    if np.isnan(tau):
        return 0.0
    return float(tau)


def compute_overall_reward(predicted_sentence, true_sentence):
    """
    Compute Kendall τ on ALL overlapping genes between predicted and true.
    Used as a secondary reward to prevent catastrophic forgetting.
    """
    pred_genes = predicted_sentence.strip().split()
    true_genes = true_sentence.strip().split()

    pred_ranks = {}
    for rank, g in enumerate(pred_genes, 1):
        if g not in pred_ranks:
            pred_ranks[g] = rank
    true_ranks = {}
    for rank, g in enumerate(true_genes, 1):
        if g not in true_ranks:
            true_ranks[g] = rank

    common = set(pred_ranks.keys()) & set(true_ranks.keys())
    if len(common) < 5:
        return 0.0

    pred_vals = [pred_ranks[g] for g in common]
    true_vals = [true_ranks[g] for g in common]

    tau, _ = scipy_stats.kendalltau(pred_vals, true_vals)
    if np.isnan(tau):
        return 0.0
    return float(tau)


# =============================================================================
# Generation with log-probs
# =============================================================================

def generate_candidates_with_logprobs(model, tokenizer, prompt, num_candidates,
                                       max_new_tokens, device, temperature=0.8):
    """
    Generate K candidate responses and compute their per-token log-probabilities
    under the model.

    Returns:
        candidates: list of str (decoded text)
        candidate_ids: list of tensor (token ids, response portion only)
        log_probs: list of tensor (per-token log prob for each candidate)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    candidates = []
    candidate_ids_list = []
    log_probs_list = []

    for _ in range(num_candidates):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract response tokens
        full_ids = outputs.sequences[0]
        response_ids = full_ids[prompt_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        candidates.append(text)
        candidate_ids_list.append(response_ids)

        # Compute log-probs from scores
        # scores is a tuple of (vocab_size,) tensors, one per generated token
        token_log_probs = []
        for t, score in enumerate(outputs.scores):
            log_prob = F.log_softmax(score[0] / temperature, dim=-1)
            token_id = response_ids[t]
            token_log_probs.append(log_prob[token_id].item())
        log_probs_list.append(torch.tensor(token_log_probs))

    return candidates, candidate_ids_list, log_probs_list


def compute_log_probs_for_sequence(model, tokenizer, prompt, response_ids, device):
    """
    Compute per-token log-probabilities for a given response under the model.
    Used to get reference model log-probs.
    """
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_ids = torch.cat([prompt_ids[0], response_ids.to(device)]).unsqueeze(0)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[0]  # (seq_len, vocab)

    # Get log-probs for response tokens
    prompt_len = prompt_ids.shape[1]
    response_logits = logits[prompt_len - 1:-1]  # predict next token
    log_probs = F.log_softmax(response_logits, dim=-1)

    token_log_probs = []
    for t, token_id in enumerate(response_ids):
        if t < len(log_probs):
            token_log_probs.append(log_probs[t, token_id].item())
    return torch.tensor(token_log_probs)


# =============================================================================
# GRPO update
# =============================================================================

def grpo_step(model, ref_model, tokenizer, prompt, true_response,
              pathway_genes, num_candidates, max_new_tokens, device,
              kl_coeff=0.05, temperature=0.8, pathway_weight=0.7):
    """
    One GRPO step:
      1. Generate K candidates from the current policy
      2. Compute combined reward: pathway_weight * pathway_τ + (1-pathway_weight) * overall_τ
      3. Normalize rewards across group (zero-mean, unit-variance)
      4. Compute policy gradient with KL penalty

    Returns:
        loss: scalar tensor (for backward)
        mean_reward: float (combined)
        best_reward: float (combined)
        mean_pathway_reward: float (pathway-only, for logging)
    """
    # Step 1: Generate candidates
    candidates, candidate_ids_list, policy_log_probs = generate_candidates_with_logprobs(
        model, tokenizer, prompt, num_candidates, max_new_tokens, device, temperature
    )

    # Step 2: Compute combined rewards
    rewards = []
    pathway_rewards = []
    for cand in candidates:
        r_pathway = compute_pathway_reward(cand, true_response, pathway_genes)
        r_overall = compute_overall_reward(cand, true_response)
        r_combined = pathway_weight * r_pathway + (1 - pathway_weight) * r_overall
        rewards.append(r_combined)
        pathway_rewards.append(r_pathway)
    rewards = np.array(rewards)
    pathway_rewards = np.array(pathway_rewards)

    # Step 3: Group-relative normalization
    if rewards.std() > 1e-8:
        normalized_rewards = (rewards - rewards.mean()) / rewards.std()
    else:
        normalized_rewards = np.zeros_like(rewards)

    # Step 4: Compute loss
    # For each candidate: loss = -normalized_reward * sum(log_prob) + kl_coeff * KL
    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    n_valid = 0

    for k in range(num_candidates):
        if len(candidate_ids_list[k]) == 0:
            continue

        # Re-compute log-probs with gradient (the generation was no_grad)
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        resp_ids = candidate_ids_list[k].to(device)
        full_ids = torch.cat([prompt_ids[0], resp_ids]).unsqueeze(0)

        outputs = model(full_ids)
        logits = outputs.logits[0]
        prompt_len = prompt_ids.shape[1]
        response_logits = logits[prompt_len - 1:-1]

        if len(response_logits) == 0:
            continue

        log_probs_policy = F.log_softmax(response_logits, dim=-1)
        token_lps = []
        for t, tid in enumerate(resp_ids):
            if t < len(log_probs_policy):
                token_lps.append(log_probs_policy[t, tid])

        if not token_lps:
            continue

        policy_lp = torch.stack(token_lps).sum()

        # Reference model log-probs (no gradient)
        ref_lp = compute_log_probs_for_sequence(
            ref_model, tokenizer, prompt, resp_ids, device
        ).sum()

        # KL divergence (approx): policy_lp - ref_lp
        kl = (policy_lp - ref_lp)

        # GRPO loss for this candidate
        candidate_loss = -(normalized_rewards[k] * policy_lp) + kl_coeff * kl
        total_loss = total_loss + candidate_loss
        n_valid += 1

    if n_valid > 0:
        total_loss = total_loss / n_valid

    return total_loss, float(rewards.mean()), float(rewards.max()), float(pathway_rewards.mean())


# =============================================================================
# Dataset (reuse prompts from training data)
# =============================================================================

class GRPODataset(Dataset):
    def __init__(self, filepath, max_examples=None):
        self.examples = []
        with open(filepath) as f:
            for line in f:
                self.examples.append(json.loads(line.strip()))
                if max_examples and len(self.examples) >= max_examples:
                    break
        logger.info(f"GRPO dataset: {len(self.examples)} examples from {filepath}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# =============================================================================
# Main training loop
# =============================================================================

def run_grpo(args):
    # --- Device ---
    if args.mode == "test":
        device = torch.device("cpu")
        args.bf16 = False
        logger.info("Test mode: CPU, bf16 disabled")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Pathway ---
    if args.target_pathway not in PATHWAY_GENE_SETS:
        logger.error(f"Unknown pathway: {args.target_pathway}")
        logger.error(f"Available: {list(PATHWAY_GENE_SETS.keys())}")
        return
    pathway_genes = PATHWAY_GENE_SETS[args.target_pathway]
    logger.info(f"Target pathway: {args.target_pathway} ({len(pathway_genes)} genes)")

    # --- Load models ---
    logger.info(f"Loading policy model from {args.model_path}...")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Reference model (frozen copy)
    logger.info("Creating frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    # --- Data ---
    max_examples = 20 if args.mode == "test" else args.max_train_examples
    dataset = GRPODataset(args.train_file, max_examples=max_examples)

    # In test mode, very small
    if args.mode == "test":
        dataset.examples = dataset.examples[:5]
        args.num_candidates = 2
        args.num_epochs = 1
        args.max_new_tokens = 50  # very short for CPU test

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Training ---
    logger.info(f"\n{'='*60}")
    logger.info(f"GRPO Training — {args.target_pathway}")
    logger.info(f"  Candidates per prompt: {args.num_candidates}")
    logger.info(f"  KL coefficient: {args.kl_coeff}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Examples: {len(dataset)}")
    logger.info(f"{'='*60}")

    reward_history = []

    for epoch in range(args.num_epochs):
        model.train()
        epoch_rewards = []
        epoch_best_rewards = []
        epoch_pathway_rewards = []

        indices = np.random.permutation(len(dataset))

        for step_i, idx in enumerate(indices):
            ex = dataset[idx]
            prompt = ex["prompt"]
            true_response = ex["response"]

            loss, mean_r, best_r, mean_pathway_r = grpo_step(
                model, ref_model, tokenizer, prompt, true_response,
                pathway_genes, args.num_candidates, args.max_new_tokens,
                device, kl_coeff=args.kl_coeff,
                pathway_weight=args.pathway_weight,
            )

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_rewards.append(mean_r)
            epoch_best_rewards.append(best_r)
            epoch_pathway_rewards.append(mean_pathway_r)

            if (step_i + 1) % max(1, len(dataset) // 10) == 0:
                recent_r = np.mean(epoch_rewards[-20:])
                recent_best = np.mean(epoch_best_rewards[-20:])
                recent_pathway = np.mean(epoch_pathway_rewards[-20:])
                logger.info(
                    f"  Epoch {epoch+1} | {step_i+1}/{len(dataset)} | "
                    f"Combined τ: {recent_r:.4f} | Pathway τ: {recent_pathway:.4f} | "
                    f"Best: {recent_best:.4f} | Loss: {loss.item():.4f}"
                )

            if args.mode == "test" and step_i >= 2:
                logger.info("  Test mode: stopping after 3 GRPO steps")
                break

        # Epoch summary
        mean_epoch_r = np.mean(epoch_rewards)
        mean_epoch_best = np.mean(epoch_best_rewards)
        mean_epoch_pathway = np.mean(epoch_pathway_rewards)
        logger.info(
            f"\n  Epoch {epoch+1} complete | "
            f"Combined τ: {mean_epoch_r:.4f} | Pathway τ: {mean_epoch_pathway:.4f} | "
            f"Mean best: {mean_epoch_best:.4f}"
        )
        reward_history.append({
            "epoch": epoch + 1,
            "mean_reward": mean_epoch_r,
            "mean_best_reward": mean_epoch_best,
            "mean_pathway_reward": mean_epoch_pathway,
        })

    # Save model
    if args.mode != "test":
        final_dir = os.path.join(args.output_dir, "final")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"\nSaved GRPO model to {final_dir}")

        # Save training log
        log_path = os.path.join(args.output_dir, "grpo_training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "target_pathway": args.target_pathway,
                "num_candidates": args.num_candidates,
                "kl_coeff": args.kl_coeff,
                "epochs": reward_history,
            }, f, indent=2)
        logger.info(f"Saved training log to {log_path}")

    logger.info("GRPO complete!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO for C2S-Scale pathway optimization")
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SFT checkpoint (starting point for GRPO)")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training JSONL from preprocessing")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_grpo")
    parser.add_argument("--target_pathway", type=str, default="apoptosis",
                        choices=list(PATHWAY_GENE_SETS.keys()),
                        help="Which pathway to optimize (pick weakest from eval)")
    parser.add_argument("--num_candidates", type=int, default=8,
                        help="K candidates per prompt (paper uses 16, we use 8 for memory)")
    parser.add_argument("--max_new_tokens", type=int, default=2000,
                        help="Max tokens to generate per candidate")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Lower LR than SFT to avoid catastrophic forgetting")
    parser.add_argument("--kl_coeff", type=float, default=0.05,
                        help="KL penalty coefficient (higher = stay closer to SFT)")
    parser.add_argument("--pathway_weight", type=float, default=0.7,
                        help="Weight for pathway reward vs overall reward (0.7 = 70%% pathway, 30%% overall)")
    parser.add_argument("--max_train_examples", type=int, default=2000,
                        help="Subset of training data for GRPO (doesn't need all)")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=4096)

    args = parser.parse_args()
    run_grpo(args)


if __name__ == "__main__":
    main()
