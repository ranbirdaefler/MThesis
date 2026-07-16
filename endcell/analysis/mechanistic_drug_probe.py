#!/usr/bin/env python
r"""
mechanistic_drug_probe.py
=========================
Does drug identity reach the model's representation, and if so, does it SURVIVE to the response
position through the layers? This probes activations (not weights — weights are input-independent
and cannot answer this). Forward passes only; no generation, no training. Fast.

Two prompt constructions (run both):
  * FIXED  : one cell line + one control cell, vary ONLY the drug (+MOA). Cleanest possible
             contrast — any activation difference is attributable to the drug text alone.
             Used for the clean PCA/UMAP separation plot.
  * REAL   : real eval prompts grouped by drug (varied cells/controls). Realistic; used for the
             per-layer linear-probe separability (can we DECODE drug from the activation?).

For each prompt we run a forward pass with output_hidden_states=True and take the residual-stream
activation at the LAST prompt position (where response generation begins) at EVERY layer.

Outputs, per model:
  * per-layer drug separability = accuracy of a logistic-regression probe classifying drug from the
    activation (cross-validated), plus a between/within variance ratio. The KEY curve: if
    separability DECAYS across layers, the model discards drug info as depth increases.
  * PCA (and UMAP if available) 2D coords of the FIXED-set activations at a few layers, for plots.
  * baseline: probe accuracy on SHUFFLED drug labels (the chance floor for the probe).

USAGE
-----
  python mechanistic_drug_probe.py \
     --eval_dir DATA_endcell_big --model_paths CKPT_endcell/final,CKPT_sft/checkpoint-10000 \
     --model_names endcell,original_sft \
     --tier tier2_unseen_drugs --n_drugs 12 --n_real_per_drug 40 \
     --out RESULTS/mechanistic_probe.json --bf16 --seed 42
"""
import argparse, json, os, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def control_from_prompt(prompt):
    marker = "Control cell:"
    i = prompt.find(marker)
    if i == -1:
        return ""
    rest = prompt[i + len(marker):]
    j = rest.find("\n")
    return (rest if j == -1 else rest[:j]).strip()


def build_fixed_prompts(examples, n_drugs, rng):
    """One cell line + one control, vary only the drug. Returns list of (prompt, drug)."""
    # group by cell line; pick the cell line with the most distinct drugs
    by_cl = defaultdict(list)
    for e in examples:
        m = e.get("metadata", {})
        by_cl[m.get("cell_line_id")].append(e)
    best_cl = max(by_cl, key=lambda cl: len({e["metadata"]["drug"] for e in by_cl[cl]}))
    cl_examples = by_cl[best_cl]
    # one fixed control cell (take the first example's control sentence + prompt template)
    template_ex = cl_examples[0]
    tp = template_ex["prompt"]
    # find the drug substring to replace: use metadata drug
    drugs = sorted({e["metadata"]["drug"] for e in cl_examples})[:n_drugs]
    # for each drug, take a representative example's prompt (keeps that drug's real MOA text)
    drug_to_prompt = {}
    for e in cl_examples:
        d = e["metadata"]["drug"]
        if d in drugs and d not in drug_to_prompt:
            drug_to_prompt[d] = e["prompt"]
    prompts = [(drug_to_prompt[d], d) for d in drugs if d in drug_to_prompt]
    logger.info(f"  FIXED set: cell line {best_cl}, {len(prompts)} drugs (one prompt each)")
    return prompts


def build_real_prompts(examples, n_drugs, n_per_drug, rng):
    """Real prompts grouped by drug (varied cells). Returns list of (prompt, drug)."""
    by_drug = defaultdict(list)
    for e in examples:
        by_drug[e["metadata"]["drug"]].append(e)
    drugs = [d for d, v in by_drug.items() if len(v) >= n_per_drug]
    rng.shuffle(drugs); drugs = drugs[:n_drugs]
    out = []
    for d in drugs:
        cells = by_drug[d]
        idx = rng.choice(len(cells), n_per_drug, replace=False)
        for i in idx:
            out.append((cells[i]["prompt"], d))
    logger.info(f"  REAL set: {len(drugs)} drugs x {n_per_drug} = {len(out)} prompts")
    return out


def extract_activations(model, tok, prompts, device, bf16):
    """Return activations[layer] = array (n_prompts, hidden) at the last prompt position."""
    import torch
    n_layers = model.config.num_hidden_layers
    acts = {L: [] for L in range(n_layers + 1)}  # +1 for embedding output
    for prompt, _ in prompts:
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple len n_layers+1, each (1, seq, hidden). Take last position.
        for L, hs in enumerate(out.hidden_states):
            acts[L].append(hs[0, -1, :].float().cpu().numpy())
    return {L: np.vstack(v) for L, v in acts.items()}


def probe_separability(acts_by_layer, labels, rng, n_splits=5):
    """Per-layer: cross-validated logistic-regression accuracy classifying drug from activation,
    plus a shuffled-label baseline, plus between/within variance ratio."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    y = np.array(labels)
    uniq = sorted(set(labels))
    y_idx = np.array([uniq.index(l) for l in labels])
    n_per_class = min(np.bincount(y_idx))
    cv = min(n_splits, n_per_class) if n_per_class >= 2 else 2
    results = {}
    y_shuf = y_idx.copy(); rng.shuffle(y_shuf)
    for L, X in acts_by_layer.items():
        if len(set(y_idx)) < 2 or X.shape[0] < 2 * cv:
            results[L] = None; continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        try:
            acc = float(np.mean(cross_val_score(clf, X, y_idx, cv=cv)))
            acc_shuf = float(np.mean(cross_val_score(clf, X, y_shuf, cv=cv)))
        except Exception as e:
            results[L] = None; continue
        # between/within variance ratio (Fisher-like, averaged over dims)
        overall = X.mean(0)
        sw, sb = 0.0, 0.0
        for c in set(y_idx):
            Xc = X[y_idx == c]
            sw += ((Xc - Xc.mean(0)) ** 2).sum()
            sb += Xc.shape[0] * ((Xc.mean(0) - overall) ** 2).sum()
        ratio = float(sb / sw) if sw > 0 else None
        results[L] = {"probe_acc": acc, "probe_acc_shuffled": acc_shuf,
                      "chance": 1.0 / len(uniq), "between_within_ratio": ratio}
    return results


def pca_coords(X, k=2):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k]).tolist(), (S[:k] ** 2 / (S ** 2).sum()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--model_paths", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--model_names", required=True, help="comma-separated names (match paths)")
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--n_drugs", type=int, default=12)
    ap.add_argument("--n_real_per_drug", type=int, default=40)
    ap.add_argument("--pca_layers", default="0,4,8,12,16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = np.random.RandomState(args.seed)
    path = os.path.join(args.eval_dir, f"eval_{args.tier}.jsonl")
    examples = [json.loads(l) for l in open(path)]
    logger.info(f"loaded {len(examples)} examples from {args.tier}")

    fixed_prompts = build_fixed_prompts(examples, args.n_drugs, rng)
    real_prompts = build_real_prompts(examples, args.n_drugs, args.n_real_per_drug, rng)
    pca_layers = [int(x) for x in args.pca_layers.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [p.strip() for p in args.model_paths.split(",")]
    names = [n.strip() for n in args.model_names.split(",")]

    result = {"tier": args.tier, "n_drugs": args.n_drugs, "models": {}}

    for name, mpath in zip(names, paths):
        logger.info(f"=== probing model '{name}' ({mpath}) ===")
        tok = AutoTokenizer.from_pretrained(mpath)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mpath, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
        model.eval()
        n_layers = model.config.num_hidden_layers
        logger.info(f"  {n_layers} layers, hidden {model.config.hidden_size}")

        # REAL set -> per-layer separability (the key curve)
        real_acts = extract_activations(model, tok, real_prompts, device, args.bf16)
        real_labels = [d for _, d in real_prompts]
        sep = probe_separability(real_acts, real_labels, rng)

        # FIXED set -> PCA coords for the clean plot
        fixed_acts = extract_activations(model, tok, fixed_prompts, device, args.bf16)
        fixed_labels = [d for _, d in fixed_prompts]
        pca = {}
        for L in pca_layers:
            if L in fixed_acts:
                coords, var = pca_coords(fixed_acts[L], 2)
                pca[L] = {"coords": coords, "labels": fixed_labels, "explained_var": var}

        # log the separability curve
        logger.info(f"  per-layer drug separability (probe acc | shuffled | chance={1.0/len(set(real_labels)):.3f}):")
        for L in sorted(sep.keys()):
            s = sep[L]
            if s:
                logger.info(f"    layer {L:2d}: acc={s['probe_acc']:.3f}  shuf={s['probe_acc_shuffled']:.3f}  "
                            f"b/w_ratio={s['between_within_ratio']:.3f}" if s['between_within_ratio'] else
                            f"    layer {L:2d}: acc={s['probe_acc']:.3f}")

        result["models"][name] = {"n_layers": n_layers, "separability": sep, "pca": pca,
                                   "n_real": len(real_prompts), "n_fixed": len(fixed_prompts)}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
