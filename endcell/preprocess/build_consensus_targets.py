#!/usr/bin/env python
r"""
build_consensus_targets.py  —  Arm 1a: the denoised-target objective fix
========================================================================
Transforms an existing [END_CELL] train.jsonl into a *consensus-target* version WITHOUT
re-streaming Tahoe. It groups the training cells you already have by biological condition
(drug x cell line x dose) and replaces each cell's response with the condition's denoised
"consensus cell" — a pseudobulk-derived typical response.

WHY (the diagnosis -> the cure)
-------------------------------
We showed the model is "read but not used": the drug is decodable from its activations, yet its
generations are drug-blind, and NIR sits at chance (~0.5) while a real replicate discriminates at
~0.88. The most likely mechanism: at SINGLE-CELL resolution the drug barely moves the target, so the
next-token (MLE) objective gets almost no gradient from drug-specific genes — it is dominated by the
generic response + dropout noise. If instead the TARGET is the denoised condition consensus, the
drug-specific genes dominate the target and cross-entropy finally rewards capturing them.

This is the cleanest test of the objective-side hypothesis: same architecture, same inputs, only the
target is denoised. If NIR moves off chance, the signal was always usable and the objective was the
bottleneck.

WHAT IT DOES
------------
- Groups examples by --group_keys (default: drug, cell_line_id, dose_float).
- For each group with >= --min_cells cells, builds a consensus response sentence:
    * per gene, a length-normalized expression proxy averaged across the group's cells
      (proxy for a gene at rank r in a cell of length L is 1 - (r-1)/L; absent genes contribute 0),
    * keep genes detected in >= --min_detect_frac of the group's cells,
    * order by mean proxy (desc) and keep the top N, where N = median expressed-genes-per-cell in the
      group (so the consensus "looks like" a real cell), then append the [END_CELL] sentinel.
- Emits one output example per input example (default), keeping the ORIGINAL prompt/control (so input
  diversity is preserved) and swapping only the response to the consensus. Singleton conditions
  (< --min_cells) pass through UNCHANGED.
- Prints and saves a diagnostics report: cells-per-condition distribution and how much of the data
  is actually denoised. This directly answers "do we need to stream more cells?" — if most conditions
  are singletons, consensus is a near-no-op and more cells per condition are needed.

USAGE
  python build_consensus_targets.py --train_file DATA/train.jsonl \
     --out DATA/train_consensus.jsonl --report RESULTS/consensus_report.json

SELFTEST (no data/network) — verifies the consensus recovers a condition's true top genes better
than any individual noisy cell does:
  python build_consensus_targets.py --selftest
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"


def parse_genes(response):
    """[END_CELL] response string -> ordered list of gene symbols (sentinel stripped)."""
    toks = []
    for t in response.strip().split():
        if t == SENTINEL:
            break
        toks.append(t)
    return toks


def build_consensus(sentences, min_detect_frac, min_detect_count):
    """List of response strings (one per cell) -> a single consensus response string.

    Length-normalized rank proxy so cells of different lengths contribute comparably; a gene must be
    detected in >= max(min_detect_count, min_detect_frac * n_cells) cells to enter the consensus."""
    n = len(sentences)
    sum_proxy = defaultdict(float)
    det = defaultdict(int)
    lengths = []
    for resp in sentences:
        genes = parse_genes(resp)
        L = len(genes)
        if L == 0:
            continue
        lengths.append(L)
        seen = set()
        for r, g in enumerate(genes, 1):          # r=1 is highest-expressed
            if g in seen:
                continue
            seen.add(g)
            sum_proxy[g] += 1.0 - (r - 1) / L      # in (0, 1]
            det[g] += 1
    if not lengths:
        return None
    thresh = max(min_detect_count, int(round(min_detect_frac * n)))
    cand = [(g, sum_proxy[g] / n) for g in sum_proxy if det[g] >= thresh]
    if not cand:
        # fall back: no gene clears the detection floor (very noisy) -> use union ordered by proxy
        cand = [(g, sum_proxy[g] / n) for g in sum_proxy]
    cand.sort(key=lambda x: -x[1])
    N = int(round(np.median(lengths)))
    N = max(1, min(N, len(cand)))
    consensus_genes = [g for g, _ in cand[:N]]
    return " ".join(consensus_genes) + " " + SENTINEL


def group_key(meta, keys):
    parts = []
    for k in keys:
        v = meta.get(k)
        if k == "dose_float" and v is not None:
            v = round(float(v), 6)
        parts.append(str(v))
    return tuple(parts)


def run(args):
    logger.info(f"Loading {args.train_file} ...")
    examples = []
    with open(args.train_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info(f"  {len(examples):,} training examples")

    keys = [k.strip() for k in args.group_keys.split(",") if k.strip()]
    groups = defaultdict(list)                     # key -> [indices]
    for i, ex in enumerate(examples):
        groups[group_key(ex.get("metadata", {}), keys)].append(i)

    sizes = np.array([len(v) for v in groups.values()])
    denoisable = sizes[sizes >= args.min_cells]
    n_denoise_conditions = int((sizes >= args.min_cells).sum())
    n_denoise_cells = int(denoisable.sum()) if len(denoisable) else 0

    logger.info("")
    logger.info("=" * 78)
    logger.info(f"  Grouping by {keys}")
    logger.info(f"  conditions: {len(groups):,}   examples: {len(examples):,}")
    logger.info(f"  cells/condition: min {sizes.min()}  p25 {np.percentile(sizes,25):.0f}  "
                f"median {np.median(sizes):.0f}  mean {sizes.mean():.1f}  "
                f"p90 {np.percentile(sizes,90):.0f}  max {sizes.max()}")
    logger.info(f"  singletons (<{args.min_cells} cells): {int((sizes < args.min_cells).sum()):,} "
                f"conditions ({100.0*(sizes<args.min_cells).mean():.1f}%)")
    logger.info(f"  DENOISABLE: {n_denoise_conditions:,} conditions / {n_denoise_cells:,} cells "
                f"({100.0*n_denoise_cells/max(1,len(examples)):.1f}% of examples)")
    if n_denoise_cells / max(1, len(examples)) < 0.5:
        logger.warning("  <50% of examples are denoisable -> consensus is close to a no-op here; "
                       "consider re-streaming with a higher --cells_per_condition before relying on this.")
    logger.info("=" * 78)

    # build consensus per denoisable condition
    consensus_by_group = {}
    out_lengths, in_lengths = [], []
    for k, idxs in groups.items():
        if len(idxs) < args.min_cells:
            continue
        sents = [examples[i]["response"] for i in idxs]
        cons = build_consensus(sents, args.min_detect_frac, args.min_detect_count)
        if cons is not None:
            consensus_by_group[k] = cons
            out_lengths.append(len(parse_genes(cons)))
            in_lengths.extend(len(parse_genes(s)) for s in sents)

    # emit
    n_denoised, n_passthrough = 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fo:
        if args.emit == "per_condition":
            for k, idxs in groups.items():
                if k in consensus_by_group:
                    ex = dict(examples[idxs[0]])   # representative prompt/control
                    ex["response"] = consensus_by_group[k]
                    ex.setdefault("metadata", {})["consensus_n_cells"] = len(idxs)
                    fo.write(json.dumps(ex) + "\n"); n_denoised += 1
                else:
                    for i in idxs:
                        fo.write(json.dumps(examples[i]) + "\n"); n_passthrough += 1
        else:  # per_cell (default): keep every prompt/control, swap only the target
            for i, ex in enumerate(examples):
                k = group_key(ex.get("metadata", {}), keys)
                if k in consensus_by_group:
                    ex = dict(ex)
                    ex["response"] = consensus_by_group[k]
                    ex.setdefault("metadata", {})["consensus_n_cells"] = len(groups[k])
                    n_denoised += 1
                else:
                    n_passthrough += 1
                fo.write(json.dumps(ex) + "\n")

    logger.info(f"  wrote {n_denoised + n_passthrough:,} examples -> {args.out}")
    logger.info(f"    denoised: {n_denoised:,}   passthrough (singletons): {n_passthrough:,}")
    if out_lengths:
        logger.info(f"    consensus length: mean {np.mean(out_lengths):.0f}  "
                    f"(vs single-cell mean {np.mean(in_lengths):.0f})")

    report = {
        "train_file": args.train_file, "out": args.out, "group_keys": keys,
        "n_examples": len(examples), "n_conditions": len(groups),
        "cells_per_condition": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                                "mean": float(sizes.mean()), "p90": float(np.percentile(sizes, 90)),
                                "max": int(sizes.max())},
        "n_denoise_conditions": n_denoise_conditions,
        "frac_examples_denoised": float(n_denoised / max(1, len(examples))),
        "emit": args.emit, "min_cells": args.min_cells,
        "consensus_len_mean": float(np.mean(out_lengths)) if out_lengths else None,
        "single_cell_len_mean": float(np.mean(in_lengths)) if in_lengths else None,
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        json.dump(report, open(args.report, "w"), indent=2)
        logger.info(f"  report -> {args.report}")


def selftest(args):
    """Synthetic: each condition has a TRUE gene ranking; we generate noisy cells (dropout + rank
    jitter) and check the consensus recovers the true top-K better than the average single cell,
    AND that consensus profiles stay drug-discriminative (own condition closest)."""
    rng = np.random.RandomState(0)
    P, n_cells, K = 200, 15, 30
    panel = [f"G{i}" for i in range(P)]

    def true_ranking(seed):
        r = np.random.RandomState(seed)
        return list(r.choice(P, 90, replace=False))         # 90 expressed genes, ordered

    def noisy_cell(true_order):
        # dropout ~40%, then jitter the order
        kept = [g for g in true_order if rng.rand() > 0.4]
        jit = np.arange(len(kept)) + rng.randn(len(kept)) * 3.0
        order = [kept[i] for i in np.argsort(jit)]
        return " ".join(panel[g] for g in order) + " " + SENTINEL

    conds = {c: true_ranking(100 + c) for c in range(6)}
    cons = {}
    single_ov, cons_ov = [], []
    for c, order in conds.items():
        cells = [noisy_cell(order) for _ in range(n_cells)]
        consensus = build_consensus(cells, args.min_detect_frac, args.min_detect_count)
        cons[c] = set(parse_genes(consensus)[:K])
        truth = set(panel[g] for g in order[:K])
        cons_ov.append(len(cons[c] & truth) / K)
        for cell in cells:
            single_ov.append(len(set(parse_genes(cell)[:K]) & truth) / K)

    # discrimination: each consensus should overlap its own truth more than other conditions' truths
    correct = 0
    for c, order in conds.items():
        own = len(cons[c] & set(panel[g] for g in order[:K]))
        others = [len(cons[c] & set(panel[g] for g in conds[c2][:K])) for c2 in conds if c2 != c]
        correct += int(own > max(others))

    m_cons, m_single = float(np.mean(cons_ov)), float(np.mean(single_ov))
    logger.info(f"  consensus top-{K} overlap with truth: {m_cons:.3f}")
    logger.info(f"  mean single-cell top-{K} overlap:     {m_single:.3f}")
    logger.info(f"  consensus own-condition discrimination: {correct}/{len(conds)}")
    ok = (m_cons > m_single + 0.08) and (correct == len(conds))
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file")
    ap.add_argument("--out")
    ap.add_argument("--report", default=None)
    ap.add_argument("--group_keys", default="drug,cell_line_id,dose_float",
                    help="metadata fields defining a biological condition (comma-separated)")
    ap.add_argument("--min_cells", type=int, default=2,
                    help="conditions with fewer cells pass through unchanged (no denoising possible)")
    ap.add_argument("--min_detect_frac", type=float, default=0.1,
                    help="a gene must be expressed in >= this fraction of a condition's cells")
    ap.add_argument("--min_detect_count", type=int, default=2,
                    help="...and in >= this many cells (absolute floor)")
    ap.add_argument("--emit", choices=["per_cell", "per_condition"], default="per_cell",
                    help="per_cell keeps every prompt/control (denoise target only); "
                         "per_condition emits one example per condition")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return
    if not args.train_file or not args.out:
        ap.error("--train_file and --out are required (unless --selftest)")
    run(args)


if __name__ == "__main__":
    main()
