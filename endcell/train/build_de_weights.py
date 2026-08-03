#!/usr/bin/env python
"""
Annotate a STANDARD cell-sentence training file with each condition's drug-specific DE gene set, so the
trainer can up-weight the token-level loss on exactly the genes that distinguish this drug from the
average drug.

WHY THIS ARM EXISTS
-------------------
Q15 diagnosed the failure as TOKEN DILUTION: only ~35 of 200 target tokens differ between two drugs, so
drug identity receives ~1% of the cross-entropy gradient and every target-CONTENT change (consensus
targets, OT targets, the whole epsilon ladder) failed identically. The residual-target arm fixed the
dilution by changing the TARGET (rank genes by residual -> 120/200 tokens differ) and produced the first
non-null model - scramble gap. But it also changed the output format, so those numbers are no longer
comparable to any prior tier result and the model can no longer emit a full profile.

This arm attacks the SAME diagnosis with a different lever: keep the standard target and standard output
format -- so every prior tier number stays comparable -- and re-weight the LOSS so the drug-discriminative
tokens dominate the gradient. It is a COMPETING HYPOTHESIS to GRPO, not a warm start for it: if
re-weighting alone moves model - scramble, the reward machinery is unnecessary.

WHAT COUNTS AS "DE"
-------------------
Per (drug, cell_line), the mean residual over that pair's conditions:
    residual = (treated - control) - mean_over_drugs(treated - control)
The generic program is subtracted by construction, so the top-k up / bottom-k down genes of the residual
are the genes this drug moves DIFFERENTLY from the average drug -- not merely the genes it moves. That
distinction is the whole point: the unweighted loss already rewards the generic program handsomely.

Keyed on (drug, cell_line) rather than the residual builder's 4-tuple (drug, cell_line, plate, dose)
deliberately: averaging over plates/doses denoises the gene SET (which is all we need to steer a weight),
and it avoids depending on plate/dose string formatting matching between the cache metadata and the
training JSONL.

Usage
-----
  python build_de_weights.py --cache_dir ot_cache \
      --in_file  data_diverse2_endcell_big/train.jsonl \
      --out_file data_diverse2_endcell_big/train_de.jsonl --k_sig 100
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--k_sig", type=int, default=100,
                    help="top-k UP and top-k DOWN residual genes per condition")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--truth_repro_thr", type=float, default=0.2,
                    help="same reliability filter as every other residual arm: conditions whose two "
                         "half-splits disagree have no measurable drug effect to up-weight")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.dirname(here) + "/ot")
    import build_residual_targets as brt

    panel = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    kept, _generic, _ctrl_rows, _X, _meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control, args.truth_repro_thr, seed=args.seed)

    # ---- (drug, cell_line) -> mean residual -> DE gene set ------------------------------------------
    by_pair = defaultdict(list)
    for k, v in kept.items():
        by_pair[(k[0], k[1])].append(v["residual"])
    by_drug = defaultdict(list)
    for (d, _c), rs in by_pair.items():
        by_drug[d].extend(rs)

    def de_set(res):
        up = [j for j in np.argsort(-res)[:args.k_sig].tolist() if res[j] > 0]
        dn = [j for j in np.argsort(res)[:args.k_sig].tolist() if res[j] < 0]
        return [panel[j] for j in up + dn]

    de_pair = {p: de_set(np.mean(np.stack(rs), 0)) for p, rs in by_pair.items()}
    de_drug = {d: de_set(np.mean(np.stack(rs), 0)) for d, rs in by_drug.items()}
    logger.info(f"DE sets: {len(de_pair)} (drug, cell_line) pairs | {len(de_drug)} drug fallbacks | "
                f"mean set size {np.mean([len(v) for v in de_pair.values()]):.0f} genes")

    # ---- annotate ------------------------------------------------------------------------------------
    n, hit_pair, hit_drug, miss = 0, 0, 0, 0
    cov = []                                    # fraction of response GENES that are DE genes
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.in_file) as fi, open(args.out_file, "w") as fo:
        for line in fi:
            ex = json.loads(line)
            md = ex.get("metadata", {})
            d, c = str(md.get("drug")), str(md.get("cell_line_id"))
            genes = de_pair.get((d, c))
            if genes is not None:
                hit_pair += 1
            else:
                genes = de_drug.get(d)
                if genes is not None:
                    hit_drug += 1
                else:
                    # No measurable drug-specific residual for this drug anywhere (failed the
                    # reliability filter). An empty set means the trainer weights this example
                    # uniformly -- it degrades to ordinary SFT rather than inventing a target.
                    genes = []
                    miss += 1
            ex["de_genes"] = genes
            resp = ex.get("response", "").split()
            if resp:
                s = set(genes)
                cov.append(sum(1 for t in resp if t in s) / len(resp))
            fo.write(json.dumps(ex) + "\n")
            n += 1
            if n % 20000 == 0:
                logger.info(f"  {n} examples")

    f = float(np.mean(cov)) if cov else 0.0
    logger.info("=" * 90)
    logger.info(f"wrote {n} examples -> {args.out_file}")
    logger.info(f"  matched: {hit_pair} (drug, cell_line) | {hit_drug} drug-level fallback | "
                f"{miss} no DE set (uniform weighting)")
    logger.info(f"  DE token share f = {100*f:.1f}% of response genes")
    # The weight that hands the DE genes a target share of the gradient. Share(w) = wf / (wf + 1 - f),
    # so w = share/(1-share) * (1-f)/f. Reported rather than assumed: f depends on k_sig and on how
    # many panel genes actually appear in a sentence, and guessing it wrong is the difference between
    # a real intervention and a no-op.
    if f > 0:
        for share in (0.3, 0.5, 0.7):
            logger.info(f"  --de_weight {share/(1-share) * (1-f)/f:6.1f}  -> DE genes carry "
                        f"~{100*share:.0f}% of the gradient")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
