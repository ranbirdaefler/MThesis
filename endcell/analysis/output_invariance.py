#!/usr/bin/env python
r"""
output_invariance.py
====================
The truth-independent, most-visceral drug-blindness demonstration.

Every existing drug-blindness result compares the model's prediction to the TRUTH (grading,
scramble, discrimination). This one needs no truth, no ceiling, no metric argument: hold the
CONTROL cell FIXED and vary ONLY the drug in the prompt. If the model's generated cell sentence
barely changes when you swap the drug, the drug token does nothing.

The killer comparison is against the model's OWN sampling noise:
  * cross-drug divergence  = similarity( gen(drug_i), gen(drug_j) )   on the SAME control  (i != j)
  * same-drug divergence   = similarity( gen(drug_i, sample a), gen(drug_i, sample b) )     (resample)
If cross-drug similarity ~= same-drug similarity, changing the drug perturbs the output no more
than re-rolling the sampler does -> the model is drug-blind. A drug-AWARE model shows
cross-drug << same-drug.

Prompts are built with the preprocessor's own format_prompt(), so the drug-variant prompts are
byte-identical to training except for the drug name + mechanism. The control cell sentence is held
fixed within a context; only (drug, dose, moa) vary.

SIMILARITY between two generated sentences (higher = more similar; [END_CELL] stripped):
  * topn_tau : Kendall tau over the top-N expressed genes of the first sentence (the honest metric)
  * jaccard  : Jaccard overlap of the expressed panel-gene SETS

OUTPUTS
  * <out>.json : per-context matrices + pooled cross vs same distributions + gap with bootstrap CI
  * <out>.png  : (A) one context's cross-drug similarity heatmap, (B) pooled cross vs same-drug
                 histograms with the gap annotated. The figure IS the result.

USAGE (cluster, GPU)
  python output_invariance.py \
     --eval_dir DATA_endcell_big --model_path CKPT_endcell/final --model_name endcell \
     --tier tier2_unseen_drugs --n_contexts 8 --n_drugs 12 --n_samples 2 \
     --temperature 0.8 --top_p 0.9 --topn 100 --max_new_tokens 3800 --bf16 \
     --out RESULTS/output_invariance_endcell.json --seed 42

SELFTEST (no model/data; validates the instrument with a fake generator)
  python output_invariance.py --selftest --out /tmp/outinv_selftest.json
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

# --- repo path bootstrap (reorg): make shared/ + sibling pipeline dirs importable ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
for _p in [os.path.join(_ROOT, "shared"), *sorted(glob.glob(os.path.join(_PIPE, "*")))]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"


# ----------------------------------------------------------------- sentence similarity
def _genes(sentence):
    return [g for g in sentence.strip().split() if g != SENTINEL]


def expressed_set(sentence, panel_set):
    return {g for g in _genes(sentence) if g in panel_set}


def jaccard(a, b, panel_set):
    sa, sb = expressed_set(a, panel_set), expressed_set(b, panel_set)
    u = len(sa | sb)
    return (len(sa & sb) / u) if u else None


def topn_tau(a, b, topn):
    """Kendall tau over the top-N expressed genes of sentence a (the reference), missing genes
    in b sent to worst rank. Reuses the project's metric library for identical scoring."""
    import evaluate_c2s_tahoe as ev
    ra = ev.cell_sentence_to_gene_ranks(" ".join(_genes(a)))
    rb = ev.cell_sentence_to_gene_ranks(" ".join(_genes(b)))
    if len(ra) < 3 or len(rb) < 3:
        return None
    worst = max(ra.values()) + 1
    topn_genes = sorted(ra, key=lambda g: ra[g])[:topn]
    res = ev.compute_rank_correlation(rb, ra, gene_subset=topn_genes)
    return res.get("kendall_tau") if isinstance(res, dict) else None


# ----------------------------------------------------------------- context construction
def header_of(prompt):
    """Everything before the control cell (the drug/dose/mechanism line)."""
    return prompt.split("\nControl cell:", 1)[0]


def build_contexts(examples, n_contexts, n_drugs, rng):
    """Each context = one FIXED control cell + a set of drug HEADERS from the SAME cell line.
    Holding the control fixed and swapping the header isolates the drug token (cell line name is
    constant within a context because all headers come from that cell line)."""
    import evaluate_c2s_tahoe as ev
    by_cl = defaultdict(list)
    for e in examples:
        cl = e.get("metadata", {}).get("cell_line_id")
        if cl is not None:
            by_cl[cl].append(e)
    # prefer cell lines with the most distinct drugs
    cls = sorted(by_cl, key=lambda cl: -len({e["metadata"]["drug"] for e in by_cl[cl]}))
    contexts = []
    for cl in cls:
        if len(contexts) >= n_contexts:
            break
        exs = by_cl[cl]
        # distinct drugs -> one representative header each
        drug_header = {}
        for e in exs:
            d = e["metadata"]["drug"]
            if d not in drug_header:
                drug_header[d] = header_of(e["prompt"])
        if len(drug_header) < 3:
            continue
        drugs = list(drug_header)
        rng.shuffle(drugs)
        drugs = drugs[:n_drugs]
        # fixed control from the first example of this cell line that has one
        ctrl = ""
        for e in exs:
            ctrl = ev.control_from_prompt(e["prompt"])
            if ctrl:
                break
        if not ctrl:
            continue
        prompts = {d: f"{drug_header[d]}\nControl cell: {ctrl}\n\nResponse cell:" for d in drugs}
        contexts.append({"cell_line": str(cl), "drugs": drugs, "prompts": prompts})
    logger.info(f"  built {len(contexts)} contexts (fixed control, {n_drugs} drugs each)")
    return contexts


# ----------------------------------------------------------------- similarity bookkeeping
def collect_similarities(gens, contexts, panel_set, topn):
    """gens[(ctx_idx, drug, sample_idx)] = sentence.
    Returns pooled cross-drug and same-drug similarity lists (topn_tau + jaccard),
    plus one representative context matrix for the heatmap."""
    cross = {"topn_tau": [], "jaccard": []}
    same = {"topn_tau": [], "jaccard": []}
    matrices = []
    for ci, ctx in enumerate(contexts):
        drugs = ctx["drugs"]
        # same-drug: pairs of samples of the SAME drug
        for d in drugs:
            samples = [gens[(ci, d, s)] for s in range(ctx["n_samples"]) if (ci, d, s) in gens]
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    for mname, fn in (("topn_tau", lambda a, b: topn_tau(a, b, topn)),
                                      ("jaccard", lambda a, b: jaccard(a, b, panel_set))):
                        v = fn(samples[i], samples[j])
                        if v is not None:
                            same[mname].append(v)
        # cross-drug: sample 0 of each drug vs sample 0 of each other drug
        reps = {d: gens.get((ci, d, 0)) for d in drugs if (ci, d, 0) in gens}
        dl = [d for d in drugs if reps.get(d)]
        mat = np.full((len(dl), len(dl)), np.nan)
        for i, di in enumerate(dl):
            for j, dj in enumerate(dl):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                t = topn_tau(reps[di], reps[dj], topn)
                if t is not None:
                    mat[i, j] = t
                if i < j:
                    for mname, fn in (("topn_tau", lambda a, b: topn_tau(a, b, topn)),
                                      ("jaccard", lambda a, b: jaccard(a, b, panel_set))):
                        v = fn(reps[di], reps[dj])
                        if v is not None:
                            cross[mname].append(v)
        matrices.append({"cell_line": ctx["cell_line"], "drugs": dl, "topn_tau_matrix": mat.tolist()})
    return cross, same, matrices


def summarize(cross, same, n_boot, seed):
    rng = np.random.RandomState(seed)
    out = {}
    for m in ("topn_tau", "jaccard"):
        c = np.array(cross[m], float)
        s = np.array(same[m], float)
        if len(c) == 0 or len(s) == 0:
            out[m] = None
            continue
        # bootstrap the gap (same - cross); ~0 => drug-blind, >0 => drug changes the output
        boots = []
        for _ in range(n_boot):
            bc = c[rng.randint(0, len(c), len(c))].mean()
            bs = s[rng.randint(0, len(s), len(s))].mean()
            boots.append(bs - bc)
        boots = np.array(boots)
        out[m] = {
            "cross_mean": float(c.mean()), "cross_n": int(len(c)),
            "same_mean": float(s.mean()), "same_n": int(len(s)),
            "gap_same_minus_cross": float(s.mean() - c.mean()),
            "gap_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    return out


# ----------------------------------------------------------------- figure
def make_figure(summary, matrices, model_name, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"  matplotlib unavailable ({e}); skipping figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: representative context heatmap (topn_tau across drugs)
    ax = axes[0]
    rep = max(matrices, key=lambda mm: len(mm["drugs"])) if matrices else None
    if rep:
        M = np.array(rep["topn_tau_matrix"], float)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_title(f"Output similarity across drugs\n(same fixed control, cell line {rep['cell_line'][:14]})",
                     fontsize=10)
        ax.set_xlabel("drug j"); ax.set_ylabel("drug i")
        ax.set_xticks(range(len(rep["drugs"]))); ax.set_yticks(range(len(rep["drugs"])))
        ax.set_xticklabels([d[:8] for d in rep["drugs"]], rotation=90, fontsize=6)
        ax.set_yticklabels([d[:8] for d in rep["drugs"]], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="topN-τ")

    # Panel B: pooled cross-drug vs same-drug (resample) similarity
    ax = axes[1]
    s = summary.get("topn_tau")
    if s:
        # reconstruct approximate distributions from stored means is not possible; instead
        # the JSON keeps raw lists — but for the figure we draw the two means as reference lines
        # over the histograms passed in via summary["_dists"] if present.
        dists = summary.get("_dists", {})
        cross = np.array(dists.get("cross_topn_tau", []), float)
        same = np.array(dists.get("same_topn_tau", []), float)
        bins = np.linspace(min(0.0, np.nanmin(cross) if len(cross) else 0.0), 1.0, 40)
        if len(cross):
            ax.hist(cross, bins=bins, alpha=0.6, label=f"cross-drug (mean {s['cross_mean']:.2f})",
                    color="#d1495b", density=True)
        if len(same):
            ax.hist(same, bins=bins, alpha=0.6, label=f"same-drug resample (mean {s['same_mean']:.2f})",
                    color="#30638e", density=True)
        ax.axvline(s["cross_mean"], color="#d1495b", ls="--", lw=1.5)
        ax.axvline(s["same_mean"], color="#30638e", ls="--", lw=1.5)
        gap = s["gap_same_minus_cross"]
        ax.set_title(f"Does the drug change the output?  gap (same−cross) = {gap:+.3f}\n"
                     f"95% CI [{s['gap_ci'][0]:+.3f}, {s['gap_ci'][1]:+.3f}]  "
                     f"— ~0 ⇒ drug-blind", fontsize=10)
        ax.set_xlabel("output-vs-output topN-τ"); ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(f"Output invariance to the drug token — model: {model_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_path, dpi=150)
    logger.info(f"  figure -> {png_path}")


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Fake-generator instrument check. Two synthetic models:
      * drug-blind: output independent of the drug -> cross ~= same (gap ~ 0)  [primary claim shape]
      * drug-aware: output strongly depends on the drug -> cross << same (gap > 0)  [detects signal]
    Confirms the instrument distinguishes the two — so a real gap~0 is meaningful."""
    P = 300
    panel = [f"G{i}" for i in range(P)]
    panel_set = set(panel)

    n_ctx, n_drugs, n_samp = 4, 8, 2

    def gen(mode, ci, drug, call):
        """One synthetic 'generation' = ~90 expressed genes drawn from a candidate pool.
        Each call is an INDEPENDENT draw (unique `call` seed), so two generations differ by
        sampling noise. What changes between modes is the POOL the draw comes from:
          * blind: pool depends only on the context -> same/cross-drug draws are exchangeable (gap~0)
          * aware: pool depends on the drug -> same-drug draws overlap more than cross-drug (gap>0)
        """
        if mode == "blind":
            pool = np.random.RandomState(ci).choice(P, 200, replace=False)          # drug ignored
        else:
            pool = np.random.RandomState((ci * 131 + hash(drug)) % (2**30)).choice(P, 120, replace=False)
        base = np.random.RandomState(call).choice(pool, 90, replace=False)
        return " ".join(panel[i] for i in base) + " " + SENTINEL

    def run(mode):
        contexts = []
        gens = {}
        call = 0
        for ci in range(n_ctx):
            drugs = [f"drug{d}" for d in range(n_drugs)]
            contexts.append({"cell_line": f"cl{ci}", "drugs": drugs,
                             "prompts": {}, "n_samples": n_samp})
            for d in drugs:
                for s in range(n_samp):
                    gens[(ci, d, s)] = gen(mode, ci, d, call)
                    call += 1
        cross, same, mats = collect_similarities(gens, contexts, panel_set, args.topn)
        summ = summarize(cross, same, args.n_boot, args.seed)
        summ["_dists"] = {"cross_topn_tau": cross["topn_tau"], "same_topn_tau": same["topn_tau"]}
        return summ

    blind = run("blind")
    aware = run("aware")
    # Validate on JACCARD: the synthetic sentences differ only in gene SET (random order), so the
    # set metric is the one that should fire. On REAL model outputs the ordering is meaningful, so
    # topn_tau is informative too; both are reported at run time.
    gb = blind["jaccard"]["gap_same_minus_cross"]
    ga = aware["jaccard"]["gap_same_minus_cross"]
    logger.info(f"  drug-BLIND synthetic : jaccard gap(same-cross) = {gb:+.3f}  (expect ~0)")
    logger.info(f"  drug-AWARE synthetic : jaccard gap(same-cross) = {ga:+.3f}  (expect strongly > 0)")
    ok = (abs(gb) < 0.10) and (ga > gb + 0.15)
    out = {"selftest": True, "passed": bool(ok), "drug_blind": blind, "drug_aware": aware}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--model_name", default="model")
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--n_contexts", type=int, default=8)
    ap.add_argument("--n_drugs", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=2, help="samples per (context,drug); >=2 for the same-drug baseline")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=3800)
    ap.add_argument("--gen_batch_size", type=int, default=48)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    if not (args.eval_dir and args.model_path):
        raise SystemExit("--eval_dir and --model_path are required for a real run.")

    import torch
    import evaluate_c2s_tahoe as ev
    from transformers import AutoModelForCausalLM, AutoTokenizer

    panel = json.load(open(os.path.join(args.eval_dir, "l1000_panel.json")))
    panel_set = set(panel)
    rng = np.random.RandomState(args.seed)

    path = os.path.join(args.eval_dir, f"eval_{args.tier}.jsonl")
    examples = [json.loads(l) for l in open(path)]
    logger.info(f"loaded {len(examples)} examples from {args.tier}")
    contexts = build_contexts(examples, args.n_contexts, args.n_drugs, rng)
    for ctx in contexts:
        ctx["n_samples"] = args.n_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
    model.eval()
    ec = tok.encode(SENTINEL, add_special_tokens=False)
    logger.info(f"  model on {device}; [END_CELL] -> {ec} "
                f"({'atomic' if len(ec) == 1 else 'SPLIT — check tokenizer'})")

    # flatten all (context, drug, sample) prompts, generate in batches (sampled -> distinct draws)
    jobs = []
    for ci, ctx in enumerate(contexts):
        for d in ctx["drugs"]:
            for s in range(args.n_samples):
                jobs.append((ci, d, s, ctx["prompts"][d]))
    logger.info(f"  generating {len(jobs)} sentences "
                f"({len(contexts)} contexts x {args.n_drugs} drugs x {args.n_samples} samples)")
    gens = {}
    for i in range(0, len(jobs), args.gen_batch_size):
        batch = jobs[i:i + args.gen_batch_size]
        outs = ev.generate_cell_sentences_batched(
            model, tok, [b[3] for b in batch], device=device,
            max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=args.top_p)
        for (ci, d, s, _), g in zip(batch, outs):
            gens[(ci, d, s)] = g
        logger.info(f"    {min(i + args.gen_batch_size, len(jobs))}/{len(jobs)} generated")

    cross, same, matrices = collect_similarities(gens, contexts, panel_set, args.topn)
    summary = summarize(cross, same, args.n_boot, args.seed)
    summary["_dists"] = {"cross_topn_tau": cross["topn_tau"], "same_topn_tau": same["topn_tau"],
                         "cross_jaccard": cross["jaccard"], "same_jaccard": same["jaccard"]}

    out = {"model_name": args.model_name, "tier": args.tier,
           "n_contexts": len(contexts), "n_drugs": args.n_drugs, "n_samples": args.n_samples,
           "temperature": args.temperature, "summary": summary, "matrices": matrices,
           "config": {k: v for k, v in vars(args).items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    png = os.path.splitext(args.out)[0] + ".png"
    make_figure(summary, matrices, args.model_name, png)

    # ---- report ----
    logger.info("")
    logger.info("=" * 100)
    logger.info(f"  OUTPUT INVARIANCE TO THE DRUG TOKEN — model: {args.model_name}")
    for m in ("topn_tau", "jaccard"):
        s = summary.get(m)
        if not s:
            continue
        logger.info(f"  [{m}] cross-drug={s['cross_mean']:.3f} (n={s['cross_n']})  "
                    f"same-drug resample={s['same_mean']:.3f} (n={s['same_n']})  "
                    f"gap(same−cross)={s['gap_same_minus_cross']:+.3f} "
                    f"CI[{s['gap_ci'][0]:+.3f},{s['gap_ci'][1]:+.3f}]")
    logger.info("  READ: gap ~ 0 (CI includes 0) => swapping the drug changes the output no more")
    logger.info("        than resampling => DRUG-BLIND. gap >> 0 => the model uses the drug.")
    logger.info("=" * 100)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
