#!/usr/bin/env python
r"""
spikein_metric_benchmark.py
===========================
Advisor's spike-in protocol: benchmark which METRIC best distinguishes two drug populations,
and how gracefully each degrades as the populations are mixed (titrated contamination).

PROTOCOL (per cell line, per pair of drugs A,B):
  * A "sample" = a pseudobulk of `sample_size` cells drawn from a population.
  * Forced-choice trial: draw a REFERENCE sample from A, a SAME candidate from A (disjoint cells),
    and a DIFFERENT candidate from B. Score both candidates vs the reference with a metric. The
    trial is CORRECT if metric(ref, same) > metric(ref, diff) (higher = more similar).
  * Discrimination ACCURACY = fraction of correct trials over many resamples. Chance = 0.50.
  * SPIKE-IN TITRATION: contaminate the "different" population by mixing in a fraction `s` of A's
    cells (s = 0, 0.1, ..., 1.0). At s=0 the different candidate is pure B (easy). At s=1.0 it is
    effectively A -> the two candidates are indistinguishable -> accuracy -> 0.50. The curve
    accuracy(s) characterises the metric's sensitivity: a better metric stays above chance to
    higher contamination.

METRICS COMPARED (all higher = more similar):
  * de_delta        : Pearson of rank-shift (vs a shared control) over top-K DE genes  [needs control]
  * panel_tau       : Kendall τ over all 946 genes (rank space)
  * topn_tau        : Kendall τ over top-N expressed genes (rank space)
  * spearman_expr   : Spearman over expression (rank of expression, whole panel)
  * cosine_expr     : COSINE similarity in expression space (advisor found this strong).
                      Expression is reconstructed from the cell sentence via the C2S-style
                      rank->expression map in linear_model.json (rank r -> slope*log-ish); if absent,
                      falls back to a rank-decreasing proxy (rank i -> (P - i)).
  * cosine_shift    : cosine of the (candidate_expr - control_expr) shift vectors  [needs control]

Also supports an optional TAIL-FIX (--tail_rank_max): assign every inactive/absent gene the SAME
max rank (P) instead of position-based distinct ranks — the advisor's suggested tie fix — and
re-run all rank metrics under it, so you can see if it improves discrimination.

USAGE
-----
  python spikein_metric_benchmark.py --data_dir DATA \
     --sources train,eval_tier1_seen_conditions --out RESULTS/spikein_benchmark.json \
     --sample_size 15 --n_trials 400 --spike_fracs 0,0.1,0.2,0.3,0.5,0.7,1.0 \
     --n_celllines 40 --drug_pairs_per_cellline 40 --de_k 50 --topn 100 \
     --tail_rank_max --seed 42
"""
# --- repo path bootstrap (reorg): make shared/ + sibling pipeline dirs importable ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
for _p in [os.path.join(_ROOT, "shared"), *sorted(glob.glob(os.path.join(_PIPE, "*")))]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, os, logging
from collections import defaultdict
import numpy as np
import evaluate_c2s_tahoe as ev

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- representation helpers
def sentence_to_rankarr(sentence, panel_index, P, mode="position", midrank=None):
    """Rank array over the panel. Inactive genes (not in the sentence) handled per mode:
      * "position"  : inactive genes get rank P+1 (they trail; active genes keep 1..k position).
                      (Original behavior; the arbitrary tail order is absent here since all
                       inactive share P+1, but active-gene positions are as emitted.)
      * "tail_max"  : inactive genes all tied at rank P (Federico's fix — equal jump zero->rank 1,
                      no arbitrary tail ordering).
      * "zero_bucket": inactive genes all tied at a per-cell mid rank = n_active+1 (our initial
                      interpretation — varies per cell depending on how many genes are expressed).
      * "zero_bucket_fixed": inactive genes all tied at a FIXED mid rank = P//2 (Francesca's exact
                      specification — "fixed mid-rank of the high ones", same for ALL cells,
                      so inactive-in-both-cells genes contribute zero shift). This is the canonical
                      version of the professor's proposal.
    """
    if mode == "tail_max":
        fill = P
    elif mode == "zero_bucket":
        fill = None  # set per-sentence below
    elif mode == "zero_bucket_fixed":
        fill = P // 2
    else:  # position
        fill = P + 1
    genes = [g for g in sentence.split() if g != "[END_CELL]"]
    if mode == "zero_bucket" and fill is None:
        n_active = len(set(g for g in genes if g in panel_index))
        fill = n_active + 1
    arr = np.full(P, fill, dtype=np.float64)
    seen = set(); k = 0
    for pos, g in enumerate(genes, 1):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        seen.add(gi); k += 1
        arr[gi] = pos
    return arr


def expressed_panel_set(sentence, panel_index):
    """The set of PANEL gene indices expressed in this sentence (sentinel stripped)."""
    return {panel_index[g] for g in sentence.split()
            if g != "[END_CELL]" and g in panel_index}


def activity_diagnostic(by_cl_drug, panel_index, P, n_pairs=4000, seed=42):
    """Report whether the END_CELL representation actually yields differing active-gene sets.
    If control and treated (and different drugs) express nearly the SAME genes, then dropping
    inactive genes changed little and the three representations will barely differ — this is the
    gate that tells us the rebuild bought us something.

    Measures, over sampled cells / matched pairs within a cell line:
      1. per-cell expressed count (how sparse are sentences)
      2. control->treated ON genes (expressed in treated, not control) and OFF genes (vice versa)
         -> motivates the professor's zero_bucket (on/off transitions)
      3. treated-vs-treated (same drug) and treated-vs-treated (different drug) Jaccard of the
         expressed sets, and how many genes differ -> whether 'absent' carries drug information
      4. how often the union of two cells' expressed sets < P (i.e. some panel gene absent in both)
         -> if ~0, every gene shows up somewhere and representations converge
    """
    rng = np.random.RandomState(seed)
    all_cells = []  # (expr_set, drug, cl)
    for cl, dd in by_cl_drug.items():
        for drug, cells in dd.items():
            for c in cells:
                all_cells.append((expressed_panel_set(c["resp"], panel_index), drug, cl,
                                  expressed_panel_set(c["ctrl"], panel_index)))
    if not all_cells:
        return None

    # 1. per-cell expressed counts (treated) + control
    treat_counts = np.array([len(s) for s, _, _, _ in all_cells])
    ctrl_counts = np.array([len(cs) for _, _, _, cs in all_cells])

    # 2. control->treated ON/OFF per cell
    on_counts, off_counts = [], []
    for tset, _, _, cset in all_cells:
        on_counts.append(len(tset - cset))   # expressed in treated, not control (turned ON)
        off_counts.append(len(cset - tset))  # expressed in control, not treated (turned OFF)
    on_counts = np.array(on_counts); off_counts = np.array(off_counts)

    # 3/4. pairwise same-drug vs diff-drug expressed-set overlap, within cell line
    by_cl = defaultdict(list)
    for s, drug, cl, _ in all_cells:
        by_cl[cl].append((s, drug))
    same_jac, diff_jac, same_symdiff, diff_symdiff, union_lt_P = [], [], [], [], []
    for _ in range(n_pairs):
        cl = rng.choice(list(by_cl.keys()))
        items = by_cl[cl]
        if len(items) < 2:
            continue
        i, j = rng.choice(len(items), 2, replace=False)
        (s1, d1), (s2, d2) = items[i], items[j]
        u = len(s1 | s2); inter = len(s1 & s2)
        jac = inter / u if u else 0.0
        sym = len(s1 ^ s2)  # genes expressed in exactly one of the two
        if u < P:
            union_lt_P.append(1)
        else:
            union_lt_P.append(0)
        if d1 == d2:
            same_jac.append(jac); same_symdiff.append(sym)
        else:
            diff_jac.append(jac); diff_symdiff.append(sym)

    def stats(a):
        a = np.array(a) if len(a) else np.array([np.nan])
        return dict(mean=float(np.nanmean(a)), median=float(np.nanmedian(a)),
                    p10=float(np.nanpercentile(a, 10)), p90=float(np.nanpercentile(a, 90)))

    return {
        "panel_size": P,
        "n_cells": len(all_cells),
        "treated_expressed_count": stats(treat_counts),
        "control_expressed_count": stats(ctrl_counts),
        "genes_turned_ON_ctrl_to_treat": stats(on_counts),
        "genes_turned_OFF_ctrl_to_treat": stats(off_counts),
        "same_drug_expressed_jaccard": stats(same_jac),
        "diff_drug_expressed_jaccard": stats(diff_jac),
        "same_drug_symdiff_genes": stats(same_symdiff),
        "diff_drug_symdiff_genes": stats(diff_symdiff),
        "frac_pairs_union_lt_panel": float(np.mean(union_lt_P)) if union_lt_P else 0.0,
    }


def rankarr_to_expr(rankarr, linear_model, P):
    """C2S-style reconstruction of expression from rank (monotone-decreasing: higher rank = lower
    expression). Fallback: expr = P - rank."""
    if linear_model and "slope" in linear_model:
        slope = linear_model["slope"]; intercept = linear_model.get("intercept", 0.0)
        return np.exp(slope * rankarr + intercept)
    return (P - rankarr)


def pseudobulk_rankarr(cell_sentences, panel_index, P, mode="position"):
    """Mean rank array over a set of cells (the sample representation)."""
    acc = np.zeros(P)
    for s in cell_sentences:
        acc += sentence_to_rankarr(s, panel_index, P, mode)
    return acc / max(len(cell_sentences), 1)


# ----------------------------------------------------------------- similarity metrics
def _pearson_on(a, b, idx):
    """Fast rank correlation = Pearson on the (already-rank) values over idx. Equivalent in
    discrimination to Spearman/Kendall for our purpose, at O(n) instead of Kendall's O(n^2)."""
    x, y = a[idx], b[idx]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def kendall_sub(a, b, idx):
    # kept for reference/compatibility; not used in the hot path (too slow)
    return _pearson_on(a, b, idx)


def spearman_all(a, b):
    return _pearson_on(a, b, np.arange(len(a)))


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return None
    return float(np.dot(a, b) / (na * nb))


def similarity(metric, ref, cand, control, panel, panel_index, P, linear_model,
               de_k, topn, tail_rank_max):
    """All metrics take pseudobulk RANK arrays (ref, cand) and optional control rank array.
    Higher = more similar. Returns float or None."""
    if metric == "de_delta":
        if control is None: return None
        # DE genes = top-k by |ref_rank - control_rank|
        de_idx = np.argsort(-np.abs(ref - control))[:de_k]
        rs = ref[de_idx] - control[de_idx]
        cs = cand[de_idx] - control[de_idx]
        if np.std(rs) < 1e-9 or np.std(cs) < 1e-9: return None
        return float(np.corrcoef(rs, cs)[0, 1])
    if metric == "panel_tau":
        return kendall_sub(ref, cand, np.arange(P))
    if metric == "topn_tau":
        topn_idx = np.argsort(ref)[:topn]   # lowest rank number = most expressed
        return kendall_sub(ref, cand, topn_idx)
    if metric == "spearman_expr":
        re, ce = rankarr_to_expr(ref, linear_model, P), rankarr_to_expr(cand, linear_model, P)
        return spearman_all(re, ce)
    if metric == "cosine_expr":
        re, ce = rankarr_to_expr(ref, linear_model, P), rankarr_to_expr(cand, linear_model, P)
        return cosine(re, ce)
    if metric == "cosine_shift":
        if control is None: return None
        ce_ctrl = rankarr_to_expr(control, linear_model, P)
        re, ce = rankarr_to_expr(ref, linear_model, P), rankarr_to_expr(cand, linear_model, P)
        return cosine(re - ce_ctrl, ce - ce_ctrl)
    return None


METRICS = ["de_delta", "panel_tau", "topn_tau", "spearman_expr", "cosine_expr", "cosine_shift"]


# ----------------------------------------------------------------- data
def load_cells(data_dir, sources):
    by_cl_drug = defaultdict(lambda: defaultdict(list))
    n = 0
    for src in sources:
        path = os.path.join(data_dir, src if src.endswith(".jsonl") else f"{src}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  missing {path}"); continue
        with open(path) as f:
            for line in f:
                ex = json.loads(line); m = ex.get("metadata", {})
                cl, drug = m.get("cell_line_id"), m.get("drug")
                if cl is None or drug is None: continue
                ctrl = ev.control_from_prompt(ex["prompt"])
                if not ctrl: continue
                by_cl_drug[cl][drug].append({"resp": ex["response"], "ctrl": ctrl})
                n += 1
    logger.info(f"  Loaded {n:,} cells, {len(by_cl_drug)} cell lines")
    return by_cl_drug


def draw_sample(cells, size, rng):
    idx = rng.choice(len(cells), size=min(size, len(cells)), replace=(len(cells) < size))
    return [cells[i] for i in idx]


def classify_de_genes(ref_arr, diff_arr, ctrl_arr, de_k, mode, P):
    """For the top-K DE genes (by |ref - ctrl|), classify each as on/off or within-expressed.
    A gene is 'on/off' if its rank in one of ref/diff is at the fill value for this mode
    (meaning it was inactive in all cells of that pseudobulk). Otherwise 'within-expressed'.
    This tests Francesca's claim that the zero-bucket lets more within-expressed genes into
    the top-K by shrinking the on/off rank jumps."""
    de_idx = np.argsort(-np.abs(ref_arr - ctrl_arr))[:de_k]
    if mode == "tail_max":
        fill = P
    elif mode == "zero_bucket_fixed":
        fill = P // 2
    elif mode == "zero_bucket":
        fill = None
    else:  # position
        fill = P + 1
    n_onoff = 0
    for gi in de_idx:
        ref_at_fill = False; diff_at_fill = False
        if fill is not None:
            ref_at_fill = abs(ref_arr[gi] - fill) < 1.5
            diff_at_fill = abs(diff_arr[gi] - fill) < 1.5
        else:
            ref_at_fill = ref_arr[gi] > P * 0.15
            diff_at_fill = diff_arr[gi] > P * 0.15
        if (ref_at_fill and not diff_at_fill) or (diff_at_fill and not ref_at_fill):
            n_onoff += 1
    return n_onoff, de_k - n_onoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--sources", default="train,eval_tier1_seen_conditions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample_size", type=int, default=15, help="cells per pseudobulk sample")
    ap.add_argument("--n_trials", type=int, default=150, help="forced-choice trials per (cl, drugpair)")
    ap.add_argument("--spike_fracs", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--n_celllines", type=int, default=40)
    ap.add_argument("--drug_pairs_per_cellline", type=int, default=25)
    ap.add_argument("--min_cells_per_drug", type=int, default=30)
    ap.add_argument("--de_k", type=int, default=50)
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--modes", default="position,tail_max,zero_bucket_fixed",
                    help="comma-sep representation modes: position, tail_max, zero_bucket, zero_bucket_fixed")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diagnostic_only", action="store_true",
                    help="run only the active-gene-set diagnostic (fast), skip the benchmark")
    ap.add_argument("--diag_pairs", type=int, default=4000,
                    help="number of within-cell-line pairs sampled for the activity diagnostic")
    args = ap.parse_args()

    if os.path.abspath(args.out).startswith(os.path.abspath(args.data_dir) + os.sep):
        raise SystemExit("Refusing to write inside data_dir.")
    rng = np.random.RandomState(args.seed)
    panel = json.load(open(os.path.join(args.data_dir, "l1000_panel.json")))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    lm_path = os.path.join(args.data_dir, "linear_model.json")
    linear_model = json.load(open(lm_path)) if os.path.exists(lm_path) else None
    spikes = [float(x) for x in args.spike_fracs.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    by_cl_drug = load_cells(args.data_dir, [s.strip() for s in args.sources.split(",")])

    # ---- ACTIVITY DIAGNOSTIC (runs FIRST): does END_CELL actually give differing active sets? ----
    logger.info("")
    logger.info("=" * 100)
    logger.info("  ACTIVITY-SET DIAGNOSTIC  (does dropping inactive genes give the representations")
    logger.info("  something to differ on? If active sets barely vary, the modes will still converge.)")
    diag = activity_diagnostic(by_cl_drug, panel_index, P, n_pairs=args.diag_pairs, seed=args.seed)
    if diag:
        def fmt(key):
            s = diag[key]; return f"mean={s['mean']:.1f} median={s['median']:.1f} [p10={s['p10']:.1f}, p90={s['p90']:.1f}]"
        def fmtf(key):
            s = diag[key]; return f"mean={s['mean']:.3f} median={s['median']:.3f} [p10={s['p10']:.3f}, p90={s['p90']:.3f}]"
        logger.info(f"  panel size P = {diag['panel_size']}, cells analyzed = {diag['n_cells']:,}")
        logger.info(f"  treated expressed genes/cell : {fmt('treated_expressed_count')}")
        logger.info(f"  control expressed genes/cell : {fmt('control_expressed_count')}")
        logger.info(f"  genes turned ON  (treat not ctrl): {fmt('genes_turned_ON_ctrl_to_treat')}")
        logger.info(f"  genes turned OFF (ctrl not treat): {fmt('genes_turned_OFF_ctrl_to_treat')}")
        logger.info(f"  same-drug expressed-set Jaccard  : {fmtf('same_drug_expressed_jaccard')}")
        logger.info(f"  diff-drug expressed-set Jaccard  : {fmtf('diff_drug_expressed_jaccard')}")
        logger.info(f"  same-drug #genes differing (symdiff): {fmt('same_drug_symdiff_genes')}")
        logger.info(f"  diff-drug #genes differing (symdiff): {fmt('diff_drug_symdiff_genes')}")
        logger.info(f"  frac of cell-pairs where union of active sets < P (some gene absent in both): "
                    f"{diag['frac_pairs_union_lt_panel']:.3f}")
        logger.info("  INTERPRET: if ON/OFF counts and symdiff are ~0 and Jaccard ~1.0, the active")
        logger.info("  sets barely differ -> representations will converge and END_CELL bought little.")
        logger.info("  If diff-drug symdiff > same-drug symdiff, 'which genes are absent' carries drug info.")
    logger.info("=" * 100)
    if diag:
        import copy
        diag_out = os.path.join(os.path.dirname(args.out), "activity_diagnostic.json")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(diag, open(diag_out, "w"), indent=2)
        logger.info(f"  -> activity diagnostic saved to {diag_out}")
    if args.diagnostic_only:
        logger.info("  --diagnostic_only set; stopping before benchmark.")
        return

    cls = [cl for cl, dd in by_cl_drug.items()
           if sum(len(v) >= args.min_cells_per_drug for v in dd.values()) >= 2]
    rng.shuffle(cls); cls = cls[:args.n_celllines]
    logger.info(f"  Using {len(cls)} cell lines; modes={modes}")

    # acc[mode][metric][spike] = {cellline: [correct, total]}
    acc = {md: {m: {s: defaultdict(lambda: [0, 0]) for s in spikes} for m in METRICS} for md in modes}

    # DE top-K composition diagnostic (Francesca's claim):
    # For each mode, what fraction of the top-K DE genes are on/off transitions vs within-expressed?
    # on/off = gene whose pseudobulk rank in one sample is at/near the fill value (inactive in all
    # cells of that sample) but well below in the other (active in most cells).
    # Tracked at s=0 only (pure drug A vs pure drug B — the clean discrimination case).
    de_composition = {md: {"n_onoff": [], "n_within": [], "n_total": []} for md in modes}

    def sample_repr(cells, md):
        return pseudobulk_rankarr([c["resp"] for c in cells], panel_index, P, md)

    def control_repr(cells, md):
        return pseudobulk_rankarr([c["ctrl"] for c in cells], panel_index, P, md)

    for cl in cls:
        dd = by_cl_drug[cl]
        drugs = [d for d, v in dd.items() if len(v) >= args.min_cells_per_drug]
        if len(drugs) < 2: continue
        pairs = []
        for _ in range(args.drug_pairs_per_cellline):
            a, b = rng.choice(len(drugs), 2, replace=False)
            pairs.append((drugs[a], drugs[b]))

        for (A, B) in pairs:
            cellsA, cellsB = dd[A], dd[B]
            for _ in range(args.n_trials):
                ref = draw_sample(cellsA, args.sample_size, rng)
                same = draw_sample(cellsA, args.sample_size, rng)
                # ref / same / control do NOT depend on spike -> compute once per mode (caching)
                ref_r = {md: sample_repr(ref, md) for md in modes}
                same_r = {md: sample_repr(same, md) for md in modes}
                ctrl_r = {md: control_repr(ref, md) for md in modes}
                # precompute same-similarity once per (mode, metric) — independent of spike
                sim_same = {md: {m: similarity(m, ref_r[md], same_r[md], ctrl_r[md], panel,
                                               panel_index, P, linear_model, args.de_k, args.topn, md)
                                 for m in METRICS} for md in modes}
                for s in spikes:
                    n_from_A = int(round(s * args.sample_size))
                    n_from_B = args.sample_size - n_from_A
                    diff = (draw_sample(cellsB, n_from_B, rng) + draw_sample(cellsA, n_from_A, rng)) \
                        if n_from_B > 0 else draw_sample(cellsA, args.sample_size, rng)
                    for md in modes:
                        diff_r = sample_repr(diff, md)
                        # DE top-K composition diagnostic at s=0 (pure A vs pure B)
                        if s == 0:
                            n_on, n_wi = classify_de_genes(ref_r[md], diff_r, ctrl_r[md],
                                                           args.de_k, md, P)
                            de_composition[md]["n_onoff"].append(n_on)
                            de_composition[md]["n_within"].append(n_wi)
                            de_composition[md]["n_total"].append(args.de_k)
                    for md in modes:
                        diff_r = sample_repr(diff, md)
                        for m in METRICS:
                            ss = sim_same[md][m]
                            if ss is None: continue
                            sd = similarity(m, ref_r[md], diff_r, ctrl_r[md], panel,
                                            panel_index, P, linear_model, args.de_k, args.topn, md)
                            if sd is None: continue
                            correct = 1 if ss > sd else (0 if ss < sd else None)
                            if correct is None: continue
                            cell = acc[md][m][s][cl]
                            cell[0] += correct; cell[1] += 1
        logger.info(f"  {str(cl)[:22]:22s} done")

    # ---- aggregate: per (tail, metric, spike): accuracy = mean over cell lines of (correct/total)
    def cl_bootstrap(percl, n_boot, seed):
        rngb = np.random.RandomState(seed)
        cls_ = list(percl.keys())
        vals = np.array([percl[c][0] / percl[c][1] for c in cls_ if percl[c][1] > 0])
        if len(vals) == 0: return None
        boots = [np.mean(vals[rngb.choice(len(vals), len(vals), replace=True)]) for _ in range(n_boot)]
        return dict(acc=float(np.mean(vals)), ci_low=float(np.percentile(boots, 2.5)),
                    ci_high=float(np.percentile(boots, 97.5)), n_cl=len(vals))

    results = {}
    for md in modes:
        results[md] = {}
        for m in METRICS:
            results[md][m] = {}
            for s in spikes:
                results[md][m][str(s)] = cl_bootstrap(acc[md][m][s], args.n_boot, args.seed)

    # ---- DE top-K composition summary (Francesca's claim) ----
    de_comp_summary = {}
    for md in modes:
        d = de_composition[md]
        if d["n_total"]:
            onoff = np.array(d["n_onoff"])
            within = np.array(d["n_within"])
            frac_onoff = onoff / np.array(d["n_total"])
            de_comp_summary[md] = {
                "mean_onoff_in_topK": float(np.mean(onoff)),
                "mean_within_in_topK": float(np.mean(within)),
                "mean_frac_onoff": float(np.mean(frac_onoff)),
                "median_frac_onoff": float(np.median(frac_onoff)),
                "de_k": args.de_k,
                "n_trials_sampled": len(onoff),
            }

    out = {"sample_size": args.sample_size, "n_trials": args.n_trials, "spikes": spikes,
           "de_k": args.de_k, "topn": args.topn, "modes": modes, "results": results,
           "de_topk_composition": de_comp_summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    # ---- print accuracy(spike) curves
    logger.info("")
    logger.info("=" * 100)
    logger.info("  DISCRIMINATION ACCURACY vs SPIKE-IN CONTAMINATION  (chance=0.50; higher/flatter=better metric)")
    for md in modes:
        logger.info(f"\n  [representation: {md}]")
        header = "  " + "metric".ljust(15) + "".join(f"s={s:<5}" for s in spikes)
        logger.info(header)
        for m in METRICS:
            row = "  " + m.ljust(15)
            for s in spikes:
                r = results[md][m][str(s)]
                row += (f"{r['acc']:.3f} " if r else " NA   ")
            logger.info(row)
    logger.info("=" * 100)
    logger.info("  Read: at s=0 (pure B) accuracy should be highest; at s=1.0 it should -> 0.50.")
    logger.info("  The metric+representation that stays HIGH to larger s is the better discriminator.")

    # ---- DE top-K composition diagnostic (Francesca's claim) ----
    logger.info("")
    logger.info("=" * 100)
    logger.info("  DE TOP-K COMPOSITION (at s=0): what fraction of the top-K DE genes are on/off")
    logger.info("  transitions vs within-expressed shifts? (Francesca's zero-bucket should let more")
    logger.info("  within-expressed genes enter the top-K by shrinking the on/off rank jumps.)")
    logger.info(f"  {'mode':22s} {'on/off':>8s} {'within':>8s} {'frac_onoff':>12s} {'(of top-K =':>12s} {args.de_k})")
    for md in modes:
        d = de_comp_summary.get(md)
        if d:
            logger.info(f"  {md:22s} {d['mean_onoff_in_topK']:>8.1f} {d['mean_within_in_topK']:>8.1f}"
                        f" {d['mean_frac_onoff']:>12.3f}   (n_trials={d['n_trials_sampled']})")
    logger.info("  INTERPRET: if zero_bucket_fixed has a LOWER frac_onoff than tail_max/position,")
    logger.info("  Francesca's proposal works as intended — the mid-rank lets within-expressed genes")
    logger.info("  enter the DE top-K instead of being crowded out by on/off events.")
    logger.info("=" * 100)

    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
