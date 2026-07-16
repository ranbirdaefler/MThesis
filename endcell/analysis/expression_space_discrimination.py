#!/usr/bin/env python
r"""
expression_space_discrimination.py
==================================
Closes the rank-vs-expression confound in the drug-blindness result.

Every core analysis so far (Part I saturation, the noise ceiling, "different-drug = ceiling")
lives in RANK / cell-sentence space. Cell sentences discard expression MAGNITUDE. So the null
"single-cell drug signal is undetectable" might really be "the rank representation discards the
drug signal", not "single-cell noise destroys it". The spike-in expression metrics never settled
this because they used expression RECONSTRUCTED from ranks via linear_model.json (lossy).

This script runs the same-drug vs different-drug discrimination on TRUE normalized expression
vectors streamed directly from Tahoe-100M — no rank round-trip, no model. It mirrors
drug_specificity_in_data.py (Part I) and spikein_metric_benchmark.py (the forced-choice protocol),
but in real expression space, and sweeps single-cell -> pseudobulk so the two regimes are directly
comparable.

DECISION THIS ANSWERS
  * TRUE-expression single-cell discrimination ~ 0.50 (like rank)  -> the bottleneck is single-cell
    NOISE, not the representation. The original thesis claim strengthens.
  * TRUE-expression single-cell discrimination >> 0.50 (unlike rank) -> the RANK representation was
    discarding the drug signal. The bottleneck is (partly) the representation; STATE-style
    expression-space models deserve a look, and the framing shifts from "noise" to "representation".
Either outcome is decisive and cheap (CPU, streaming).

METRICS (all: higher = more similar; on TRUE log-normalized expression over the 946-gene panel)
  * cosine_expr    : cosine similarity of the two expression vectors (Federico's proposal, done right)
  * pearson_expr   : Pearson r of the two expression vectors
  * spearman_expr  : Spearman (Pearson on the rank-of-expression) — a bridge to the rank metrics
  * cosine_shift   : cosine of (treated - control) shift vectors  [only if --collect_controls]

WHAT IT REPORTS (per metric, per pseudobulk size)
  * forced-choice discrimination ACCURACY  (ref=drug A, same=drug A disjoint, diff=drug B;
    correct if sim(ref,same) > sim(ref,diff); chance 0.50), cell-line-clustered bootstrap CI.
  * MOA positive control accuracy (same-MOA vs different-MOA, different drugs) — the test-validity
    check: if the instrument can't see mechanism it can't see drug either.
  * same/diff GAP + Cohen's d + within-cell-line permutation p (mirrors Part I), for magnitude.

USAGE (cluster)
  python expression_space_discrimination.py \
     --out RESULTS/expr_space_discrimination.json \
     --num_shards 8 --rows_per_shard 150000 --shard_seed 7 \
     --cells_per_drug 40 --min_cells_per_drug 30 \
     --n_celllines 30 --drugs_per_cellline 25 \
     --pb_sizes 1,15 --n_trials 200 --n_boot 1000 --n_perm 2000 \
     --collect_controls --seed 42

SELFTEST (no network / no data; validates the instrument end-to-end)
  python expression_space_discrimination.py --selftest --out /tmp/expr_selftest.json
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

# --- make both the flat cluster layout (~/tahoe/*.py) and the split repo layout
#     (src/evaluate_c2s_tahoe.py, ./tahoe_c2s_preprocess_endcell_v2.py) importable ---
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

EXPR_METRICS = ["cosine_expr", "pearson_expr", "spearman_expr"]
MLABEL = {"cosine_expr": "cos-expr", "pearson_expr": "pear-expr",
          "spearman_expr": "spear-expr", "cosine_shift": "cos-shift",
          "topn_tau": "topn-tau", "de_delta": "de-delta"}


# ----------------------------------------------------------------- expression vector builder
def panel_expr_vector(genes, exprs, gene_id_to_symbol, panel_index, P,
                      min_expressed=200, mito_frac_max=0.20):
    """TRUE log-normalized expression vector over the fixed panel (length P; 0 = unexpressed).

    Replicates the QC + normalization of build_panel_sentence()/raw_to_cell_sentence() exactly:
      * keep expr > 0, require >= min_expressed genes, drop cells with >20% mito counts,
      * library-size normalize to 1e4 over ALL positive genes, then log10(1 + x),
      * scatter the normalized value of each expressed PANEL gene into its canonical slot.
    Returns None on QC failure. This is the ONLY place magnitude is retained (vs the sentence).
    """
    gid = np.asarray(genes)
    ex = np.asarray(exprs, dtype=np.float64)
    m = ex > 0
    gid, ex = gid[m], ex[m]
    if len(ex) < min_expressed:
        return None
    tot = ex.sum()
    if tot <= 0:
        return None
    # mito fraction QC (MT- genes), matching raw_to_cell_sentence
    mito = 0.0
    for g, v in zip(gid, ex):
        s = gene_id_to_symbol.get(g, "")
        if s.startswith("MT-"):
            mito += v
    if mito / tot > mito_frac_max:
        return None
    norm = np.log10(1.0 + (ex / tot) * 1e4)
    vec = np.zeros(P, dtype=np.float32)
    for g, v in zip(gid, norm):
        sym = gene_id_to_symbol.get(g, None)
        idx = panel_index.get(sym) if sym is not None else None
        if idx is not None:
            vec[idx] = v  # first occurrence wins the slot; duplicates are negligible
    return vec


# ----------------------------------------------------------------- expression-space metrics
def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(a, b) / (na * nb))


def _pearson(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(a):
    """Average-rank of a 1-D array (ties broken to mean rank), no scipy dependency."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0  # mean rank per unique value (1-indexed)
    return avg[inv]


def expr_to_rank(vec):
    """Convert a (pseudobulk) expression vector to the project's cell-sentence rank profile:
    rank 1 = highest expression; unexpressed (0) genes fall to the tail in stable panel order.
    This is exactly the rank representation the MODEL consumes, derived from the SAME true-
    expression pseudobulk — so rank vs expression is compared on identical cells."""
    order = np.argsort(-vec, kind="stable")     # highest expression first
    ranks = np.empty(len(vec), dtype=np.float64)
    ranks[order] = np.arange(1, len(vec) + 1, dtype=np.float64)
    return ranks


def make_bundle(vec):
    """Both representations of one pseudobulk: the true-expression vector and its rank profile."""
    return {"expr": vec, "rank": expr_to_rank(vec)}


def sim(metric, ref, cand, ctrl, topn, de_k):
    """Unified similarity (higher = more similar). ref/cand/ctrl are bundles from make_bundle().
      * expression-space: cosine_expr, pearson_expr, spearman_expr, cosine_shift (needs ctrl)
      * rank-space (what the model sees): topn_tau (top-N expressed of ref), de_delta (top movers
        vs ctrl; needs ctrl) — the project's headline rank metrics, on the SAME pseudobulk."""
    re, ce = ref["expr"], cand["expr"]
    if metric == "cosine_expr":
        return _cosine(re, ce)
    if metric == "pearson_expr":
        return _pearson(re, ce)
    if metric == "spearman_expr":
        return _pearson(_rankdata(re), _rankdata(ce))
    if metric == "cosine_shift":
        if ctrl is None:
            return None
        return _cosine(re - ctrl["expr"], ce - ctrl["expr"])
    if metric == "topn_tau":
        rr, cr = ref["rank"], cand["rank"]
        idx = np.argsort(rr)[:topn]             # smallest rank number = most expressed in ref
        return _pearson(rr[idx], cr[idx])       # Pearson-on-ranks == discrimination-equivalent to τ
    if metric == "de_delta":
        if ctrl is None:
            return None
        rr, cr, kr = ref["rank"], cand["rank"], ctrl["rank"]
        de = np.argsort(-np.abs(rr - kr))[:de_k]
        rs, cs = rr[de] - kr[de], cr[de] - kr[de]
        if rs.std() < 1e-12 or cs.std() < 1e-12:
            return None
        return float(np.corrcoef(rs, cs)[0, 1])
    return None


# ----------------------------------------------------------------- streaming
def stream_panel_vectors(args, panel_index, P):
    """Stream Tahoe treated (and optionally DMSO) cells -> TRUE expression vectors.

    Returns:
      by_cl_drug : {cell_line_id: {drug: {"vecs":[np.array], "moa":str, "dose":float|None}}}
      ctrl_by_cl : {cell_line_id: mean control expression vector}  (pooled DMSO; None if not collected)
    Reuses the preprocessor's metadata + shard helpers so the data contract is identical.
    """
    import tahoe_c2s_preprocess_endcell_v2 as pp

    gene_id_to_symbol, sample_to_conc, drug_info, _cvcl = pp.load_metadata()
    all_shards = pp.discover_expression_shards()
    treated_shards = pp.select_shards(all_shards, args.num_shards, args.shard_seed)
    logger.info(f"Tahoe exposes {len(all_shards)} shards; streaming {len(treated_shards)} "
                f"(<= {args.rows_per_shard:,} rows each, seed={args.shard_seed})")

    held_out = None
    if args.held_out_drugs_file and os.path.exists(args.held_out_drugs_file):
        held_out = set(json.load(open(args.held_out_drugs_file)))
        logger.info(f"  restricting treated cells to {len(held_out)} held-out drugs (tier2 mode)")

    from datasets import load_dataset
    by_cl_drug = defaultdict(lambda: defaultdict(lambda: {"vecs": [], "moa": None, "dose": None}))
    ctrl_pool = defaultdict(list)   # cell_line -> [ctrl vecs]  (pooled DMSO fallback)
    n_treated = n_ctrl = n_seen = 0
    MAX_CTRL_PER_CL = 32

    for shard in treated_shards:
        url = f"hf://datasets/{pp.TAHOE_REPO}/{shard}"
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        sc = 0
        for row in ds:
            n_seen += 1
            sc += 1
            if sc > args.rows_per_shard:
                break
            drug = row["drug"]
            cl = row["cell_line_id"]
            is_dmso = (drug == "DMSO_TF" or drug == "DMSO")
            if is_dmso:
                if args.collect_controls and len(ctrl_pool[cl]) < MAX_CTRL_PER_CL:
                    v = panel_expr_vector(row["genes"], row["expressions"],
                                          gene_id_to_symbol, panel_index, P)
                    if v is not None:
                        ctrl_pool[cl].append(v); n_ctrl += 1
                continue
            if held_out is not None and drug not in held_out:
                continue
            slot = by_cl_drug[cl][drug]
            if len(slot["vecs"]) >= args.cells_per_drug:
                continue
            v = panel_expr_vector(row["genes"], row["expressions"],
                                  gene_id_to_symbol, panel_index, P)
            if v is None:
                continue
            slot["vecs"].append(v)
            n_treated += 1
            if slot["moa"] is None:
                slot["moa"] = (drug_info.get(drug, {}) or {}).get("moa", "unclear")
                try:
                    slot["dose"] = float(pp.parse_dose(
                        sample_to_conc.get(row["sample"], "unknown")).split()[0])
                except Exception:
                    slot["dose"] = None
            if n_treated >= args.max_cells_total:
                break
        logger.info(f"  shard {shard}: cumulative treated={n_treated:,} ctrl={n_ctrl:,} "
                    f"seen={n_seen:,}")
        if n_treated >= args.max_cells_total:
            logger.info("  reached --max_cells_total; stopping stream")
            break

    ctrl_by_cl = {}
    if args.collect_controls:
        for cl, vs in ctrl_pool.items():
            if vs:
                ctrl_by_cl[cl] = np.mean(np.stack(vs), axis=0).astype(np.float32)
        logger.info(f"  pooled DMSO controls for {len(ctrl_by_cl)} cell lines "
                    f"(mean vector per cell line; NOT plate-matched — documented approximation)")
    logger.info(f"Collected {n_treated:,} treated vectors across "
                f"{len(by_cl_drug)} cell lines")
    return by_cl_drug, ctrl_by_cl


# ----------------------------------------------------------------- stats (mirror Part I)
def cluster_bootstrap(per_cl_vals, n_boot, seed):
    rng = np.random.RandomState(seed)
    vals = np.array([v for v in per_cl_vals if v is not None and v == v], dtype=float)
    if len(vals) == 0:
        return None
    boots = [np.mean(vals[rng.choice(len(vals), len(vals), replace=True)]) for _ in range(n_boot)]
    return dict(mean=float(np.mean(vals)), ci_low=float(np.percentile(boots, 2.5)),
                ci_high=float(np.percentile(boots, 97.5)), n_cl=len(vals))


def perm_p(per_cl_gaps, obs, n_perm, seed):
    rng = np.random.RandomState(seed + 1)
    g = np.array([v for v in per_cl_gaps if v is not None and v == v], dtype=float)
    if len(g) == 0:
        return None
    null = np.array([np.nanmean(g * rng.choice([-1, 1], size=len(g))) for _ in range(n_perm)])
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def cohens_d(same, diff):
    same, diff = np.asarray(same, float), np.asarray(diff, float)
    if len(same) < 2 or len(diff) < 2:
        return None
    sp = np.sqrt(((len(same) - 1) * same.var(ddof=1) + (len(diff) - 1) * diff.var(ddof=1))
                 / (len(same) + len(diff) - 2))
    return float((same.mean() - diff.mean()) / sp) if sp > 0 else None


# ----------------------------------------------------------------- core benchmark
def pseudobulk(vecs, idx):
    return np.mean(np.stack([vecs[i] for i in idx]), axis=0)


def run_benchmark(by_cl_drug, ctrl_by_cl, args, rng):
    pb_sizes = [int(x) for x in args.pb_sizes.split(",")]
    use_shift = bool(ctrl_by_cl)
    # rank metrics (topn_tau, de_delta) are computed on the SAME pseudobulks as the expression
    # metrics, so the one table is a direct rank-vs-expression comparison on identical cells.
    metrics = list(EXPR_METRICS) + ["topn_tau"]
    if use_shift:
        metrics += ["cosine_shift", "de_delta"]
    topn, de_k = args.topn, args.de_k

    # eligible cell lines: >=2 drugs each with enough cells for disjoint ref/same at max pb size
    max_pb = max(pb_sizes)
    cls = []
    for cl, dd in by_cl_drug.items():
        good = [d for d, s in dd.items() if len(s["vecs"]) >= max(args.min_cells_per_drug, 2 * max_pb)]
        if len(good) >= 2:
            cls.append(cl)
    rng.shuffle(cls)
    cls = cls[:args.n_celllines]
    logger.info(f"Benchmark on {len(cls)} cell lines; sizes={pb_sizes}; "
                f"metrics={metrics}; shift={'on' if use_shift else 'off'}")

    # accuracy[size][metric] -> {cl: [correct, total]}; moa positive control likewise
    acc = {s: {m: defaultdict(lambda: [0, 0]) for m in metrics} for s in pb_sizes}
    moa_acc = {s: {m: defaultdict(lambda: [0, 0]) for m in metrics} for s in pb_sizes}
    # gap accumulators (single-cell + each pb size), mirroring Part I
    gap = {s: {m: {"per_cl": {}, "same": [], "diff": []} for m in metrics} for s in pb_sizes}

    def draw(vecs, size, rng, exclude=None):
        n = len(vecs)
        pool = [i for i in range(n) if exclude is None or i not in exclude]
        idx = rng.choice(pool, size=min(size, len(pool)),
                         replace=(len(pool) < size)).tolist()
        return idx

    for cl in cls:
        dd = by_cl_drug[cl]
        drugs = [d for d, s in dd.items() if len(s["vecs"]) >= max(args.min_cells_per_drug, 2 * max_pb)]
        rng.shuffle(drugs)
        drugs = drugs[:args.drugs_per_cellline]
        _ctrl_vec = ctrl_by_cl.get(cl) if use_shift else None
        ctrl = make_bundle(_ctrl_vec) if _ctrl_vec is not None else None
        moa = {d: dd[d]["moa"] for d in drugs}

        for size in pb_sizes:
            same_pool = {m: [] for m in metrics}
            diff_pool = {m: [] for m in metrics}
            # ---- forced-choice discrimination + gap ----
            for _ in range(args.n_trials):
                A = drugs[rng.randint(len(drugs))]
                B = drugs[rng.randint(len(drugs))]
                if A == B:
                    continue
                va, vb = dd[A]["vecs"], dd[B]["vecs"]
                ref_i = draw(va, size, rng)
                same_i = draw(va, size, rng, exclude=set(ref_i))   # disjoint same-drug candidate
                diff_i = draw(vb, size, rng)
                ref = make_bundle(pseudobulk(va, ref_i))
                same = make_bundle(pseudobulk(va, same_i))
                diff = make_bundle(pseudobulk(vb, diff_i))
                for m in metrics:
                    ss = sim(m, ref, same, ctrl, topn, de_k)
                    sd = sim(m, ref, diff, ctrl, topn, de_k)
                    if ss is None or sd is None:
                        continue
                    c = acc[size][m][cl]
                    c[1] += 1
                    c[0] += 1 if ss > sd else (0 if ss < sd else 0.5)
                    same_pool[m].append(ss)
                    diff_pool[m].append(sd)
            for m in metrics:
                if same_pool[m] and diff_pool[m]:
                    gap[size][m]["per_cl"][cl] = float(np.mean(same_pool[m]) - np.mean(diff_pool[m]))
                    gap[size][m]["same"] += same_pool[m]
                    gap[size][m]["diff"] += diff_pool[m]

            # ---- MOA positive control: same-MOA vs diff-MOA, DIFFERENT drugs ----
            for _ in range(args.n_trials):
                A = drugs[rng.randint(len(drugs))]
                same_moa_drugs = [d for d in drugs if d != A and moa[d] == moa[A]
                                  and moa[A] not in (None, "unclear", "unknown")]
                diff_moa_drugs = [d for d in drugs if moa[d] != moa[A]
                                  and moa[d] not in (None, "unclear", "unknown")]
                if not same_moa_drugs or not diff_moa_drugs:
                    continue
                S = same_moa_drugs[rng.randint(len(same_moa_drugs))]
                D = diff_moa_drugs[rng.randint(len(diff_moa_drugs))]
                ref = make_bundle(pseudobulk(dd[A]["vecs"], draw(dd[A]["vecs"], size, rng)))
                s_c = make_bundle(pseudobulk(dd[S]["vecs"], draw(dd[S]["vecs"], size, rng)))
                d_c = make_bundle(pseudobulk(dd[D]["vecs"], draw(dd[D]["vecs"], size, rng)))
                for m in metrics:
                    ss = sim(m, ref, s_c, ctrl, topn, de_k)
                    sd = sim(m, ref, d_c, ctrl, topn, de_k)
                    if ss is None or sd is None:
                        continue
                    c = moa_acc[size][m][cl]
                    c[1] += 1
                    c[0] += 1 if ss > sd else (0 if ss < sd else 0.5)
        logger.info(f"  {str(cl)[:26]:26s} done ({len(drugs)} drugs)")

    # ---- aggregate ----
    def agg_acc(table):
        out = {}
        for s in pb_sizes:
            out[str(s)] = {}
            for m in metrics:
                per_cl = [v[0] / v[1] for v in table[s][m].values() if v[1] > 0]
                out[str(s)][m] = cluster_bootstrap(per_cl, args.n_boot, args.seed)
        return out

    gap_out = {}
    for s in pb_sizes:
        gap_out[str(s)] = {}
        for m in metrics:
            g = gap[s][m]
            if not g["per_cl"]:
                gap_out[str(s)][m] = None
                continue
            b = cluster_bootstrap(list(g["per_cl"].values()), args.n_boot, args.seed)
            gap_out[str(s)][m] = {
                "same_mean": float(np.mean(g["same"])), "diff_mean": float(np.mean(g["diff"])),
                "gap": b, "cohens_d": cohens_d(g["same"], g["diff"]),
                "perm_p": perm_p(list(g["per_cl"].values()), b["mean"], args.n_perm, args.seed),
                "n_cl": len(g["per_cl"])}

    return {"pb_sizes": pb_sizes, "metrics": metrics,
            "discrimination_accuracy": agg_acc(acc),
            "moa_poscontrol_accuracy": agg_acc(moa_acc),
            "same_diff_gap": gap_out}


# ----------------------------------------------------------------- printing
def print_report(res):
    pb = res["pb_sizes"]
    logger.info("")
    logger.info("=" * 100)
    logger.info("  RANK-vs-EXPRESSION DRUG DISCRIMINATION  (forced choice; chance=0.50; cell-line bootstrap)")
    logger.info("  Same pseudobulks scored under both representations. KEY: single-cell (size 1) is the crux.")
    logger.info("    expression ~ rank ~ 0.50 at size 1  => noise-limited; the representation is NOT the cause.")
    logger.info("    expression >> rank at size 1         => rank was discarding drug signal (representation matters).")
    for tbl, title in [("discrimination_accuracy", "same-drug vs diff-drug"),
                       ("moa_poscontrol_accuracy", "MOA positive control (test validity)")]:
        logger.info(f"\n  [{title}]")
        logger.info("  " + "metric".ljust(14) + "".join(f"size={s:<12}" for s in pb))
        for m in res["metrics"]:
            row = "  " + m.ljust(14)
            for s in pb:
                r = res[tbl][str(s)][m]
                row += (f"{r['mean']:.3f}[{r['ci_low']:.2f},{r['ci_high']:.2f}] " if r else "  NA          ")
            logger.info(row)
    logger.info("\n  [same/diff GAP + Cohen's d  (magnitude, mirrors Part I)]")
    for s in pb:
        for m in res["metrics"]:
            g = res["same_diff_gap"][str(s)][m]
            if not g:
                continue
            logger.info(f"    size={s:<3} {MLABEL.get(m, m):9s} same={g['same_mean']:.3f} "
                        f"diff={g['diff_mean']:.3f} gap={g['gap']['mean']:+.3f}"
                        f"[{g['gap']['ci_low']:+.3f},{g['gap']['ci_high']:+.3f}] "
                        f"d={g['cohens_d'] if g['cohens_d'] is None else round(g['cohens_d'], 2)} "
                        f"p={g['perm_p']}")
    logger.info("=" * 100)


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic instrument check — no network, no data, no model.
    Build expression vectors with a KNOWN, tunable drug effect and confirm:
      * signal present  -> single-cell accuracy well above 0.50 AND gap>0 (instrument sees drug)
      * signal absent   -> single-cell accuracy ~0.50 (instrument does not hallucinate signal)
    This is the analogue of the MOA positive control: proves the benchmark can detect drug signal
    when it exists, so a real null is interpretable.
    """
    rng = np.random.RandomState(args.seed)
    P = 200
    panel_index = {f"G{i}": i for i in range(P)}

    def synth(n_cl, n_drug, cells, drug_scale, noise=1.0):
        by = defaultdict(lambda: defaultdict(lambda: {"vecs": [], "moa": None, "dose": 1.0}))
        for cl in range(n_cl):
            cl_base = rng.randn(P) * 0.5
            for d in range(n_drug):
                sig = rng.randn(P) * drug_scale            # drug-specific mean shift
                for _ in range(cells):
                    v = cl_base + sig + rng.randn(P) * noise
                    by[f"cl{cl}"][f"drug{d}"]["vecs"].append(v.astype(np.float32))
                by[f"cl{cl}"][f"drug{d}"]["moa"] = f"moa{d % 3}"
        return by

    class NS:  # lightweight args holder
        pass
    def run(drug_scale):
        a = NS()
        for k, v in vars(args).items():
            setattr(a, k, v)
        a.pb_sizes = "1,8"
        a.n_celllines = 6
        a.drugs_per_cellline = 6
        a.min_cells_per_drug = 20
        a.n_trials = 150
        by = synth(6, 6, 24, drug_scale, noise=1.0)
        return run_benchmark(by, {}, a, np.random.RandomState(args.seed))

    logger.info("SELFTEST: signal-present (drug_scale=1.0) ...")
    r_sig = run(1.0)
    logger.info("SELFTEST: signal-absent (drug_scale=0.0) ...")
    r_null = run(0.0)

    acc_sig = r_sig["discrimination_accuracy"]["1"]["cosine_expr"]["mean"]
    acc_null = r_null["discrimination_accuracy"]["1"]["cosine_expr"]["mean"]
    gap_sig = r_sig["same_diff_gap"]["1"]["cosine_expr"]["gap"]["mean"]
    logger.info(f"  signal-present  single-cell cosine acc = {acc_sig:.3f}  (expect >> 0.5), "
                f"gap = {gap_sig:+.3f} (expect > 0)")
    logger.info(f"  signal-absent   single-cell cosine acc = {acc_null:.3f}  (expect ~ 0.5)")
    ok = (acc_sig > 0.75) and (abs(acc_null - 0.5) < 0.06) and (gap_sig > 0)
    out = {"selftest": True, "passed": bool(ok),
           "signal_present": r_sig, "signal_absent": r_null}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic instrument check (no network/data/model) and exit")
    # streaming
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--rows_per_shard", type=int, default=150000)
    ap.add_argument("--shard_seed", type=int, default=7)
    ap.add_argument("--cells_per_drug", type=int, default=40, help="per (cell_line,drug) collection cap")
    ap.add_argument("--max_cells_total", type=int, default=400000, help="global collection budget")
    ap.add_argument("--held_out_drugs_file", default=None,
                    help="optional held_out_drugs.json -> restrict to tier2 (unseen) drugs")
    ap.add_argument("--collect_controls", action="store_true",
                    help="also pool DMSO controls per cell line and add the cosine_shift metric "
                         "(pooled, not plate-matched — a documented approximation)")
    # benchmark
    ap.add_argument("--pb_sizes", default="1,15", help="pseudobulk sizes; include 1 for single-cell")
    ap.add_argument("--min_cells_per_drug", type=int, default=30)
    ap.add_argument("--n_celllines", type=int, default=30)
    ap.add_argument("--drugs_per_cellline", type=int, default=25)
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--topn", type=int, default=100, help="top-N expressed genes for the topn_tau rank metric")
    ap.add_argument("--de_k", type=int, default=50, help="top-K movers vs control for the de_delta rank metric")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    # panel (for selftest it is synthesized; for real runs default to the standard filename)
    ap.add_argument("--panel_file", default=None,
                    help="path to l1000_panel.json (defaults to ../../shared/l1000_panel.json)")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    # locate the panel
    panel_file = args.panel_file
    if panel_file is None:
        for cand in ("l1000_panel.json", os.path.join(_ROOT, "shared", "l1000_panel.json"),
                     os.path.join(_HERE, "l1000_panel.json")):
            if os.path.exists(cand):
                panel_file = cand
                break
    if not panel_file or not os.path.exists(panel_file):
        raise SystemExit("Could not find l1000_panel.json; pass --panel_file explicitly.")
    panel = json.load(open(panel_file))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    logger.info(f"Panel: {P} genes from {panel_file}")

    rng = np.random.RandomState(args.seed)
    by_cl_drug, ctrl_by_cl = stream_panel_vectors(args, panel_index, P)
    res = run_benchmark(by_cl_drug, ctrl_by_cl, args, rng)
    res["config"] = {k: v for k, v in vars(args).items()}
    res["panel_size"] = P
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print_report(res)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
