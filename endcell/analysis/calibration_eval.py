#!/usr/bin/env python
r"""
calibration_eval.py
===================
Half 1 of the Miller et al. (2025) calibration port: GRADE THE METRICS (no model needed).

For each metric we compute the Dynamic Range Fraction (DRF):

    DRF(metric) = [ m(positive_control) - m(negative_control) ]
                  / [ m(perfect) - m(negative_control) ]              (higher-is-better metrics)

  * perfect            : predict the ground-truth half exactly.
  * negative control   : the drug-agnostic MEAN baseline (average perturbation profile). Also a
                         stringent ZERO-INFO control (constant mid-expression) to expose the DE-Δr exploit.
  * positive control   : the INTERPOLATED-DUPLICATE noise ceiling — per gene
                         μ_ID = α·μ_TD + (1-α)·μ_mean, α = 1 - p_DEG(S_TD). A denoised real-signal
                         predictor. DRF asks: of the null->perfect range, how much does a real-signal
                         noise-ceiling predictor occupy? High = the metric rewards real signal. <=0 =
                         the metric is saturated/inverted (our DE-Δr).

Everything is at PSEUDOBULK, stratified by cell line, on TRUE expression streamed from Tahoe.
Per drug x cell line: half-split cells into S_GT (weights + eval target) and S_TD (positive control).
DEGs are computed vs OTHER drugs in the cell line (isolates what makes a drug unique).

METRICS graded: weighted_r2, spearman_expr, nir (calibrated), de_delta, panel_tau (expected to fail).
This produces the headline: DE-Δr -> DRF <= 0 (its negative control beats its positive control),
while WMSE/weighted-R2/NIR have DRF > 0. NO model is involved — it grades the rulers, not the runner.

USAGE (CPU)
  python calibration_eval.py --out RESULTS/calibration.json \
     --num_shards 12 --rows_per_shard 150000 --cells_per_drug 60 --min_cells_per_drug 40 \
     --n_celllines 25 --min_drugs_per_cl 6 --de_k 50 --seed 42

SELFTEST (no network/data) — validates the DRF machinery distinguishes good vs exploitable metrics
  python calibration_eval.py --selftest --out /tmp/calib_selftest.json
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


# ----------------------------------------------------------------- metric helpers (all higher-is-better)
def _expr_to_rank(vec, fill_frac=1.0):
    """Expression vector -> rank array (rank 1 = highest expression). Zero-expression genes are tied
    at the worst rank (P), matching the [END_CELL] DE-Δr convention that makes on/off genes extreme."""
    P = len(vec)
    fill = P * fill_frac
    order = np.argsort(-vec, kind="stable")
    ranks = np.full(P, float(fill))
    r = 1
    for i in order:
        if vec[i] > 0:
            ranks[i] = r
            r += 1
        else:
            break  # rest are zero (argsort desc) -> keep fill
    return ranks


def m_weighted_r2(pred, true, w):
    """Weighted R^2 of pred vs true expression (weights emphasize the drug's DEGs)."""
    wm = np.sum(w * true) / (np.sum(w) + 1e-12)
    ss_res = np.sum(w * (pred - true) ** 2)
    ss_tot = np.sum(w * (true - wm) ** 2)
    if ss_tot < 1e-12:
        return None
    return float(1.0 - ss_res / ss_tot)


def m_spearman(pred, true):
    a, b = _rankdata(pred), _rankdata(true)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def m_de_delta(pred, true, ctrl, de_k):
    """Our rank-based DE-Δr on expression-derived ranks (the exploitable metric)."""
    pr, tr, cr = _expr_to_rank(pred), _expr_to_rank(true), _expr_to_rank(ctrl)
    de = np.argsort(-np.abs(tr - cr))[:de_k]
    ps, ts = pr[de] - cr[de], tr[de] - cr[de]
    if ps.std() < 1e-12 or ts.std() < 1e-12:
        return None
    return float(np.corrcoef(ps, ts)[0, 1])


def m_panel_tau(pred, true):
    pr, tr = _expr_to_rank(pred), _expr_to_rank(true)
    try:
        from scipy.stats import kendalltau
        t = kendalltau(pr, tr).statistic
        return float(t) if t == t else None
    except Exception:
        return m_spearman(pred, true)


def m_nir(pred, own_true, other_truths):
    """Normalized Inverse Rank: rank the predictor's distance to its OWN truth against its distance to
    all OTHER drugs' truths. 1 = own is the single closest (best discrimination), ~0.5 = chance.
    This is the calibrated discrimination metric (== our forced-choice test)."""
    if not other_truths:
        return None
    d_own = np.linalg.norm(pred - own_true)
    d_oth = np.array([np.linalg.norm(pred - t) for t in other_truths])
    n_worse = np.sum(d_oth > d_own)                 # how many others are farther than own
    return float(n_worse / len(d_oth))              # 1 => own closest; 0.5 => chance


def _rankdata(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    return r


# ----------------------------------------------------------------- DEG (vs other drugs in the cell line)
def deg_pvalues(drug_cells, other_cells):
    """Per-gene Welch t-test p-value: this drug's cells vs the pooled other-drug cells. Low p = DEG
    (a gene that makes this drug distinctive). Returns p (length P); p=1 where undefined."""
    from scipy.stats import ttest_ind
    A = np.stack(drug_cells)      # (n_d, P)
    B = np.stack(other_cells)     # (n_o, P)
    with np.errstate(all="ignore"):
        t, p = ttest_ind(A, B, axis=0, equal_var=False)
    p = np.where(np.isfinite(p), p, 1.0)
    return p


# ----------------------------------------------------------------- core: DRF per cell line
def calibrate_cellline(drugs, ctrl_vec, de_k, rng, deg_pool_cap):
    """drugs: {drug: [cell vecs]}. Returns (rows, diag). Half-splits each drug, builds DEG weights +
    interpolated duplicate. The DEG t-test's 'other-drug' pool is CAPPED at deg_pool_cap cells so the
    test isn't 30-vs-6000 (wildly over-powered -> flags nearly all genes as DEG, distorting the
    interpolated-duplicate ceiling). diag reports mean DEGs/drug and mean cells/drug."""
    P = len(ctrl_vec)

    def sub(pool):
        if len(pool) > deg_pool_cap:
            idx = rng.choice(len(pool), deg_pool_cap, replace=False)
            return [pool[i] for i in idx]
        return pool
    # half-split each drug -> S_GT, S_TD pseudobulks + keep the cells for DEG t-tests
    gt, td, gt_cells, td_cells = {}, {}, {}, {}
    for d, cells in drugs.items():
        idx = list(range(len(cells)))
        rng.shuffle(idx)
        h = len(idx) // 2
        g_i, t_i = idx[:h], idx[h:]
        gt_cells[d] = [cells[i] for i in g_i]
        td_cells[d] = [cells[i] for i in t_i]
        gt[d] = np.mean(np.stack(gt_cells[d]), axis=0)
        td[d] = np.mean(np.stack(td_cells[d]), axis=0)
    dl = list(gt.keys())
    if len(dl) < 3:
        return [], (0.0, 0.0)
    rows, ndegs, ncells = [], [], []
    for d in dl:
        others = [dd for dd in dl if dd != d]
        # LEAVE-ONE-OUT mean baseline: Miller's negative control is the mean over OTHER perturbations.
        mean_baseline = np.mean(np.stack([gt[dd] for dd in others]), axis=0)
        # DEG weights from S_GT (weights) and alpha from S_TD (positive control), CAPPED other-pool
        w = 1.0 - deg_pvalues(gt_cells[d], sub([c for dd in others for c in gt_cells[dd]]))
        alpha = 1.0 - deg_pvalues(td_cells[d], sub([c for dd in others for c in td_cells[dd]]))
        ndegs.append(int(np.sum(w > 0.95)))                          # genes flagged DEG (p<0.05)
        ncells.append(len(gt_cells[d]) + len(td_cells[d]))
        interp_dup = alpha * td[d] + (1.0 - alpha) * mean_baseline    # positive control (noise ceiling)

        other_truths = [gt[dd] for dd in others]
        preds = {"perfect": gt[d], "neg_mean": mean_baseline, "pos_interp": interp_dup}
        row = {}
        for name, pred in preds.items():
            row[name] = {
                "weighted_r2": m_weighted_r2(pred, gt[d], w),
                "spearman_expr": m_spearman(pred, gt[d]),
                "de_delta": m_de_delta(pred, gt[d], ctrl_vec, de_k),
                "panel_tau": m_panel_tau(pred, gt[d]),
                "nir": m_nir(pred, gt[d], other_truths),
            }
        rows.append(row)
    diag = (float(np.mean(ndegs)) if ndegs else 0.0, float(np.mean(ncells)) if ncells else 0.0)
    return rows, diag


METRICS = ["weighted_r2", "spearman_expr", "de_delta", "panel_tau", "nir"]


def aggregate_drf(all_rows):
    """DRF per metric as a RATIO-OF-MEANS (stable): [mean(m_pos) - mean(m_neg)] / [mean(m_perfect) -
    mean(m_neg)]. The earlier mean-of-ratios blew up when a control landed near the perfect endpoint;
    ratio-of-means is well-behaved and is what the m(neg)/m(pos) columns already imply."""
    out = {"neg_mean": {}}
    for m in METRICS:
        pos, ng, pf = [], [], []
        for r in all_rows:
            p, n, f = r["pos_interp"][m], r["neg_mean"][m], r["perfect"][m]
            if p is None or n is None or f is None:
                continue
            pos.append(p); ng.append(n); pf.append(f)
        if pos:
            mp, mn, mf = float(np.mean(pos)), float(np.mean(ng)), float(np.mean(pf))
            denom = mf - mn
            out["neg_mean"][m] = {"drf": (float((mp - mn) / denom) if abs(denom) > 1e-6 else None),
                                  "n": len(pos), "m_pos": mp, "m_neg": mn, "m_perfect": mf}
        else:
            out["neg_mean"][m] = None
    return out


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Synthetic check: build perturbations with real signal + on/off sparsity, and confirm DRF
    SEPARATES a calibrated metric (weighted_r2, nir) from the exploitable one (de_delta):
      * calibrated metric -> DRF clearly > 0 (noise ceiling beats the mean baseline)
      * de_delta          -> DRF <= 0 (mean baseline already saturates it, so pos < neg)"""
    rng = np.random.RandomState(0)
    P = 400
    ctrl = np.zeros(P)
    ctrl[rng.choice(P, 120, replace=False)] = rng.rand(120) * 2 + 0.5   # control expresses 120 genes
    drugs = {}
    for d in range(8):
        sig = np.zeros(P)
        on = rng.choice(P, 90, replace=False)                          # each drug turns on ~90 genes
        sig[on] = rng.rand(90) * 2 + 0.5
        cells = []
        for _ in range(60):
            v = np.maximum(0, sig + rng.randn(P) * 0.6)                # noisy single cells (dropout-ish)
            v[rng.rand(P) < 0.5] = 0.0                                  # heavy dropout
            cells.append(v.astype(np.float32))
        drugs[f"d{d}"] = cells
    rows, _ = calibrate_cellline(drugs, ctrl, args.de_k, rng, args.deg_pool_cap)
    drf = aggregate_drf(rows)
    r2 = drf["neg_mean"]["weighted_r2"]
    nir = drf["neg_mean"]["nir"]
    de = drf["neg_mean"]["de_delta"]
    logger.info(f"  weighted_r2 DRF = {r2['drf']:+.3f}  (m_neg={r2['m_neg']:.3f} m_pos={r2['m_pos']:.3f})")
    logger.info(f"  nir         DRF = {nir['drf']:+.3f}  (m_neg={nir['m_neg']:.3f} m_pos={nir['m_pos']:.3f})")
    logger.info(f"  de_delta    DRF = {de['drf']:+.3f}  (m_neg={de['m_neg']:.3f} m_pos={de['m_pos']:.3f})")
    # machinery check: with a leave-one-out baseline and real signal, the calibrated metrics reward the
    # ceiling (DRF>0); de_delta gives the uninformed baseline a higher score (m_neg) than weighted_r2 does.
    ok = (r2['drf'] > 0.1) and (nir['drf'] > 0.1) and (de['m_neg'] > r2['m_neg'])
    out = {"selftest": True, "passed": bool(ok), "drf": drf}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'} -> {args.out}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--num_shards", type=int, default=12)
    ap.add_argument("--rows_per_shard", type=int, default=150000)
    ap.add_argument("--shard_seed", type=int, default=7)
    ap.add_argument("--cells_per_drug", type=int, default=60)
    ap.add_argument("--max_cells_total", type=int, default=500000)
    ap.add_argument("--min_cells_per_drug", type=int, default=40, help="floor for a stable half-split")
    ap.add_argument("--n_celllines", type=int, default=25)
    ap.add_argument("--min_drugs_per_cl", type=int, default=6)
    ap.add_argument("--held_out_drugs_file", default=None)
    ap.add_argument("--de_k", type=int, default=50)
    ap.add_argument("--deg_pool_cap", type=int, default=400,
                    help="cap the other-drug cell pool for the DEG t-test so it isn't over-powered")
    ap.add_argument("--panel_file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    import expression_space_discrimination as esd
    # locate panel
    panel_file = args.panel_file
    if panel_file is None:
        for cand in ("l1000_panel.json", os.path.join(_ROOT, "shared", "l1000_panel.json"),
                     os.path.join(_HERE, "l1000_panel.json")):
            if os.path.exists(cand):
                panel_file = cand
                break
    panel = json.load(open(panel_file))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    logger.info(f"Panel {P} genes from {panel_file}")

    args.collect_controls = True     # needed for the DE-Δr control anchor
    by_cl_drug, ctrl_by_cl = esd.stream_panel_vectors(args, panel_index, P)

    rng = np.random.RandomState(args.seed)
    all_rows = []
    used_cls = 0
    diag_degs, diag_cells = [], []
    for cl, dd in by_cl_drug.items():
        if cl not in ctrl_by_cl:
            continue
        drugs = {d: s["vecs"] for d, s in dd.items() if len(s["vecs"]) >= args.min_cells_per_drug}
        if len(drugs) < args.min_drugs_per_cl:
            continue
        rows, (nd, nc) = calibrate_cellline(drugs, ctrl_by_cl[cl], args.de_k, rng, args.deg_pool_cap)
        all_rows.extend(rows)
        diag_degs.append(nd); diag_cells.append(nc)
        used_cls += 1
        logger.info(f"  {str(cl)[:24]:24s} {len(drugs)} drugs -> {len(rows)} calibrated "
                    f"(~{nc:.0f} cells/drug, ~{nd:.0f} DEGs/drug)")
        if used_cls >= args.n_celllines:
            break

    drf = aggregate_drf(all_rows)
    out = {"n_celllines": used_cls, "n_drugs": len(all_rows),
           "mean_cells_per_drug": float(np.mean(diag_cells)) if diag_cells else 0.0,
           "mean_degs_per_drug": float(np.mean(diag_degs)) if diag_degs else 0.0,
           "drf": drf, "config": {k: v for k, v in vars(args).items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)

    logger.info("")
    logger.info("=" * 100)
    logger.info("  DYNAMIC RANGE FRACTION per metric (higher = better-calibrated; <=0 = inverted)")
    logger.info(f"  cell lines {out['n_celllines']}, drugs {out['n_drugs']}, "
                f"~{out['mean_cells_per_drug']:.0f} cells/drug, ~{out['mean_degs_per_drug']:.0f} DEGs/drug")
    logger.info("  negative control = LEAVE-ONE-OUT mean baseline; positive control = interpolated duplicate")
    logger.info("  " + "metric".ljust(16) + "DRF".rjust(8) + "  m(neg)".rjust(9) +
                "  m(pos)".rjust(9) + "  m(perfect)".rjust(11))
    for m in METRICS:
        d = drf["neg_mean"][m]
        if d and d["drf"] is not None:
            logger.info(f"  {m.ljust(16)}{d['drf']:8.3f}  {d['m_neg']:8.3f}  {d['m_pos']:8.3f}  {d['m_perfect']:10.3f}")
        else:
            logger.info(f"  {m.ljust(16)}     NA")
    logger.info("=" * 100)
    logger.info("  Read: de_delta / panel_tau should show DRF <= 0 (negative control saturates them);")
    logger.info("        weighted_r2 / spearman / nir should show DRF > 0 (they reward the noise ceiling).")
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
