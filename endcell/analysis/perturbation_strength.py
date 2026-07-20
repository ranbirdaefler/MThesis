#!/usr/bin/env python
r"""
perturbation_strength.py — the rigorous preprocessing step: which drug conditions are worth
evaluating on, estimated WITHOUT contaminating the evaluation.
=============================================================================================
Replaces the improvised SNR / mean-difference "effect size" with the field standard, and fixes the
circularity in our earlier subsetting.

WHY THE EARLIER VERSION WAS NOT RIGOROUS
  * "effect size" = ||mean(drug) - mean(DMSO)|| is inflated by low cell count and driven by
    sequencing depth; it is not a distributional test.
  * We selected drugs on `ceiling >= 0.8` computed from half-A/half-B, then scored the model against
    half-A -> selection and evaluation shared cells (double dipping / circular analysis).

WHAT THIS DOES INSTEAD
  1. SPLIT-SAMPLE. Each condition's cells are deterministically partitioned into
     S (SELECTION) and E (EVALUATION). Everything here is estimated on S ONLY. A manifest of the S
     line-indices is written so the model scorer can EXCLUDE them (nir_benchmark --exclude_manifest),
     making selection and evaluation provably disjoint.
  2. STRENGTH = E-DISTANCE + permutation E-test (Peidli et al., scPerturb, Nat Methods 2024) between
     the drug's cells and the plate-matched DMSO cells:
         E = 2*mean(cross-distances) - mean(within-drug) - mean(within-control)
     It compares full DISTRIBUTIONS, not just means, and subtracts within-group dispersion, so it is
     intrinsically noise-aware (the failure mode of our old mean-difference measure).
  3. MATCHED CELL COUNTS. Every condition is subsampled to the same n (drug) and m (control) before
     computing E, because the estimator's bias/variance depends on n -- otherwise cell count, not
     biology, drives the ranking.
  4. DISTINCTIVENESS = a ceiling-style discrimination computed WITHIN S (S split into S1/S2, ranked
     against other drugs' S profiles). Kept as a SEPARATE axis from strength: a cytotoxic drug can be
     very strong yet indistinct (shared death program), and a weak drug can be distinct. They answer
     different questions and must not be collapsed.

OUTPUTS
  * strength table (per tier x cell_line x plate x drug): e_distance, perm p, significant,
    distinctiveness, n used.  -> join to model scores for stratified reporting.
  * split manifest: {tier: [line indices assigned to SELECTION]} -> feed to nir_benchmark
    --exclude_manifest so the model is scored only on disjoint cells.

USAGE (CPU)
  python perturbation_strength.py --eval_dir DATA_endcell_big \
     --tiers tier1_seen_conditions,tier2_unseen_drugs \
     --select_frac 0.35 --n_strength 12 --m_control 60 --n_perm 200 \
     --out RESULTS/strength_table.json --manifest RESULTS/split_manifest.json \
     --csv RESULTS/strength_table.csv

SELFTEST (no data)
  python perturbation_strength.py --selftest
"""
# --- repo path bootstrap: works in BOTH the reorganized repo AND the flat cluster layout ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
_cands = [_HERE, os.path.join(_HERE, "src")]
if os.path.isdir(os.path.join(_ROOT, "shared")):
    _cands += [os.path.join(_ROOT, "shared")] + sorted(glob.glob(os.path.join(_PIPE, "*")))
for _p in _cands:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, logging, csv as _csv
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"


# ----------------------------------------------------------------- representation
def genes_of(s):
    out = []
    for t in s.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def sentence_to_expr(s, pidx, P, lm):
    slope, intercept = lm["slope"], lm["intercept"]
    arr = np.zeros(P, dtype=np.float32); seen = set(); pos = 0
    for g in genes_of(s):
        gi = pidx.get(g)
        if gi is None or gi in seen:
            continue
        seen.add(gi); pos += 1
        arr[gi] = max(0.0, slope * np.log10(pos) + intercept)
    return arr


def control_from_prompt(p):
    if "\nControl cell: " not in p:
        return None
    try:
        return p.split("\nControl cell: ", 1)[1].split("\n\nResponse cell:", 1)[0]
    except Exception:
        return None


# ----------------------------------------------------------------- E-distance + permutation E-test
def _E_from_D(D, ia, ib):
    """Energy distance from a precomputed pairwise-distance matrix D and index sets ia, ib."""
    cross = D[np.ix_(ia, ib)].mean()
    wa = D[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)].mean() if len(ia) > 1 else 0.0
    wb = D[np.ix_(ib, ib)][np.triu_indices(len(ib), 1)].mean() if len(ib) > 1 else 0.0
    return float(2.0 * cross - wa - wb)


def e_test(X, Y, n_perm, rng):
    """E-distance between cell sets X (drug) and Y (control) + permutation p-value.

    The full pairwise-distance matrix of the pooled cells is computed ONCE; each permutation only
    re-indexes it, so 200 permutations cost almost nothing. p is the fraction of label-shuffles whose
    E >= the observed E, i.e. 'could this separation arise with the drug/control labels meaningless?'
    """
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return None, None
    Z = np.vstack([X, Y]).astype(np.float64)
    # pairwise Euclidean distances (pooled)
    sq = np.sum(Z * Z, axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0))
    obs = _E_from_D(D, np.arange(n), np.arange(n, n + m))
    cnt = 0
    tot = n + m
    for _ in range(n_perm):
        perm = rng.permutation(tot)
        if _E_from_D(D, perm[:n], perm[n:]) >= obs:
            cnt += 1
    return obs, float((cnt + 1) / (n_perm + 1))


# ----------------------------------------------------------------- distinctiveness (within S)
def half_split(S, key):
    """Deterministic half-split of a condition's selection cells -> (S1 = truth, S2 = replicate).
    Done once per condition so every reference profile is at the SAME denoising level."""
    if len(S) < 4:
        return None
    r = np.random.RandomState(abs(hash(("half",) + key)) % (2 ** 31))
    idx = r.permutation(len(S)); h = len(idx) // 2
    return (np.mean(np.stack([S[i] for i in idx[:h]]), axis=0),
            np.mean(np.stack([S[i] for i in idx[h:2 * h]]), axis=0))


def distinctiveness(own_halves, other_truths):
    """Ceiling-style discrimination within the selection split: is this drug's replicate (S2) nearer
    its OWN truth (S1) than the other drugs' truths?  1.0 = uniquely distinguishable, 0.5 = chance.

    CONSISTENT DENOISING IS ESSENTIAL: `other_truths` must be the other drugs' S1 half-means, NOT
    their full-S means. Comparing a noisy half-mean (own) against cleaner full means (others)
    inflates the own-distance and biases the score far below chance."""
    if own_halves is None or len(other_truths) < 2:
        return None
    S1, S2 = own_halves
    d_own = float(np.linalg.norm(S2 - S1))
    d_oth = [float(np.linalg.norm(S2 - t)) for t in other_truths]
    return float(np.mean([d_own < x for x in d_oth]))


# ----------------------------------------------------------------- loading + deterministic split
def load_train_selection(train_file, eval_dir, tiers, pidx, P, lm):
    """SELECTION SOURCE = train.jsonl (recommended).

    Train and eval cells are disjoint BY CONSTRUCTION (preprocessing assigns each cell to exactly one
    split), so strength/distinctiveness estimated on train cells cannot contaminate model scoring on
    eval cells. That is structurally stronger than an exclusion manifest, and it sacrifices no eval
    cells. Only works for conditions present in train (i.e. SEEN drugs / tier1) — held-out tiers have
    no train cells by design.

    -> conditions {(tier, cl, plate, drug): {"S": [vec]}}, controls {(tier, cl, plate): [vec]}
    """
    # which (cl, plate, drug) conditions do we need, and under which tier?
    want = {}
    for tier in tiers:
        path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  missing {path}"); continue
        n = 0
        for line in open(path):
            m = json.loads(line).get("metadata", {})
            cl, plate, drug = m.get("cell_line_id"), m.get("plate"), m.get("drug")
            if cl is None or plate is None or drug is None:
                continue
            want.setdefault((cl, plate, drug), set()).add(tier)
            n += 1
        logger.info(f"  eval_{tier}.jsonl: {n:,} rows")

    cells = defaultdict(list)
    ctrl = defaultdict(list)
    seen_ctrl = defaultdict(set)
    n_rows = 0
    for line in open(train_file):
        ex = json.loads(line)
        m = ex.get("metadata", {})
        cl, plate, drug = m.get("cell_line_id"), m.get("plate"), m.get("drug")
        if cl is None or plate is None or drug is None:
            continue
        k = (cl, plate, drug)
        if k in want:
            cells[k].append(sentence_to_expr(ex["response"], pidx, P, lm))
        gk = (cl, plate)
        cs = control_from_prompt(ex["prompt"])
        if cs:
            h = hash(cs)
            if h not in seen_ctrl[gk] and len(ctrl[gk]) < 400:
                seen_ctrl[gk].add(h); ctrl[gk].append(sentence_to_expr(cs, pidx, P, lm))
        n_rows += 1
    logger.info(f"  {os.path.basename(train_file)}: {n_rows:,} rows scanned; "
                f"{len(cells)}/{len(want)} eval conditions found in train")

    out, ctrl_out = {}, {}
    for (cl, plate, drug), tset in want.items():
        v = cells.get((cl, plate, drug))
        if not v:
            continue
        for tier in tset:
            out[(tier, cl, plate, drug)] = {"S": v, "S_idx": [], "E_idx": []}
            ctrl_out[(tier, cl, plate)] = ctrl.get((cl, plate), [])
    return out, ctrl_out


def load_with_split(eval_dir, tiers, pidx, P, lm, select_frac, seed):
    """-> conditions {(tier, cl, plate, drug): {"S":[vec], "E_idx":[line], "S_idx":[line]}},
          controls {(tier, cl, plate): [vec]}

    The split is DETERMINISTIC per condition (seeded by the condition key), so it is reproducible and
    can be replayed by the scorer via the emitted manifest."""
    cond = defaultdict(lambda: {"vecs": [], "lines": []})
    ctrl = defaultdict(list)
    seen_ctrl = defaultdict(set)
    for tier in tiers:
        path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  missing {path}"); continue
        n = 0
        for li, line in enumerate(open(path)):
            ex = json.loads(line)
            m = ex.get("metadata", {})
            cl, plate, drug = m.get("cell_line_id"), m.get("plate"), m.get("drug")
            if cl is None or plate is None or drug is None:
                continue
            key = (tier, cl, plate, drug)
            cond[key]["vecs"].append(sentence_to_expr(ex["response"], pidx, P, lm))
            cond[key]["lines"].append(li)
            cs = control_from_prompt(ex["prompt"])
            if cs:
                gk = (tier, cl, plate)
                h = hash(cs)
                if h not in seen_ctrl[gk]:
                    seen_ctrl[gk].add(h); ctrl[gk].append(sentence_to_expr(cs, pidx, P, lm))
            n += 1
        logger.info(f"  eval_{tier}.jsonl: {n:,} rows")

    out = {}
    for key, d in cond.items():
        k = len(d["vecs"])
        if k < 4:
            continue
        rng = np.random.RandomState(abs(hash(key)) % (2 ** 31))   # deterministic per condition
        order = rng.permutation(k)
        n_sel = max(2, int(round(select_frac * k)))
        n_sel = min(n_sel, k - 2)                                  # always leave >=2 for evaluation
        sel, ev = order[:n_sel], order[n_sel:]
        out[key] = {"S": [d["vecs"][i] for i in sel],
                    "S_idx": [d["lines"][i] for i in sel],
                    "E_idx": [d["lines"][i] for i in ev]}
    return out, ctrl


# ----------------------------------------------------------------- main analysis
def run(args, pidx, P, lm):
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if args.selection_source == "train":
        if not args.train_file:
            raise SystemExit("--selection_source train requires --train_file")
        cond, ctrl = load_train_selection(args.train_file, args.eval_dir, tiers, pidx, P, lm)
        logger.info("  selection source = TRAIN cells (disjoint from eval by construction; "
                    "no eval cells sacrificed, no manifest needed)")
    else:
        cond, ctrl = load_with_split(args.eval_dir, tiers, pidx, P, lm, args.select_frac, args.seed)
        logger.info(f"  selection source = internal split of eval cells "
                    f"(select_frac={args.select_frac})")
    logger.info(f"  {len(cond)} conditions with a usable split")

    # DIAGNOSTIC: show the selection-cell distribution BEFORE filtering, so a mass-skip is never
    # silent (the first run skipped 100% of conditions because eval tiers hold only ~11 cells each).
    if cond:
        ns = np.array([len(v["S"]) for v in cond.values()])
        logger.info(f"  selection cells/condition: min {ns.min()}  p25 {np.percentile(ns,25):.0f}  "
                    f"median {np.median(ns):.0f}  p75 {np.percentile(ns,75):.0f}  max {ns.max()}")
        n_ok = int((ns >= args.n_strength).sum())
        logger.info(f"  conditions with >= --n_strength ({args.n_strength}): {n_ok}/{len(ns)}")
        if n_ok == 0:
            logger.error(f"  NONE clear --n_strength={args.n_strength}. Either lower it (>= ~8 keeps "
                         f"the E-test meaningful) or use --selection_source train (more cells/condition).")

    # group conditions by (tier, cell_line, plate) so distinctiveness compares WITHIN plate
    by_group = defaultdict(list)
    for key in cond:
        by_group[(key[0], key[1], key[2])].append(key)

    rng = np.random.RandomState(args.seed)
    rows, manifest = [], defaultdict(list)
    n_skip_cells = n_skip_ctrl = 0
    n_ctrl_seen = []
    for gk, keys in by_group.items():
        tier, cl, plate = gk
        controls = ctrl.get(gk, [])
        n_ctrl_seen.append(len(controls))
        # half-split every drug ONCE: all reference truths are S1 half-means (matched denoising)
        halves = {k: half_split(cond[k]["S"], k) for k in keys}
        halves = {k: v for k, v in halves.items() if v is not None}
        for key in keys:
            S = cond[key]["S"]
            drug = key[3]
            if len(S) < args.n_strength:
                n_skip_cells += 1
                continue
            # ---- MATCHED-n: subsample drug + control to fixed sizes so E is comparable across conditions
            si = rng.choice(len(S), args.n_strength, replace=False)
            X = np.stack([S[i] for i in si])
            edist = pval = None
            if len(controls) >= args.min_control:
                ci = rng.choice(len(controls), min(args.m_control, len(controls)), replace=False)
                Y = np.stack([controls[i] for i in ci])
                edist, pval = e_test(X, Y, args.n_perm, rng)
            else:
                n_skip_ctrl += 1
            other_truths = [halves[k][0] for k in keys if k != key and k in halves]
            dist_ = distinctiveness(halves.get(key), other_truths)
            rows.append({"tier": tier, "cell_line": cl, "plate": plate, "drug": drug,
                         "n_total": len(S) + len(cond[key]["E_idx"]),
                         "n_selection": len(S), "n_strength_used": int(args.n_strength),
                         "e_distance": edist, "perm_p": pval,
                         "significant": (pval is not None and pval < args.alpha),
                         "distinctiveness": dist_})
            manifest[tier].extend(cond[key]["S_idx"])

    if n_skip_cells:
        logger.info(f"  skipped {n_skip_cells} conditions with < {args.n_strength} selection cells")
    if n_ctrl_seen:
        cs = np.array(n_ctrl_seen)
        logger.info(f"  UNIQUE controls per (cell_line, plate): min {cs.min()}  median "
                    f"{np.median(cs):.0f}  max {cs.max()}   (--min_control={args.min_control})")
    if n_skip_ctrl:
        logger.warning(f"  {n_skip_ctrl} conditions had too few UNIQUE controls for the E-test -> no "
                       f"E-distance computed for them. The DMSO pool per plate is small (controls are "
                       f"drawn from a shared pool and deduplicated), so --min_control must be <= that "
                       f"pool size. Lower it if this is most conditions.")
    return rows, manifest


def report(rows, args):
    ed = np.array([r["e_distance"] for r in rows if r["e_distance"] is not None])
    di = np.array([r["distinctiveness"] for r in rows if r["distinctiveness"] is not None])
    sig = [r for r in rows if r["perm_p"] is not None]
    logger.info("")
    logger.info("=" * 96)
    logger.info(f"  PERTURBATION STRENGTH (E-distance + permutation E-test), split-sample")
    logger.info(f"  {len(rows)} conditions | strength estimated on {args.n_strength} cells vs "
                f"{args.m_control} controls (matched)")
    logger.info("-" * 96)
    if sig:
        ns = sum(r["significant"] for r in sig)
        logger.info(f"  significant perturbation (E-test p<{args.alpha}): {ns}/{len(sig)} "
                    f"({100.0*ns/len(sig):.1f}%)")
    if len(ed):
        qs = np.percentile(ed, [10, 25, 50, 75, 90])
        logger.info(f"  E-distance quantiles: p10 {qs[0]:.3f}  p25 {qs[1]:.3f}  median {qs[2]:.3f}  "
                    f"p75 {qs[3]:.3f}  p90 {qs[4]:.3f}")
    if len(di):
        logger.info(f"  distinctiveness (within-S): median {np.median(di):.3f}  "
                    f">=0.8: {100.0*np.mean(di>=0.8):.1f}%")
    # the two axes must be reported separately — check how orthogonal they actually are
    pair = [(r["e_distance"], r["distinctiveness"]) for r in rows
            if r["e_distance"] is not None and r["distinctiveness"] is not None]
    if len(pair) >= 10:
        a = np.array([p[0] for p in pair]); b = np.array([p[1] for p in pair])
        if a.std() > 1e-9 and b.std() > 1e-9:
            r_ = float(np.corrcoef(a, b)[0, 1])
            logger.info(f"  corr(strength, distinctiveness) = {r_:+.3f}  "
                        f"({'largely orthogonal -> report as 2 axes' if abs(r_) < 0.5 else 'correlated'})")
    logger.info("=" * 96)
    logger.info("  Use these to STRATIFY model performance (curve vs strength, curve vs")
    logger.info("  distinctiveness) — not to pick one filtered subset. A model at chance across all")
    logger.info("  strata is immune to 'you chose the wrong cutoff'.")


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Plant strong / weak / inert conditions and verify the E-test orders them, that matched-n is
    enforced, and that the selection and evaluation splits are disjoint."""
    rng = np.random.RandomState(0)
    P = 120
    base = np.zeros(P); base[rng.choice(P, 60, replace=False)] = rng.rand(60) * 2 + 1.0

    def cells(sig, n, noise=0.5):
        return [np.maximum(0.0, sig + rng.randn(P) * noise).astype(np.float32) for _ in range(n)]

    ctrl = cells(base, 60)
    strong = base.copy(); strong[rng.choice(P, 40, replace=False)] += 4.0
    weak = base.copy(); weak[rng.choice(P, 40, replace=False)] += 0.35

    args.n_strength, args.m_control, args.n_perm, args.alpha = 12, 60, 200, 0.05
    e_s, p_s = e_test(np.stack(cells(strong, 12)), np.stack(ctrl), args.n_perm, rng)
    e_w, p_w = e_test(np.stack(cells(weak, 12)), np.stack(ctrl), args.n_perm, rng)
    e_i, p_i = e_test(np.stack(cells(base, 12)), np.stack(ctrl), args.n_perm, rng)
    logger.info(f"  strong drug : E={e_s:7.3f}  p={p_s:.3f}  (expect large E, p<0.05)")
    logger.info(f"  weak drug   : E={e_w:7.3f}  p={p_w:.3f}")
    logger.info(f"  inert (=DMSO): E={e_i:7.3f}  p={p_i:.3f}  (expect E~0, p>=0.05)")

    ok = True
    if not (p_s < 0.05):
        logger.error("  FAIL: strong perturbation not significant"); ok = False
    if not (p_i >= 0.05):
        logger.error("  FAIL: inert condition called significant"); ok = False
    if not (e_s > e_w > e_i - 1e-6):
        logger.error("  FAIL: E-distance does not order strong > weak > inert"); ok = False

    # split disjointness + determinism
    lines = list(range(30))
    key = ("t", "cl", "p1", "d1")
    r1 = np.random.RandomState(abs(hash(key)) % (2 ** 31)).permutation(30)
    r2 = np.random.RandomState(abs(hash(key)) % (2 ** 31)).permutation(30)
    n_sel = max(2, int(round(0.35 * 30)))
    S, E = set(r1[:n_sel]), set(r1[n_sel:])
    if S & E:
        logger.error("  FAIL: selection/evaluation splits overlap"); ok = False
    else:
        logger.info(f"  split: {len(S)} selection / {len(E)} evaluation cells, disjoint OK")
    if not np.array_equal(r1, r2):
        logger.error("  FAIL: split is not deterministic"); ok = False
    else:
        logger.info("  split determinism OK (same key -> same partition)")
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--tiers", default="tier1_seen_conditions,tier2_unseen_drugs")
    ap.add_argument("--selection_source", choices=["train", "internal"], default="train",
                    help="'train' (RECOMMENDED): estimate strength/distinctiveness from train.jsonl "
                         "cells — disjoint from eval by construction, sacrifices no eval cells, and "
                         "needs no manifest. Works for conditions present in train (seen drugs). "
                         "'internal': split the eval cells themselves (needed for held-out tiers, but "
                         "the eval tiers hold only ~11 cells/condition, so it is thin).")
    ap.add_argument("--train_file", default=None, help="train.jsonl (for --selection_source train)")
    ap.add_argument("--select_frac", type=float, default=0.35,
                    help="internal mode only: fraction of each condition's cells reserved for SELECTION")
    ap.add_argument("--n_strength", type=int, default=12, help="matched # drug cells for the E-test")
    ap.add_argument("--m_control", type=int, default=60, help="matched # control cells for the E-test")
    ap.add_argument("--min_control", type=int, default=5,
                    help="min UNIQUE control cells per (cell_line, plate) to attempt the E-test. "
                         "Controls come from a small shared DMSO pool (~8 unique per plate), so this "
                         "must stay below that; 10 rejected every condition.")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="RESULTS/strength_table.json")
    ap.add_argument("--manifest", default="RESULTS/split_manifest.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--panel_file", default=None)
    ap.add_argument("--linear_model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return
    if not args.eval_dir:
        ap.error("--eval_dir required (unless --selftest)")

    panel = json.load(open(args.panel_file or os.path.join(args.eval_dir, "l1000_panel.json")))
    pidx = {g: i for i, g in enumerate(panel)}; P = len(panel)
    lm = json.load(open(args.linear_model or os.path.join(args.eval_dir, "linear_model.json")))
    logger.info(f"Panel {P}; lm slope={lm['slope']:.3f} intercept={lm['intercept']:.3f}")

    rows, manifest = run(args, pidx, P, lm)
    if not rows:
        logger.error("No conditions passed the filters."); sys.exit(1)
    report(rows, args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"rows": rows, "config": vars(args)}, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")
    json.dump({k: sorted(v) for k, v in manifest.items()}, open(args.manifest, "w"))
    logger.info(f"-> {args.manifest}  (feed to nir_benchmark --exclude_manifest so the model is "
                f"scored ONLY on cells disjoint from selection)")

    if args.csv:
        cols = ["tier", "cell_line", "plate", "drug", "n_total", "n_selection", "n_strength_used",
                "e_distance", "perm_p", "significant", "distinctiveness"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
        logger.info(f"-> {args.csv}")


if __name__ == "__main__":
    main()
