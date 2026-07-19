#!/usr/bin/env python
r"""
drug_difficulty_atlas.py  —  TEST 1: which drugs actually do anything, and which are confusable?
=================================================================================================
Answers the question raised by the advisor:

  "All of these experiments seem to assume that each drug induces a substantial change in the
   expression profile, but that may not always be true. Some drugs might produce only subtle
   changes, or different drugs may induce very similar expression profiles."

This is a CONFOUND CHECK on every aggregate NIR number we report. NIR asks "is this profile closer
to its own drug than to other drugs", which silently assumes each drug (a) does something and
(b) does something distinct. If many drugs are INERT (treated ~ DMSO) or REDUNDANT (two drugs with
near-identical profiles), the task is impossible for those drugs and they drag the average toward
chance no matter how good the model is. This script measures that structure with NO MODEL involved.

PER DRUG x CELL LINE (at pseudobulk, on real cells), it computes:
  * effect_size      : || drug_pb - DMSO_control_pb ||           (does the drug do anything?)
  * replicate_noise  : || halfA_pb - halfB_pb ||                 (intrinsic noise at this cell count)
  * snr              : effect_size / replicate_noise             -> INERT if < 1 (drug moves the
                       profile less than its own replicate noise = undetectable by anyone)
  * nn_dist          : distance to the NEAREST OTHER drug in the same cell line
  * isolation        : nn_dist / replicate_noise                 -> REDUNDANT if < 1 (another drug is
                       closer than your own replicate -> not identifiable by anyone)
  * ceiling_nir      : NIR of a real held-out replicate (halfB vs every drug's halfA) -> the PER-DRUG
                       ceiling. This is the same quantity nir_benchmark reports, but per drug instead
                       of averaged, so we can see the distribution behind the 0.88.
  * n_deg            : # genes differentially expressed vs the DMSO controls (Welch t-test)

It also reports MoA structure (within-MoA vs between-MoA distances) and, optionally, whether
CHEMICAL similarity predicts RESPONSE similarity (--drug_features) — which is the prerequisite gate
for the structure-injection arm: if chemically similar drugs do NOT have similar responses, a
structure/CLIP embedding cannot help, and we learn that for free before building it.

SELF-CONTAINED: no local imports, so it runs in the flat cluster layout and the reorganized repo.
Reads only the eval/train jsonl + linear_model.json + l1000_panel.json in --eval_dir.

USAGE (CPU)
  python drug_difficulty_atlas.py --eval_dir DATA_endcell_big \
     --tiers tier1_seen_conditions,tier2_unseen_drugs,tier3_unseen_combos,tier4_dose_interpolation \
     --min_cells 8 --out RESULTS/drug_atlas.json --csv RESULTS/drug_atlas.csv \
     --profiles RESULTS/drug_atlas_profiles.npz

SELFTEST (no data/network) — plants inert / redundant / distinct drugs and checks we flag them:
  python drug_difficulty_atlas.py --selftest
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENTINEL = "[END_CELL]"


# ----------------------------------------------------------------- representation (self-contained)
def genes_of(sentence):
    out = []
    for t in sentence.strip().split():
        if t == SENTINEL:
            break
        out.append(t)
    return out


def control_from_prompt(prompt):
    """Prompt format: '...\nControl cell: <sentence>\n\nResponse cell:'"""
    if "\nControl cell: " not in prompt:
        return None
    try:
        return prompt.split("\nControl cell: ", 1)[1].split("\n\nResponse cell:", 1)[0]
    except Exception:
        return None


def sentence_to_expr(sentence, panel_index, P, lm):
    """Decode a cell sentence -> expression vector (expr = slope*log10(rank)+intercept, absent=0).
    Identical to the decode used by nir_benchmark's expr-NIR, so numbers are comparable."""
    slope, intercept = lm["slope"], lm["intercept"]
    arr = np.zeros(P)
    seen, pos = set(), 0
    for g in genes_of(sentence):
        gi = panel_index.get(g)
        if gi is None or gi in seen:
            continue
        seen.add(gi); pos += 1
        arr[gi] = max(0.0, slope * np.log10(pos) + intercept)
    return arr


def pb(vecs):
    return np.mean(np.stack(vecs), axis=0)


# ----------------------------------------------------------------- core per-cell-line analysis
def analyze_cellline(drug_cells, ctrl_vecs, rng, min_cells, deg_max_cells=400):
    """drug_cells: {drug: [expr vectors]}. ctrl_vecs: [expr vectors] (pooled DMSO for this cell line).
    Returns per-drug dict rows + the per-drug halfA profiles (for the geometry matrix)."""
    drugs = {d: v for d, v in drug_cells.items() if len(v) >= min_cells}
    if len(drugs) < 3:
        return [], {}
    ctrl_pb = pb(ctrl_vecs) if ctrl_vecs else None

    A, B, full = {}, {}, {}
    for d, vecs in drugs.items():
        idx = list(range(len(vecs))); rng.shuffle(idx)
        h = len(idx) // 2
        A[d] = pb([vecs[i] for i in idx[:h]])
        B[d] = pb([vecs[i] for i in idx[h:]])
        full[d] = pb(vecs)

    dl = sorted(A.keys())
    rows = []
    for d in dl:
        others = [o for o in dl if o != d]
        rep_noise = float(np.linalg.norm(A[d] - B[d]))
        eff = float(np.linalg.norm(full[d] - ctrl_pb)) if ctrl_pb is not None else None
        # cosine distance to control (scale-free view of the effect)
        eff_cos = None
        if ctrl_pb is not None:
            na, nb = np.linalg.norm(full[d]), np.linalg.norm(ctrl_pb)
            if na > 1e-9 and nb > 1e-9:
                eff_cos = float(1.0 - np.dot(full[d], ctrl_pb) / (na * nb))
        # nearest other drug (compare like-for-like: halfA vs halfA)
        dists = {o: float(np.linalg.norm(A[d] - A[o])) for o in others}
        nn = min(dists, key=dists.get)
        nn_dist = dists[nn]
        # per-drug ceiling NIR: a REAL held-out replicate (halfB) vs every drug's halfA
        d_own = float(np.linalg.norm(B[d] - A[d]))
        d_oth = [float(np.linalg.norm(B[d] - A[o])) for o in others]
        ceiling_nir = float(np.mean([d_own < x for x in d_oth]))

        rows.append({
            "drug": d, "n_cells": len(drugs[d]),
            "effect_size": eff, "effect_cosine": eff_cos,
            "replicate_noise": rep_noise,
            "snr": (eff / rep_noise) if (eff is not None and rep_noise > 1e-9) else None,
            "nn_drug": nn, "nn_dist": nn_dist,
            "isolation": (nn_dist / rep_noise) if rep_noise > 1e-9 else None,
            "ceiling_nir": ceiling_nir,
        })
    return rows, A


def ceiling_at_n(drug_cells, n, rng, n_reps=3):
    """Mean per-drug CEILING NIR when each drug is subsampled to exactly n cells.

    This is the achievable-discrimination bar as a function of how many cells you aggregate. Run it
    with same_plate grouping and it is LEAK-FREE, so the curve answers the only question that sets a
    real target: does honest drug identifiability grow with aggregation (task winnable), or does it
    saturate near chance (there is almost nothing to capture)?"""
    drugs = [d for d, v in drug_cells.items() if len(v) >= n]
    if len(drugs) < 3:
        return None
    vals = []
    for _ in range(n_reps):
        A, B = {}, {}
        for d in drugs:
            v = drug_cells[d]
            idx = rng.choice(len(v), n, replace=False)
            h = n // 2
            A[d] = pb([v[i] for i in idx[:h]])
            B[d] = pb([v[i] for i in idx[h:2 * h]])
        for d in drugs:
            others = [o for o in drugs if o != d]
            d_own = float(np.linalg.norm(B[d] - A[d]))
            d_oth = [float(np.linalg.norm(B[d] - A[o])) for o in others]
            vals.append(float(np.mean([d_own < x for x in d_oth])))
    return (float(np.mean(vals)), len(drugs)) if vals else None


def run_cell_sweep(by_cl, ns, rng, min_drugs):
    """Ceiling NIR vs cells-per-drug, pooled over comparison groups."""
    out = {}
    for n in ns:
        per_group, n_drugs = [], 0
        for g, dd in by_cl.items():
            if len([d for d, v in dd.items() if len(v) >= n]) < min_drugs:
                continue
            r = ceiling_at_n(dd, n, rng)
            if r is not None:
                per_group.append(r[0]); n_drugs += r[1]
        if per_group:
            out[n] = {"ceiling_nir": float(np.mean(per_group)),
                      "n_groups": len(per_group), "n_drugs": n_drugs}
    return out


def add_degs(rows, drug_cells, ctrl_vecs, min_cells, rng, cap=400):
    """Welch t-test per gene: drug cells vs DMSO control cells -> # DEG (p<0.05)."""
    try:
        from scipy.stats import ttest_ind
    except Exception:
        return
    if not ctrl_vecs:
        return
    C = np.stack(ctrl_vecs if len(ctrl_vecs) <= cap else
                 [ctrl_vecs[i] for i in rng.choice(len(ctrl_vecs), cap, replace=False)])
    by_drug = {r["drug"]: r for r in rows}
    for d, vecs in drug_cells.items():
        if d not in by_drug or len(vecs) < min_cells:
            continue
        D = np.stack(vecs if len(vecs) <= cap else
                     [vecs[i] for i in rng.choice(len(vecs), cap, replace=False)])
        with np.errstate(all="ignore"):
            _, p = ttest_ind(D, C, axis=0, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        by_drug[d]["n_deg"] = int((p < 0.05).sum())


# ----------------------------------------------------------------- loading
def load(eval_dir, tiers, train_file, panel_index, P, lm, max_per_tier=None, same_plate=False):
    """-> {group: {drug: [expr vecs]}}, {group: [ctrl expr vecs]}, {drug: moa}
    group = (cell_line, plate) if same_plate else (cell_line, None).

    same_plate closes the PLATE LEAK: drug and plate are confounded by the experimental design (each
    drug sits on its own plate), so cross-plate comparisons let batch identity stand in for drug
    identity. Holding plate constant across the comparison set makes that impossible.
    Measured: a zero-drug-info control scores 0.551 cross-plate vs 0.510 same-plate."""
    by_cl = defaultdict(lambda: defaultdict(list))
    ctrl_by_cl = defaultdict(list)
    moa_of = {}
    files = [os.path.join(eval_dir, f"eval_{t}.jsonl") for t in tiers]
    if train_file:
        files.append(train_file)
    seen_ctrl = defaultdict(set)
    for path in files:
        if not os.path.exists(path):
            logger.warning(f"  missing {path} (skipping)")
            continue
        n = 0
        for line in open(path):
            ex = json.loads(line)
            m = ex.get("metadata", {})
            cl, drug, plate = m.get("cell_line_id"), m.get("drug"), m.get("plate")
            if cl is None or drug is None:
                continue
            if same_plate and plate is None:
                continue
            cl = (cl, plate) if same_plate else (cl, None)
            by_cl[cl][drug].append(sentence_to_expr(ex["response"], panel_index, P, lm))
            if m.get("moa"):
                moa_of[drug] = m["moa"]
            cs = control_from_prompt(ex["prompt"])
            if cs:
                key = hash(cs)
                if key not in seen_ctrl[cl]:      # dedup: the same control cell is reused often
                    seen_ctrl[cl].add(key)
                    ctrl_by_cl[cl].append(sentence_to_expr(cs, panel_index, P, lm))
            n += 1
            if max_per_tier and n >= max_per_tier:
                break
        logger.info(f"  {os.path.basename(path)}: {n:,} rows")
    return by_cl, ctrl_by_cl, moa_of


# ----------------------------------------------------------------- reporting
def summarize(rows, moa_of, args):
    snr = np.array([r["snr"] for r in rows if r.get("snr") is not None])
    iso = np.array([r["isolation"] for r in rows if r.get("isolation") is not None])
    ceil = np.array([r["ceiling_nir"] for r in rows if r.get("ceiling_nir") is not None])

    n = len(rows)
    inert = int((snr < 1.0).sum()) if len(snr) else 0
    redundant = int((iso < 1.0).sum()) if len(iso) else 0
    identifiable = int((ceil >= args.identifiable_nir).sum()) if len(ceil) else 0
    ncells = np.array([r["n_cells"] for r in rows])

    logger.info("")
    logger.info("=" * 92)
    logger.info(f"  DRUG DIFFICULTY ATLAS — {n} (drug x cell line) entries")
    logger.info("-" * 92)
    logger.info(f"  cells per (drug x cell line): median {np.median(ncells):.0f}  "
                f"p10 {np.percentile(ncells,10):.0f}  p90 {np.percentile(ncells,90):.0f}  "
                f"(each pseudobulk half gets ~{np.median(ncells)/2:.0f})")
    if np.median(ncells) < args.snr_reliable_cells:
        logger.warning("  *** UNDERPOWERED: with this few cells the split-half replicate noise is huge, ")
        logger.warning(f"  *** so SNR (=effect/noise) and the 'INERT' count below are CELL-COUNT LIMITED,")
        logger.warning(f"  *** not statements about drug biology. A near-constant SNR across a diverse ")
        logger.warning(f"  *** panel is the signature of this floor. Trust the per-drug CEILING NIR and the")
        logger.warning(f"  *** RANKING of drugs; re-run with more cells/drug before quoting SNR as biology.")
    if len(snr):
        logger.info(f"  effect / replicate-noise (SNR):  median {np.median(snr):.2f}   "
                    f"p10 {np.percentile(snr,10):.2f}  p90 {np.percentile(snr,90):.2f}")
        logger.info(f"    INERT (SNR < 1: moves less than its own replicate noise): "
                    f"{inert}/{n}  ({100.0*inert/max(1,n):.1f}%)")
    if len(iso):
        logger.info(f"  isolation (nn-drug dist / replicate noise): median {np.median(iso):.2f}   "
                    f"p10 {np.percentile(iso,10):.2f}  p90 {np.percentile(iso,90):.2f}")
        logger.info(f"    REDUNDANT (isolation < 1: another drug is closer than own replicate): "
                    f"{redundant}/{n}  ({100.0*redundant/max(1,n):.1f}%)")
    if len(ceil):
        logger.info(f"  per-drug CEILING NIR (real replicate):  median {np.median(ceil):.3f}   "
                    f"mean {ceil.mean():.3f}   at-chance(<0.6): "
                    f"{int((ceil<0.6).sum())}/{n} ({100.0*(ceil<0.6).mean():.1f}%)")
        logger.info(f"    IDENTIFIABLE (ceiling NIR >= {args.identifiable_nir}): "
                    f"{identifiable}/{n}  ({100.0*identifiable/max(1,n):.1f}%)")
        logger.info("    ^ THIS is the subset on which 'the model is drug-blind' is a fair claim.")
    logger.info("=" * 92)

    # hardest / easiest drugs by ceiling
    srt = sorted([r for r in rows if r.get("ceiling_nir") is not None], key=lambda r: r["ceiling_nir"])
    logger.info("  LEAST identifiable drugs (ceiling NIR lowest — task ~impossible for anyone):")
    for r in srt[:8]:
        logger.info(f"    {str(r['drug'])[:28]:28s} cl={str(r.get('cell_line'))[:12]:12s} "
                    f"ceil={r['ceiling_nir']:.2f} snr={_f(r.get('snr'))} iso={_f(r.get('isolation'))} "
                    f"nn={str(r.get('nn_drug'))[:20]}")
    logger.info("  MOST identifiable drugs:")
    for r in srt[-8:][::-1]:
        logger.info(f"    {str(r['drug'])[:28]:28s} cl={str(r.get('cell_line'))[:12]:12s} "
                    f"ceil={r['ceiling_nir']:.2f} snr={_f(r.get('snr'))} iso={_f(r.get('isolation'))}")

    return {"n_entries": n, "n_inert": inert, "n_redundant": redundant,
            "n_identifiable": identifiable,
            "frac_inert": float(inert / max(1, n)), "frac_redundant": float(redundant / max(1, n)),
            "frac_identifiable": float(identifiable / max(1, n)),
            "median_snr": float(np.median(snr)) if len(snr) else None,
            "median_isolation": float(np.median(iso)) if len(iso) else None,
            "median_ceiling_nir": float(np.median(ceil)) if len(ceil) else None}


def _f(x):
    return "  NA" if x is None else f"{x:.2f}"


def moa_structure(profiles_by_cl, moa_of):
    """Within-MoA vs between-MoA pseudobulk distance. If within << between, drugs of the same
    mechanism are redundant and drug-level discrimination is intrinsically harder than MoA-level."""
    win, btw = [], []
    for cl, A in profiles_by_cl.items():
        dl = [d for d in A if d in moa_of]
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                di, dj = dl[i], dl[j]
                dist = float(np.linalg.norm(A[di] - A[dj]))
                (win if moa_of[di] == moa_of[dj] else btw).append(dist)
    if not win or not btw:
        return None
    r = {"within_moa_mean": float(np.mean(win)), "between_moa_mean": float(np.mean(btw)),
         "n_within": len(win), "n_between": len(btw)}
    r["ratio"] = r["within_moa_mean"] / max(1e-9, r["between_moa_mean"])
    logger.info(f"  MoA structure: within-MoA dist {r['within_moa_mean']:.3f} (n={len(win)})  vs  "
                f"between-MoA {r['between_moa_mean']:.3f} (n={len(btw)})   ratio={r['ratio']:.3f}")
    logger.info("    ratio << 1 => same-MoA drugs are near-duplicates (drug-level ID intrinsically hard);"
                " ~1 => MoA does not explain response similarity.")
    return r


def chem_gate(profiles_by_cl, feats_path):
    """OPTIONAL Arm-2 gate: does CHEMICAL similarity predict RESPONSE similarity?
    --drug_features: JSON {drug: [binary fingerprint bits]}. Correlates Tanimoto similarity with
    response similarity across drug pairs. If ~0, structure/CLIP injection cannot help."""
    feats = json.load(open(feats_path))
    feats = {k: np.asarray(v, dtype=float) for k, v in feats.items()}
    xs, ys = [], []
    for cl, A in profiles_by_cl.items():
        dl = [d for d in A if d in feats]
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                a, b = feats[dl[i]], feats[dl[j]]
                inter = float(np.sum(np.minimum(a, b))); union = float(np.sum(np.maximum(a, b)))
                if union <= 0:
                    continue
                tani = inter / union
                resp_sim = -float(np.linalg.norm(A[dl[i]] - A[dl[j]]))   # higher = more similar
                xs.append(tani); ys.append(resp_sim)
    if len(xs) < 10:
        logger.warning("  chem gate: too few overlapping drugs with features")
        return None
    xs, ys = np.array(xs), np.array(ys)
    r = float(np.corrcoef(xs, ys)[0, 1])
    logger.info(f"  CHEM GATE: corr(chemical Tanimoto, response similarity) = {r:+.3f}  "
                f"over {len(xs):,} drug pairs")
    logger.info("    >0.2 => structure predicts response; structure-injection (Arm 2) is well-founded."
                "  ~0 => it cannot help; do not build the CLIP arm on this data.")
    return {"corr_tanimoto_vs_response_sim": r, "n_pairs": int(len(xs))}


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Plant known structure and check we recover it:
      - 'inert'   : treated == control (+noise)          -> SNR < 1
      - 'twinA/B' : two drugs with identical true profile -> isolation < 1, ceiling_nir low
      - 'distinct*': well-separated real effects          -> ceiling_nir high, isolation > 1"""
    rng = np.random.RandomState(0)
    P = 300
    lm = {"slope": -0.4, "intercept": 1.6}
    panel = [f"G{i}" for i in range(P)]
    pidx = {g: i for i, g in enumerate(panel)}

    def cell(sig, noise=0.25):
        v = np.maximum(0.0, sig + rng.randn(P) * noise)
        order = np.argsort(-v)
        genes = [panel[i] for i in order if v[i] > 0][:120]
        return " ".join(genes) + " " + SENTINEL

    base = np.zeros(P); base[rng.choice(P, 120, replace=False)] = rng.rand(120) * 2 + 0.5
    twin = base.copy(); twin[rng.choice(P, 60, replace=False)] += 2.0     # twinA == twinB profile
    sigs = {"inert": base, "twinA": twin, "twinB": twin}
    for k in range(4):
        s = base.copy(); s[rng.choice(P, 60, replace=False)] += 4.0 + k    # distinct, strong
        sigs[f"distinct{k}"] = s

    drug_cells = {d: [sentence_to_expr(cell(s), pidx, P, lm) for _ in range(24)]
                  for d, s in sigs.items()}
    ctrl = [sentence_to_expr(cell(base), pidx, P, lm) for _ in range(40)]

    rows, A = analyze_cellline(drug_cells, ctrl, np.random.RandomState(1), min_cells=8)
    by = {r["drug"]: r for r in rows}
    for r in rows:
        logger.info(f"  {r['drug']:10s} snr={_f(r['snr'])} iso={_f(r['isolation'])} "
                    f"ceil={r['ceiling_nir']:.2f} nn={r['nn_drug']}")

    ok = True
    # inert drug: moves no more than its own replicate noise
    if not (by["inert"]["snr"] is not None and by["inert"]["snr"] < 1.5):
        logger.error("  FAIL: inert drug not flagged as low-SNR"); ok = False
    # twins: each other's nearest neighbour, and not identifiable
    if by["twinA"]["nn_drug"] != "twinB" or by["twinB"]["nn_drug"] != "twinA":
        logger.error("  FAIL: twins are not each other's nearest neighbour"); ok = False
    if not (by["twinA"]["ceiling_nir"] < 0.9 or by["twinB"]["ceiling_nir"] < 0.9):
        logger.error("  FAIL: twins look perfectly identifiable"); ok = False
    # distinct drugs: identifiable
    dis = [by[f"distinct{k}"]["ceiling_nir"] for k in range(4)]
    if np.mean(dis) < 0.8:
        logger.error(f"  FAIL: distinct drugs not identifiable (mean ceiling {np.mean(dis):.2f})"); ok = False
    # the headline contrast: twins must be less identifiable than distinct drugs
    if not (min(by["twinA"]["ceiling_nir"], by["twinB"]["ceiling_nir"]) < np.mean(dis)):
        logger.error("  FAIL: twins not less identifiable than distinct drugs"); ok = False

    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--tiers", default="tier1_seen_conditions,tier2_unseen_drugs,"
                                       "tier3_unseen_combos,tier4_dose_interpolation")
    ap.add_argument("--train_file", default=None, help="optionally fold train.jsonl in for more cells")
    ap.add_argument("--min_cells", type=int, default=8)
    ap.add_argument("--n_celllines", type=int, default=30)
    ap.add_argument("--min_drugs_per_cl", type=int, default=3)
    ap.add_argument("--identifiable_nir", type=float, default=0.8,
                    help="per-drug ceiling NIR above which a drug counts as identifiable-in-principle")
    ap.add_argument("--same_plate_only", action="store_true",
                    help="LEAKAGE FIX: compare drugs only within the same (cell_line, plate), so "
                         "batch identity carries no drug information.")
    ap.add_argument("--cell_sweep", default=None,
                    help="e.g. '4,6,8,12,16,20' — report ceiling NIR vs cells/drug and exit. Use with "
                         "--same_plate_only for the leak-free achievable bar.")
    ap.add_argument("--snr_reliable_cells", type=int, default=20,
                    help="below this median cells/(drug x cell line), SNR is cell-count limited and the "
                         "'inert' count is flagged as unreliable (not a biological statement)")
    ap.add_argument("--drug_features", default=None,
                    help="optional JSON {drug: [fingerprint bits]} -> runs the Arm-2 chemical gate")
    ap.add_argument("--out", default="RESULTS/drug_atlas.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--profiles", default=None,
                    help="optional .npz of per-(cell_line,drug) halfA pseudobulk profiles, for the "
                         "geometry test (drug_stratify_geometry.py)")
    ap.add_argument("--panel_file", default=None)
    ap.add_argument("--linear_model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return
    if not args.eval_dir:
        ap.error("--eval_dir is required (unless --selftest)")

    panel_file = args.panel_file or os.path.join(args.eval_dir, "l1000_panel.json")
    lm_file = args.linear_model or os.path.join(args.eval_dir, "linear_model.json")
    panel = json.load(open(panel_file))
    pidx = {g: i for i, g in enumerate(panel)}
    P = len(panel)
    lm = json.load(open(lm_file))
    logger.info(f"Panel {P} from {panel_file}; linear_model slope={lm['slope']:.3f} "
                f"intercept={lm['intercept']:.3f}")

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    by_cl, ctrl_by_cl, moa_of = load(args.eval_dir, tiers, args.train_file, pidx, P, lm,
                                     same_plate=args.same_plate_only)
    _grp = ("(cell_line, plate) — LEAK-FREE" if args.same_plate_only
            else "(cell_line) — cross-plate, batch can leak drug identity")
    logger.info(f"  grouping: {_grp}")

    rng = np.random.RandomState(args.seed)

    if args.cell_sweep:
        ns = [int(x) for x in args.cell_sweep.split(",") if x.strip()]
        logger.info("")
        logger.info("=" * 92)
        logger.info("  CEILING NIR vs CELLS PER DRUG  (chance = 0.50)")
        logger.info(f"  {'(same-plate: leak-free)' if args.same_plate_only else '(CROSS-PLATE: inflated by batch leakage)'}")
        logger.info("-" * 92)
        sweep = run_cell_sweep(by_cl, ns, rng, args.min_drugs_per_cl)
        logger.info(f"  {'cells/drug':>12} {'ceiling NIR':>12} {'headroom':>10} {'groups':>8} {'drugs':>8}")
        for n, r in sorted(sweep.items()):
            logger.info(f"  {n:>12} {r['ceiling_nir']:>12.3f} {r['ceiling_nir']-0.5:>+10.3f} "
                        f"{r['n_groups']:>8} {r['n_drugs']:>8}")
        logger.info("=" * 92)
        logger.info("  RISING with cells => real drug signal, noise-limited: the task IS winnable and")
        logger.info("                      the target is whatever this curve reaches at high n.")
        logger.info("  SATURATING near chance => there is very little honest drug signal to capture;")
        logger.info("                      the model cannot be blamed for missing what isn't there.")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"cell_sweep": sweep, "same_plate_only": args.same_plate_only,
                   "config": {k: v for k, v in vars(args).items()}},
                  open(args.out, "w"), indent=2, default=float)
        logger.info(f"-> {args.out}")
        return
    all_rows, profiles_by_cl = [], {}
    used = 0
    for cl, dd in by_cl.items():
        if len([d for d, v in dd.items() if len(v) >= args.min_cells]) < args.min_drugs_per_cl:
            continue
        if not ctrl_by_cl.get(cl):
            logger.warning(f"  {str(cl)[:20]}: no controls found -> effect sizes will be NA")
        rows, A = analyze_cellline(dd, ctrl_by_cl.get(cl, []), rng, args.min_cells)
        if not rows:
            continue
        add_degs(rows, {d: v for d, v in dd.items()}, ctrl_by_cl.get(cl, []), args.min_cells, rng)
        for r in rows:
            r["cell_line"] = cl
            r["moa"] = moa_of.get(r["drug"])
        all_rows.extend(rows)
        profiles_by_cl[cl] = A
        used += 1
        logger.info(f"  {str(cl)[:22]:22s} {len(rows)} drugs analyzed")
        if used >= args.n_celllines:
            break

    if not all_rows:
        logger.error("No drugs analyzed — check --eval_dir/--min_cells."); sys.exit(1)

    summ = summarize(all_rows, moa_of, args)
    summ["moa_structure"] = moa_structure(profiles_by_cl, moa_of)
    if args.drug_features:
        summ["chem_gate"] = chem_gate(profiles_by_cl, args.drug_features)

    out = {"summary": summ, "rows": all_rows, "n_celllines": used,
           "config": {k: v for k, v in vars(args).items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")

    if args.csv:
        # NOTE: drug names contain commas (e.g. "Dapagliflozin ((2S)-1,2-propanediol, hydrate)"),
        # so the csv module (proper quoting) is mandatory here — hand-joining shifts columns.
        import csv as _csv
        cols = ["cell_line", "drug", "moa", "n_cells", "effect_size", "effect_cosine", "n_deg",
                "replicate_noise", "snr", "nn_drug", "nn_dist", "isolation", "ceiling_nir"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(cols)
            for r in all_rows:
                w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
        logger.info(f"-> {args.csv}")

    if args.profiles:
        flat = {f"{cl}||{d}": A[d] for cl, A in profiles_by_cl.items() for d in A}
        np.savez_compressed(args.profiles, **flat)
        logger.info(f"-> {args.profiles}  ({len(flat)} real halfA profiles for the geometry test)")


if __name__ == "__main__":
    main()
