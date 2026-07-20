#!/usr/bin/env python
r"""
dose_response_analysis.py — does a drug become more IDENTIFIABLE at higher dose?
================================================================================
Follow-up to the drug-biology atlas, which found 62% of drugs have 2-3 distinct doses. Natural
question: are the "subtle / unidentifiable" drugs simply being measured at a LOW dose? If potency
and identifiability rise with dose, then dose (not the drug) explains a chunk of the heterogeneity.

DESIGN (paired, within-drug, cell-count matched, within-plate) — every one of these matters:
  * PAIRED WITHIN (drug x cell_line x plate): each drug is its own control, so the comparison is not
    confounded by drugs differing in intrinsic potency.
  * DOSE RANK, not absolute uM: drugs differ hugely in potency, so 10 uM of drug A is not comparable
    to 10 uM of drug B. We rank each drug's OWN doses (low -> high) and compare rank-to-rank.
  * CELL-COUNT MATCHED: identifiability rises steeply with cells (0.61@n=10 -> 0.85@n=120). If the
    high dose had more cells it would look more identifiable for that reason alone. We SUBSAMPLE every
    dose of a drug to the SAME n before measuring. This is the single most important guard here.
  * WITHIN-PLATE comparison set: drugs are compared only against other drugs on the same
    (cell_line, plate), so batch identity cannot masquerade as drug identity.
  * SAME-DRUG OTHER DOSES ARE EXCLUDED from the comparison set: we are asking "is this drug
    distinguishable from OTHER DRUGS", not "can you tell its doses apart".
  * CLUSTERED CIs: resample CELL LINES (drugs within a cell line are not independent).

MEASURES per (drug x cell_line x plate x dose), at matched n:
  * effect   : || pseudobulk(drug,dose) - pseudobulk(plate-matched DMSO) ||   (potency)
  * ceiling  : identifiability — a held-out replicate ranked against same-plate OTHER drugs
Then per drug: does effect / ceiling increase from its lowest to its highest dose?

NOTE ON ABSOLUTE VALUES: the comparison-set profiles are pooled (more cells) than the per-dose
half-splits, so the ABSOLUTE ceiling here is not comparable to the atlas. That bias is identical
across a drug's doses, so it cancels in the paired within-drug comparison — the DOSE TREND is what
this script measures, not the absolute level.

Reads jsonl only (no streaming). Self-contained; runs in minutes on CPU.

USAGE
  python dose_response_analysis.py --eval_dir DATA_endcell_big \
     --sources train,eval_tier1_seen_conditions,eval_tier2_unseen_drugs \
     --min_cells_per_dose 10 --out RESULTS/dose_response.json --csv RESULTS/dose_response.csv

SELFTEST (no data) — plants a real dose-response and a flat (no-response) drug:
  python dose_response_analysis.py --selftest
"""
import argparse, json, os, sys, logging, csv as _csv
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
    arr = np.zeros(P); seen = set(); pos = 0
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


def pb(vs):
    return np.mean(np.stack(vs), axis=0)


def boot_clustered(vals, clusters, n_boot=2000, seed=0):
    """Resample CELL LINES (not individual drugs) — drugs within a line are not independent."""
    v = np.asarray(vals, float); cl = np.asarray(clusters)
    uniq = np.unique(cl)
    if len(uniq) < 3 or len(v) < 3:
        return None, None, len(uniq)
    groups = [v[cl == c] for c in uniq]
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        cat = np.concatenate([groups[i] for i in pick])
        if len(cat):
            means.append(cat.mean())
    if not means:
        return None, None, len(uniq)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(uniq)


# ----------------------------------------------------------------- core
def analyze_group(drug_dose_cells, ctrl_vecs, rng, args):
    """drug_dose_cells: {drug: {dose: [vecs]}} for ONE (cell_line, plate).
    Returns per (drug,dose) rows measured at a cell count matched within each drug."""
    # reference profiles for the comparison set: pooled over all of a drug's cells (all doses)
    pooled = {d: pb([v for dd in dv.values() for v in dd])
              for d, dv in drug_dose_cells.items() if any(dv.values())}
    if len(pooled) < 3:
        return []
    ctrl_pb = pb(ctrl_vecs) if ctrl_vecs else None
    rows = []
    for drug, dv in drug_dose_cells.items():
        doses = sorted([d for d, v in dv.items() if len(v) >= args.min_cells_per_dose])
        if len(doses) < 2:
            continue
        # CELL-COUNT MATCH: every dose of this drug is subsampled to the same n
        n_match = min(len(dv[d]) for d in doses)
        n_match = min(n_match, args.max_cells_per_dose)
        if n_match < args.min_cells_per_dose:
            continue
        # comparison set = OTHER drugs on this plate (same-drug other doses excluded by construction)
        others = [o for o in pooled if o != drug]
        if len(others) < 2:
            continue
        for rank_i, dose in enumerate(doses):
            v = dv[dose]
            idx = rng.choice(len(v), n_match, replace=False)
            sub = [v[i] for i in idx]
            h = n_match // 2
            A = pb(sub[:h]); B = pb(sub[h:2 * h])          # truth / held-out replicate
            full = pb(sub)
            d_own = float(np.linalg.norm(B - A))
            d_oth = [float(np.linalg.norm(B - pooled[o])) for o in others]
            ceiling = float(np.mean([d_own < x for x in d_oth]))
            effect = float(np.linalg.norm(full - ctrl_pb)) if ctrl_pb is not None else None
            rows.append({"drug": drug, "dose": float(dose), "dose_rank": rank_i + 1,
                         "n_doses": len(doses), "n_cells_matched": n_match,
                         "ceiling": ceiling, "effect": effect})
    return rows


def summarize(rows, args):
    """Paired within-drug: compare each drug's HIGHEST vs LOWEST dose."""
    by_drug = defaultdict(list)
    for r in rows:
        by_drug[(r["cell_line"], r["plate"], r["drug"])].append(r)

    d_ceil, d_eff, clusters, trends = [], [], [], []
    for key, rs in by_drug.items():
        rs = sorted(rs, key=lambda r: r["dose_rank"])
        if len(rs) < 2:
            continue
        lo, hi = rs[0], rs[-1]
        d_ceil.append(hi["ceiling"] - lo["ceiling"])
        if lo["effect"] is not None and hi["effect"] is not None:
            d_eff.append(hi["effect"] - lo["effect"])
        clusters.append(key[0])
        if len(rs) >= 3:
            ranks = [r["dose_rank"] for r in rs]; ce = [r["ceiling"] for r in rs]
            if np.std(ce) > 1e-9:
                trends.append(float(np.corrcoef(ranks, ce)[0, 1]))

    logger.info("")
    logger.info("=" * 96)
    logger.info(f"  DOSE RESPONSE — paired within (drug x cell_line x plate), cell-count matched")
    logger.info(f"  {len(by_drug)} multi-dose drug-conditions across {len(set(clusters))} cell lines")
    logger.info("-" * 96)

    # mean by dose rank
    by_rank = defaultdict(list)
    for r in rows:
        by_rank[r["dose_rank"]].append(r)
    logger.info(f"  {'dose rank':>10} {'n':>7} {'mean ceiling':>14} {'mean effect':>13}")
    for k in sorted(by_rank):
        rs = by_rank[k]
        ce = np.mean([r["ceiling"] for r in rs])
        ef = [r["effect"] for r in rs if r["effect"] is not None]
        logger.info(f"  {k:>10} {len(rs):>7} {ce:>14.3f} "
                    f"{(np.mean(ef) if ef else float('nan')):>13.3f}")

    out = {"n_conditions": len(by_drug), "n_celllines": len(set(clusters))}
    logger.info("")
    if d_ceil:
        lo_, hi_, ncl = boot_clustered(d_ceil, clusters, seed=args.seed)
        m = float(np.mean(d_ceil))
        logger.info(f"  IDENTIFIABILITY  highest-dose minus lowest-dose = {m:+.3f}" +
                    (f"   CLUSTERED 95% CI [{lo_:+.3f}, {hi_:+.3f}]  ({ncl} cell lines)"
                     if lo_ is not None else ""))
        out["delta_ceiling"] = {"mean": m, "ci": [lo_, hi_], "n": len(d_ceil)}
        if lo_ is not None and lo_ > 0:
            logger.info("     => HIGHER DOSE IS MORE IDENTIFIABLE (CI excludes 0). Dose explains part "
                        "of the drug-difficulty heterogeneity.")
        elif lo_ is not None and lo_ <= 0 <= hi_:
            logger.info("     => NO reliable dose effect on identifiability (CI spans 0): the "
                        "'unidentifiable' drugs are not merely low-dose.")
    if d_eff:
        lo2, hi2, _ = boot_clustered(d_eff, clusters, seed=args.seed)
        m2 = float(np.mean(d_eff))
        logger.info(f"  POTENCY (effect vs DMSO)  highest minus lowest = {m2:+.3f}" +
                    (f"   CLUSTERED 95% CI [{lo2:+.3f}, {hi2:+.3f}]" if lo2 is not None else ""))
        out["delta_effect"] = {"mean": m2, "ci": [lo2, hi2], "n": len(d_eff)}
    if trends:
        logger.info(f"  monotonic trend (corr of ceiling vs dose rank, drugs with >=3 doses): "
                    f"mean {np.mean(trends):+.3f} over {len(trends)} drugs")
        out["mean_monotonic_trend"] = float(np.mean(trends))
    logger.info("=" * 96)
    out["by_rank"] = {int(k): {"n": len(v), "mean_ceiling": float(np.mean([r['ceiling'] for r in v]))}
                      for k, v in by_rank.items()}
    return out


# ----------------------------------------------------------------- loading
def load(eval_dir, sources, pidx, P, lm):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # (cl,plate)->drug->dose->[v]
    ctrl = defaultdict(list)
    seen_ctrl = defaultdict(set)
    for src in sources:
        path = os.path.join(eval_dir, src if src.endswith(".jsonl") else f"{src}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"  missing {path}"); continue
        n = 0
        for line in open(path):
            ex = json.loads(line)
            m = ex.get("metadata", {})
            cl, plate, drug, dose = (m.get("cell_line_id"), m.get("plate"),
                                     m.get("drug"), m.get("dose_float"))
            if cl is None or plate is None or drug is None or dose is None:
                continue
            g = (cl, plate)
            groups[g][drug][round(float(dose), 6)].append(
                sentence_to_expr(ex["response"], pidx, P, lm))
            cs = control_from_prompt(ex["prompt"])
            if cs:
                k = hash(cs)
                if k not in seen_ctrl[g]:
                    seen_ctrl[g].add(k); ctrl[g].append(sentence_to_expr(cs, pidx, P, lm))
            n += 1
        logger.info(f"  {os.path.basename(path)}: {n:,} rows")
    return groups, ctrl


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Plant a genuine dose-response drug (effect grows with dose) and a flat drug (no response);
    verify the paired test detects the first and not the second."""
    rng = np.random.RandomState(0)
    P = 200
    lm = {"slope": -0.4, "intercept": 1.6}
    panel = [f"G{i}" for i in range(P)]
    pidx = {g: i for i, g in enumerate(panel)}

    def cell(sig, noise=0.6):
        v = np.maximum(0.0, sig + rng.randn(P) * noise)
        order = np.argsort(-v)
        return " ".join(panel[i] for i in order if v[i] > 0) + " " + SENTINEL

    base = np.zeros(P); base[rng.choice(P, 100, replace=False)] = rng.rand(100) * 2 + 1.0
    hit = rng.choice(P, 50, replace=False)

    def vecs(sig, n):
        return [sentence_to_expr(cell(sig), pidx, P, lm) for _ in range(n)]

    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ctrl = defaultdict(list)
    for cl in range(4):
        g = (f"cl{cl}", "p1")
        ctrl[g] = vecs(base, 60)
        # dose-responsive drug: effect scales with dose
        for k, dose in enumerate([0.1, 1.0, 10.0]):
            s = base.copy(); s[hit] += 1.0 + 3.0 * k
            groups[g]["doseDrug"][dose] = vecs(s, 40)
        # flat drug: same (weak) effect at every dose
        for dose in [0.1, 1.0, 10.0]:
            s = base.copy(); s[rng.choice(P, 40, replace=False)] += 1.0
            groups[g]["flatDrug"][dose] = vecs(s, 40)
        # filler drugs so the comparison set is non-trivial
        for j in range(4):
            s = base.copy(); s[rng.choice(P, 50, replace=False)] += 2.0 + j
            groups[g][f"filler{j}"][1.0] = vecs(s, 40)

    args.min_cells_per_dose = 10; args.max_cells_per_dose = 40; args.seed = 0
    rows = []
    for g, dd in groups.items():
        rs = analyze_group(dd, ctrl[g], np.random.RandomState(1), args)
        for r in rs:
            r["cell_line"], r["plate"] = g
        rows.extend(rs)

    dose_rows = [r for r in rows if r["drug"] == "doseDrug"]
    flat_rows = [r for r in rows if r["drug"] == "flatDrug"]
    def paired(rs):
        by = defaultdict(list)
        for r in rs: by[r["cell_line"]].append(r)
        d = []
        for k, v in by.items():
            v = sorted(v, key=lambda r: r["dose_rank"])
            if len(v) >= 2: d.append(v[-1]["effect"] - v[0]["effect"])
        return float(np.mean(d)) if d else 0.0
    de_dose, de_flat = paired(dose_rows), paired(flat_rows)
    logger.info(f"  dose-responsive drug: effect(high)-effect(low) = {de_dose:+.3f} (expect >0)")
    logger.info(f"  flat drug:            effect(high)-effect(low) = {de_flat:+.3f} (expect ~0)")
    ok = de_dose > 0.5 and abs(de_flat) < de_dose / 2
    # cell-count matching sanity: every dose of a drug measured at identical n
    ns = {(r["cell_line"], r["drug"]): set() for r in rows}
    for r in rows: ns[(r["cell_line"], r["drug"])].add(r["n_cells_matched"])
    if any(len(s) > 1 for s in ns.values()):
        logger.error("  FAIL: cell counts not matched across doses within a drug"); ok = False
    else:
        logger.info("  cell-count matching: OK (identical n across a drug's doses)")
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--sources", default="train,eval_tier1_seen_conditions,eval_tier2_unseen_drugs")
    ap.add_argument("--min_cells_per_dose", type=int, default=10)
    ap.add_argument("--max_cells_per_dose", type=int, default=60)
    ap.add_argument("--out", default="RESULTS/dose_response.json")
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

    groups, ctrl = load(args.eval_dir, [s.strip() for s in args.sources.split(",")], pidx, P, lm)
    logger.info(f"  {len(groups)} (cell_line, plate) groups")

    rng = np.random.RandomState(args.seed)
    rows = []
    for g, dd in groups.items():
        rs = analyze_group(dd, ctrl.get(g, []), rng, args)
        for r in rs:
            r["cell_line"], r["plate"] = g
        rows.extend(rs)
    if not rows:
        logger.error("No multi-dose conditions passed the filters — try lowering --min_cells_per_dose.")
        sys.exit(1)

    summ = summarize(rows, args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summ, "rows": rows, "config": vars(args)},
              open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")

    if args.csv:
        cols = ["cell_line", "plate", "drug", "dose", "dose_rank", "n_doses",
                "n_cells_matched", "ceiling", "effect"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
        logger.info(f"-> {args.csv}")


if __name__ == "__main__":
    main()
