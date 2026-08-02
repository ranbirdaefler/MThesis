#!/usr/bin/env python
r"""
kappa_channel.py -- is the drug x cell-line INTERACTION a property of the mechanism,
                    or of the individual molecule?
=============================================================================================
The transfer coefficient says roughly 45% of the drug-specific residual is interaction: real,
noise-corrected, plate-robust, and reached by neither a per-drug lookup nor the model. The thesis
currently calls that "a quantified, unclaimed target" without knowing which of two very different
things it is:

  (a) STRUCTURED -- two drugs sharing a mechanism interact with a given cell line in similar ways.
      Then the 45% is reachable, the feature that reaches it is named, and a mechanism-conditioned
      model is well motivated.
  (b) IDIOSYNCRATIC -- the interaction belongs to the individual molecule and no annotation we hold
      predicts it. Then the honest closing claim is that half the drug-specific signal is real and
      unreachable from available covariates, which is a stronger negative than "unclaimed".

WHY THE CHANNEL GATE DOES NOT ALREADY ANSWER THIS
  channel_gate.py asked whether an unseen drug's residual can be predicted from similar drugs, and
  found mechanism worth +0.078. But it predicted the WHOLE residual r, which is ~55% per-drug main
  effect beta and ~45% interaction kappa. A positive result there is consistent with "two EGFR
  inhibitors have similar AVERAGE effects", which is close to tautological. This script runs the
  same logic on kappa alone. That single substitution is the experiment.

THE ESTIMAND
  For drugs A != B and a cell line c that saw both,
      s(A,B,c) = cos( kappa(A,c), kappa(B,c) )
  compared between mechanism-matched and mechanism-mismatched pairs. kappa is built
  LEAVE-ONE-CELL-LINE-OUT:
      kappa(d,c) = r(d,c) - mean_{c' != c} r(d,c')
  Without the leave-one-out the subtraction plants a -1/(m-1) correlation and we would be measuring
  our own arithmetic.

CONTROL DISCIPLINE (the lesson that cost this project two retractions)
  A comparison differing from the estimand in more than one way cannot bound it. Mechanism-matched
  and mismatched pairs plausibly differ in several ways besides mechanism, so each is neutralised:

    same cell line     -- by construction; both arms are within-line
    plate              -- --plate_policy different draws BOTH arms from different-plate pairs only,
                          so class-structured plating cannot inflate the mechanism arm. Both
                          policies are computed and reported; `different` is the headline unless it
                          retains too little to support one, which the script decides and says.
    reliability / SNR  -- each mechanism-matched pair is paired with a mismatched pair from the SAME
                          cell line at the closest reliability product. cosine rewards well-resolved
                          kappa, and if mechanism classes skew potent the arm would win on SNR alone.
    annotation         -- no hand-coarsening (fragile). Per-class results and class sizes are
                          reported, so a null caused by a broad bucket is visible rather than buried.

  Negative control: permute the mechanism labels across drugs and re-run. Positive control: the
  split-half reliability of kappa itself, which is the CEILING any cross-drug similarity is read
  against -- a cosine of 0.15 is meaningless until you know whether the maximum attainable is 0.20
  or 0.90.

  python kappa_channel.py --selftest
  python kappa_channel.py --cache_dir /data/.../ot_cache --out RESULTS/kappa_channel.json
"""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, json, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- small numerics
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def spearman_brown(r_half):
    """Reliability of a full-length estimate from the correlation between its two halves.

    Non-positive half-correlations return 0, not a negative reliability. The formula 2r/(1+r) sends
    r = -0.9 to -18, and a negative reliability is not a quantity: it would be averaged into the
    reported mean and, worse, would pass a `relprod > 0` guard as a product of two negatives.
    """
    r_half = min(float(r_half), 0.999)
    return (2 * r_half) / (1 + r_half) if r_half > 0.0 else 0.0


def clustered_ci(values, clusters, rng, n_boot=2000):
    """Bootstrap the mean by resampling whole clusters. Pairs share drugs and lines, so resampling
    pairs would treat one cell line's idiosyncrasy as many independent observations."""
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    n_drop = int((~keep).sum())
    if n_drop:
        logger.warning(f"  clustered_ci: dropped {n_drop} non-finite value(s)")
    values, clusters = values[keep], [c for c, k in zip(clusters, keep) if k]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    g = defaultdict(list)
    for v, c in zip(values, clusters):
        g[c].append(v)
    keys = sorted(g)
    arrs = [np.array(g[k]) for k in keys]
    if len(keys) < 2:
        return float(values.mean()), float("nan"), float("nan"), len(keys)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(arrs), len(arrs))
        boots[b] = np.mean(np.concatenate([arrs[i] for i in pick]))
    return (float(values.mean()), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)), len(keys))


# ----------------------------------------------------------------- kappa at (drug, cell line)
def aggregate_to_drug_line(kept):
    """Collapse conditions to (drug, cell line), keeping the disjoint halves so the reliability of
    the aggregate is still measurable, and keeping the plate set so pairs can be plate-screened.

    Conditions within a line differ by plate and dose. Dose is a real effect -- same drug, same line,
    different dose transfers at 0.703 -- so averaging over it here is a deliberate choice: it removes
    dose as a nuisance that would otherwise differ between the two arms, at the cost of not resolving
    dose-specific interaction. That trade is stated rather than hidden.
    """
    by = defaultdict(list)
    for (d, c, p, dose), v in kept.items():
        by[(d, c)].append((p, v))
    out = {}
    for (d, c), items in by.items():
        out[(d, c)] = {
            "r": np.mean([v["residual"] for _, v in items], axis=0),
            "rA": np.mean([v["residual_A"] for _, v in items], axis=0),
            "rB": np.mean([v["residual_B"] for _, v in items], axis=0),
            "plates": {p for p, _ in items},
            "n_cond": len(items),
            "n_cells": int(sum(v["n_cells"] for _, v in items)),
        }
    return out


def build_kappa(dl, min_lines=2):
    """kappa(d,c) = r(d,c) - mean over OTHER cell lines of r(d,.), computed per half as well so the
    split-half reliability of kappa itself is available as the ceiling.

    A NOTE ON THE LEAVE-ONE-OUT, because it does less than it appears to. Writing S for the sum over
    a drug's m lines,

        kappa_LOO(c) = r(c) - (S - r(c))/(m-1) = [m*r(c) - S]/(m-1) = (m/(m-1)) * (r(c) - mean_c r(c))

    so it is EXACTLY a positive scalar multiple of grand-mean centring. Every statistic here is a
    cosine, and cosine is scale-invariant, so the leave-one-out changes nothing. In particular it
    does NOT remove the -1/(m-1) correlation that centring induces among one drug's kappas across
    cell lines. That artefact is a within-drug, across-line effect: it biases the CONTINUITY arm
    (same drug, two lines), which is therefore reported against its analytic null rather than
    against zero. It does NOT touch the headline contrast, which compares two DIFFERENT drugs in the
    SAME line, where each kappa carries its own scalar and cosine cancels both.
    """
    lines_of = defaultdict(list)
    for (d, c) in dl:
        lines_of[d].append(c)
    kap = {}
    for (d, c), v in dl.items():
        others = [x for x in lines_of[d] if x != c]
        if len(others) < min_lines - 1 or not others:
            continue
        rec = {}
        for h in ("r", "rA", "rB"):
            beta_loo = np.mean([dl[(d, o)][h] for o in others], axis=0)
            rec["k" + h] = v[h] - beta_loo
        rec["rel_half"] = cos(rec["krA"], rec["krB"])
        rec["rel"] = spearman_brown(rec["rel_half"])
        rec["plates"] = v["plates"]
        rec["n_other_lines"] = len(others)
        rec["norm"] = float(np.linalg.norm(rec["kr"]))
        kap[(d, c)] = rec
    return kap


# ----------------------------------------------------------------- pairing
def same_channel(d1, d2, ann, channel):
    """Mechanism match. For `targets` a match is any shared protein symbol; for `moa` it is an exact
    label match. Returns None when either drug is unannotated, so unannotated drugs are excluded
    rather than silently counted as mismatched -- which would dilute the control arm with pairs that
    might in fact share a mechanism."""
    a, b = ann.get(d1), ann.get(d2)
    if a is None or b is None:
        return None
    if channel == "target":
        return len(set(a) & set(b)) > 0
    return str(a) == str(b)


def build_pairs(kap, ann, channel, plate_policy, min_rel, rng, max_per_line=4000):
    """All within-line drug pairs, split into mechanism-matched and mismatched."""
    by_line = defaultdict(list)
    for (d, c), v in kap.items():
        if v["rel"] >= min_rel:
            by_line[c].append(d)
    same, diff = [], []
    for c, drugs in by_line.items():
        drugs = sorted(drugs)
        seen = 0
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                d1, d2 = drugs[i], drugs[j]
                m = same_channel(d1, d2, ann, channel)
                if m is None:
                    continue
                if plate_policy == "different" and (kap[(d1, c)]["plates"] & kap[(d2, c)]["plates"]):
                    continue
                relprod = kap[(d1, c)]["rel"] * kap[(d2, c)]["rel"]
                s = cos(kap[(d1, c)]["kr"], kap[(d2, c)]["kr"])
                # disattenuate: cosine between two noisy estimates is pulled toward zero by the
                # noise in both, and the two arms need not be equally noisy
                sd = s / np.sqrt(relprod) if relprod > 1e-6 else np.nan
                rec = {"line": c, "d1": d1, "d2": d2, "cos": s, "cos_dis": sd,
                       "relprod": relprod, "moa": str(ann.get(d1))}
                (same if m else diff).append(rec)
                seen += 1
                if seen >= max_per_line:
                    break
            if seen >= max_per_line:
                break
    return same, diff


def match_on_reliability(same, diff, rng):
    """Give every mechanism-matched pair a mismatched partner from the SAME cell line at the nearest
    reliability product, and drop matched pairs that have no partner. Cosine rewards well-resolved
    kappa, so without this the mechanism arm could win purely by being better estimated."""
    pool = defaultdict(list)
    for r in diff:
        pool[r["line"]].append(r)
    for c in pool:
        pool[c].sort(key=lambda r: r["relprod"])
    used = defaultdict(set)
    out = []
    for r in same:
        cands = pool.get(r["line"], [])
        best, bi = None, None
        for i, q in enumerate(cands):
            if i in used[r["line"]]:
                continue
            dd = abs(q["relprod"] - r["relprod"])
            if best is None or dd < best:
                best, bi = dd, i
        if bi is None:
            continue
        used[r["line"]].add(bi)
        out.append((r, cands[bi]))
    return out


# ----------------------------------------------------------------- the report
def run_channel(kap, ann, channel, args, rng):
    res = {"channel": channel, "n_annotated_drugs": len({d for (d, _) in kap if d in ann})}
    # For the `target` channel a pair matches on ANY shared symbol, so a drug annotated with many
    # targets has proportionally more matched partners and the matched arm is enriched for
    # promiscuous compounds. Reliability matching does not neutralise this -- it matches on how well
    # kappa is estimated, not on how many targets a drug has. Report the composition so the reader
    # can see it, and provide a low-degree sensitivity arm.
    if channel == "target":
        deg = {d: len(set(ann[d])) for d in ann}
        res["target_degree"] = {"median": float(np.median(list(deg.values()))) if deg else None,
                                "p90": float(np.percentile(list(deg.values()), 90)) if deg else None,
                                "max": int(max(deg.values())) if deg else None}
    for policy in ("different", "any"):
        same, diff = build_pairs(kap, ann, channel, policy, args.min_rel, rng)
        pairs = match_on_reliability(same, diff, rng)
        if not pairs:
            res[policy] = {"n_pairs": 0, "note": "no matched pairs"}
            continue
        d_same = [a["cos_dis"] for a, b in pairs]
        d_diff = [b["cos_dis"] for a, b in pairs]
        delta = [a["cos_dis"] - b["cos_dis"] for a, b in pairs]
        clus = [a["line"] for a, b in pairs]
        m, lo, hi, nc = clustered_ci(delta, clus, rng, args.n_boot)
        res[policy] = {
            "n_pairs": len(pairs), "n_lines": nc,
            "same_mech": float(np.nanmean(d_same)), "diff_mech": float(np.nanmean(d_diff)),
            "delta": m, "ci": [lo, hi],
            "relprod_gap": float(np.mean([abs(a["relprod"] - b["relprod"]) for a, b in pairs])),
            "pool_same": len(same), "pool_diff": len(diff),
        }
        # per-class, so a null produced by one broad bucket is visible
        byc = defaultdict(list)
        for a, b in pairs:
            byc[a["moa"]].append(a["cos_dis"] - b["cos_dis"])
        res[policy]["per_class"] = sorted(
            [{"class": k, "n": len(v), "delta": float(np.nanmean(v))} for k, v in byc.items()
             if len(v) >= args.min_class_pairs],
            key=lambda z: -z["n"])[:20]

    # Label-permutation null on the headline policy. This is the DECISION statistic, not a
    # decoration. The clustered bootstrap interval is measurably too narrow here (its spread runs
    # ~0.92 of the true sampling spread while the permutation reference runs ~1.01), so testing the
    # CI against zero over-rejects. The permutation preserves the pair structure, the matching and
    # the reliability distribution exactly, and destroys only the mechanism labelling -- which is
    # the single thing the estimand is about.
    perm = []
    drugs = sorted({d for (d, _) in kap})
    annotated = [d for d in drugs if d in ann]
    for _ in range(args.n_perm):
        vals = [ann[d] for d in annotated]
        rng.shuffle(vals)
        shuffled = {d: v for d, v in zip(annotated, vals)}
        s2, d2 = build_pairs(kap, shuffled, channel, args.plate_policy, args.min_rel, rng)
        pp = match_on_reliability(s2, d2, rng)
        if pp:
            perm.append(float(np.nanmean([a["cos_dis"] - b["cos_dis"] for a, b in pp])))
    obs = res.get(args.plate_policy, {}).get("delta", float("nan"))
    p_perm = float("nan")
    if perm and np.isfinite(obs):
        # one-sided: the claim is that same-mechanism kappa is MORE similar
        p_perm = (sum(1 for x in perm if x >= obs) + 1) / (len(perm) + 1)
    res["perm_null"] = {"n": len(perm),
                        "mean": float(np.mean(perm)) if perm else float("nan"),
                        "sd": float(np.std(perm, ddof=1)) if len(perm) > 1 else float("nan"),
                        "se": float(np.std(perm, ddof=1) / np.sqrt(len(perm))) if len(perm) > 1
                              else float("nan"),
                        "p_one_sided": p_perm}
    return res


def report(kept, ann_all, args, rng):
    dl = aggregate_to_drug_line(kept)
    logger.info(f"(drug, cell line) cells: {len(dl)}")
    kap = build_kappa(dl)
    rels = np.array([v["rel"] for v in kap.values()])
    logger.info(f"kappa built for {len(kap)} (drug, line) pairs | mean reliability {rels.mean():+.3f}"
                f" | frac >= {args.min_rel}: {100*(rels >= args.min_rel).mean():.0f}%")

    out = {"n_drug_line": len(kap), "min_rel": args.min_rel,
           "ceiling_kappa_reliability": float(rels[rels >= args.min_rel].mean())
           if (rels >= args.min_rel).any() else float("nan")}

    logger.info("")
    logger.info("CEILING -- split-half reliability of kappa itself: %.3f" % out["ceiling_kappa_reliability"])
    logger.info("  (every cross-drug cosine below is disattenuated by this, so the arms are on a")
    logger.info("   scale where 1.0 would mean 'as similar as kappa is to its own replicate')")

    # continuity arm: same drug, different cell line
    # CONTINUITY arm. Read against its ANALYTIC null, not against zero. Defining kappa as a
    # deviation from a mean over that drug's own cell lines forces the m kappas of one drug to sum
    # to (a multiple of) zero, so two of them are negatively correlated by construction at
    # approximately -1/(m-1) before any biology enters. Zero is not the null here; -1/(m-1) is.
    same_drug, null_drug = [], []
    bydrug = defaultdict(list)
    for (d, c), v in kap.items():
        if v["rel"] >= args.min_rel:
            bydrug[d].append((c, v))
    for d, items in bydrug.items():
        m = len(items) + 0  # lines retained for this drug
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (c1, v1), (c2, v2) = items[i], items[j]
                rp = v1["rel"] * v2["rel"]
                if rp > 1e-6 and m > 1:
                    same_drug.append(cos(v1["kr"], v2["kr"]))
                    null_drug.append(-1.0 / (m - 1))
    obs = float(np.nanmean(same_drug)) if same_drug else float("nan")
    nul = float(np.nanmean(null_drug)) if null_drug else float("nan")
    out["same_drug_cross_line_raw"] = obs
    out["same_drug_cross_line_null"] = nul
    out["same_drug_cross_line_excess"] = obs - nul
    out["n_same_drug_pairs"] = len(same_drug)
    logger.info("CONTINUITY -- same drug, different line (RAW cosine): %.4f  vs analytic null "
                "%.4f  ->  excess %+.4f  (n=%d)" % (obs, nul, obs - nul, len(same_drug)))
    logger.info("  The null is -1/(m-1), not zero: centring a drug's kappas on their own mean makes")
    logger.info("  any two of them negatively correlated by construction. Read the EXCESS.")

    out["channels"] = {}
    for ch, ann in ann_all.items():
        if not ann:
            continue
        logger.info("")
        logger.info("=" * 78)
        logger.info(f"CHANNEL: {ch}  ({len(ann)} drugs annotated)")
        r = run_channel(kap, ann, ch, args, rng)
        out["channels"][ch] = r
        for policy in ("different", "any"):
            p = r.get(policy, {})
            if not p.get("n_pairs"):
                logger.info(f"  plate={policy:<9} no matched pairs")
                continue
            logger.info(f"  plate={policy:<9} n={p['n_pairs']:<5} lines={p['n_lines']:<3} "
                        f"same {p['same_mech']:+.4f}  diff {p['diff_mech']:+.4f}  "
                        f"DELTA {p['delta']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}]")
        pn = r["perm_null"]
        logger.info(f"  label-permutation null: {pn['mean']:+.4f} +- {pn['sd']:.4f} (n={pn['n']})")
        head = r.get(args.plate_policy, {})
        if head.get("n_pairs", 0) < args.min_pairs or head.get("n_lines", 0) < args.min_lines:
            logger.info(f"  ** the '{args.plate_policy}' policy retains too little to headline "
                        f"(need >={args.min_pairs} pairs over >={args.min_lines} lines) -- read the "
                        f"'any' row and treat plate structure as UNCONTROLLED **")
            r["headline_valid"] = False
        else:
            r["headline_valid"] = True
            lo, hi = head["ci"]
            p = r["perm_null"]["p_one_sided"]
            # The decision is the permutation p-value; the CI is reported as a spread, not a test.
            # An interval lying entirely BELOW zero gets its own verdict: same-mechanism kappa being
            # significantly LESS similar than mismatched is not evidence of idiosyncrasy, it is a
            # sign the comparison is broken, and it must not be quotable as a null result.
            if np.isfinite(hi) and hi < 0:
                verdict = ("CONTROL FAILURE: the matched arm is significantly LESS similar than the "
                           "mismatched arm -- do not interpret, the contrast is not measuring what "
                           "it claims")
            elif np.isfinite(p) and p < 0.05:
                verdict = "MECHANISM-STRUCTURED: same-mechanism kappa is more similar (permutation p=%.4f)" % p
            elif np.isfinite(p):
                verdict = ("NOT DETECTED: no mechanism structure at this n (permutation p=%.4f). "
                           "This bounds the effect, it does not establish absence." % p)
            else:
                verdict = "NO VERDICT: permutation null unavailable"
            logger.info(f"  VERDICT ({args.plate_policy} plates): {verdict}")
            r["verdict"] = verdict
    return out


# ----------------------------------------------------------------- selftest
def selftest():
    """Two planted worlds. The estimator must separate them; if it cannot, a real null means nothing.

    Structured world: kappa(d,c) = M(moa(d), c) + small idiosyncratic noise -- the interaction is a
    property of the mechanism and the line.
    Idiosyncratic world: kappa(d,c) drawn independently per (drug, line) -- same magnitude, no
    mechanism structure. Both worlds are given the same reliability, so the estimator cannot
    separate them on SNR.
    """
    G, NL, NM, DPM = 200, 12, 8, 4          # genes, lines, mechanisms, drugs per mechanism
    ok = True
    res = {}
    for structured in (True, False):
        rng = np.random.RandomState(0)
        M = {(m, c): rng.randn(G) for m in range(NM) for c in range(NL)}
        kept = {}
        for m in range(NM):
            for k in range(DPM):
                d = f"D{m}_{k}"
                beta = rng.randn(G) * 2.0
                for c in range(NL):
                    core = M[(m, c)] if structured else rng.randn(G)
                    for half in range(1):
                        noise_a, noise_b = rng.randn(G) * 0.8, rng.randn(G) * 0.8
                        # the full-length residual must be as noisy as its two halves imply --
                        # a NOISELESS `residual` beside noisy halves means the disattenuation is
                        # dividing a clean cosine by a reliability estimated from dirty ones, so the
                        # selftest could never detect a mis-calibrated correction
                        kept[(d, f"C{c}", f"P{k%3}", "1.0")] = {
                            "residual": beta + core + 0.5 * (noise_a + noise_b),
                            "residual_A": beta + core + noise_a,
                            "residual_B": beta + core + noise_b,
                            "n_cells": 50,
                        }
        ann = {f"D{m}_{k}": f"MOA{m}" for m in range(NM) for k in range(DPM)}
        args = argparse.Namespace(min_rel=0.05, n_boot=500, n_perm=100, plate_policy="any",
                                  min_class_pairs=3, min_pairs=10, min_lines=3)
        out = report(kept, {"moa": ann}, args, np.random.RandomState(1))
        d = out["channels"]["moa"]["any"]["delta"]
        lo, hi = out["channels"]["moa"]["any"]["ci"]
        tag = "structured" if structured else "idiosyncratic"
        res[tag] = (d, lo, hi)
        logger.info(f"  [{tag}] delta {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
        if structured and not lo > 0:
            logger.error("  FAIL: planted mechanism structure NOT detected -- estimator has no power")
            ok = False
        if structured:
            # CALIBRATION, not just detection. In the structured world two same-mechanism drugs share
            # their core exactly, so a correctly disattenuated cosine should land near 1.0. A value
            # far from 1 means the reliability correction is mis-scaled even though the sign is right.
            sm = out["channels"]["moa"]["any"]["same_mech"]
            logger.info(f"  [structured] disattenuated same-mech cosine {sm:+.3f} (want ~1.0)")
            if not (0.7 <= sm <= 1.4):
                logger.error(f"  FAIL: disattenuation mis-calibrated -- same-mech cosine {sm:.3f} "
                             f"is far from the 1.0 the planted world implies"); ok = False
        if not structured and not (lo <= 0 <= hi):
            logger.error("  FAIL: mechanism structure detected where none was planted")
            ok = False
    if res["structured"][0] <= res["idiosyncratic"][0]:
        logger.error("  FAIL: the two planted worlds are not separated"); ok = False
    else:
        logger.info(f"  SEPARATION: structured {res['structured'][0]:+.4f} vs "
                    f"idiosyncratic {res['idiosyncratic'][0]:+.4f}")
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--min_rel", type=float, default=0.05,
                    help="drop (drug, line) cells whose kappa is not reliably estimated")
    ap.add_argument("--plate_policy", choices=["different", "any"], default="different",
                    help="which policy is the headline; both are always computed and reported")
    ap.add_argument("--min_pairs", type=int, default=60)
    ap.add_argument("--min_lines", type=int, default=8)
    ap.add_argument("--min_class_pairs", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=200,
                    help="permutation draws; this is the DECISION statistic, so the floor "
                         "on the achievable p-value is 1/(n_perm+1)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/kappa_channel.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")

    import variance_decomposition as vd
    import channel_gate as cg

    rng = np.random.RandomState(args.seed)
    # plate-scoped generic and leave-one-out, matching the configuration the transfer coefficient
    # headline uses -- this analysis decomposes the same kappa that number is about
    kept = vd.build_conditions(args.cache_dir, args.min_treated, args.min_control, seed=args.seed,
                               split_controls=True, loo_generic=True, generic_scope="plate")
    feats, cov = cg.load_drug_channels()
    ann_all = {k: v for k, v in feats.items() if k in ("moa", "target")}
    logger.info(f"annotation coverage: " + ", ".join(f"{k}={len(v)}" for k, v in ann_all.items()))

    out = report(kept, ann_all, args, rng)
    out["config"] = vars(args)
    out["annotation_coverage"] = {k: len(v) for k, v in ann_all.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info("")
    logger.info("READ: the DELTA line is the result -- how much more similar two same-mechanism")
    logger.info("      drugs' interactions are with a given cell line than two mismatched drugs',")
    logger.info("      disattenuated, reliability-matched, and (headline) drawn only from pairs on")
    logger.info("      different plates. Zero means the interaction belongs to the molecule.")
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
