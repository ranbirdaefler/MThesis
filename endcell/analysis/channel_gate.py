#!/usr/bin/env python
"""
channel_gate.py — is unseen-drug generalisation reachable through ANY drug-side channel?

THE ARGUMENT
------------
An unseen drug can only be predicted through a property it SHARES with drugs seen in training.
So enumerate the channels, and for each ask the same question in the same frame:

    given drug d in cell line c, how well does the mean residual of the OTHER drugs in c that
    share property X with d predict d's own residual?

This is exactly the construction of `moa_lookup` in residual_eval.py, generalised over channels.
It uses only drugs that WERE in training, so it is a fair contest in every split -- unlike
`drug_lookup`, which on an unseen drug is an oracle.

The best channel is an upper bound on similarity-transfer through that channel. It is NOT a
universal upper bound on any model: a model could in principle combine channels, or learn a
non-smooth structure->response map that a nearest-neighbour average cannot express. The combined
leave-drug-out ridge (below) closes the first escape. The second is bounded by sample size rather
than argument: with a few hundred drugs, learning a non-smooth map is hopeless, so at OUR n the
lookup is effectively the bound. That scope is stated in the output and must be stated in the write-up.

WHY THIS EXISTS
---------------
Two design documents in this repo conclude that unseen-drug generalisation is out of reach, and
both infer it from the SAR gate -- which tested Morgan fingerprints and MolFormer, i.e. chemical
STRUCTURE. But `pubchem_drug_injection_spec.md` ranks raw structure the LOWEST-value channel and
protein targets "the key feature", on the reasoning that transcriptional response is driven by what
protein the drug hits, which structure only indirectly encodes. Tahoe's own `drug_metadata.parquet`
carries a `targets` column that no script in this repository has ever read. The pre-registration
generalised from the weakest channel to the strongest; this script measures the strongest.

THE CONTROL THAT MAKES IT READABLE
----------------------------------
Every channel arm is paired with a COUNT-MATCHED random null: the same number of partner drugs,
drawn at random from the same cell line, averaged the same way. Without it, a channel that scores
above chance is indistinguishable from "averaging k residuals denoises, and denoised residuals beat
chance" -- which is true of any k and says nothing about the channel.

USAGE (CPU, ~1-2 h)
  python channel_gate.py --cache_dir /data/.../ot_cache --out RESULTS/channel_gate.json

SELFTEST (no data, no network -- plants a channel that works and one that does not)
  python channel_gate.py --selftest
"""
import argparse, json, logging, os, sys
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ot"))

TAHOE_REPO = "tahoebio/Tahoe-100M"


# --------------------------------------------------------------------------- primitives
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def signed_rank(res, P, k=100):
    """Same encoding residual_eval uses, so these numbers sit on its scale."""
    v = np.zeros(P, dtype=np.float32)
    up = [j for j in np.argsort(-res)[:k].tolist() if res[j] > 0]
    dn = [j for j in np.argsort(res)[:k].tolist() if res[j] < 0]
    for blk, sgn in ((up, 1.0), (dn, -1.0)):
        for r, j in enumerate(blk, 1):
            v[j] = sgn / np.log2(r + 1.0)
    return v


def nir(own, others):
    """Tie-aware, as in residual_eval: a degenerate predictor must score 0.5, not 0."""
    o = [x for x in others if x is not None]
    if not o:
        return None
    return float(np.mean([1.0 if own > x else (0.5 if own == x else 0.0) for x in o]))


def two_way_ci(recs_vals, lines, wells):
    """Two-way cluster-robust interval, so the gate carries the same estimator as everything
    downstream of ERRATA defect 19. The channel arms are condition-level means over a crossed design
    (about 48.6 cell lines per treated well), so a one-way clustered interval is too narrow -- and
    the exposed row is `moa` under different-plate, which is barely positive already."""
    try:
        import inference as _inf
    except ImportError:
        return None
    return _inf.two_way_cluster_ci(list(recs_vals), list(lines), list(wells))


def clustered_ci(vals, clusters, rng, n_boot=2000):
    v = np.asarray(vals, float)
    cl = np.asarray(clusters)
    if len(v) == 0:
        return None
    u = sorted(set(cl.tolist()))
    idx = {k: np.where(cl == k)[0] for k in u}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.randint(0, len(u), len(u))
        boot[b] = v[np.concatenate([idx[u[t]] for t in take])].mean()
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(v), len(u)


# --------------------------------------------------------------------------- drug metadata
def load_drug_channels(repo=TAHOE_REPO, morgan_bits=2048):
    """-> {channel_name: {drug: feature}}, plus a coverage report.

    Coverage is reported BEFORE any verdict. A channel annotated on 15% of drugs cannot be closed
    by a null result -- it was never tested.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download(repo, "metadata/drug_metadata.parquet", repo_type="dataset"))
    logger.info(f"drug_metadata columns: {list(df.columns)}")
    out, cov = {}, {}

    def col(*names):
        return next((c for c in names if c in df.columns), None)

    # ---- TARGETS: the channel the design docs call "the key feature" and never measured
    tcol = col("targets", "target", "gene_targets", "target_genes")
    if tcol:
        tgt = {}
        for _, r in df.iterrows():
            d, v = r.get("drug"), r.get(tcol)
            if not d or v is None:
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                syms = [str(x).strip() for x in v if str(x).strip()]
            else:
                syms = [s.strip() for s in str(v).replace(";", ",").replace("|", ",").split(",")
                        if s.strip() and s.strip().lower() not in ("nan", "none", "unknown", "")]
            if syms:
                tgt[str(d)] = frozenset(syms)
        out["target"] = tgt
        cov["target"] = len(tgt)
        allsyms = set().union(*tgt.values()) if tgt else set()
        logger.info(f"  targets column '{tcol}': {len(tgt)} drugs, {len(allsyms)} distinct symbols, "
                    f"{sum(len(v) for v in tgt.values())} edges")
    else:
        logger.warning("  NO targets column found -- the target channel CANNOT be tested")

    # ---- MoA: already measured elsewhere; included so all channels sit on one axis
    mcol = col("moa-fine", "moa_fine", "moa")
    if mcol:
        moa = {str(r.get("drug")): str(r.get(mcol)) for _, r in df.iterrows()
               if r.get("drug") and str(r.get(mcol)).lower() not in ("nan", "none", "unclear", "")}
        out["moa"] = moa
        cov["moa"] = len(moa)

    # ---- chemical structure: re-measured HERE in the NIR frame so it is comparable to the others
    scol = col("canonical_smiles", "smiles", "SMILES")
    if scol:
        smiles = {str(r.get("drug")): str(r.get(scol)) for _, r in df.iterrows()
                  if r.get("drug") and r.get(scol)}
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import ConvertToNumpyArray
            fps = {}
            for d, s in smiles.items():
                m = Chem.MolFromSmiles(s)
                if m is None:
                    continue
                arr = np.zeros((morgan_bits,), dtype=np.int8)
                ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=morgan_bits), arr)
                fps[d] = arr
            out["chem"] = fps
            cov["chem"] = len(fps)
            logger.info(f"  Morgan fingerprints for {len(fps)} drugs")
        except Exception as e:
            logger.warning(f"  rdkit unavailable ({e}) -> chem channel skipped")
    return out, cov


# --------------------------------------------------------------------------- partner selection
def partners_target(d, cands, feats):
    """Drugs sharing at least one annotated protein target."""
    t = feats.get(d)
    if not t:
        return []
    return [o for o in cands if feats.get(o) and (feats[o] & t)]


def partners_moa(d, cands, feats):
    m = feats.get(d)
    if not m:
        return []
    return [o for o in cands if feats.get(o) == m]


def _draw_matched(rng, same_pool, diff_pool, n_same, n_diff):
    """Random partner conditions preserving the channel arm's same-plate / different-plate split.

    Returns [] when either pool is too thin to honour the split, which is the honest outcome: a
    plate-matched null that quietly falls back to an unmatched draw would reintroduce exactly the
    confound it exists to remove."""
    if n_same > len(same_pool) or n_diff > len(diff_pool) or (n_same + n_diff) == 0:
        return []
    out = []
    if n_same:
        out += [same_pool[i] for i in rng.choice(len(same_pool), n_same, replace=False)]
    if n_diff:
        out += [diff_pool[i] for i in rng.choice(len(diff_pool), n_diff, replace=False)]
    return out


def partners_chem(d, cands, feats, k=5):
    """Top-k by Tanimoto. k is fixed so the count-matched null is well defined."""
    f = feats.get(d)
    if f is None:
        return []
    scored = []
    for o in cands:
        g = feats.get(o)
        if g is None:
            continue
        inter = float(np.bitwise_and(f, g).sum())
        union = float(np.bitwise_or(f, g).sum())
        if union > 0:
            scored.append((inter / union, o))
    scored.sort(reverse=True)
    return [o for _, o in scored[:k]]


CHANNELS = {"target": partners_target, "moa": partners_moa, "chem": partners_chem}


# --------------------------------------------------------------------------- main measurement
def run(args):
    import build_residual_targets as brt

    rng = np.random.RandomState(args.seed)

    # THE FRAME. This used to call build_residuals with bare defaults, which are the PUBLISHED ones:
    # cell-line scope, no leave-one-drug-out, shared reliability controls, and a generic fitted over
    # every condition. That is ERRATA defect 24 surviving in a second consumer -- the gate was
    # measuring channel availability in a residual frame the repaired pipeline no longer uses, and
    # the gate is load-bearing for Q18. --truth_from pins it to the build's own report, exactly as
    # residual_eval does; without it the old frame is used and the run says so.
    tcfg = dict(generic_scope="cell_line", loo=False, split_controls=False,
                repro_thr=args.repro_thr, holdout=None, eval_filter=True, split=None)
    if getattr(args, "truth_from", None):
        rd = json.load(open(args.truth_from))
        tcfg.update(generic_scope=rd.get("scope", "plate"),
                    loo=bool(rd.get("leave_one_drug_out", True)),
                    split_controls=not bool(rd.get("reliability_controls", {})
                                            .get("shared_control_reliability", False)),
                    repro_thr=float(rd.get("repro_thr", args.repro_thr)),
                    eval_filter=bool(rd.get("eval_repro_filter", False)),
                    holdout=(args.holdout if getattr(args, "holdout", None)
                             and os.path.exists(args.holdout) else None))
        if tcfg["holdout"]:
            hm = json.load(open(tcfg["holdout"]))
            if hm.get("split_all"):
                # manifest string keys; build_residuals normalises them (tuple(...) is type-lossy)
                tcfg["split"] = dict(hm["split_all"])
        logger.info(f"truth frame from {args.truth_from}: scope={tcfg['generic_scope']} "
                    f"loo={tcfg['loo']} split_controls={tcfg['split_controls']} "
                    f"repro_thr={tcfg['repro_thr']:.4f} eval_filter={tcfg['eval_filter']}")
    else:
        logger.warning("=" * 96)
        logger.warning("NO --truth_from: the channel gate is measuring the PUBLISHED residual frame "
                       "(cell-line scope, no leave-one-drug-out, shared controls, generic over every "
                       "condition). Correct only when reproducing a published number.")
        logger.warning("=" * 96)

    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control, tcfg["repro_thr"], seed=args.seed,
        generic_scope=tcfg["generic_scope"], loo=tcfg["loo"],
        split_controls=tcfg["split_controls"], holdout=tcfg["holdout"],
        eval_filter=tcfg["eval_filter"], split=tcfg["split"])
    P = len(next(iter(kept.values()))["residual"])
    logger.info(f"conditions: {len(kept)}  panel: {P}")

    feats, cov = load_drug_channels()
    drugs_in_cache = {k[0] for k in kept}
    logger.info("=" * 100)
    logger.info("COVERAGE (reported BEFORE any verdict -- a channel annotated on few drugs cannot "
                "be closed by a null):")
    for ch, f in feats.items():
        n = len(drugs_in_cache & set(f))
        logger.info(f"  {ch:8s} annotated for {n}/{len(drugs_in_cache)} cached drugs "
                    f"({100*n/max(1,len(drugs_in_cache)):.0f}%)")

    by_cl = defaultdict(list)
    for k in kept:
        by_cl[k[1]].append(k)
    truth = {k: signed_rank(kept[k]["residual"], P, args.k_sig) for k in kept}

    recs = []
    for c, keys in by_cl.items():
        if len(keys) < args.min_drugs:
            continue
        drug_of = {k: k[0] for k in keys}
        for key in keys:
            d = drug_of[key]
            others = [k2 for k2 in keys if k2[0] != d]
            if len(others) < 3:
                continue
            oth_truth = [truth[k2] for k2 in others]
            # the WELL, so the gate can use the two-way estimator the rest of the stack uses.
            # key = (drug, cell_line_id, plate, sample_id); a treated well spans ~48.6 cell lines,
            # so cell line alone is not the unit of independent assignment.
            row = {"drug": d, "cell_line": c, "n_others": len(others),
                   "plate": key[2], "well": (key[3] if len(key) > 3 else key[2])}
            cand_drugs = sorted({k2[0] for k2 in others})
            p_key = key[2]
            same_pool = [k2 for k2 in others if k2[2] == p_key]
            diff_pool = [k2 for k2 in others if k2[2] != p_key]
            for ch, picker in CHANNELS.items():
                if ch not in feats:
                    continue
                sel = picker(d, cand_drugs, feats[ch])
                if not sel:
                    continue
                sel_keys = [k2 for k2 in others if k2[0] in set(sel)]
                if not sel_keys:
                    continue
                score = lambda ks: nir(cos(signed_rank(np.mean(np.stack(
                    [kept[k3]["residual"] for k3 in ks]), 0), P, args.k_sig), truth[key]),
                    [cos(signed_rank(np.mean(np.stack([kept[k3]["residual"] for k3 in ks]), 0),
                                     P, args.k_sig), t) for t in oth_truth])
                row[ch] = score(sel_keys)
                row[ch + "_k"] = len(set(sel))

                # ---- the plate confound, measured rather than assumed ------------------------
                # Mechanism- and target-paired drugs are largely CO-PLATED, while drugs drawn at
                # random are co-plated at the background rate. The residual frame these vectors
                # live in is cell-line-scoped, which FINDINGS.md:523 measures as plate-retaining
                # (+0.478 same-plate null). So a channel arm and a count-matched random null differ
                # in TWO respects -- shared mechanism AND shared plate -- and this script's own
                # estimand requires exactly one. Same class of defect as the two probe retractions.
                #
                # The co-plate rates below say whether that is actually happening here. If the two
                # arms are co-plated at the same rate, there is no confound to correct.
                n_same = sum(1 for k2 in sel_keys if k2[2] == p_key)
                n_diff = len(sel_keys) - n_same
                row[ch + "_coplate"] = n_same / len(sel_keys)

                # ---- null 1: COUNT-MATCHED (the original, kept for continuity) ----------------
                # It must NOT exclude the channel's own picks: excluding them depletes the null pool
                # of exactly the partners the channel selected, which anti-correlates the two arms
                # and biases the gap. (Measured on synthetic data where the channel is known to be
                # useless, exclusion produced a spurious -0.18.)
                rsel = set(np.asarray(cand_drugs)[
                    rng.choice(len(cand_drugs), min(len(set(sel)), len(cand_drugs)),
                               replace=False)].tolist())
                rkeys = [k2 for k2 in others if k2[0] in rsel]
                if rkeys:
                    row[ch + "_null"] = score(rkeys)
                    row[ch + "_null_coplate"] = sum(1 for k2 in rkeys if k2[2] == p_key) / len(rkeys)

                # ---- null 2: PLATE-MATCHED (the fix) ------------------------------------------
                # Same number of partner conditions AND the same same-plate/different-plate split
                # as the channel arm, drawn at random within each pool. Now the only thing the two
                # arms do not share is the annotation, which is the estimand.
                pm = _draw_matched(rng, same_pool, diff_pool, n_same, n_diff)
                if pm:
                    row[ch + "_null_pm"] = score(pm)

                # ---- arm 3: DIFFERENT-PLATE ONLY ---------------------------------------------
                # Both arms restricted to partners sharing no plate with the target. The cleanest
                # construction and the one most likely to be underpowered, so n is reported and the
                # verdict says "untestable" rather than "closed" when the pool is too thin.
                dp_sel = [k2 for k2 in sel_keys if k2[2] != p_key]
                if dp_sel and len(diff_pool) >= len(dp_sel):
                    row[ch + "_dp"] = score(dp_sel)
                    row[ch + "_dp_n"] = len(dp_sel)
                    dn = [diff_pool[i] for i in rng.choice(len(diff_pool), len(dp_sel), replace=False)]
                    row[ch + "_dp_null"] = score(dn)
            recs.append(row)
        if len(recs) and len(recs) % 500 < len(keys):
            logger.info(f"  scored {len(recs)} conditions")

    # ---------------------------------------------------------------- report
    out = {"config": vars(args), "coverage": cov, "n_conditions": len(recs)}

    def gap_vs(ch, arm_key, null_key):
        use = [r for r in recs if r.get(arm_key) is not None and r.get(null_key) is not None]
        if not use:
            return None
        diffs = [r[arm_key] - r[null_key] for r in use]
        lines = [r["cell_line"] for r in use]
        wells = [r.get("well", r["cell_line"]) for r in use]

        # TWO-WAY IS PRIMARY. `two_way_ci` was added for ERRATA defect 19 and then never called, so
        # the gate kept reporting one-way cell-line intervals -- too narrow for a crossed design, and
        # the exposed row (`moa` under different-plate) is barely positive as it is.
        tw = two_way_ci(diffs, lines, wells)
        ci = clustered_ci(diffs, lines, rng, args.n_boot)
        out_ = {"arm": float(np.mean([r[arm_key] for r in use])),
                "null": float(np.mean([r[null_key] for r in use])),
                "n": len(use), "n_cell_lines": len(set(lines)), "n_wells": len(set(wells))}
        if tw:
            out_.update(gap=tw["point"], ci=[tw["lo"], tw["hi"]], unit=tw["unit"],
                        df=tw.get("df"),
                        ci_one_way_cell_line=([ci[1], ci[2]] if ci else None))
        elif ci:
            g, lo, hi, n_, ncl = ci
            out_.update(gap=g, ci=[lo, hi], unit="one-way cell line (two-way unavailable)",
                        ci_one_way_cell_line=[lo, hi])
        else:
            return None
        return out_

    logger.info("=" * 100)
    logger.info("CO-PLATING -- how often a partner shares the target's plate. If the channel and the")
    logger.info("count-matched null differ here, the count-matched gap is part plate and part channel.")
    for ch in CHANNELS:
        cp = [r[ch + "_coplate"] for r in recs if r.get(ch + "_coplate") is not None]
        np_ = [r[ch + "_null_coplate"] for r in recs if r.get(ch + "_null_coplate") is not None]
        if cp:
            logger.info(f"  {ch:8s} channel {np.mean(cp):.3f}   count-matched null "
                        f"{(np.mean(np_) if np_ else float('nan')):.3f}   "
                        f"excess {np.mean(cp) - (np.mean(np_) if np_ else float('nan')):+.3f}")
            out.setdefault("coplate", {})[ch] = {
                "channel": float(np.mean(cp)),
                "count_matched_null": float(np.mean(np_)) if np_ else None}

    logger.info("=" * 100)
    logger.info("  channel   construction        n      arm     null      arm - null (clustered CI)")
    verdicts = {}
    for ch in CHANNELS:
        if not any(r.get(ch) is not None for r in recs):
            logger.info(f"  {ch:8s}  --- not measurable (no annotated partners) ---")
            continue
        v = {"mean_partners": float(np.mean([r[ch + "_k"] for r in recs if r.get(ch + "_k")]))}
        for label, arm_key, null_key in ((("count_matched"), ch, ch + "_null"),
                                         (("plate_matched"), ch, ch + "_null_pm"),
                                         (("different_plate"), ch + "_dp", ch + "_dp_null")):
            g = gap_vs(ch, arm_key, null_key)
            v[label] = g
            if g:
                logger.info(f"  {ch:8s}  {label:18s} {g['n']:6d}  {g['arm']:.3f}   {g['null']:.3f}   "
                            f"{g['gap']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]")
            else:
                logger.info(f"  {ch:8s}  {label:18s}    ---  no matched partner sets available")
        # THE VERDICT IS THE PLATE-MATCHED GAP. The count-matched gap is retained for continuity
        # with the published number, but it is not what decides "live".
        pm = v.get("plate_matched")
        if pm is None:
            v["live"], v["verdict"] = None, "UNTESTABLE -- no plate-matched null could be built"
        elif pm["n_cell_lines"] < args.min_lines_for_verdict:
            v["live"], v["verdict"] = None, (
                f"UNTESTABLE -- plate-matched null spans only {pm['n_cell_lines']} cell lines")
        else:
            v["live"] = bool(pm["ci"][0] > args.min_margin)
            v["verdict"] = "LIVE" if v["live"] else "closed"
        verdicts[ch] = v
        logger.info(f"  {ch:8s}  -> {v['verdict']}")
    out["channels"] = verdicts

    logger.info("-" * 100)
    any_live = [c for c, v in verdicts.items() if v.get("live")]
    untestable = [c for c, v in verdicts.items() if v.get("live") is None]
    if any_live:
        logger.info(f"  >>> {', '.join(any_live)} beats its PLATE-MATCHED null by more than "
                    f"{args.min_margin}. That is a channel result, not a plate result; unseen-drug "
                    "generalisation is live through it.")
    else:
        logger.info("  >>> NO channel beats its plate-matched null. Any gap against the "
                    "count-matched null above is plate structure, not mechanism.")
    if untestable:
        logger.info(f"  >>> {', '.join(untestable)}: UNTESTABLE in this atlas. Mechanism partners are "
                    "too co-plated to build a matched null at adequate coverage. This is NOT a null "
                    "result and must not be written as one.")
    logger.info("=" * 100)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({**out, "records": recs}, open(args.out, "w"), indent=2, default=float)
    logger.info(f"wrote {args.out}")


# --------------------------------------------------------------------------- selftest
def _trial(seed, n_drug=14, n_cl=6, P=200, n_fam=4, k=40):
    """One synthetic world: 'target' groups drugs into families that genuinely SHARE a response,
    'moa' is a random relabelling that carries nothing. Returns the channel-minus-null gap for each."""
    rng = np.random.RandomState(seed)
    fam = {f"d{j}": j % n_fam for j in range(n_drug)}
    famsig = {f: rng.randn(P) for f in set(fam.values())}
    kept = {}
    for c in range(n_cl):
        for j in range(n_drug):
            kept[(f"d{j}", f"c{c}")] = famsig[fam[f"d{j}"]] + rng.randn(P) * 0.8
    truth = {kk: signed_rank(v, P, k) for kk, v in kept.items()}
    feats = {"target": {f"d{j}": frozenset({f"T{fam[f'd{j}']}"}) for j in range(n_drug)},
             "moa": {f"d{j}": f"M{rng.randint(n_fam)}" for j in range(n_drug)}}
    out = {}
    for ch, picker in (("target", partners_target), ("moa", partners_moa)):
        gaps = []
        for c in range(n_cl):
            keys = [kk for kk in kept if kk[1] == f"c{c}"]
            for key in keys:
                others = [k2 for k2 in keys if k2[0] != key[0]]
                cands = sorted({k2[0] for k2 in others})
                sel = set(picker(key[0], cands, feats[ch]))
                if not sel:
                    continue
                ot = [truth[k2] for k2 in others]
                pred = signed_rank(np.mean(np.stack(
                    [kept[k2] for k2 in others if k2[0] in sel]), 0), P, k)
                a = nir(cos(pred, truth[key]), [cos(pred, t) for t in ot])
                rsel = set(np.asarray(cands)[rng.choice(
                    len(cands), min(len(sel), len(cands)), replace=False)].tolist())
                rp = signed_rank(np.mean(np.stack(
                    [kept[k2] for k2 in others if k2[0] in rsel]), 0), P, k)
                b = nir(cos(rp, truth[key]), [cos(rp, t) for t in ot])
                gaps.append(a - b)
        out[ch] = float(np.mean(gaps)) if gaps else float("nan")
    return out


def _trial_plate(seed, n_cl=6, n_drug=40, n_plate=5, P=200, k=40, plate_strength=1.0, coplate=0.9):
    """A world with a PLATE effect and NO channel biology whatsoever.

    Drugs carry an annotation, and same-annotation drugs are co-plated with probability `coplate` --
    which is what Tahoe looks like, because a mechanism series is typically laid out together. Their
    residuals share the plate term and nothing else: there is no family signature to find.

    A count-matched random null draws partners at the background co-plating rate, so the channel arm
    enjoys a plate-similarity advantage the null does not, and the gap comes out positive for a
    channel that carries no information. A plate-matched null removes that advantage and must read
    ~0. This is the defect the audit found in the real gate, reproduced where it can be measured.
    """
    rng = np.random.RandomState(seed)
    plate_sig = {p: rng.randn(P) for p in range(n_plate)}
    fam = {f"d{j}": j % n_plate for j in range(n_drug)}
    kept, plate_of = {}, {}
    for c in range(n_cl):
        for j in range(n_drug):
            d = f"d{j}"
            p = fam[d] if rng.rand() < coplate else rng.randint(n_plate)
            key = (d, f"c{c}", f"p{p}", f"smp{c}_{j}")
            plate_of[key] = p
            kept[key] = plate_sig[p] * plate_strength + rng.randn(P)     # no family signature
    truth = {kk: signed_rank(v, P, k) for kk, v in kept.items()}
    feats = {f"d{j}": frozenset({f"T{fam[f'd{j}']}"}) for j in range(n_drug)}

    cm, pm = [], []
    for c in range(n_cl):
        keys = [kk for kk in kept if kk[1] == f"c{c}"]
        for key in keys:
            others = [k2 for k2 in keys if k2[0] != key[0]]
            cands = sorted({k2[0] for k2 in others})
            sel = set(partners_target(key[0], cands, feats))
            sel_keys = [k2 for k2 in others if k2[0] in sel]
            if not sel_keys:
                continue
            ot = [truth[k2] for k2 in others]
            sc = lambda ks: nir(cos(signed_rank(np.mean(np.stack([kept[x] for x in ks]), 0), P, k),
                                    truth[key]),
                                [cos(signed_rank(np.mean(np.stack([kept[x] for x in ks]), 0), P, k), t)
                                 for t in ot])
            a = sc(sel_keys)
            pk = key[2]
            same_pool = [k2 for k2 in others if k2[2] == pk]
            diff_pool = [k2 for k2 in others if k2[2] != pk]
            n_same = sum(1 for k2 in sel_keys if k2[2] == pk)
            rsel = set(np.asarray(cands)[rng.choice(len(cands), min(len(sel), len(cands)),
                                                    replace=False)].tolist())
            rkeys = [k2 for k2 in others if k2[0] in rsel]
            if rkeys:
                cm.append(a - sc(rkeys))
            mk = _draw_matched(rng, same_pool, diff_pool, n_same, len(sel_keys) - n_same)
            if mk:
                pm.append(a - sc(mk))
    return (float(np.mean(cm)) if cm else float("nan"),
            float(np.mean(pm)) if pm else float("nan"))


def selftest():
    """Plant a channel that works and one that does not; both must be recovered.

    Averaged over MANY independent worlds, not one. The channel-minus-null gap for a USELESS
    channel is ~0 only IN EXPECTATION: on a single small world its standard deviation is ~0.10, so
    a single draw of -0.18 is unremarkable and asserting |gap| < 0.10 on one seed tests nothing but
    luck. That variance is itself the reason the real run gates on a clustered CI rather than on a
    point estimate.
    """
    ok = True
    n_worlds = 24
    tg = [_trial(s)["target"] for s in range(n_worlds)]
    mo = [_trial(s)["moa"] for s in range(n_worlds)]
    print(f"  over {n_worlds} independent worlds:")
    print(f"    planted-WORKING channel  gap = {np.mean(tg):+.3f}  (sd {np.std(tg):.3f})")
    print(f"    planted-USELESS channel  gap = {np.mean(mo):+.3f}  (sd {np.std(mo):.3f})")
    print(f"    single-world sd on the useless channel is {np.std(mo):.3f} -- which is why the "
          f"real run needs its clustered CI, not its point estimate")

    good = np.mean(tg) > 0.15
    ok &= good
    print(f"  planted-WORKING channel recovered: {'OK' if good else 'FAIL'}")
    good = abs(np.mean(mo)) < 0.05
    ok &= good
    print(f"  planted-USELESS channel reads ~0:  {'OK' if good else 'FAIL'}")
    good = np.mean(tg) - np.mean(mo) > 0.25
    ok &= good
    print(f"  the two are separable:             {'OK' if good else 'FAIL'}")
    d0 = nir(0.5, [0.5, 0.5, 0.5])
    good = abs(d0 - 0.5) < 1e-9
    ok &= good
    print(f"  degenerate predictor scores {d0:.3f} (expect 0.500): {'OK' if good else 'FAIL'}")

    # ---- the plate confound: a channel with NO biology, whose partners are merely co-plated ----
    cm, pm = zip(*[_trial_plate(s) for s in range(12)])
    print(f"\n  PLATE-ONLY worlds (no channel biology planted, partners 90% co-plated), 12 worlds:")
    print(f"    vs count-matched null  gap = {np.mean(cm):+.3f}  (sd {np.std(cm):.3f})")
    print(f"    vs plate-matched null  gap = {np.mean(pm):+.3f}  (sd {np.std(pm):.3f})")
    good = np.mean(cm) > 0.05
    ok &= good
    print(f"  count-matched null is FOOLED by plate: {'OK' if good else 'FAIL'} "
          f"(if this fails the fixture plants no confound and the next check proves nothing)")
    good = abs(np.mean(pm)) < 0.03
    ok &= good
    print(f"  plate-matched null closes it:          {'OK' if good else 'FAIL'}")
    good = np.mean(cm) - np.mean(pm) > 0.05
    ok &= good
    print(f"  the correction is the difference:      {'OK' if good else 'FAIL'}")

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2)
    ap.add_argument("--truth_from", default=None,
                    help="path to the target build's report.json. Reproduces that build's residual "
                         "frame exactly -- scope, leave-one-drug-out, split controls, reliability "
                         "threshold and held-out filtering. Without it the PUBLISHED frame is used, "
                         "which is correct only when reproducing a published number.")
    ap.add_argument("--holdout", default=None,
                    help="path to the build's holdout.json, so the generic is fitted on training "
                         "conditions only and the real split labels are used")
    ap.add_argument("--min_drugs", type=int, default=6,
                    help="cell lines with fewer drugs give a comparison set too small to score")
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_margin", type=float, default=0.03,
                    help="a channel must beat its COUNT-MATCHED null by this much, with the "
                         "clustered CI clearing it, to count as live")
    ap.add_argument("--min_lines_for_verdict", type=int, default=5,
                    help="a plate-matched null spanning fewer cell lines than this yields "
                         "UNTESTABLE rather than a verdict. Mechanism partners are heavily "
                         "co-plated, so the matched pool can be too thin to decide anything, and "
                         "'we could not build the control' must not be reported as 'closed'.")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="RESULTS/channel_gate.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.cache_dir:
        ap.error("--cache_dir required (or use --selftest)")
    run(args)


if __name__ == "__main__":
    main()
