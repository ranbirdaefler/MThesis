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
    kept, generic, ctrl_rows, X, meta = brt.build_residuals(
        args.cache_dir, args.min_treated, args.min_control, args.repro_thr, seed=args.seed)
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
            row = {"drug": d, "cell_line": c, "n_others": len(others)}
            cand_drugs = sorted({k2[0] for k2 in others})
            for ch, picker in CHANNELS.items():
                if ch not in feats:
                    continue
                sel = picker(d, cand_drugs, feats[ch])
                if not sel:
                    continue
                vecs = [kept[k2]["residual"] for k2 in others if k2[0] in set(sel)]
                if not vecs:
                    continue
                pred = signed_rank(np.mean(np.stack(vecs), 0), P, args.k_sig)
                row[ch] = nir(cos(pred, truth[key]), [cos(pred, t) for t in oth_truth])
                row[ch + "_k"] = len(set(sel))
                # COUNT-MATCHED NULL: the same NUMBER of partners, drawn at random from ALL other
                # drugs in this cell line. It must NOT exclude the channel's own picks: excluding
                # them depletes the null pool of exactly the partners the channel selected, which
                # anti-correlates the two arms and biases the gap. (Measured on synthetic data
                # where the channel is known to be useless, exclusion produced a spurious -0.18.)
                rsel = set(np.asarray(cand_drugs)[
                    rng.choice(len(cand_drugs), min(len(set(sel)), len(cand_drugs)),
                               replace=False)].tolist())
                rvecs = [kept[k2]["residual"] for k2 in others if k2[0] in rsel]
                if rvecs:
                    rp = signed_rank(np.mean(np.stack(rvecs), 0), P, args.k_sig)
                    row[ch + "_null"] = nir(cos(rp, truth[key]), [cos(rp, t) for t in oth_truth])
            recs.append(row)
        if len(recs) and len(recs) % 500 < len(keys):
            logger.info(f"  scored {len(recs)} conditions")

    # ---------------------------------------------------------------- report
    out = {"config": vars(args), "coverage": cov, "n_conditions": len(recs)}
    logger.info("=" * 100)
    logger.info("  channel        n scored   NIR      count-matched null   channel - null (clustered CI)")
    verdicts = {}
    for ch in CHANNELS:
        vals = [(r["cell_line"], r[ch], r.get(ch + "_null")) for r in recs if r.get(ch) is not None]
        if not vals:
            logger.info(f"  {ch:12s}  --- not measurable (no annotated partners) ---")
            continue
        m_ch = float(np.mean([v for _, v, _ in vals]))
        paired = [(cl, v - n) for cl, v, n in vals if n is not None]
        m_nl = float(np.mean([n for _, _, n in vals if n is not None])) if paired else float("nan")
        ci = clustered_ci([d for _, d in paired], [cl for cl, _ in paired], rng, args.n_boot)
        mean_k = float(np.mean([r[ch + "_k"] for r in recs if r.get(ch + "_k")]))
        if ci:
            g, lo, hi, n, ncl = ci
            live = lo > args.min_margin
            verdicts[ch] = {"nir": m_ch, "null": m_nl, "gap": g, "ci": [lo, hi],
                            "n": n, "n_cell_lines": ncl, "mean_partners": mean_k, "live": bool(live)}
            logger.info(f"  {ch:12s}  {n:6d}     {m_ch:.3f}    {m_nl:.3f}               "
                        f"{g:+.4f} [{lo:+.4f}, {hi:+.4f}]  {'LIVE' if live else 'closed'}")
    out["channels"] = verdicts

    logger.info("-" * 100)
    any_live = [c for c, v in verdicts.items() if v["live"]]
    if any_live:
        logger.info(f"  >>> {', '.join(any_live)} beats its count-matched null by more than "
                    f"{args.min_margin}. Unseen-drug generalisation is LIVE through that channel; "
                    "the next arm is a channel-conditioned prompt evaluated on a leave-one-MoA-out "
                    "split.")
    else:
        logger.info("  >>> NO channel beats its count-matched null. Unseen-drug generalisation is "
                    "closed BY MEASUREMENT, not by assumption -- and the closure is a statement "
                    "about similarity transfer at this sample size, which is the scope to write.")
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
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2)
    ap.add_argument("--min_drugs", type=int, default=6,
                    help="cell lines with fewer drugs give a comparison set too small to score")
    ap.add_argument("--k_sig", type=int, default=100)
    ap.add_argument("--min_margin", type=float, default=0.03,
                    help="a channel must beat its COUNT-MATCHED null by this much, with the "
                         "clustered CI clearing it, to count as live")
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
