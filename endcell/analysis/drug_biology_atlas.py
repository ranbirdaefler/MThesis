#!/usr/bin/env python
r"""
drug_biology_atlas.py — the clean, high-cell, within-plate answer to the advisor's question
============================================================================================
"Understand which drugs the model performs well on and which it doesn't. These experiments assume
each drug induces a substantial change; that may not be true — some drugs produce only subtle
changes, or different drugs induce very similar profiles. Test this before thinking of other models."

This is the DEFINITIVE version of the drug-difficulty analysis: it STREAMS ~100+ cells per drug from
Tahoe (so effect vs noise is real, not cell-count-limited) and compares drugs ONLY within the same
(cell_line, plate) (so identifiability is real drug signal, not batch). The earlier eval-tier atlas
(`drug_difficulty_atlas.py`) was cross-plate and ~10 cells/drug, so its "% inert / % redundant" were
artifacts; this replaces them with proper statistics.

PER DRUG x CELL LINE x PLATE (at ~100 cells), it reports:
  (a) POTENCY vs plate-matched DMSO — does the drug do ANYTHING?
        * n_DEG   : Welch t-test per gene vs DMSO, Benjamini-Hochberg FDR, count at q<0.05
        * effect  : ||pseudobulk(drug) - pseudobulk(DMSO)||
        * PERMUTATION p : shuffle drug/DMSO labels; is the real effect bigger than the label-shuffle
                          null? p>=0.05 => statistically INERT (not distinguishable from control)
        * snr = effect / replicate-noise (now meaningful at high cells)
  (b) IDENTIFIABILITY — is it distinguishable from its plate-mates?
        * ceiling NIR (same-plate) : a real replicate's rank against same-plate other drugs. >=0.8 =>
                                     identifiable-in-principle by anyone.
        * isolation = nearest-other-drug distance / replicate noise. <1 => REDUNDANT (a plate-mate is
                     closer than the drug's own replicate).
  (c) MoA STRUCTURE — do same-mechanism drugs have similar responses? (within- vs between-MoA distance)
  (d) DRUG x CELL-LINE INTERACTION — is a drug potent in some lines and inert in others?
  (e) DOSE — how many doses per drug, and (if a series exists) potency vs dose.
  (f) CHEM GATE (optional, gates the structure-injection arm) — does chemical (Tanimoto) similarity
      predict response similarity? Uses Tahoe's own canonical_smiles + rdkit if available.
  (g) CELL-COUNT SWEEP — ceiling NIR vs cells/drug (where identifiability saturates).

SELF-CONTAINED except for the Tahoe streaming helpers (reused from the preprocessor + esd, exactly
as calibration_eval does). Dual path-bootstrap: runs in the flat cluster layout and the split repo.

USAGE (CPU, streams — multi-hour)
  python drug_biology_atlas.py --num_shards 24 --rows_per_shard 250000 --cells_per_drug 120 \
     --min_cells 30 --min_drugs_per_group 3 --n_perm 200 --fdr_q 0.05 \
     --out RESULTS/drug_biology_atlas.json --csv RESULTS/drug_biology_atlas.csv

SELFTEST (no data/network) — plants inert / redundant-twin / distinct-potent drugs and checks the
permutation test, isolation, and ceiling recover them:
  python drug_biology_atlas.py --selftest
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
from collections import defaultdict, Counter
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- metric helpers
def pb(vs):
    return np.mean(np.stack(vs), axis=0)


def _cap(vs, cap, rng):
    if len(vs) <= cap:
        return np.stack(vs)
    idx = rng.choice(len(vs), cap, replace=False)
    return np.stack([vs[i] for i in idx])


def n_deg_fdr(D, C, q):
    """# genes differentially expressed drug-vs-DMSO at BH-FDR q (Welch t-test)."""
    try:
        from scipy.stats import ttest_ind
    except Exception:
        return None
    with np.errstate(all="ignore"):
        _, p = ttest_ind(D, C, axis=0, equal_var=False)
    p = np.where(np.isfinite(p), p, 1.0)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    bh = ranked * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    return int((bh < q).sum())


def perm_potency(D, C, n_perm, rng):
    """Effect size ||pb(drug)-pb(DMSO)|| and a label-permutation p-value. p>=0.05 => the drug's
    profile is not distinguishable from the control beyond chance = statistically inert."""
    real = float(np.linalg.norm(D.mean(0) - C.mean(0)))
    pool = np.vstack([D, C]); nd = len(D); n = len(pool)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        a = pool[idx[:nd]].mean(0); b = pool[idx[nd:]].mean(0)
        if np.linalg.norm(a - b) >= real:
            cnt += 1
    return real, float((cnt + 1) / (n_perm + 1))


# ----------------------------------------------------------------- per (cell_line, plate) group
def analyze_group(gcells, ctrl, rng, args):
    """gcells: {drug: [vecs]}. ctrl: [DMSO vecs] for this (cell_line, plate). Returns per-drug rows
    and the per-drug half-A pseudobulk profiles (for MoA / geometry)."""
    drugs = [d for d, v in gcells.items() if len(v) >= args.min_cells]
    if len(drugs) < args.min_drugs_per_group:
        return [], {}
    A, B, full = {}, {}, {}
    for d in drugs:
        v = gcells[d]; idx = rng.permutation(len(v)); h = len(idx) // 2
        A[d] = pb([v[i] for i in idx[:h]]); B[d] = pb([v[i] for i in idx[h:]]); full[d] = pb(v)
    Cm = _cap(ctrl, args.deg_cap, rng) if ctrl else None
    rows = []
    for d in drugs:
        others = [o for o in drugs if o != d]
        rep_noise = float(np.linalg.norm(A[d] - B[d]))
        potency = pval = ndeg = snr = None
        if Cm is not None:
            Dm = _cap(gcells[d], args.deg_cap, rng)
            potency, pval = perm_potency(Dm, Cm, args.n_perm, rng)
            ndeg = n_deg_fdr(Dm, Cm, args.fdr_q)
            snr = (potency / rep_noise) if rep_noise > 1e-9 else None
        # identifiability (same-plate): real replicate vs same-plate other drugs
        d_own = float(np.linalg.norm(B[d] - A[d]))
        d_oth = [float(np.linalg.norm(B[d] - A[o])) for o in others]
        ceiling_nir = float(np.mean([d_own < x for x in d_oth]))
        nn_dist = min(float(np.linalg.norm(A[d] - A[o])) for o in others)
        nn_drug = min(others, key=lambda o: float(np.linalg.norm(A[d] - A[o])))
        rows.append({
            "drug": d, "n_cells": len(gcells[d]),
            "effect": potency, "perm_p": pval, "significant": (pval is not None and pval < 0.05),
            "n_deg": ndeg, "replicate_noise": rep_noise, "snr": snr,
            "ceiling_nir": ceiling_nir, "identifiable": ceiling_nir >= args.identifiable_nir,
            "nn_drug": nn_drug, "nn_dist": nn_dist,
            "isolation": (nn_dist / rep_noise) if rep_noise > 1e-9 else None,
            "redundant": (rep_noise > 1e-9 and nn_dist < rep_noise),
        })
    return rows, A


# ----------------------------------------------------------------- aggregate reports
def _frac(xs, cond):
    xs = [x for x in xs if x is not None]
    return (float(np.mean([cond(x) for x in xs])), len(xs)) if xs else (None, 0)


def summarize(rows, args):
    logger.info("")
    logger.info("=" * 100)
    logger.info(f"  DRUG BIOLOGY ATLAS — {len(rows)} (drug x cell_line x plate) entries, within-plate")
    logger.info("-" * 100)
    ncell = np.array([r["n_cells"] for r in rows])
    logger.info(f"  cells/drug: median {np.median(ncell):.0f}  p10 {np.percentile(ncell,10):.0f}  "
                f"p90 {np.percentile(ncell,90):.0f}")

    # (a) potency / inert
    sig_frac, n_sig = _frac(rows, lambda r: r["significant"] if isinstance(r, dict) else r)
    # note: pass full rows for readability
    sig = [r for r in rows if r["perm_p"] is not None]
    if sig:
        n_signif = sum(r["significant"] for r in sig)
        logger.info("")
        logger.info(f"  (a) POTENCY vs plate-matched DMSO (permutation test, n={len(sig)}):")
        logger.info(f"      statistically ACTIVE (perm p<0.05): {n_signif}/{len(sig)} "
                    f"({100.0*n_signif/len(sig):.1f}%)")
        logger.info(f"      INERT (not distinguishable from DMSO): {len(sig)-n_signif}/{len(sig)} "
                    f"({100.0*(len(sig)-n_signif)/len(sig):.1f}%)")
        degs = np.array([r["n_deg"] for r in sig if r["n_deg"] is not None])
        if len(degs):
            logger.info(f"      #DEG vs DMSO (BH-FDR<{args.fdr_q}): median {np.median(degs):.0f}  "
                        f"p90 {np.percentile(degs,90):.0f}  (of {args._P} panel genes)")
        snr = np.array([r["snr"] for r in sig if r["snr"] is not None])
        if len(snr):
            logger.info(f"      effect/replicate-noise (SNR): median {np.median(snr):.2f}  "
                        f"p10 {np.percentile(snr,10):.2f}  p90 {np.percentile(snr,90):.2f}")

    # (b) identifiability / redundancy
    ident = [r for r in rows if r["ceiling_nir"] is not None]
    n_ident = sum(r["identifiable"] for r in ident)
    iso = [r for r in rows if r["isolation"] is not None]
    n_red = sum(r["redundant"] for r in iso)
    logger.info("")
    logger.info(f"  (b) IDENTIFIABILITY (same-plate ceiling NIR, chance 0.50):")
    ceil = np.array([r["ceiling_nir"] for r in ident])
    logger.info(f"      median ceiling {np.median(ceil):.3f}   at-chance(<0.6): "
                f"{int((ceil<0.6).sum())}/{len(ceil)} ({100.0*(ceil<0.6).mean():.1f}%)")
    logger.info(f"      IDENTIFIABLE (ceiling>={args.identifiable_nir}): {n_ident}/{len(ident)} "
                f"({100.0*n_ident/max(1,len(ident)):.1f}%)  <- the subset the model can fairly be graded on")
    logger.info(f"      REDUNDANT (isolation<1: a plate-mate closer than own replicate): "
                f"{n_red}/{len(iso)} ({100.0*n_red/max(1,len(iso)):.1f}%)")

    # biological ranking
    srt = sorted(ident, key=lambda r: r["ceiling_nir"])
    logger.info("")
    logger.info("      MOST identifiable drugs:")
    for r in srt[-8:][::-1]:
        logger.info(f"        {str(r['drug'])[:30]:30s} ceil={r['ceiling_nir']:.2f} "
                    f"deg={r['n_deg']} snr={_f(r['snr'])} moa={str(r.get('moa'))[:22]}")
    logger.info("      LEAST identifiable drugs:")
    for r in srt[:8]:
        logger.info(f"        {str(r['drug'])[:30]:30s} ceil={r['ceiling_nir']:.2f} "
                    f"deg={r['n_deg']} snr={_f(r['snr'])} moa={str(r.get('moa'))[:22]}")

    return {
        "n_entries": len(rows),
        "median_cells_per_drug": float(np.median(ncell)),
        "frac_active_perm": (float(sum(r['significant'] for r in sig)/len(sig)) if sig else None),
        "n_active": (int(sum(r['significant'] for r in sig)) if sig else None), "n_tested": len(sig),
        "median_n_deg": (float(np.median([r['n_deg'] for r in sig if r['n_deg'] is not None]))
                         if sig else None),
        "median_snr": (float(np.median([r['snr'] for r in sig if r['snr'] is not None])) if sig else None),
        "frac_identifiable": float(n_ident/max(1,len(ident))), "n_identifiable": n_ident,
        "median_ceiling_nir": float(np.median(ceil)),
        "frac_redundant": float(n_red/max(1,len(iso))),
    }


def _f(x):
    return " NA" if x is None else f"{x:.2f}"


def moa_structure(profiles_by_group, moa_of):
    win, btw = [], []
    for g, A in profiles_by_group.items():
        dl = [d for d in A if d in moa_of and moa_of[d] not in (None, "unknown", "unclear")]
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                dist = float(np.linalg.norm(A[dl[i]] - A[dl[j]]))
                (win if moa_of[dl[i]] == moa_of[dl[j]] else btw).append(dist)
    if not win or not btw:
        return None
    r = {"within_moa_mean": float(np.mean(win)), "between_moa_mean": float(np.mean(btw)),
         "n_within": len(win), "n_between": len(btw)}
    r["ratio"] = r["within_moa_mean"] / max(1e-9, r["between_moa_mean"])
    logger.info("")
    logger.info(f"  (c) MoA STRUCTURE: within-MoA dist {r['within_moa_mean']:.3f} (n={len(win)}) vs "
                f"between-MoA {r['between_moa_mean']:.3f} (n={len(btw)})  ratio {r['ratio']:.3f}")
    logger.info("      ratio << 1 => same-MoA drugs are near-duplicates (drug-level ID intrinsically "
                "hard; MoA-level is the right granularity). ~1 => MoA does not explain response.")
    return r


def interaction(rows):
    """Same drug across cell lines: is it context-dependent (potent in some lines, inert in others)?"""
    by_drug = defaultdict(list)
    for r in rows:
        by_drug[r["drug"]].append(r)
    multi = {d: rs for d, rs in by_drug.items() if len({r["cell_line"] for r in rs}) >= 3}
    if not multi:
        return None
    swings = []
    for d, rs in multi.items():
        ce = [r["ceiling_nir"] for r in rs if r["ceiling_nir"] is not None]
        if len(ce) >= 3:
            swings.append((d, float(np.min(ce)), float(np.max(ce)), float(np.max(ce) - np.min(ce))))
    if not swings:
        return None
    swings.sort(key=lambda x: -x[3])
    logger.info("")
    logger.info(f"  (d) DRUG x CELL-LINE INTERACTION ({len(multi)} drugs in >=3 lines): identifiability")
    logger.info(f"      swing (max-min ceiling) median {np.median([s[3] for s in swings]):.2f} — "
                f"many drugs are potent in some lines and inert in others (target dependency).")
    for d, lo, hi, sw in swings[:6]:
        logger.info(f"        {str(d)[:30]:30s} ceiling {lo:.2f} -> {hi:.2f} (swing {sw:.2f})")
    return {"n_multiline_drugs": len(multi),
            "median_identifiability_swing": float(np.median([s[3] for s in swings])),
            "top_context_dependent": [{"drug": d, "min": lo, "max": hi} for d, lo, hi, _ in swings[:10]]}


def dose_report(dose_sets):
    """dose_sets: {drug: set(doses seen)}."""
    counts = Counter(len(v) for v in dose_sets.values())
    n = len(dose_sets)
    multi = sum(1 for v in dose_sets.values() if len(v) >= 2)
    logger.info("")
    logger.info(f"  (e) DOSE DESIGN: {n} drugs with dose info; {multi} ({100.0*multi/max(1,n):.0f}%) "
                f"have >=2 distinct doses.")
    logger.info(f"      doses-per-drug distribution: {dict(sorted(counts.items()))}")
    if multi < 0.1 * n:
        logger.info("      => essentially single-dose; dose is a covariate, not a series. 'Subtle' "
                    "drugs are genuinely subtle (not just low-dose).")
    else:
        logger.info("      => a real dose series exists; dose-stratified potency is worth reporting.")
    return {"n_drugs": n, "frac_multidose": float(multi / max(1, n)),
            "doses_per_drug_hist": dict(counts)}


def cell_sweep(gcells_by_group, ns, rng, min_drugs):
    logger.info("")
    logger.info("  (g) CEILING NIR vs CELLS/DRUG (same-plate; where identifiability saturates):")
    out = {}
    for n in ns:
        vals = []
        for g, gc in gcells_by_group.items():
            drugs = [d for d, v in gc.items() if len(v) >= n]
            if len(drugs) < min_drugs:
                continue
            A, B = {}, {}
            for d in drugs:
                idx = rng.choice(len(gc[d]), n, replace=False); h = n // 2
                A[d] = pb([gc[d][i] for i in idx[:h]]); B[d] = pb([gc[d][i] for i in idx[h:2*h]])
            for d in drugs:
                others = [o for o in drugs if o != d]
                d_own = np.linalg.norm(B[d] - A[d]); d_oth = [np.linalg.norm(B[d] - A[o]) for o in others]
                vals.append(float(np.mean([d_own < x for x in d_oth])))
        if vals:
            out[n] = float(np.mean(vals))
            logger.info(f"        n={n:>4}  ceiling {np.mean(vals):.3f}  (headroom {np.mean(vals)-0.5:+.3f})")
    return out


# ----------------------------------------------------------------- chem gate (optional)
def load_drug_smiles():
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        path = hf_hub_download("tahoebio/Tahoe-100M", "metadata/drug_metadata.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        col = next((c for c in ("canonical_smiles", "smiles", "SMILES") if c in df.columns), None)
        if col is None:
            logger.warning(f"  chem gate: no SMILES column in drug_metadata ({list(df.columns)})")
            return {}
        return {r["drug"]: r[col] for _, r in df.iterrows() if r.get("drug") and r.get(col)}
    except Exception as e:
        logger.warning(f"  chem gate: could not load SMILES ({e})")
        return {}


def chem_gate(profiles_by_group, smiles_of):
    """Does chemical (Tanimoto) similarity predict response similarity? Gates the structure/CLIP arm."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except Exception:
        logger.warning("  (f) chem gate: rdkit not available -> skipping (install rdkit to enable)")
        return None
    fps = {}
    for d, smi in smiles_of.items():
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps[d] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        except Exception:
            continue
    if len(fps) < 10:
        logger.warning(f"  (f) chem gate: only {len(fps)} usable fingerprints -> skipping")
        return None
    from rdkit.Chem import DataStructs
    chem, resp = [], []
    seen = set()
    for g, A in profiles_by_group.items():
        dl = [d for d in A if d in fps]
        for i in range(len(dl)):
            for j in range(i + 1, len(dl)):
                key = tuple(sorted((dl[i], dl[j])))
                if key in seen:
                    continue
                seen.add(key)
                chem.append(DataStructs.TanimotoSimilarity(fps[dl[i]], fps[dl[j]]))
                resp.append(-float(np.linalg.norm(A[dl[i]] - A[dl[j]])))
    if len(chem) < 20:
        return None
    r = float(np.corrcoef(chem, resp)[0, 1])
    logger.info("")
    logger.info(f"  (f) CHEM GATE: corr(chemical Tanimoto, response similarity) = {r:+.3f} "
                f"over {len(chem):,} drug pairs, {len(fps)} drugs")
    logger.info("      >0.2 => structure predicts response; the structure/CLIP arm is well-founded.")
    logger.info("      ~0   => structure does NOT predict response here; do not build the CLIP arm.")
    return {"corr_tanimoto_response": r, "n_pairs": len(chem), "n_drugs": len(fps)}


# ----------------------------------------------------------------- streaming
def stream(args, panel_index, P):
    """-> gcells_by_group {(cl,plate): {drug: [vecs]}}, ctrl_by_group {(cl,plate): [vecs]},
          moa_of {drug: moa}, dose_sets {drug: set(dose)}."""
    import tahoe_c2s_preprocess_endcell_v2 as pp
    import expression_space_discrimination as esd
    from datasets import load_dataset

    gene_id_to_symbol, sample_to_conc, drug_info, _ = pp.load_metadata()
    moa_of = {d: (info or {}).get("moa") for d, info in drug_info.items()}
    all_shards = pp.discover_expression_shards()
    shards = pp.select_shards(all_shards, args.num_shards, args.shard_seed)
    logger.info(f"Streaming {len(shards)}/{len(all_shards)} shards (<= {args.rows_per_shard:,} rows each)")

    gcells = defaultdict(lambda: defaultdict(list))
    ctrl = defaultdict(list)
    dose_sets = defaultdict(set)
    MAX_CTRL = 300
    n_t = n_c = 0
    for shard in shards:
        url = f"hf://datasets/{pp.TAHOE_REPO}/{shard}"
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        sc = 0
        for row in ds:
            sc += 1
            if sc > args.rows_per_shard:
                break
            drug = row["drug"]; cl = row["cell_line_id"]; plate = row.get("plate")
            if plate is None:
                continue
            g = (cl, plate)
            is_dmso = (drug in ("DMSO_TF", "DMSO"))
            if is_dmso:
                if len(ctrl[g]) < MAX_CTRL:
                    v = esd.panel_expr_vector(row["genes"], row["expressions"],
                                              gene_id_to_symbol, panel_index, P)
                    if v is not None:
                        ctrl[g].append(v); n_c += 1
                continue
            if len(gcells[g][drug]) >= args.cells_per_drug:
                continue
            v = esd.panel_expr_vector(row["genes"], row["expressions"], gene_id_to_symbol, panel_index, P)
            if v is None:
                continue
            gcells[g][drug].append(v); n_t += 1
            try:
                dose_sets[drug].add(round(float(pp.parse_dose(
                    sample_to_conc.get(row["sample"], "unknown")).split()[0]), 6))
            except Exception:
                pass
            if n_t >= args.max_cells_total:
                break
        logger.info(f"  {shard}: treated={n_t:,} ctrl={n_c:,}")
        if n_t >= args.max_cells_total:
            break
    return gcells, ctrl, moa_of, dose_sets


# ----------------------------------------------------------------- selftest
def selftest(args):
    """Plant known biology and check recovery: inert (==DMSO), redundant twins, distinct potent."""
    rng = np.random.RandomState(0)
    P = 200
    base = np.zeros(P); base[rng.choice(P, 100, replace=False)] = rng.rand(100) * 2 + 1.0
    twin = base.copy(); twin[rng.choice(P, 40, replace=False)] += 3.0

    def cells(sig, n=80, noise=0.5):
        return [np.maximum(0.0, sig + rng.randn(P) * noise).astype(np.float32) for _ in range(n)]

    gc = {"inert": cells(base), "twinA": cells(twin), "twinB": cells(twin)}
    for k in range(3):
        s = base.copy(); s[rng.choice(P, 50, replace=False)] += 5.0 + k
        gc[f"distinct{k}"] = cells(s)
    ctrl = cells(base, n=120)

    args.min_cells = 30; args.min_drugs_per_group = 3; args.n_perm = 200
    args.deg_cap = 200; args.fdr_q = 0.05; args.identifiable_nir = 0.8; args._P = P
    rows, A = analyze_group(gc, ctrl, np.random.RandomState(1), args)
    by = {r["drug"]: r for r in rows}
    for r in rows:
        logger.info(f"  {r['drug']:10s} perm_p={r['perm_p']:.3f} sig={r['significant']} "
                    f"snr={_f(r['snr'])} iso={_f(r['isolation'])} ceil={r['ceiling_nir']:.2f} "
                    f"redundant={r['redundant']} nn={r['nn_drug']}")
    ok = True
    if by["inert"]["significant"]:
        logger.error("  FAIL: inert drug (==DMSO) flagged as active"); ok = False
    if not all(by[f"distinct{k}"]["significant"] for k in range(3)):
        logger.error("  FAIL: distinct potent drugs not flagged active"); ok = False
    if not (by["twinA"]["redundant"] or by["twinB"]["redundant"]):
        logger.error("  FAIL: twins not flagged redundant"); ok = False
    if by["twinA"]["nn_drug"] != "twinB":
        logger.error("  FAIL: twinA nearest neighbour is not twinB"); ok = False
    if np.mean([by[f"distinct{k}"]["ceiling_nir"] for k in range(3)]) < 0.8:
        logger.error("  FAIL: distinct drugs not identifiable"); ok = False
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_shards", type=int, default=24)
    ap.add_argument("--rows_per_shard", type=int, default=250000)
    ap.add_argument("--shard_seed", type=int, default=7)
    ap.add_argument("--cells_per_drug", type=int, default=120)
    ap.add_argument("--max_cells_total", type=int, default=3000000)
    ap.add_argument("--min_cells", type=int, default=30)
    ap.add_argument("--min_drugs_per_group", type=int, default=3)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--deg_cap", type=int, default=200, help="cap cells for the t-test / permutation")
    ap.add_argument("--fdr_q", type=float, default=0.05)
    ap.add_argument("--identifiable_nir", type=float, default=0.8)
    ap.add_argument("--cell_sweep", default="10,20,40,80,120")
    ap.add_argument("--chem_gate", action="store_true", help="run the SMILES/Tanimoto structure gate")
    ap.add_argument("--panel_file", default=None)
    ap.add_argument("--out", default="RESULTS/drug_biology_atlas.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return

    panel_file = args.panel_file
    if panel_file is None:
        for c in ("l1000_panel.json", os.path.join("src", "l1000_panel.json"),
                  os.path.join(_ROOT, "shared", "l1000_panel.json")):
            if os.path.exists(c):
                panel_file = c; break
    panel = json.load(open(panel_file))
    panel_index = {g: i for i, g in enumerate(panel)}
    P = len(panel); args._P = P
    logger.info(f"Panel {P} from {panel_file}")

    gcells, ctrl, moa_of, dose_sets = stream(args, panel_index, P)
    logger.info(f"Collected {len(gcells)} (cell_line, plate) groups")

    rng = np.random.RandomState(args.seed)
    all_rows, profiles_by_group = [], {}
    for g, gc in gcells.items():
        rows, A = analyze_group(gc, ctrl.get(g, []), rng, args)
        if not rows:
            continue
        cl, plate = g
        for r in rows:
            r["cell_line"] = cl; r["plate"] = plate; r["moa"] = moa_of.get(r["drug"])
        all_rows.extend(rows)
        profiles_by_group[g] = A
    if not all_rows:
        logger.error("No drugs analyzed — check streaming volume / --min_cells."); sys.exit(1)

    summ = summarize(all_rows, args)
    summ["moa_structure"] = moa_structure(profiles_by_group, moa_of)
    summ["interaction"] = interaction(all_rows)
    summ["dose"] = dose_report(dose_sets)
    summ["cell_sweep"] = cell_sweep(gcells, [int(x) for x in args.cell_sweep.split(",") if x],
                                    np.random.RandomState(args.seed), args.min_drugs_per_group)
    if args.chem_gate:
        summ["chem_gate"] = chem_gate(profiles_by_group, load_drug_smiles())

    out = {"summary": summ, "rows": all_rows, "n_groups": len(profiles_by_group),
           "config": {k: v for k, v in vars(args).items() if not k.startswith("_")}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")

    if args.csv:
        cols = ["cell_line", "plate", "drug", "moa", "n_cells", "effect", "perm_p", "significant",
                "n_deg", "snr", "ceiling_nir", "identifiable", "isolation", "redundant", "nn_drug"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f); w.writerow(cols)
            for r in all_rows:
                w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
        logger.info(f"-> {args.csv}")


if __name__ == "__main__":
    main()
