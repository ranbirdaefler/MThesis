#!/usr/bin/env python
r"""
experimental_unit_audit.py -- what is the physical experiment, and what can it support?
=============================================================================================
Every analysis in this repository keys on `(drug, cell_line, plate, dose)`. Tahoe does not assign
treatments that way: it treats a spheroid **sample/well** containing a mixture of cell lines, and the
per-cell-line profiles are deconvolved observations nested inside one treatment. Two cell lines from
one treated well are one assignment of that drug, not two.

This is a DISCOVERY step and its output decides the cost of everything downstream, so it answers
questions rather than asserting conclusions:

  1. Is `dose` in the cache actually holding sample identifiers? If so the sample identity is
     MISLABELLED, not lost, and the cache can be enriched in place instead of rebuilt -- which is the
     single largest cost fork in the remediation.
  2. How many cell lines nest inside one treatment? That number is the pseudoreplication factor, and
     it sets the correct clustering unit for every confidence interval in the thesis.
  3. Do replicate treatments exist -- the same drug at the same dose in independent wells or on
     independent plates? Without them, `repro_cos` measures split-half sampling precision within one
     well and must not be called biological reproducibility. With them, several claims get stronger.
  4. How many samples are COMBINATION treatments? The shipped parser takes `parsed[0]`, so any
     multi-drug sample has been analysed as if only its first component were applied.
  5. What fraction of conditions have a recoverable molar dose, once parsed properly?
  6. Does any sample cross the existing holdout? If one physical treatment has conditions on both
     sides of the split, the split is broken at the assignment level and no amount of careful fitting
     repairs it.

  python experimental_unit_audit.py --selftest
  python experimental_unit_audit.py --cache_dir /data/.../ot_cache \
      --holdout /data/.../residual_targets_holdout2/holdout.json \
      --out RESULTS/experimental_unit_audit.json
"""
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_ROOT, "shared"), os.path.join(os.path.dirname(_HERE), "..", "shared")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, json, logging
from collections import defaultdict, Counter
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import tahoe_design as td
except ImportError:  # allow running from the repo root
    sys.path.insert(0, os.path.join(os.getcwd(), "shared"))
    import tahoe_design as td

TAHOE_REPO = "tahoebio/Tahoe-100M"


# --------------------------------------------------------------------------- inputs
def load_cache_meta(cache_dir):
    import pandas as pd
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    logger.info(f"cache meta: {len(meta)} rows, columns {list(meta.columns)}")
    return meta


def load_sample_metadata(repo=TAHOE_REPO):
    """sample -> SampleTreatment, parsed properly (combinations retained, dose in molar)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, "metadata/sample_metadata.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    logger.info(f"sample metadata: {len(df)} rows, columns {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        s = r.get("sample")
        if s is None:
            continue
        out[str(s)] = td.parse_treatment(r.get("drugname_drugconc", ""), str(s))
    return out, df


# --------------------------------------------------------------------------- the audit
def audit(meta, samples, split=None):
    rep = {}

    # ---- 1. is `dose` holding sample identifiers? --------------------------------------------
    dose_vals = meta["dose"].astype(str).tolist() if "dose" in meta.columns else []
    is_sample = td.looks_like_sample_id(dose_vals)
    rep["dose_column_holds_sample_ids"] = bool(is_sample)
    ex = sorted(set(dose_vals))[:5]
    logger.info("")
    logger.info("=" * 78)
    logger.info("(1) IS THE `dose` COLUMN ACTUALLY A SAMPLE IDENTIFIER?")
    logger.info(f"    verdict: {'YES' if is_sample else 'no'}   examples: {ex}")
    if is_sample:
        logger.info("    -> sample identity is MISLABELLED, not lost. The cache can be enriched in")
        logger.info("       place: read `dose` as sample_id, resolve the real dose from metadata.")
    else:
        logger.info("    -> the sample identity is NOT recoverable from this column; a rebuild may")
        logger.info("       be required to obtain it.")

    sample_col = "dose" if is_sample else ("sample_id" if "sample_id" in meta.columns else None)
    if sample_col is None:
        rep["fatal"] = "no column carries a sample identifier"
        logger.error("    no sample identifier anywhere -- the treatment unit cannot be reconstructed")
        return rep

    treated = meta[~meta["is_control"].astype(bool)] if "is_control" in meta.columns else meta
    smp = treated[sample_col].astype(str).values
    cl = treated["cell_line_id"].astype(str).values
    plate = treated["plate"].astype(str).values
    drug = treated["drug"].astype(str).values

    # ---- 2. nesting: cell lines per treatment -------------------------------------------------
    lines_per_sample = defaultdict(set)
    plates_per_sample = defaultdict(set)
    drugs_per_sample = defaultdict(set)
    for s, c, p, d in zip(smp, cl, plate, drug):
        lines_per_sample[s].add(c)
        plates_per_sample[s].add(p)
        drugs_per_sample[s].add(d)
    npl = np.array([len(v) for v in lines_per_sample.values()])
    rep["n_samples"] = len(lines_per_sample)
    rep["cell_lines_per_sample"] = {"mean": float(npl.mean()), "median": float(np.median(npl)),
                                    "min": int(npl.min()), "max": int(npl.max())}
    logger.info("")
    logger.info("(2) NESTING -- how many cell lines share one treatment?")
    logger.info(f"    {len(lines_per_sample)} treatment samples, {len(treated)} treated rows")
    logger.info(f"    cell lines per sample: mean {npl.mean():.1f}  median {np.median(npl):.0f}  "
                f"range {npl.min()}-{npl.max()}")
    logger.info(f"    -> an interval that resamples CONDITIONS overstates the evidence by up to "
                f"{npl.mean():.1f}x; the assignment unit is the sample.")

    bad_plate = {s: sorted(v) for s, v in plates_per_sample.items() if len(v) > 1}
    bad_drug = {s: sorted(v) for s, v in drugs_per_sample.items() if len(v) > 1}
    rep["samples_on_multiple_plates"] = len(bad_plate)
    rep["samples_with_multiple_drugs"] = len(bad_drug)
    logger.info(f"    samples spanning >1 plate: {len(bad_plate)}   >1 drug: {len(bad_drug)}")

    # ---- 3. replicate treatments --------------------------------------------------------------
    # A replicate is the SAME drug at the SAME dose in a DIFFERENT sample. Plate labels in this
    # cache are numbered within cell line, so "plate 6" is not one physical plate across lines --
    # replication is therefore counted by sample, and by plate only within a cell line.
    tre = {}
    for s in lines_per_sample:
        st = samples.get(s)
        if st is None or not st.treatments:
            continue
        prim = st.primary
        if prim is None:
            continue
        tre[s] = (prim.drug, td.molar_key(prim.dose.molar))
    by_treat = defaultdict(set)
    for s, key in tre.items():
        by_treat[key].add(s)
    rep_counts = Counter(len(v) for v in by_treat.values())
    n_replicated = sum(1 for v in by_treat.values() if len(v) > 1)
    rep["distinct_treatments"] = len(by_treat)
    rep["treatments_with_replicate_samples"] = n_replicated
    rep["samples_per_treatment_hist"] = {str(k): int(v) for k, v in sorted(rep_counts.items())}
    logger.info("")
    logger.info("(3) REPLICATE TREATMENTS -- same drug, same molar dose, different sample")
    logger.info(f"    distinct (drug, dose) treatments: {len(by_treat)}")
    logger.info(f"    of which replicated in >1 sample: {n_replicated} "
                f"({100.0*n_replicated/max(1,len(by_treat)):.0f}%)")
    logger.info(f"    samples per treatment: {dict(sorted(rep_counts.items()))}")
    if n_replicated == 0:
        logger.info("    -> NO independent-well replication. `repro_cos` measures split-half")
        logger.info("       sampling precision within ONE well and must not be called biological")
        logger.info("       reproducibility. Workstream H's replicate analysis is not available.")
    else:
        logger.info("    -> independent-well replication EXISTS; biological reproducibility can be")
        logger.info("       estimated on that subset rather than inferred from split halves.")

    # within-cell-line plate replication, which is what the cache's plate labels can support
    by_dcl = defaultdict(set)
    for s, c, p, d in zip(smp, cl, plate, drug):
        if s in tre:
            by_dcl[(tre[s][0], tre[s][1], c)].add(p)
    multi_plate = sum(1 for v in by_dcl.values() if len(v) > 1)
    rep["drug_dose_line_on_multiple_plates"] = multi_plate
    rep["drug_dose_line_groups"] = len(by_dcl)
    logger.info(f"    (drug, dose, cell line) seen on >1 plate: {multi_plate}/{len(by_dcl)}")

    # ---- 4. combination treatments ------------------------------------------------------------
    combos = [s for s in lines_per_sample if samples.get(s) and samples[s].is_combination]
    rep["combination_samples"] = len(combos)
    rep["combination_examples"] = [samples[s].raw for s in combos[:5]]
    logger.info("")
    logger.info("(4) COMBINATION TREATMENTS -- silently truncated by the shipped parser")
    logger.info(f"    samples carrying >1 compound: {len(combos)} of {len(lines_per_sample)}")
    for s in combos[:3]:
        logger.info(f"      e.g. {s}: {samples[s].raw}")
    if combos:
        logger.info("    -> these have been analysed as single-drug conditions. Either exclude them")
        logger.info("       explicitly or model them as combinations; do not keep taking the first.")

    # ---- 5. dose recoverability ---------------------------------------------------------------
    ok = sum(1 for s in lines_per_sample
             if samples.get(s) and samples[s].primary and samples[s].primary.dose.ok)
    known = sum(1 for s in lines_per_sample if samples.get(s))
    reasons = Counter()
    for s in lines_per_sample:
        st = samples.get(s)
        if st and st.primary and not st.primary.dose.ok:
            reasons[st.primary.dose.reason] += 1
    rep["samples_in_metadata"] = known
    rep["samples_with_molar_dose"] = ok
    rep["dose_failure_reasons"] = {k: int(v) for k, v in reasons.most_common(10)}
    logger.info("")
    logger.info("(5) DOSE RECOVERABILITY once parsed properly")
    logger.info(f"    samples found in metadata : {known}/{len(lines_per_sample)}")
    logger.info(f"    with a usable molar dose  : {ok} ({100.0*ok/max(1,len(lines_per_sample)):.0f}%)")
    for r, n in reasons.most_common(5):
        logger.info(f"      unusable: {n:>5}  {r}")

    # ---- 5b. does unit-stripping corrupt the ORIGINAL preprocessing's dose? --------------------
    # tahoe_c2s_preprocess_endcell.py does `dose_float = float(dose_str.split()[0])`, which throws
    # the unit away. Two failure modes follow, and which one bites is a property of the data:
    #   COLLISION  1 uM and 1 nM both become 1.0 -> two concentrations treated as one. The tier-4
    #              held-out-dose test then holds out a dose it also trained on.
    #   SPLIT      0.05 uM and 50 nM are one concentration but become 0.05 and 50.0 -> one dose
    #              counted as two, and `--group_keys drug,cell_line_id,dose_float` splits a group.
    per_drug = defaultdict(set)
    for st in samples.values():
        if st and st.primary and st.primary.dose.value is not None:
            per_drug[st.primary.drug].add((float(st.primary.dose.value), st.primary.dose.unit,
                                           td.molar_key(st.primary.dose.molar)))
    collisions, splits = [], []
    for d, entries in per_drug.items():
        by_float, by_molar = defaultdict(set), defaultdict(set)
        for val, unit, mk in entries:
            by_float[round(val, 6)].add(mk)
            by_molar[mk].add(round(val, 6))
        for f, mks in by_float.items():
            if len(mks) > 1:
                collisions.append({"drug": d, "dose_float": f, "distinct_molar": len(mks)})
        for mk, fs in by_molar.items():
            if len(fs) > 1:
                splits.append({"drug": d, "molar_key": str(mk), "distinct_dose_float": sorted(fs)})
    rep["dose_float_collisions"] = len(collisions)
    rep["dose_float_splits"] = len(splits)
    rep["dose_float_collision_examples"] = collisions[:5]
    rep["dose_float_split_examples"] = splits[:5]
    logger.info("")
    logger.info("(5b) UNIT-STRIPPED `dose_float` in the ORIGINAL preprocessing")
    logger.info(f"    drugs with >=1 numeric dose : {len(per_drug)}")
    logger.info(f"    COLLISIONS (one float, several concentrations): {len(collisions)}")
    logger.info(f"    SPLITS     (one concentration, several floats): {len(splits)}")
    for e in collisions[:3]:
        logger.info(f"      collision: {e['drug']} dose_float={e['dose_float']} covers "
                    f"{e['distinct_molar']} real concentrations")
    for e in splits[:3]:
        logger.info(f"      split: {e['drug']} one concentration appears as {e['distinct_dose_float']}")
    if collisions:
        logger.info("    -> tier 4 (held-out dose) is unsound for those drugs: the held-out dose was")
        logger.info("       also trained on under a different unit. Any tier-4 number needs a caveat.")
    if splits:
        logger.info("    -> dose-keyed grouping splits one concentration in two, which deflates every")
        logger.info("       per-dose sample size it touches.")
    if not collisions and not splits:
        logger.info("    -> every drug states its doses in a single unit, so unit-stripping happened")
        logger.info("       to be harmless here. Worth stating explicitly rather than assuming.")

    # ---- 6. does any sample cross the holdout? -------------------------------------------------
    if split:
        cond_split = defaultdict(set)
        for s, c, p, d in zip(smp, cl, plate, drug):
            v = split.get((d, c, p, s))
            if v is not None:
                cond_split[s].add(v)
        crossing = {s: sorted(v) for s, v in cond_split.items() if len(v) > 1}
        rep["samples_crossing_split"] = len(crossing)
        rep["samples_with_split_info"] = len(cond_split)
        logger.info("")
        logger.info("(6) DOES ONE PHYSICAL TREATMENT SIT ON BOTH SIDES OF THE SPLIT?")
        logger.info(f"    samples with split info: {len(cond_split)}")
        logger.info(f"    samples CROSSING the split: {len(crossing)}")
        if crossing:
            for s, v in list(crossing.items())[:5]:
                logger.info(f"      {s}: {v}")
            logger.info("    -> the split is broken at the ASSIGNMENT level. No amount of careful")
            logger.info("       fitting repairs this; the new split must be assigned by sample.")
        else:
            logger.info("    -> no sample crosses the split, so the existing assignment is at least")
            logger.info("       coherent with the treatment unit.")
    return rep


# --------------------------------------------------------------------------- selftest
def selftest():
    import pandas as pd
    ok = True

    def check(c, m):
        nonlocal ok
        logger.info(("  ok   " if c else "  FAIL ") + m)
        ok = ok and c

    # a cache whose `dose` column holds sample IDs, four cell lines per sample, one replicate pair
    rows = []
    for si, (drug, smp_id) in enumerate([("D1", "smp_1"), ("D2", "smp_2"), ("D1", "smp_3")]):
        for c in range(4):
            rows.append({"drug": drug, "cell_line_id": f"C{c}", "plate": f"p{si%2}",
                         "dose": smp_id, "is_control": False})
    meta = pd.DataFrame(rows)
    samples = {
        "smp_1": td.parse_treatment("[('D1', 0.05, 'uM')]", "smp_1"),
        "smp_2": td.parse_treatment("[('D2', 1.0, 'uM'), ('X', 2.0, 'uM')]", "smp_2"),
        "smp_3": td.parse_treatment("[('D1', 50, 'nM')]", "smp_3"),   # == 0.05 uM, different unit
    }
    r = audit(meta, samples)

    check(r["dose_column_holds_sample_ids"], "sample IDs in the dose column are detected")
    check(r["n_samples"] == 3, "three treatment samples found")
    check(r["cell_lines_per_sample"]["mean"] == 4.0, "four cell lines nested per sample")
    check(r["combination_samples"] == 1, "the combination sample is flagged")
    check(r["treatments_with_replicate_samples"] == 1,
          "smp_1 and smp_3 recognised as replicates despite uM vs nM units")
    check(r["samples_with_molar_dose"] == 2, "the two single-drug samples yield a molar dose")
    # D1 is dosed at 0.05 uM and 50 nM: one concentration, two unit-stripped floats
    check(r["dose_float_splits"] == 1 and r["dose_float_collisions"] == 0,
          "unit-stripping splits D1's single concentration into dose_float 0.05 and 50.0")

    # and the mirror-image failure: one float standing for two different concentrations
    coll = {"smp_a": td.parse_treatment("[('D9', 1.0, 'uM')]", "smp_a"),
            "smp_b": td.parse_treatment("[('D9', 1.0, 'nM')]", "smp_b")}
    rc = audit(meta, {**samples, **coll})
    check(rc["dose_float_collisions"] == 1,
          "dose_float 1.0 standing for both 1 uM and 1 nM is caught")

    # a sample crossing the split must be caught
    split = {("D1", "C0", "p0", "smp_1"): "train", ("D1", "C1", "p0", "smp_1"): "unseen_combo"}
    r2 = audit(meta, samples, split)
    check(r2["samples_crossing_split"] == 1, "a sample straddling the split is caught")

    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--holdout")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--out", default="RESULTS/experimental_unit_audit.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return
    if not a.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")

    meta = load_cache_meta(a.cache_dir)
    samples, sample_df = load_sample_metadata(a.repo)

    split = None
    if a.holdout and os.path.exists(a.holdout):
        ho = json.load(open(a.holdout))
        raw = ho.get("split", ho)
        split = {}
        for k, v in raw.items():
            parts = k.rsplit("|", 3)
            if len(parts) == 4:
                split[tuple(parts)] = v
        logger.info(f"holdout manifest: {len(split)} conditions")

    rep = audit(meta, samples, split)
    rep["cache_dir"] = a.cache_dir

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2, default=float)

    logger.info("")
    logger.info("=" * 78)
    logger.info("WHAT THIS DECIDES")
    logger.info("  (1) enrich vs rebuild the cache -- the largest cost fork in the remediation")
    logger.info("  (2) the clustering unit for every interval in the thesis")
    logger.info("  (3) whether biological reproducibility can be measured at all, or only")
    logger.info("      split-half sampling precision")
    logger.info("  (4) whether combination samples must be excluded before the retrain")
    logger.info("  (6) whether the new split can be assigned by condition or must be by sample")
    logger.info(f"-> {a.out}")


if __name__ == "__main__":
    main()
