#!/usr/bin/env python
"""
Position-paired real-vs-scrambled DE-Δr delta. The scramble changes the prompt (hence
the content-hash example_id), so ID matching fails; but both runs used the same
subsample_seed, so row i is the SAME cell in both files. We pair by position, sanity-check
alignment via cell_line_name (must match) and drug (must differ, since it was scrambled),
then bootstrap the paired delta clustered by drug.

Δ = real - scrambled.  Positive ⇒ the real drug helped ⇒ model USES the drug token.
"""
import json, glob, os, sys
import numpy as np

OUT = "/data/BuffaF-Projetcs/florian_c2s/eval_results"
TIERS = ["tier1_seen_conditions","tier2_unseen_drugs","tier3_unseen_combos","tier4_dose_interpolation"]
METRIC = "de_delta_pearson"

def load(d, tier):
    f = os.path.join(OUT, d, f"{tier}_per_example.json")
    if not os.path.exists(f): return None
    a = json.load(open(f)); a = a if isinstance(a, list) else a.get("per_example", a)
    return a

def getm(e):
    m = e.get("metrics", {})
    v = m.get(METRIC)
    return v.get("mean") if isinstance(v, dict) else v  # per-example is a scalar here

def cluster_boot(deltas, drugs, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    by = {}
    for d, dr in zip(deltas, drugs):
        if d is None or d != d: continue
        by.setdefault(dr, []).append(d)
    groups = list(by.keys())
    if not groups: return None
    means = []
    for _ in range(n):
        gs = rng.choice(groups, size=len(groups), replace=True)
        vals = [v for g in gs for v in by[g]]
        means.append(np.mean(vals))
    allv = [v for g in groups for v in by[g]]
    return dict(mean=float(np.mean(allv)),
                ci_low=float(np.percentile(means,2.5)),
                ci_high=float(np.percentile(means,97.5)),
                n=len(allv), n_drugs=len(groups))

def run(real_dir, scram_dir):
    print(f"\n### {real_dir}  −  {scram_dir}   (Δ=real−scram; +ve ⇒ uses drug) ###")
    for tier in TIERS:
        R = load(real_dir, tier); S = load(scram_dir, tier)
        if R is None or S is None:
            print(f"  {tier:26s} MISSING"); continue
        if len(R) != len(S):
            print(f"  {tier:26s} LENGTH MISMATCH {len(R)} vs {len(S)} — position pairing unsafe"); continue
        # alignment sanity
        cl_match = sum(1 for a,b in zip(R,S) if a.get("cell_line_name")==b.get("cell_line_name"))
        drug_diff = sum(1 for a,b in zip(R,S) if a.get("drug")!=b.get("drug"))
        deltas, drugs = [], []
        for a,b in zip(R,S):
            ra, sa = getm(a), getm(b)
            if ra is None or sa is None: continue
            deltas.append(ra - sa); drugs.append(a.get("drug"))
        res = cluster_boot(deltas, drugs)
        flag = "" if cl_match==len(R) else f"  [!! cell_line match {cl_match}/{len(R)}]"
        if res:
            print(f"  {tier:26s} Δ=%+.3f [%+.3f, %+.3f]  n=%d n_drugs=%d  (cl_match=%d/%d, drug_diff=%d/%d)%s"
                  % (res["mean"],res["ci_low"],res["ci_high"],res["n"],res["n_drugs"],
                     cl_match,len(R),drug_diff,len(S),flag))
        else:
            print(f"  {tier:26s} no valid paired deltas")

if __name__ == "__main__":
    for real, scram in [("sft10k_model","c2s_scram_diff_moa"),
                        ("sft10k_model","c2s_scram_rand_drug"),
                        ("base_sft10k_model","base_scram_diff_moa"),
                        ("base_sft10k_model","base_scram_rand_drug")]:
        run(real, scram)
