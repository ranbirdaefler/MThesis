"""Number ledger for the J-space thesis section. Read-only over the copied results.

Every number the draft quotes that is NOT printed in report.md is computed here,
from phi_pairs.csv and the secondary_*.csv tables, at the primary specification
(k=16, dictionary D, full lens, union support U, pos_last_prompt) unless stated.
These are pair-level (unweighted) summaries: exploratory, not pre-registered.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import os
# repo-relative: tools/jspace_numbers -> thesis_v5 -> MThesis
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "RESULTS_cluster", "jspace_frames_v1_analysis", "analysis", "")
def spearman(a, b):
    """Pearson on average ranks (scipy is broken locally)."""
    return float(np.corrcoef(a.rank().to_numpy(), b.rank().to_numpy())[0, 1])


ORDER = ["expression", "residual_repaired", "residual_long", "pretrained"]
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

cols = ["model", "pair_id", "pair_kind", "a_prompt_id", "base_drug", "name_only",
        "layer", "k", "dictionary", "position", "lens", "support", "phi",
        "phi_null_mean", "phi_adj", "U_size", "delta_norm", "support_token_ids", "status"]
phi = pd.read_csv(A + "phi_pairs.csv", usecols=cols)
prim = phi[(phi.k == 16) & (phi.dictionary == "D") & (phi.position == "last_prompt")
           & (phi.lens == "full") & (phi.support == "U") & (phi.status == "ok")].copy()
print("primary-spec ok rows:", len(prim), "| by model:", prim.model.value_counts().to_dict())

rec = pd.read_csv(A + "secondary_reconstruction.csv",
                  usecols=["model", "prompt_id", "layer", "k", "dictionary", "position",
                           "lens", "R", "h_norm2", "status"])
recp = rec[(rec.k == 16) & (rec.dictionary == "D") & (rec.position == "last_prompt")
           & (rec.lens == "full") & (rec.status == "ok")]
hn = recp.set_index(["model", "layer", "prompt_id"]).h_norm2
prim["h_a"] = np.sqrt(hn.reindex(list(zip(prim.model, prim.layer, prim.a_prompt_id))).values)
assert prim.h_a.notna().all(), "missing ||h_a||"
prim["rel"] = prim.delta_norm / prim.h_a

print("\n== 1. pair-level levels (primary spec) ==")
g = prim.groupby(["layer", "model", "pair_kind"])
t1 = pd.DataFrame({
    "n": g.size(),
    "mean_phi": g.phi.mean(),
    "mean_null": g.phi_null_mean.mean(),
    "mean_phi_adj": g.phi_adj.mean(),
    "median_U": g.U_size.median(),
    "mean_delta": g.delta_norm.mean(),
    "median_rel": g.rel.median(),
    "mean_rel": g.rel.mean(),
})
t1["phi_over_null"] = t1.mean_phi / t1.mean_null
print(t1.round(4).to_string())

print("\n== 2. Spearman within drug pairs, layer 11 (block) ==")
for m in ORDER:
    s = prim[(prim.layer == 11) & (prim.model == m) & (prim.pair_kind == "drug")]
    r_rel = spearman(s.phi_adj, s.rel)
    r_u = spearman(s.phi_adj, s.U_size)
    print(f"{m:18s} n={len(s):4d}  rho(phi_adj, rel)={r_rel:+.3f}  rho(phi_adj, |U|)={r_u:+.3f}")

print("\n== 3. matched relative-delta bins, layer 11, all drug+ctrl pairs ==")
bins = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
s = prim[prim.layer == 11].copy()
s["bin"] = pd.cut(s.rel, bins, right=False)
t3 = s.groupby(["bin", "model"], observed=True).agg(n=("phi_adj", "size"), mean_phi_adj=("phi_adj", "mean"))
print(t3.round(4).unstack("model").to_string())
t3k = s.groupby(["bin", "model", "pair_kind"], observed=True).agg(n=("phi_adj", "size"), mean_phi_adj=("phi_adj", "mean"))
print(t3k.round(4).to_string())

print("\n== 4. lens readout Spearman, D, last_prompt, full lens: pair means ==")
ro = pd.read_csv(A + "secondary_readout.csv",
                 usecols=["model", "pair_kind", "layer", "dictionary", "position", "lens", "spearman", "status"])
ro = ro[(ro.position == "last_prompt") & (ro.lens == "full") & (ro.status == "ok")]
print(ro.groupby(["dictionary", "layer", "model", "pair_kind"]).spearman.agg(["size", "mean", "median"]).round(3).unstack("pair_kind").to_string())

print("\n== 5. output diff (no lens), last_prompt: pair means ==")
od = pd.read_csv(A + "secondary_output_diff.csv", usecols=["model", "pair_kind", "position", "dz_true_norm", "status"])
od = od[(od.status == "ok")]
print(od.groupby(["position", "model", "pair_kind"]).dz_true_norm.agg(["size", "mean", "median"]).round(2).to_string())

print("\n== 6. most frequent atoms in U, layer 11, primary spec ==")
tokjson = None
for pat in [os.path.expanduser("~/.cache/huggingface/hub/models--vandijklab--C2S-Scale-Pythia-1b-pt/snapshots/*/tokenizer.json"),
            os.path.expanduser("~/.cache/huggingface/hub/models--*Pythia*/snapshots/*/tokenizer.json"),
            os.path.expanduser("~/.cache/huggingface/hub/models--*pythia*/snapshots/*/tokenizer.json"),            ]:
    hits = glob.glob(pat, recursive=True)
    if hits:
        tokjson = hits[0]
        break
id2tok = {}
if tokjson:
    vocab = json.load(open(tokjson, encoding="utf-8"))["model"]["vocab"]
    id2tok = {v: k for k, v in vocab.items()}
    print("tokenizer:", tokjson)
for m in ORDER:
    for kind in ["drug", "ctrl"]:
        s = prim[(prim.layer == 11) & (prim.model == m) & (prim.pair_kind == kind)]
        cnt = {}
        for ids in s.support_token_ids:
            for t in set(str(ids).split(";")):
                cnt[t] = cnt.get(t, 0) + 1
        top = sorted(cnt.items(), key=lambda kv: -kv[1])[:5]
        pretty = ", ".join(f"{id2tok.get(int(t), t)!r}({t}) {c / len(s):.0%}" for t, c in top)
        print(f"{m:18s} {kind:4s} n={len(s):3d}: {pretty}")

print("\n== 7. reconstruction R, k=16, last_prompt, full lens: prompt means ==")
r16 = rec[(rec.k == 16) & (rec.position == "last_prompt") & (rec.lens == "full") & (rec.status == "ok")]
print(r16.groupby(["dictionary", "layer", "model"]).R.agg(["size", "mean", "min", "max"]).round(4).unstack("dictionary").to_string())

print("\n== 8. name-only vs MoA-changed drug pairs, layer 11: pair-mean phi_adj ==")
s = prim[(prim.layer == 11) & (prim.pair_kind == "drug")]
print(s.groupby(["model", "name_only"]).phi_adj.agg(["size", "mean"]).round(4).to_string())
