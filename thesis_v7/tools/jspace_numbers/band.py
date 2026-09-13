"""Inside the shared size band at layer 12: composition, drug-first gap (plan-style averaging), sizes."""
import numpy as np, pandas as pd
import os
# repo-relative: tools/jspace_numbers -> thesis_v5 -> MThesis
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "RESULTS_cluster", "jspace_frames_v1_analysis", "analysis", "")
cols = ["model", "pair_kind", "a_prompt_id", "base_drug", "layer", "k", "dictionary", "position", "lens", "support", "phi_adj", "delta_norm", "status"]
phi = pd.read_csv(A + "phi_pairs.csv", usecols=cols)
p = phi[(phi.k == 16) & (phi.dictionary == "D") & (phi.position == "last_prompt") & (phi.lens == "full") & (phi.support == "U") & (phi.status == "ok") & (phi.pair_kind.isin(["drug", "ctrl"])) & (phi.layer == 11)].copy()
rec = pd.read_csv(A + "secondary_reconstruction.csv", usecols=["model", "prompt_id", "layer", "k", "dictionary", "position", "lens", "h_norm2"])
rec = rec[(rec.k == 16) & (rec.dictionary == "D") & (rec.position == "last_prompt") & (rec.lens == "full")]
hn = rec.set_index(["model", "layer", "prompt_id"]).h_norm2
p["rel"] = p.delta_norm / np.sqrt(hn.reindex(list(zip(p.model, p.layer, p.a_prompt_id))).values)
rng = np.random.default_rng(20260912)
for m in ["expression", "residual_repaired", "pretrained"]:
    s = p[p.model == m]
    d, c = s[s.pair_kind == "drug"], s[s.pair_kind == "ctrl"]
    lo, hi = max(d.rel.min(), c.rel.min()), min(d.rel.max(), c.rel.max())
    o = s[(s.rel >= lo) & (s.rel <= hi)]
    od, oc = o[o.pair_kind == "drug"], o[o.pair_kind == "ctrl"]
    per = o.groupby(["base_drug", "pair_kind"]).phi_adj.mean().unstack()
    both = per.dropna()
    gaps = (both["drug"] - both["ctrl"]).values
    bs = [gaps[rng.integers(0, len(gaps), len(gaps))].mean() for _ in range(2000)]
    print(f"{m}: band [{lo:.4f},{hi:.4f}] drug n={len(od)} from {od.base_drug.nunique()} drugs (max {od.base_drug.value_counts().max()} from {od.base_drug.value_counts().idxmax()}), ctrl n={len(oc)} from {oc.base_drug.nunique()} drugs")
    print(f"   mean rel in band: drug {od.rel.mean():.4f} ctrl {oc.rel.mean():.4f} | pooled gap {od.phi_adj.mean() - oc.phi_adj.mean():+.4f}")
    print(f"   drug-first gap over {len(both)} drugs with both kinds: {gaps.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
