"""Exploratory size-matched check (not in DESIGN.md). Primary spec: k=16, D, full lens, U, last_prompt.
Within each checkpoint x layer: (1) OLS phi_adj ~ 1 + is_drug + log(rel) on all pairs;
(2) mean(drug) - mean(ctrl) restricted to the overlap of relative size; 95% base-drug cluster bootstrap."""
import numpy as np, pandas as pd
import os
# repo-relative: tools/jspace_numbers -> thesis_v5 -> MThesis
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "RESULTS_cluster", "jspace_frames_v1_analysis", "analysis", "")
cols = ["model", "pair_kind", "a_prompt_id", "base_drug", "layer", "k", "dictionary", "position", "lens", "support", "phi_adj", "delta_norm", "status"]
phi = pd.read_csv(A + "phi_pairs.csv", usecols=cols)
p = phi[(phi.k == 16) & (phi.dictionary == "D") & (phi.position == "last_prompt") & (phi.lens == "full") & (phi.support == "U") & (phi.status == "ok") & (phi.pair_kind.isin(["drug", "ctrl"]))].copy()
rec = pd.read_csv(A + "secondary_reconstruction.csv", usecols=["model", "prompt_id", "layer", "k", "dictionary", "position", "lens", "h_norm2", "status"])
rec = rec[(rec.k == 16) & (rec.dictionary == "D") & (rec.position == "last_prompt") & (rec.lens == "full")]
hn = rec.set_index(["model", "layer", "prompt_id"]).h_norm2
p["rel"] = p.delta_norm / np.sqrt(hn.reindex(list(zip(p.model, p.layer, p.a_prompt_id))).values)
p["is_drug"] = (p.pair_kind == "drug").astype(float)
p["lrel"] = np.log(p.rel)
rng = np.random.default_rng(20260912)
B = 2000
def stats(s, lo, hi):
    X = np.column_stack([np.ones(len(s)), s.is_drug.values, s.lrel.values])
    beta = np.linalg.lstsq(X, s.phi_adj.values, rcond=None)[0][1]
    o = s[(s.rel >= lo) & (s.rel <= hi)]
    dd, cc = o[o.is_drug == 1].phi_adj, o[o.is_drug == 0].phi_adj
    diff = dd.mean() - cc.mean() if len(dd) and len(cc) else np.nan
    return beta, diff
rows = []
for m in ["expression", "residual_repaired", "residual_long", "pretrained"]:
    for L in [3, 7, 11]:
        s = p[(p.model == m) & (p.layer == L)]
        d, c = s[s.is_drug == 1], s[s.is_drug == 0]
        lo, hi = max(d.rel.min(), c.rel.min()), min(d.rel.max(), c.rel.max())
        o = s[(s.rel >= lo) & (s.rel <= hi)]
        nd, nc = int((o.is_drug == 1).sum()), int((o.is_drug == 0).sum())
        b0, d0 = stats(s, lo, hi)
        groups = {k: g for k, g in s.groupby("base_drug")}
        keys = list(groups)
        bs_b, bs_d = [], []
        for _ in range(B):
            pick = rng.choice(len(keys), len(keys), replace=True)
            ss = pd.concat([groups[keys[i]] for i in pick])
            bb, dd_ = stats(ss, lo, hi)
            bs_b.append(bb); bs_d.append(dd_)
        qb = np.nanpercentile(bs_b, [2.5, 97.5]); qd = np.nanpercentile(bs_d, [2.5, 97.5])
        rows.append(dict(model=m, layer=L + 1, med_rel_drug=d.rel.median(), med_rel_ctrl=c.rel.median(),
                         overlap=f"[{lo:.4f},{hi:.4f}]", n_drug_ov=nd, n_ctrl_ov=nc,
                         beta_drug=b0, beta_lo=qb[0], beta_hi=qb[1],
                         ovdiff=d0, ov_lo=qd[0], ov_hi=qd[1]))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
