import pandas as pd
import os
# repo-relative: tools/jspace_numbers -> thesis_v5 -> MThesis
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "RESULTS_cluster", "jspace_frames_v1_analysis", "analysis", "")
c = pd.read_csv(A + "contrasts.csv")
c = c[c.status == "ok"]
f = lambda x: "NA" if pd.isna(x) else f"{x:+.6f}"
def show(df, title):
    print(f"\n== {title}")
    for _, r in df.iterrows():
        print(f"{r.metric:12s} {r.contrast:10s} a={str(r.model_a):17s} L{r.layer!s:>3} k={r.k!s:>3} {r.dictionary!s:3s} {r.position:11s} {r.support!s:4s} {r.restriction:12s} "
              f"est={f(r.estimate)} [{f(r.ci_lo)}, {f(r.ci_hi)}] p={r.p_raw!s:.6} holm={r.p_holm!s:.6} gate={r.gate_outcome} word={r.wording_outcome} "
              f"loo=[{f(r.loo_min)}, {f(r.loo_max)}] hA={f(r.C_half_A)} hB={f(r.C_half_B)}")
prim = c[(c.metric == "phi_adj") & (c.k == 16) & (c.dictionary == "D") & (c.position == "last_prompt") & (c.support == "U") & (c.lens == "full") & (c.restriction == "all")]
show(prim[prim.contrast.isin(["M_drug", "M_ctrl", "W"])].sort_values(["layer", "model_a", "contrast"]), "levels and W, primary spec")
show(prim[prim.contrast.str.match(r"^(C_rr|C_rl|DiD_rr|DiD_rl)")].sort_values(["layer", "contrast"]), "primary family and companions")
sens = c[(c.metric == "phi_adj") & (c.layer == 11) & (c.contrast.isin(["C_rl", "C_rr", "DiD_rl"])) & (c.position == "last_prompt")]
show(sens.sort_values(["contrast", "restriction", "k", "dictionary", "support", "lens"]), "layer-11 C_rl/C_rr/DiD_rl, all specs at last_prompt")
sec = c[c.metric.isin(["readout", "output_diff", "reconstruction"]) & (c.position == "last_prompt") & (c.contrast.isin(["W", "M_drug", "M_ctrl"]))]
sec = sec[(sec.k.isna()) | (sec.k == 16)]
show(sec[(sec.dictionary.isna()) | (sec.dictionary == "D")].sort_values(["metric", "layer", "model_a", "contrast"]), "secondary W and levels, last_prompt")
print("\nlens values present:", sorted(c.lens.dropna().unique()), "| restrictions:", sorted(c.restriction.dropna().unique()))
