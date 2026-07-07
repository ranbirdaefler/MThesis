"""Offline test for eval baseline helpers (no model/network). Safe to delete."""
import json, os, random
from evaluate_c2s_tahoe import (
    control_from_prompt, compute_mean_shift, predict_mean_shift,
    cell_sentence_to_gene_ranks, compute_rank_correlation,
)

panel = json.load(open("l1000_panel.json"))
panel_index = {g: i for i, g in enumerate(panel)}
rng = random.Random(0)

def make_prompt(cell_line, drug, control_sentence):
    return (f"Predict the response of {cell_line} to {drug} at 0.1 uM. "
            f"Mechanism: unclear.\nControl cell: {control_sentence}\n\nResponse cell:")

# Build a tiny synthetic train.jsonl: response = control shifted by a fixed permutation,
# so a mean-shift baseline should recover structure.
tmp = "_tmp_train.jsonl"
with open(tmp, "w") as f:
    for n in range(50):
        ctrl_order = panel[:]; rng.shuffle(ctrl_order)
        resp_order = panel[:]; rng.shuffle(resp_order)
        ex = {
            "prompt": make_prompt("MCF7" if n % 2 else "A549", f"Drug{n%5}", " ".join(ctrl_order)),
            "response": " ".join(resp_order),
            "metadata": {"cell_line_name": "MCF7" if n % 2 else "A549", "drug": f"Drug{n%5}"},
        }
        f.write(json.dumps(ex) + "\n")

# 1. control_from_prompt round-trips the control sentence
ex0 = json.loads(open(tmp).readline())
ctrl = control_from_prompt(ex0["prompt"])
assert len(ctrl.split()) == len(panel), "control_from_prompt did not recover full control"
assert set(ctrl.split()) == set(panel), "control genes != panel"
print(f"OK control_from_prompt: recovered {len(ctrl.split())} genes")

# 2. global mean shift -> tuple of dicts, panel-length
gmap, gfallback = compute_mean_shift(tmp, panel, per_cellline=False, limit=50)
assert isinstance(gmap, dict) and len(gmap) == len(panel), "global shift map malformed"
assert gmap is gfallback or gmap == gfallback, "global fallback should equal global map"
print(f"OK global mean shift: {len(gmap)} gene shifts")

# 3. per-cell-line mean shift -> per-cl dict + global fallback
pmap, pfallback = compute_mean_shift(tmp, panel, per_cellline=True, limit=50)
assert set(pmap.keys()) == {"MCF7", "A549"}, f"unexpected cell lines: {pmap.keys()}"
assert len(pfallback) == len(panel)
print(f"OK per-cell-line mean shift: cell lines={list(pmap.keys())}")

# 4. predict_mean_shift -> panel-length permutation (mirrors evaluate_tier global path)
pred = predict_mean_shift(ex0["prompt"], gfallback, panel, panel_index)
assert len(pred.split()) == len(panel) and set(pred.split()) == set(panel), "prediction malformed"
print(f"OK predict_mean_shift: {len(pred.split())}-gene panel permutation")

# 5. per-cell-line path (mirrors evaluate_tier per-cl path)
cl = ex0["metadata"]["cell_line_name"]
sm = pmap.get(cl, pfallback)
pred_cl = predict_mean_shift(ex0["prompt"], sm, panel, panel_index)
assert len(pred_cl.split()) == len(panel)
print("OK per-cell-line predict path works")

# 6. overall metric over panel returns a real number (worst-rank handles full panel)
m = compute_rank_correlation(cell_sentence_to_gene_ranks(pred),
                             cell_sentence_to_gene_ranks(ex0["response"]),
                             gene_subset=panel)
assert m["kendall_tau"] is not None and m["n_genes"] == len(panel)
print(f"OK panel-scored tau computed over {m['n_genes']} genes: tau={m['kendall_tau']:.4f}")

os.remove(tmp)
print("\nALL EVAL-BASELINE TESTS PASSED")
