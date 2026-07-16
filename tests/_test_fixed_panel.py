"""Offline unit test for the new fixed-panel functions. No network. Safe to delete."""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "shared"), os.path.join(_ROOT, "legacy_whole_panel", "preprocess")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tahoe_c2s_preprocess import (
    make_paired_cell_sentences_fixed_panel,
    fit_expression_linear_model_panel,
)

# Build a small panel from the real panel file, plus non-panel genes.
full_panel = json.load(open(os.path.join(_ROOT, "shared", "l1000_panel.json")))
panel = full_panel[:100]
panel_index = {g: i for i, g in enumerate(panel)}
nonpanel = [f"NONPANEL{i}" for i in range(250)]
all_syms = panel + nonpanel
gene_id_to_symbol = {i: s for i, s in enumerate(all_syms)}  # token_id -> symbol

rng = np.random.default_rng(0)

def make_cell(expressed_panel_idx, expressed_nonpanel_n=220):
    """Return (genes, exprs) Tahoe-style arrays. token ids 0..99 = panel, 100+ = nonpanel."""
    genes, exprs = [], []
    for k, idx in enumerate(expressed_panel_idx):
        genes.append(idx)                      # token id == panel position here
        exprs.append(100.0 - k)                # decreasing expression by listed order
    for j in range(expressed_nonpanel_n):
        genes.append(100 + j)
        exprs.append(float(rng.integers(1, 50)))
    return np.array(genes), np.array(exprs, dtype=np.float64)

# control expresses panel genes 0..69 (in that order = decreasing expr)
ctrl_genes, ctrl_exprs = make_cell(list(range(70)))
# treated expresses a DIFFERENT panel subset/order: 60..5 reversed -> different ranking
treat_genes, treat_exprs = make_cell(list(range(60, 5, -1)))

ctrl_s, resp_s = make_paired_cell_sentences_fixed_panel(
    ctrl_genes, ctrl_exprs, treat_genes, treat_exprs,
    gene_id_to_symbol, panel, panel_index,
)

assert ctrl_s is not None and resp_s is not None, "QC unexpectedly failed"
cg, rg = ctrl_s.split(), resp_s.split()
print(f"control genes: {len(cg)}  response genes: {len(rg)}  panel: {len(panel)}")
assert len(cg) == len(panel), "control != panel length"
assert len(rg) == len(panel), "response != panel length"
assert set(cg) == set(panel), "control is not a permutation of the panel"
assert set(rg) == set(panel), "response is not a permutation of the panel"

# Leak-free: control ordering must depend ONLY on the control cell, not the treated one.
ctrl_s2, _ = make_paired_cell_sentences_fixed_panel(
    ctrl_genes, ctrl_exprs, *make_cell(list(range(10, 60))),  # different treated cell
    gene_id_to_symbol=gene_id_to_symbol, panel_symbols=panel, panel_index=panel_index,
)
assert ctrl_s == ctrl_s2, "LEAK: control sentence changed when only the treated cell changed!"
print("OK leak-free: control ordering independent of treated cell")

# Deterministic canonical tail: the unexpressed panel genes (>=70) must appear in
# canonical (panel_index) order at the end of the control sentence.
tail = cg[70:]
expected_tail = [g for g in panel if panel_index[g] >= 70]
assert tail == expected_tail, "unexpressed tail is not in canonical order"
print("OK canonical worst-rank tail is deterministic")

# QC: a treated cell with <50 expressed panel genes is rejected.
none_c, none_r = make_paired_cell_sentences_fixed_panel(
    ctrl_genes, ctrl_exprs, *make_cell(list(range(10))),  # only 10 panel genes expressed
    gene_id_to_symbol=gene_id_to_symbol, panel_symbols=panel, panel_index=panel_index,
)
assert none_c is None and none_r is None, "min_panel_expressed QC did not trigger"
print("OK QC rejects treated cells with <50 expressed panel genes")

# Linear-model panel fit on a few control cells.
controls = [make_cell(list(range(rng.integers(55, 70)))) for _ in range(20)]
lm = fit_expression_linear_model_panel(controls, gene_id_to_symbol, panel, panel_index, n_cells=20)
assert lm is not None and lm["fit"] == "panel_restricted", "panel fit failed"
print(f"OK panel linear fit: slope={lm['slope']:.3f} intercept={lm['intercept']:.3f} "
      f"R2={lm['r_squared']:.3f} (n_cells={lm['n_control_cells']})")

print("\nALL FIXED-PANEL UNIT TESTS PASSED")
