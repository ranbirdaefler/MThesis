# `shared/` — representation-agnostic core

Code and reference data used by **both** the legacy full-panel pipeline and the current
[END_CELL] pipeline. Nothing here is specific to one cell-sentence representation, which is
exactly why it lives outside `legacy_whole_panel/` and `endcell/`.

| File | Role |
|---|---|
| `evaluate_c2s_tahoe.py` | ⭐ **The metric library.** DE-Δr, panel-τ, rank utilities, mean-shift baselines, noise-ceiling and bootstrap-CI helpers. Imported (as `evaluate_c2s_tahoe`) by ~13 scripts across both pipelines, so every reported number comes from the *same* scoring functions. |
| `l1000_panel.json` | The fixed 946-gene panel (LINCS L1000 ∩ Tahoe vocabulary). The single source of gene order for every representation. |
| `l1000_landmark_genes.txt` | Raw L1000 landmark symbols the panel is built from. |
| `build_l1000_panel.py` | Rebuilds `l1000_panel.json` from the landmark list ∩ Tahoe genes. Run once, up front. |
| `inspect_generation.py` | Small debug helper for eyeballing model generations. |

**Import path.** Every pipeline script adds `shared/` to `sys.path` via its own path bootstrap,
so `import evaluate_c2s_tahoe` resolves regardless of which subfolder the caller lives in.
`evaluate_c2s_tahoe.py` itself imports no local modules — it is the leaf of the dependency graph.
