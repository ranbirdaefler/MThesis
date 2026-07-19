# Plate/batch confound control

All discrimination results in this project are reported under **within-plate comparison sets**. This
folder documents *why* that matters, the audit that established it, and the exact re-run scripts. If
you only read one thing: **the conclusions are unchanged under plate control; some magnitudes shrink.**

## The confound

In Tahoe-100M, **drug and plate are partially confounded by the experimental design.** Within a cell
line, each drug is assayed on its own plate(s) (~4–12 drugs per 96-well plate, ~8 plates per cell
line), and the DMSO controls we condition on are **plate-matched**. Cells that share a plate also
share technical structure — day, reagent lot, sequencing run, well effects — a *plate signature* that
has nothing to do with the drug.

Any discrimination metric that compares a **"same drug"** profile (which shares a plate with its
reference) against a **"different drug"** profile (on a *different* plate) can therefore be won by
recognising the **batch signature** instead of the drug's biology. This is the classic shortcut where
a classifier reads the scanner watermark, not the pathology.

## Diagnosis (how we know it was real)

Two checks made it concrete:

- **Control-copy probe.** A predictor that simply copies a drug's own plate-matched DMSO control —
  carrying **zero drug information** — scored NIR **0.766** on a leak-exposed subset (chance 0.50),
  matching the model itself. A zero-information predictor should score 0.50; scoring 0.766 means it
  was reading the plate.
- **Pure-batch synthetic.** A synthetic dataset with *no drug-specific signal at all* (every drug
  drawn from an identical distribution, differing only by plate signature) reproduced spike-in
  discrimination of **~0.93** cross-plate — i.e. the benchmark, as originally built, could not
  distinguish "detects drugs" from "detects plates."

## The fix

Every discrimination instrument gained a **`--same_plate_only`** flag that restricts each comparison
set to drugs on the **same `(cell_line, plate)`**. With the plate held constant across every
candidate, batch identity carries no information about which drug is which, so being closest to your
own drug can only come from real drug-specific signal.

Scripts with the flag (in `endcell/analysis/`): `nir_benchmark.py`, `calibration_eval.py`,
`spikein_metric_benchmark.py`, `expression_space_discrimination.py`, `drug_difficulty_atlas.py`,
`leak_audit.py`.

## Result — conclusions hold, magnitudes adjust

| claim | cross-plate | within-plate | verdict |
|---|---|---|---|
| **Spike-in** — metric separates real drug populations (pb15, spike=0) | panel-τ 0.991 / de_delta 0.949 | panel-τ **1.000** / de_delta **0.995** | ✅ holds (within-plate ≥ cross-plate → the discrimination was real drug signal, not batch) |
| **DRF** — NIR is the *only* calibrated metric | nir **+0.80** (@121 cells); all prediction metrics inverted | nir **+0.446** (@53 cells); all prediction metrics inverted (weighted_r2 −0.08, spearman −0.42, de_delta −0.45, panel_τ −0.23) | ✅ holds qualitatively (NIR sole positive metric). Magnitude lower — partly fewer cells (53 vs 121; NIR-DRF rises with cells), partly the removed batch |
| **Model vs the calibrated metric** — is the model drug-blind? | (contaminated by control leakage) | model − scramble **+0.014**, CI [−0.016, +0.042] → **NULL** | ✅ model does not use the drug (scramble is leak-immune by construction: same control cell, only the drug token changes) |

**Take-home:** plate leakage changed the *magnitude* of the calibration numbers, never the
*conclusion*. Within-plate, NIR is still the only calibrated metric, the metric still separates real
drug populations, and the model is still drug-blind.

Two comparisons are worth understanding because they behave differently:
- The **spike-in** and **scramble** arms are *leak-immune* — the spike-in compares real drug
  populations (no control in the loop), and scramble holds the control cell fixed while changing only
  the drug token. Both were correct all along.
- The **model − linear** comparison *was* contaminated (the model inherits its plate-matched control's
  batch, which a smoothed linear baseline discards), so we never anchor the drug-use claim to it — we
  use the scramble arm instead.

## Reproduce

Job scripts here (flat paths, for the cluster's flat `~/tahoe` layout):

| job | what it runs |
|---|---|
| `sameplate_final.sbatch` | the definitive within-plate model benchmark: ceiling / model / linear / control / scramble on NIR, per-drug rows + stratification |
| `corroborate_spikein.sbatch` | spike-in cross-plate **and** within-plate (A/B) |
| `corroborate_spikein_sameplate.sbatch` | spike-in within-plate only (lean; use when cross is already on record) |
| `corroborate_drf.sbatch` | DRF within-plate (streams ~53 cells/drug at 12 shards) |

Or add `--same_plate_only` to any instrument directly, e.g.:

```bash
python calibration_eval.py --same_plate_only --num_shards 12 --cells_per_drug 150 \
    --n_celllines 200 --min_drugs_per_cl 3 --de_k 50 --out RESULTS/drf_sameplate.json
python spikein_metric_benchmark.py --same_plate_only --sources train,eval_tier1_seen_conditions \
    --sample_size 15 --modes tail_max --out RESULTS/spikein_sameplate.json
python nir_benchmark.py --same_plate_only --eval_dir DATA --scram_dir DATA_scram \
    --model_path CKPT/final --out RESULTS/nir_sameplate.json
```

## Open item

The within-plate DRF above used ~53 cells/drug (same-plate grouping thins the streamed pool), while
the +0.80 on record was at ~121 cells. NIR-DRF rises with cell count, so the current +0.446 is an
underpowered estimate. A high-cell-count within-plate re-run (`--num_shards 32 --rows_per_shard
250000`) is the pending step to nail the definitive magnitude (expected ~+0.55–0.65).
