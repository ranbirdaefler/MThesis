# Two experiments: which field is read, and does moving the drug closer help

```bash
scp endcell/analysis/residual_eval.py endcell/ot/build_residual_targets.py \
    3180408@login.hpc.unibocconi.it:~/tahoe/
```

Both scripts have changed. `residual_eval.py` on the cluster is also **two patches stale** — it still
prints the retracted `HEADROOM ... has essentially vanished` line — so this scp fixes that too.

---

## 1. `fielddecomp.sbatch` — which part of the prompt does the model read?  (GPU, ~1.5 h, NO retrain)

The channel gate showed that **mechanism carries real signal** (+0.078 over its count-matched null).
The prompt has always contained `Mechanism: {moa}`. And the model's unseen-drug gap was null. Those
three facts sit uneasily together, and the only thing currently reconciling them is "that arm was
underpowered" — which is true but is a dodge.

This decomposes the scramble by field. Each arm swaps **one** part of the instruction to the
opposite-signature drug and leaves every other byte identical:

| arm | swaps | isolates |
|---|---|---|
| `scramble_drugonly` | the drug name, mechanism kept | does it read the identity token? |
| `scramble_moaonly` | the mechanism, drug name kept | **does it read the knowledge channel it already has?** |
| `scramble_opposite` | both | the existing number, for continuity |

```bash
cd ~/tahoe && cat > fielddecomp.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=fielddecomp
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fielddecomp_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
$PY residual_eval.py --cache_dir "$D/ot_cache" \
    --model_path "$D/checkpoints/pythia_sft_residual_holdout2/final" --model_kind residual \
    --holdout "$D/residual_targets_holdout2/holdout.json" \
    --train_file "$D/residual_targets_holdout2/residual.jsonl" \
    --field_decomp --split_quota "train=200,unseen_combo=250,unseen_drug=120" \
    --k_samples 4 --max_new_tokens 1400 --bf16 --seed 42 \
    --out RESULTS/field_decomp.json
echo done
EOF
sbatch fielddecomp.sbatch
```

```bash
LOG=$(ls -t logs/fielddecomp_*.out | head -1); grep -E "WHICH FIELD|swap |>>>|scramble_|model |ceiling|drug_lookup|train |unseen_" $LOG
```

**What decides it — `swap MECHANISM only`:**

- **CI clears zero** → the model reads the mechanism string it is given. A channel-conditioned
  prompt adding protein targets is then well motivated, and the future-work section becomes a
  concrete proposal.
- **CI spans zero** → the model does *not* measurably read a field that demonstrably carries signal.
  The bottleneck for unseen drugs is the **readout**, not the information — which is Q13 reappearing
  exactly where it matters most, makes the closing argument considerably stronger, and saves two
  weeks on an arm that would have failed.

Either outcome is worth having. That is what makes it worth running before the reorder.

---

## 2. `reorder.sbatch` — is the drug simply too far away?  (GPU, ~5 h, full rebuild + retrain)

A third hypothesis, independent of the two already tested. Q13 says the readout is direction-blind;
Q15 says the target's tokens are diluted. Neither addresses **distance**: the prompt is

```
Predict the response of MCF7 to Lapatinib at 0.5 uM. Mechanism: EGFR inhibitor.
Control cell: <~123 gene symbols, several hundred BPE tokens>

Response cell:
```

so the drug name sits several hundred tokens upstream of the first generated token, behind a wall of
gene symbols. `--prompt_order drug_last` moves the instruction to sit immediately before generation:

```
Control cell: <~123 gene symbols>
Predict the response of MCF7 to Lapatinib at 0.5 uM. Mechanism: EGFR inhibitor.

Response cell:
```

**Identical fields, identical values, identical targets.** The only thing that changes is the
distance between the drug token and the token that has to condition on it. That is what makes it a
clean test rather than a confounded one.

```bash
cd ~/tahoe && cat > reorder.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=reorder
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/reorder_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
CACHE=/data/BuffaF-Projetcs/florian_c2s/ot_cache
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
TGT=/data/BuffaF-Projetcs/florian_c2s/residual_targets_reorder
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_residual_reorder

echo "=== [1] build targets with the instruction adjacent to generation ==="
$PY build_residual_targets.py --cache_dir "$CACHE" --out_dir "$TGT" \
    --prompt_order drug_last \
    --tier2_file "$DATA/eval_tier2_unseen_drugs.jsonl" \
    --tier3_file "$DATA/eval_tier3_unseen_combos.jsonl" \
    --holdout_combos 0.15 --holdout_drugs 0.10 --min_combo_conditions 250 --seed 42

echo "=== [2] retrain -- IDENTICAL recipe, prompt order is the only variable ==="
$PY train_c2s_tahoe_endcell.py --mode full \
    --model_name vandijklab/C2S-Scale-Pythia-1b-pt \
    --train_file "$TGT/residual.jsonl" \
    --eval_file "$DATA/eval_tier1_seen_conditions.jsonl" \
    --output_dir "$CKPT" \
    --num_epochs 1 --batch_size 1 --grad_accum 16 \
    --bf16 --gradient_checkpointing --max_length 8192 \
    --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 \
    --log_every 50 --save_every 1000 --keep_checkpoints 3

echo "=== [3] eval -- prompt_order MUST match training ==="
$PY residual_eval.py --cache_dir "$CACHE" --model_path "$CKPT/final" --model_kind residual \
    --prompt_order drug_last --field_decomp \
    --holdout "$TGT/holdout.json" --train_file "$TGT/residual.jsonl" \
    --split_quota "train=200,unseen_combo=250,unseen_drug=120" \
    --k_samples 4 --max_new_tokens 1400 --bf16 --seed 42 \
    --out RESULTS/re_reorder.json
echo done
EOF
sbatch reorder.sbatch
```

```bash
LOG=$(ls -t logs/reorder_*.out | head -1); grep -E "===|stratified sample|validity|ceiling|model |scramble_|drug_lookup|WHICH FIELD|swap |train |unseen_combo|unseen_drug|CROSS-CONTEXT|>>>" $LOG
```

**The comparison** is against `holdout2`, which is the same construction at `drug_first`:

| | drug\_first (holdout2) | drug\_last |
|---|---|---|
| opposite-swap gap, train | +0.0898 [+0.0530, +0.1269] | ? |
| opposite-swap gap, `unseen_combo` | +0.1002 [+0.0661, +0.1368] | ? |
| model NIR | 0.657 / 0.650 | ? |

A clear rise means the drug was partly a **distance** problem, which would be a genuinely new
finding and a fourth mechanism for the thesis. No change means distance was not the bottleneck, and
that is worth stating too — it closes the last cheap architectural explanation and leaves the
readout account standing alone.

---

## 3. `probe2.sbatch` — RERUN the mechanistic probe with a corrected control  (GPU, ~2 h)

**This one is not optional.** A structural audit found that the headline swap in Q13 --- the
thesis's most novel result --- was compared against a control that is not matched in the way the
claim requires.

`comp_b` is drug B's projection **into** the purified slab, at most 10 dimensions. The old control
drew `rng.randn(H)` in the **full 2048-dimensional** residual space and matched only the L2 norm.
The hook ablates `V_drug_pure` before adding, so the swap arm perturbs only inside the ablated slab
while roughly **99.5\% of the random vector's energy lands in directions that were never ablated**
--- including the cell-line directions this same experiment measures at ~0.6 variance share and
15--20$\times$ the KL. "The swap moves the output less than matched noise" is exactly what that
confound predicts, whatever the readout actually does.

Three fixes, all local:

1. the random vector is now drawn **inside** `V_drug_pure` (`V_drug_pure @ randn(d_pure)`, norm-matched),
   randomising direction while holding subspace and norm fixed;
2. a **paired** bootstrap difference with a CI replaces two bare means --- the arms share prompts and
   drug means, so the pairing is real, and the previous version stored no dispersion at all;
3. the removal gate now also runs on `V_drug_pure`, the subspace the causal claims actually use. The
   old gate ran on the **raw** slab, certifying something other than what needed certifying.

```bash
scp endcell/analysis/workspace_probe.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > probe2.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=probe2
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/probe2_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s
$PY workspace_probe.py --selftest
$PY workspace_probe.py \
    --model_path "$D/checkpoints/pythia_sft_endcell/final" \
    --data_dir "$D/data_diverse2_endcell_big" \
    --layers 2,4,6,8,9,12,16 --n_drugs 12 --cells_per_drug 40 --do_swap \
    --out RESULTS/workspace_probe_v3.json
echo done
EOF
sbatch probe2.sbatch
```

```bash
LOG=$(ls -t logs/probe2_*.out | head -1); grep -E "SELFTEST|GATE|CONFOUND|pure-drug subspace|causal KL|SWAP|paired difference|control sanity|HEADLINE" $LOG
```

**Read `control sanity` first** --- it reports what fraction of the random vector's energy lies inside
the purified subspace, and it must be ~100\%. If it is not, the control has drifted out of the slab
again and nothing below it is readable.

Then the **paired difference and its CI**, which is now the headline rather than two bare means:

- **CI excludes zero, positive** --- injecting the right drug moves the output MORE than a random
  direction of the same norm in the same slab. The readout *is* drug-direction sensitive, and Q13's
  conclusion **inverts**: the failure is a routing problem, not direction-blindness.
- **CI spans zero** --- the original conclusion survives, now on a control that supports it.
- **CI excludes zero, negative** --- the strangest outcome and worth reporting as such.

Any of the three is publishable. The current number is not, because its control cannot distinguish
them.

---

## Validated locally before sending

| check | result |
|---|---|
| `drug_first` places the drug before the control sentence | OK |
| `drug_last` places it after, immediately before generation | OK |
| field swap `drug` / `moa` / `both`, under **both** orders | all OK, rest of prompt byte-identical |
| same-mechanism partner with `fields='moa'` returns `None` | OK |

That last one matters more than it looks. If a swap partner happens to share the original's
mechanism, a mechanism-only scramble produces a prompt identical to the model's own — a gap of
exactly zero by construction, which would have read as "the model ignores mechanism". It now returns
`None` and the arm is dropped for that condition rather than silently scoring zero.

The order-agnostic rewrite of `scramble_prompt` was also necessary rather than cosmetic: the old
version split the prompt on `Control cell:` and scrambled only what preceded it, so under
`drug_last` it would have found no drug name to replace, returned `None`, and dropped every scramble
arm without a word.
