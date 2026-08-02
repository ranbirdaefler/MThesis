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

## 3. `probe3.sbatch` — RERUN the mechanistic probe, v3 estimator  (GPU)

**This one is not optional.** Q13 --- the thesis's most novel result --- has now been retracted
twice for the same species of defect, and the v2 "fix" was never actually run: the `probe2` sbatch
passed `--data_dir` and `--cells_per_drug`, neither of which exists in the argparse, so it exited 2
under `set -euo pipefail` and only the selftest ever ran.

That failure was lucky, because the v2 control was still confounded. The hook did **ablate-then-add**,
so the delivered perturbation was `(added - removed)` while only `added` was norm-matched. Since
`drug_mean` was the **raw, uncentered** class mean, the injected vector and the removed component
shared the global-mean term: the swap arm put back roughly what it removed (a near-restore) while
the random arm put back something orthogonal (a real displacement). Simulated with no model in the
loop, the random arm delivers **5.3x** the perturbation and is larger in **100%** of draws. "Swap
moves the output less than matched noise" was forced by geometry before the model was loaded.

v3 abandons ablate-and-replace for the identity test. It injects a **displacement**
`P(mu_B - mu_A)` --- the global mean cancels exactly --- and asks whether that moves the output
**toward drug B's own held-out response**, measured as $\Delta \log P(r_B)$, against two controls of
identical norm (isotropic-in-slab, and a permuted real drug-pair displacement). KL-against-clean was
abandoned because norm-matching equalises distance-moved by construction: the selftest showed it had
**no power** in the one world where power matters --- a world where the readout genuinely reads the
drug subspace.

The selftest now plants **both** truths and runs the real `run_swap()` code path, so `SELFTEST
PASSED` finally means something. It separates the worlds at $-0.18$ (inert) vs $+5.42$ (drug read).
Also fixed: subspaces fit out-of-sample, leakage measured on the purified slab, the deleted
context-shared drug component measured, cluster bootstrap over drugs, TOST equivalence margin,
per-layer headroom gate, BH across depths, stratified prompt sampling.

```bash
scp endcell/analysis/workspace_probe.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > probe3.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=probe3
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/probe3_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final
$PY workspace_probe.py --selftest
$PY workspace_probe.py \
    --eval_dir "$DATA" --model_path "$CKPT" --tier tier2_unseen_drugs \
    --layers 2,4,6,8,9,12,16 --n_drugs 24 --n_per_drug 60 --n_dims 23 \
    --n_kl_prompts 60 --do_swap --alphas 0.5,1,2,5,10 --bf16 --seed 42 \
    --out RESULTS/workspace_probe_v3.json
echo "done -> RESULTS/workspace_probe_v3.json"
EOF
sbatch probe3.sbatch
```

Flag names are taken verbatim from the last invocation that actually ran
(`endcell/jobs/workspace_probe.sbatch:39-43`). `--bf16` matters: without it the model loads in fp32
and the run is 2--3x slower. `--n_drugs 24` (up from 12) because the bootstrap resamples **drugs**,
so the cluster count sets the interval width, and at 12 the selftest lands on "underpowered" rather
than "equivalent".

```bash
LOG=$(ls -t logs/probe3_*.out | head -1); grep -nE "SELFTEST|SEPARATION|FAIL|raising --n_dims|testing hidden_states|drugs x|split:|===== hidden_states|GATE|LEAKAGE|CONFOUND|PURE drug|random @|DELETED|cell_line  |context \(|raw drug|HEADROOM|VARIANCE SHARE|SWAP|delta\[|vs A->C|vs isotropic|norm ratio|margin|SCALE|LADDER|NOT MEASURED|-> RESULTS" $LOG
```

**Read in this order. The first three are gates --- if any fails, nothing below it is readable.**

0. **`raising --n_dims`**, if present. The class-mean subspace of $C$ drugs spans up to $C-1$
   directions; the first probe3 run capped it at 10 with 24 drugs and therefore ablated fewer than
   half of them. It now scales automatically.
1. `SELFTEST PASSED` **and** the `SEPARATION` line. The separation is the real assertion: the
   estimator must return a near-zero difference in the planted inert world and a large positive one
   in the planted drug-reading world. A pass with no separation line means the swap never ran.
2. `norm match` must read exactly `1.000000`. Unlike v2's `control sanity`, this one *can* fail ---
   that number was `||V^T V z||^2 / ||V z||^2`, which is identically 1 for an orthonormal basis, a
   tautology printed to three decimals and quoted as proof the confound was fixed.
3. `HEADROOM` per layer. Any layer marked `SATURATED` is excluded automatically and reports no swap;
   at layer 12 the v2 instrument's cell-line/random ratio was 1.02x, meaning no dynamic range at all,
   and it still carried a `<<< HEADLINE` label. The first probe3 run put layer 12 at 2.83x and
   everything else between 9.9x and 133x.

**The `GATE` lines no longer gate the identity test, and this is deliberate.** The removal gate
exists to stop an unfalsifiable *ablation* null. It does not transfer to an *injection*: the hook
demonstrably moves the output, and swap-vs-permuted is internally controlled. It also cannot be
required --- a swept synthetic check shows that with the drug planted as exactly a 23-dim subspace
and all 23 dims ablated, out-of-sample removal still reaches only 22%, because a subspace fit on
held-out rows is slightly misaligned and the surviving fraction stays decodable. v2 cleared 0.8 only
because it fit the subspace **in-sample on the rows it then probed**. The identity test is now gated
on what it actually needs: a slab that carries drug identity, a live instrument, and no leakage.

Then the **headline**, `vs A->C`, read as a verdict rather than a sign. Both arms inject a real
drug displacement from A, of identical norm, differing only in which drug they point at:

- **`identity IS read`** --- injecting drug B's displacement raises B's response relative to A's own
  more than pointing at some third drug C does. Q13 **inverts**: the failure is a routing problem,
  not direction-blindness.
- **`EQUIVALENT to control within margin`** --- the original conclusion survives, and for the first
  time on a control that can support it. This is *earned*: the whole CI must sit inside the margin
  **and** contain zero. v2 read any interval containing zero as an affirmative null.
- **`INCONCLUSIVE / underpowered`** --- the honest third outcome, and a likely one.
- **`NO VERDICT --- margin below resolution`** --- the experiment cannot resolve its own margin.
  Report as such; do not reach for the nearer of the two directional readings.

`vs isotropic` is a **damage diagnostic, not the headline**. An isotropic direction is off the
class-mean manifold, so it disrupts the model in ways an on-manifold displacement does not --- that
asymmetry is exactly what sank the previous design, where it inflated the apparent effect by 36%
with the identity information removed. Read it only to confirm the swap arm is not simply doing more
damage than the controls.

Also read `LEAKAGE in the purified slab` --- cell line, plate and dose should all sit near chance. If
they do not, "pure" is a misnomer at that layer. And `DELETED drug component`: if the part
purification threw away carries the causal effect, the correct claim is *the drug is read only via
context-shared directions*, which is a materially weaker statement than *the drug is inert*.

---

## 4. `probearms.sbatch` — run the identity probe across every training arm  (GPU, ~2.5 h)

Everything measured so far is on **one** checkpoint: `pythia_sft_endcell`, the single-cell
`[END_CELL]` model --- the arm we already know is drug-blind. So the null we obtained is a null in
the one place we most expected one, which is the weakest position a null can occupy.

**The residual arm is a positive control on real data, and it is worth more than any synthetic
world.** We know behaviourally that it uses the drug: $+0.1429$ on the opposite stratum, $+0.1002$
on unseen combinations with no memorisation premium. If the probe is sound it *must* return
`identity IS read` there. The decision rule, fixed before the run:

| single cell | residual | reading |
|---|---|---|
| not read | **IS read** | The instrument works on a model known to use the drug, and the dissociation is the chapter's headline. This is the outcome that makes the null on the single-cell arm mean something. |
| not read | not read | **The probe is insensitive and our null is uninformative.** Report the probing and behavioural results only, and drop the causal claim. This falsifies the instrument, which is exactly what a positive control is for. |
| IS read | IS read | The readout does read identity everywhere and the single-cell failure lives further downstream than this probe reaches. |

The $\varepsilon$-ladder arms (consensus, optimal transport) come along for free and locate where on
the target-construction sweep drug-direction sensitivity appears, if it appears at all.

**Each arm is probed on the target format it was trained to emit.** `resp_logprob` scores
$\log P(\text{response})$, so feeding a residual-trained model a raw cell sentence would score a
format it was never trained to produce and the number would be meaningless. That is what
`--eval_file` is for. It also means every arm here is measured **in-distribution**, which is the
fair setting for "does this readout read drug identity" --- and it differs from the earlier
single-cell run, which used the unseen-drug tier, so expect that number to move.

Raw effect sizes are **not** comparable across arms, because each target format puts $\log P$ on a
different scale. The `<<< CROSS-ARM` line is: the effect divided by that model's own clean
A-vs-B preference gap, which is unit-free.

```bash
scp endcell/analysis/workspace_probe.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > probearms.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=probearms
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/probearms_%j.out
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
DATA=$D/data_diverse2_endcell_big

$PY workspace_probe.py --selftest

# arm|checkpoint|eval file in THAT arm's target format
ARMS=(
  "single_cell|$D/checkpoints/pythia_sft_endcell/final|$DATA/train.jsonl"
  "consensus|$D/checkpoints/pythia_sft_endcell_consensus/checkpoint-25500|$DATA/train_consensus.jsonl"
  "ot_T2|$D/checkpoints/pythia_sft_ot_T2/final|$D/ot_targets/T2.jsonl"
  "residual|$D/checkpoints/pythia_sft_residual/final|$D/residual_targets/residual.jsonl"
  "residual_holdout|$D/checkpoints/pythia_sft_residual_holdout2/final|$D/residual_targets_holdout2/residual.jsonl"
)

for entry in "${ARMS[@]}"; do
  ARM="${entry%%|*}"; rest="${entry#*|}"
  CKPT="${rest%%|*}"; EVAL="${rest#*|}"
  echo ""
  echo "############ ARM: $ARM ############"
  if [ ! -d "$CKPT" ]; then echo "SKIP $ARM: no checkpoint at $CKPT"; continue; fi
  if [ ! -f "$EVAL" ]; then echo "SKIP $ARM: no eval file at $EVAL"; continue; fi
  $PY workspace_probe.py \
      --arm "$ARM" --model_path "$CKPT" --eval_file "$EVAL" \
      --layers 2,4,6,8,9,12,16 --n_drugs 24 --n_per_drug 60 --n_dims 23 \
      --n_kl_prompts 60 --do_swap --alphas 0.5,1,2,5,10 --bf16 --seed 42 \
      --out "RESULTS/probe_arm_${ARM}.json" || echo "ARM $ARM FAILED (continuing)"
done
echo ""
echo "done -> RESULTS/probe_arm_*.json"
EOF
sbatch probearms.sbatch
```

Missing checkpoints and missing target files are skipped rather than aborting the job, and a failing
arm does not take the others down with it --- so a wrong path costs one arm, not the run.

```bash
LOG=$(ls -t logs/probearms_*.out | head -1); grep -nE "ARM:|SKIP |FAILED|drugs x|===== hidden|HEADROOM|SCALE|CROSS-ARM|LADDER|<<< HEADLINE|NOT MEASURED" $LOG
```

Then bring the results back and let the comparison be assembled rather than eyeballed:

```bash
scp '3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/probe_arm_*.json' ./RESULTS/
```

**Read the `<<< CROSS-ARM` line for each arm, at layer 9** (deepest layer with both a live instrument
and high drug decodability across every arm so far). The single-cell arm currently sits at
$-0.0005$ to $+0.0008$ raw, roughly $0.005$ of its clean gap. If the residual arm returns a
normalised effect an order of magnitude larger with an interval clear of zero, the instrument is
validated and the dissociation is real.

---

## 5. `probereplicate.sbatch` — does the residual layer-12 effect survive a reseed?  (GPU, ~5 h)

The multi-arm run produced exactly one result that survives BH correction across all 26 headline
tests: the **residual** arm at **layer 12**, $+0.0197$ of that model's own clean A-vs-B preference
gap, $[+0.0063, +0.0335]$, $p = 0.0010$, $q = 0.026$. It carries a monotone dose-response across the
ladder --- detected at $\alpha = 0.5, 1, 2$ ($+0.0013$, $+0.0017$, $+0.0044$) --- which noise does
not produce. The `single_cell` arm shows nothing at any of six layers.

That is the dissociation the probe was built to find, and it is exactly one layer in exactly one
arm. Three things stop it carrying a chapter as it stands:

1. **`residual_holdout` does not reproduce it.** Same encoding, 738 conditions withheld, and at
   layer 12 it gives $+0.0028$, $p = 0.62$. Its largest effect is at layer 16 ($+0.0160$,
   $p = 0.106$) --- same sign, not significant.
2. **`single_cell` has no layer-12 measurement.** It was excluded there for instrument saturation
   ($3.88\times$, under the $5\times$ gate), so the cleanest comparison --- same layer, two arms ---
   does not exist.
3. **One seed.** The seed governs which 24 drugs are drawn, the three-way split, the response pool
   and every (A,B) pairing, so a reseed is a genuine replication rather than a rerun.

This job reseeds the two residual arms. `--n_swap_prompts 240` gives the identity test four times
the prompts without touching the ablation sweep, which is where the runtime actually goes.

\medskip
\noindent\textbf{Pre-registered, before the numbers are seen.} The layer-12 effect is
\emph{replicated} if it is positive with an interval excluding zero in at least two of the three new
seeds on the \texttt{residual} arm. Anything else is one of:

- \emph{not replicated} --- layer 12 fails in two or more seeds. The mechanistic claim is dropped and
  the chapter reports the probing and behavioural results only.
- \emph{relocated} --- layer 12 fails but some other layer is positive in all three seeds. Reportable,
  but as a weaker, exploratory finding, and explicitly labelled as one.
- \emph{unmeasurable} --- layer 12 is excluded for saturation in two or more seeds. Then the
  instrument, not the model, is the limit, and that is what gets written down.

Do not lower `--headroom` to keep layer 12 in. If the instrument has no dynamic range there under a
new seed, that is a result about the instrument and belongs in the write-up.

```bash
scp endcell/analysis/workspace_probe.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > probereplicate.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=probrep
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/probereplicate_%j.out
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

ARMS=(
  "residual|$D/checkpoints/pythia_sft_residual/final|$D/residual_targets/residual.jsonl"
  "residual_holdout|$D/checkpoints/pythia_sft_residual_holdout2/final|$D/residual_targets_holdout2/residual.jsonl"
)

for SEED in 43 44 45; do
  for entry in "${ARMS[@]}"; do
    ARM="${entry%%|*}"; rest="${entry#*|}"
    CKPT="${rest%%|*}"; EVAL="${rest#*|}"
    echo ""
    echo "############ ARM: $ARM  SEED: $SEED ############"
    if [ ! -d "$CKPT" ] || [ ! -f "$EVAL" ]; then echo "SKIP $ARM: missing inputs"; continue; fi
    $PY workspace_probe.py \
        --arm "${ARM}_s${SEED}" --model_path "$CKPT" --eval_file "$EVAL" \
        --layers 2,4,6,8,9,12,16 --n_drugs 24 --n_per_drug 60 --n_dims 23 \
        --n_kl_prompts 60 --n_swap_prompts 240 \
        --do_swap --alphas 0.5,1,2,5,10 --bf16 --seed $SEED \
        --out "RESULTS/probe_rep_${ARM}_s${SEED}.json" || echo "ARM $ARM SEED $SEED FAILED (continuing)"
  done
done
echo ""
echo "done -> RESULTS/probe_rep_*.json"
EOF
sbatch probereplicate.sbatch
```

```bash
LOG=$(ls -t logs/probereplicate_*.out | head -1); grep -nE "ARM:|SEED|SKIP |FAILED|===== hidden|HEADROOM|CROSS-ARM|<<< HEADLINE|SATURATED" $LOG
```

Layer 12 is the line to read, on the three `residual` seeds. Bring the JSON back so the three seeds
can be pooled rather than eyeballed:

```bash
scp '3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/probe_rep_*.json' ./RESULTS/
```

---

## 6. `kappachannel.sbatch` — is the interaction a property of the mechanism, or of the molecule?  (CPU, ~1 h)

The transfer coefficient says roughly $45\%$ of the drug-specific residual is drug$\times$cell-line
interaction, reached by neither the per-drug lookup nor the model. The thesis calls that a
*quantified, unclaimed target* without knowing which of two things it is, and they lead to different
closing claims:

- **structured** — two drugs sharing a mechanism interact with a given cell line in similar ways.
  The $45\%$ is then reachable, the feature that reaches it is named, and the Limitations paragraph
  currently advising *against* a mechanism-conditioned model becomes wrong and gets rewritten.
- **idiosyncratic** — the interaction belongs to the individual molecule and no annotation we hold
  predicts it. The closing claim then sharpens from "unclaimed" to "real, and unreachable from
  available covariates", which is a stronger negative than the one we currently have.

The channel gate does not answer this. It predicted the whole residual, which is $\approx 55\%$
per-drug main effect, so its $+0.078$ for mechanism is consistent with two EGFR inhibitors merely
having similar *average* effects. This runs the same logic on $\kappa$ alone.

No GPU. It reads the residual cache and fetches Tahoe's `drug_metadata.parquet` for the `moa-fine`
and `targets` columns.

```bash
scp endcell/analysis/kappa_channel.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > kappachannel.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=kapchan
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kappachannel_%j.out
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

$PY kappa_channel.py --selftest

$PY kappa_channel.py --cache_dir "$D/ot_cache" \
    --min_rel 0.05 --plate_policy different --n_perm 200 --n_boot 2000 --seed 42 \
    --out RESULTS/kappa_channel.json

echo "=== sensitivity: a stricter reliability floor, same everything else ==="
$PY kappa_channel.py --cache_dir "$D/ot_cache" \
    --min_rel 0.20 --plate_policy different --n_perm 200 --n_boot 2000 --seed 42 \
    --out RESULTS/kappa_channel_rel20.json

echo done
EOF
sbatch kappachannel.sbatch
```

The second run is not a fishing expedition. Disattenuation divides by $\sqrt{\text{rel}_1
\text{rel}_2}$, so at `min_rel 0.05` a pair can be amplified up to $20\times$; an audit established
that this costs variance rather than bias, but if the two floors disagree the answer is being
carried by badly-estimated pairs and neither number should be quoted.

```bash
LOG=$(ls -t logs/kappachannel_*.out | head -1); grep -nE "SELFTEST|SEPARATION|FAIL|CEILING|CONTINUITY|analytic null|CHANNEL:|plate=|permutation|VERDICT|sensitivity|coverage" $LOG
```

**Read in this order.**

1. `SELFTEST PASSED` and the `SEPARATION` line. The estimator must recover a planted
   mechanism-structured world and a planted idiosyncratic one; without that a null means nothing.
2. `CEILING` — $\kappa$'s own split-half reliability. Every cross-drug cosine is disattenuated by it,
   so the scale is one where $1.0$ means "as similar as $\kappa$ is to its own replicate". If the
   ceiling is very low, $\kappa$ is barely estimable and no arm below it is interpretable.
3. `CONTINUITY` — read the **excess** over the analytic null, never the raw value. Centring a drug's
   $\kappa$s on their own mean makes any two of them correlate at $\approx -1/(m-1)$ before biology
   enters; the selftest reproduces this to three decimals ($-0.0907$ observed against $-0.0909$
   predicted).
4. `plate=different` — the headline. If the script prints that this policy retains too little to
   support a headline, read `plate=any` and treat plate structure as uncontrolled.
5. `VERDICT`, which is decided by the **permutation p-value**, not by the bootstrap interval. The
   interval is reported as a spread only, because its width runs $\approx 0.92$ of the true sampling
   spread and testing it against zero over-rejects.

A `CONTROL FAILURE` verdict — the mechanism-matched arm significantly *less* similar than the
mismatched arm — means the comparison is broken, not that the interaction is idiosyncratic. It is not
quotable either way.

```bash
scp '3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/kappa_channel*.json' ./RESULTS/
```

---

## 7. `leakmag.sbatch` — how bad is the transductive holdout, really?  (CPU, ~20 min)

An external audit found `build_residual_targets.build_residuals` to be **transductive**: it computes
the generic shift as a mean over *all* drugs in a cell line, applies the reliability filter
$\cos(r_A, r_B) > 0.2$ to *all* conditions, and only then assigns the split. Held-out outcomes
therefore entered both the **centring** of the training targets and the **selection** of which
conditions exist. The audit's recommendation is to rebuild train-only, retrain, and only then make a
generalisation claim.

That is the right fix and it costs about a day. This measures the contamination first, because the
two halves of it have different consequences and only one of them is plausibly fatal:

- **Centring.** Each held-out drug supplies $1/m$ of its cell line's generic, and cell lines carry
  20--204 drugs, so the held-out set contributes roughly 10--15\%. But the generalisation claim is a
  \gap{} --- model minus scramble --- and both arms are scored against the *same* target, so a shared
  additive perturbation of that target cancels to first order.
- **Selection.** The reliability filter decided which conditions exist to be scored at all, and that
  does **not** cancel in a difference. If the retained set changes materially when the generic is
  built train-only, the evaluation population itself was chosen using held-out information.

```bash
scp endcell/analysis/leak_magnitude.py 3180408@login.hpc.unibocconi.it:~/tahoe/
```

```bash
cd ~/tahoe && cat > leakmag.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=leakmag
#SBATCH --account=3180408
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/leakmag_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
D=/data/BuffaF-Projetcs/florian_c2s

$PY leak_magnitude.py --cache_dir "$D/ot_cache" \
    --holdout "$D/residual_targets_holdout2/holdout.json" \
    --repro_thr 0.2 --seed 42 \
    --out RESULTS/leak_magnitude.json
echo done
EOF
sbatch leakmag.sbatch
```

```bash
LOG=$(ls -t logs/leakmag_*.out | head -1); grep -nE "conditions with enough|holdout manifest|of .* conditions in the cache|cos\(r_all|retained|Jaccard|kept under|READ" $LOG
```

**The decision rule, fixed before the numbers.**

| observation | reading | action |
|---|---|---|
| target cosine $\geq 0.999$ **and** retained-set Jaccard $\geq 0.99$ | the contamination is real but immaterial | report it as a measured limitation, keep the generalisation claim, no retrain |
| target cosine $\geq 0.999$ but Jaccard $< 0.99$ | the training signal is clean, the evaluation **population** is not | rebuild train-only and retrain; the current number is not quotable |
| target cosine $< 0.999$ | the training targets themselves carry held-out information | rebuild train-only and retrain |

Note the asymmetry deliberately built into that table: two of the three outcomes lead to a retrain.
The cheap measurement exists to find out whether we are in the one case that does not, not to look
for an excuse to skip it. If the answer is ambiguous, retrain.

**If the retrain is needed** it is one build plus one train plus one eval, using the existing
`reorder.sbatch` recipe with `--prompt_order drug_first` and a `build_residual_targets` patched so
that the generic and the reliability filter see training conditions only. That patch is the real
deliverable of this section, and it should be written whether or not this measurement excuses the
current run --- a transductive builder is a defect regardless of how large its effect turns out to be.

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
