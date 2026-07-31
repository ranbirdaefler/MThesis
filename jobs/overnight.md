# Overnight jobs — three cheap analyses

Copy the two changed scripts up, then submit all three. They are independent; nothing waits on
anything else. Total: ~1 GPU-hour and two short CPU jobs.

```bash
scp endcell/analysis/channel_gate.py endcell/analysis/variance_decomposition.py \
    3180408@login.hpc.unibocconi.it:~/tahoe/
```

---

## 1. `ctrl1400.sbatch` — close the token-budget confound  (GPU, ~1 h)

The Q15 control columns ran at 600 generation tokens against the residual model's 1400. The budget
alone doubles the measured effect, so the three-column comparison in the draft is currently flagged
with a `\todo`. This re-runs **both controls at 1400** so the comparison is matched.

Nothing new to write — it is `residual_eval.py` with `--model_kind cellsentence` at the correct
budget, against the two control checkpoints.

```bash
cd ~/tahoe && cat > ctrl1400.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=ctrl1400
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ctrl1400_%j.out
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
for arm in singlecell:pythia_sft_endcell ot:pythia_sft_ot_T2; do
  name="${arm%%:*}"; ckpt="${arm#*:}"
  echo ""; echo "############ CONTROL: $name ($ckpt) at 1400 tokens ############"
  $PY residual_eval.py --cache_dir "$D/ot_cache" \
      --model_path "$D/checkpoints/$ckpt/final" --model_kind cellsentence \
      --n_conditions 250 --k_samples 4 --max_new_tokens 1400 --bf16 --seed 42 \
      --out "RESULTS/re_${name}_1400.json"
done
echo done
EOF
sbatch ctrl1400.sbatch
```

Read with:
```bash
LOG=$(ls -t logs/ctrl1400_*.out | head -1); grep -E "CONTROL:|validity|near|orth|opposite|model |scramble_|ceiling" $LOG
```

**What decides it:** the `opposite` stratum for each control. At 600 they read −0.0204 (single-cell)
and +0.0263 (OT). If they stay put at 1400, the three-column comparison stands and the `\todo`
comes out. If the single-cell control rises to clear zero, the claim "only the residual model shows
a gradient" weakens and the chapter needs rewording — which is exactly why this has to be run
rather than argued.

---

## 2. `channelgate.sbatch` — is unseen-drug generalisation reachable?  (CPU, ~1–2 h)

The one genuinely untested question in the thesis. `drug_metadata.parquet` carries a `targets`
column that **no script in this repository has ever read**, and both design documents concluded
unseen drugs were out of reach from a gate that tested chemical *structure* only — the channel our
own PubChem spec ranks lowest.

```bash
cd ~/tahoe && cat > channelgate.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=channelgate
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/channelgate_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
$PY channel_gate.py --selftest
$PY channel_gate.py --cache_dir /data/BuffaF-Projetcs/florian_c2s/ot_cache \
    --seed 42 --out RESULTS/channel_gate.json
echo done
EOF
sbatch channelgate.sbatch
```

Read with:
```bash
LOG=$(ls -t logs/channelgate_*.out | head -1); grep -E "columns|targets column|COVERAGE|annotated|channel |target |moa |chem |LIVE|closed|>>>" $LOG
```

**Read COVERAGE first.** A channel annotated on a small fraction of the cached drugs cannot be
closed by a null result — it was never tested. Only then read the gap against each channel's
**count-matched null**, which is what separates "the channel works" from "averaging k residuals
denoises".

---

## 3. `kappa.sbatch` — is the 45% interaction learnable?  (CPU, ~30 min)

Q17 established that ~45% of the drug-specific residual is drug × cell-line interaction. It did
**not** establish that anything can learn it. If each cell line bends drug responses in a consistent
direction, that direction is estimable from the line's own cells and a conditional model has a
concrete target; if the interaction is specific to each pairing, the variance is real but there is
nothing to generalise from.

```bash
cd ~/tahoe && cat > kappa.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=kappa
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/kappa_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
cd ~/tahoe
mkdir -p RESULTS logs
C=/data/BuffaF-Projetcs/florian_c2s/ot_cache
$PY variance_decomposition.py --selftest
for cfg in "plate:--generic_scope plate" "cellline:"; do
  name="${cfg%%:*}"; flags="${cfg#*:}"
  echo ""; echo "############ CONFIG: $name ############"
  $PY variance_decomposition.py --cache_dir "$C" --seed 42 --kappa_structure $flags \
      --out "RESULTS/kappa_${name}.json"
done
echo done
EOF
sbatch kappa.sbatch
```

Read with:
```bash
LOG=$(ls -t logs/kappa_*.out | head -1); grep -E "CONFIG|SELFTEST|KAPPA|kappa built|same_cell_line|different_cell_lines|EXCESS|STRUCTURED|IDIOSYNCRATIC|VERDICT|NEGATIVE CONTROL" $LOG
```

**What decides it:** `EXCESS within-line consistency`. Large and positive → the interaction is
structured and the 45% is a claimable target. Near zero → it is idiosyncratic, real but with
nothing to learn it from at this sample size. Both are publishable; they are different chapters.

---

## Validation already done locally

Neither new measurement is trusted on argument alone; both were checked against planted ground truth
in **both** directions before being sent up.

| check | result |
|---|---|
| `channel_gate --selftest`, planted working channel | gap **+0.397** (sd 0.029 over 24 worlds) |
| `channel_gate --selftest`, planted useless channel | gap **−0.016** (sd 0.091) |
| `kappa_structure` on a planted structured world | same-line +0.977 vs cross-line −0.108, excess **+1.085** |
| `kappa_structure` on a planted idiosyncratic world | excess **+0.001** |
| `variance_decomposition --selftest` | still passes, T recovered to within 0.005 |

Two bugs were found and fixed by that validation rather than by review:

1. The count-matched null originally drew from a pool **excluding** the channel's own picks, which
   depletes it of exactly the partners the channel selected and anti-correlates the two arms.
2. The useless-channel gap has a standard deviation of ~0.09 on a single small world, so the
   original selftest — asserting one draw falls within 0.10 — tested luck rather than correctness.
   It now averages over 24 independent worlds. That variance is also why the real run gates on a
   clustered CI and not on a point estimate.
