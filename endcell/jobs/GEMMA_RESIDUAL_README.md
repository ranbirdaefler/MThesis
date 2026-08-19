# GEMMA-2-2B RESIDUAL-TARGET ARM — runbook

Everything below is run from `cd ~/tahoe` on the cluster. Nothing here runs on the login
node except `sbatch`, `squeue`, `sacct`, `cat` and `chmod`.

---

## WHAT THIS ARM ESTABLISHES, AND WHAT IT DOES NOT

**It establishes:** the effect of swapping the training TARGET — standard next-cell-sentence
vs. residual `(treated − control) − generic` — while holding the backbone, the tokenizer,
the trainer, the optimizer schedule, the seed and the sequence budget fixed. One backbone,
two targets. That is the cleanest form of the residual comparison the thesis makes, because
the only thing that moves is the string the model is asked to produce.

**It does not establish:**

- **Not a scale ablation.** Gemma-2-2B (26 blocks, hidden 2304) vs. Pythia-1b (16 blocks,
  hidden 2048) differ in tokenizer, pretraining corpus, architecture and C2S pretraining
  status. Nothing here isolates parameter count.
- **Not a comparison to Pythia.** The Pythia residual arm is used here only as the *authority
  on which residual build to train on* (Stage 1 matches `sha256(residual.jsonl)` against the
  digest recorded inside the Pythia checkpoint's contract). Gemma-vs-Pythia numbers put side
  by side would be confounded on every axis listed above.
- **Not a mechanistic claim.** The activation-geometry result does not gate this run and this
  run does not test it. On Pythia the geometry said the representation carried no drug biology
  and the residual re-encoding still produced a real gain in prompt sensitivity; the
  mechanistic story did not predict the empirical result, so it is not the thing that decides
  whether to run it.

The valid comparison is `gemma2b_sft_endcell_residual` against `gemma2b_sft_endcell`,
evaluated in their own respective target spaces.

---

## PRECONDITIONS (check once, before Stage 1)

The standard arm's model-provenance certificate must already exist. Every residual stage
reuses it for MODEL provenance, and none of them modifies it.

```bash
cd ~/tahoe
ls -l RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json
```

If that file is absent, stop: run the standard preflight first. The residual arm deliberately
does not re-derive model provenance.

**Why the standard certificate is reused without its generation cap:** every residual stage
calls the existing `verify` subcommand *without* `--require-generation-cap`. `verify` still
checks snapshot inventory, authoritative revision bytes, the protobuf tree, pip freeze,
tokenizer length 256000/256002 and sentinels 256000/256001 — everything about the MODEL —
while `1800` and the five canonical data digests stay bound to the standard data, where they
belong. Do not "fix" the standard constants to accommodate this arm; it breaks the live
standard arm's certificate and `tests/test_gemma_phase1a_training_contract.py`.

---

## STAGE MAP

| Stage | Script | Where | Cost |
|---|---|---|---|
| 1 | `gemma2_residual_inventory.sh` | `srun` defq, 2 cpu, 8G | 5–20 min CPU |
| 2 | `gemma2_residual_data_audit.py` | `srun` defq, 2 cpu, 16G | 2–10 min CPU |
| 3 | `gemma2_residual_preflight.sbatch` | `sbatch` defq, 8 cpu, 32G | 30–50 min CPU |
| 4 | `gemma2_residual_smoke.sbatch` | `sbatch` gpuh200, 1×H200 | 1–2 GPU-hours |
| 5 | `gemma2_residual_train.sbatch` | `sbatch` gpuh200, 1×H200 | **several 24h allocations** |
| 6 | eval — **not yet built**, see below | gpuh200 | — |

Stages 1–3 are the CPU preflight. All three exit non-zero on refusal, so Stage 4 and Stage 5
gate on their artifacts and no GPU is allocated against a build that cannot train.

---

## STAGE 1 — resolve WHICH residual build, by digest

Six `residual_targets*` directories exist, built with different `--split_unit`, `--repro_thr`
and `--generic_scope`. All of them train cleanly and all of them produce a plausible number.
This picks the one the thesis actually reports, by comparing `sha256(residual.jsonl)` against
the `train_sha256` recorded in the Pythia residual checkpoint's `training_state.pt` contract.
It also prints the Pythia arm's lr / warmup / weight-decay / seed next to this arm's intended
values, so a hyperparameter divergence cannot ship by accident.

```bash
cd ~/tahoe
srun -p defq -c 2 --mem 8G --time 00:30:00 \
    bash endcell/jobs/gemma2_residual_inventory.sh \
    2>&1 | tee logs/gemma2_residual_inventory.log
```

**Proves it worked:**

```
[PASS] resolved residual target directory: /data/.../residual_targets_<name>
```
plus `RESULTS/gemma2_residual/RESOLVED_TGT.txt` containing that one path, and a
`HYPERPARAMETER DIFF` block reading `identical on every compared field -> matched replication`.

**If it refuses:**

- `exit 2` — not inside a Slurm allocation, or `$PY` missing. You ran it on the login node;
  re-run under `srun`.
- `exit 3` — `checkpoints/gemma2b_sft_endcell_residual` already exists, or < 150 GiB free.
  Do **not** delete it. Set `GEMMA_RESIDUAL_OUT=<new absolute path>` and re-run. Note the
  contention: the standard arm still needs allocations and wants the same 150 GiB on the same
  filesystem.
- `exit 4` — zero or more than one candidate matched. **Stop and decide.** Zero matched means
  the Gemma arm would not be mirroring the arm the thesis reports; that is a rebuild-or-rescope
  decision, not something to override. If you rebuild, note that `fit_digest` reporting SAME is
  **not** evidence of equivalence — it is computed before `--repro_thr` resolves. Only the
  `residual.jsonl` sha256 is evidence.
- `exit 5` — no Pythia residual checkpoint readable, so there is no authority at all.

**Cost:** 5–20 minutes on 2 CPUs. It sha256s up to twelve JSONLs and `torch.load`s the Pythia
optimizer states, which is why it must not run on the login node.

---

## STAGE 2 — row-schema audit (tokenizer-free)

Ten times cheaper than the tokenizer stage and catches the coarse failures first: a stray
`de_genes` key, a leading/trailing space on `response` (the trainer encodes `' ' + response`,
so a leading space double-spaces every target and shifts the whole token stream), a missing
trailing `[END_CELL]`, val rows sharing `sample_id`s with train, and an ambiguous prompt order.
`report.json` records **neither** `--prompt_order` **nor** `--val_frac`, so both are derived
from the rows themselves rather than asserted.

```bash
cd ~/tahoe
srun -p defq -c 2 --mem 16G --time 01:00:00 \
    /data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python \
    endcell/jobs/gemma2_residual_data_audit.py \
    --target_dir "$(cat RESULTS/gemma2_residual/RESOLVED_TGT.txt)" \
    2>&1 | tee logs/gemma2_residual_data_audit.log
```

**Proves it worked:** `exit 0` and `RESULTS/gemma2_residual/row_audit.json`. Read three fields
before moving on: the derived `prompt_order` (a single recognized value across train **and**
val), the `[DOWN]` emission rate (should be exactly `1.0` — `residual_to_sentence` always emits
the literal; anything else is a real signal about the build, and FINDINGS.md already flags
`[DOWN]` emission rate as an open Pythia-side defect), and `rows_with_de_genes == 0`.

**If it refuses:**

- `exit 2` — usage or missing input; re-run Stage 1.
- `exit 3` — a schema violation, which is the audit's reason for existing. The message names
  the file, the row number and the kind, capped at 20 examples so a systematically broken build
  prints a diagnosis instead of 150,000 lines. A schema violation is a **build** problem: fix
  it by rebuilding to a NEW `--out_dir`, never by editing the resolved directory in place
  (that changes `residual.jsonl` and invalidates the Stage 1 digest match).

**Cost:** 2–10 minutes on 2 CPUs. No torch, no transformers, no protobuf.

---

## STAGE 3 — tokenizer preflight and the MEASURED generation cap

This is the stage that exists because of the 1800 problem. The `1800` cap was derived from a
measured maximum truth response of **1699 tokens under the Gemma tokenizer FOR THE STANDARD
TARGET** (~946–1500 gene symbols). The residual response is a different string: ~200 gene
symbols split by `[DOWN]`. Reusing 1800 would almost certainly be non-binding and therefore
harmless-looking, which is exactly why it would survive review — but the arm would have shipped
a cap nobody measured. This measures it.

It is also the **only UNK detector in the pipeline**. The residual prompt embeds an `ot_cache`
panel control sentence — a gene universe never passed through the Gemma tokenizer in this
project. The trainer does not check for unknown tokens; `--strict_token_contract` only checks
sentinels and truncation. The base Gemma tokenizer's UNK path is demonstrably live: before
sentinel registration it maps `[END_CELL]` to id 3.

```bash
cd ~/tahoe
sbatch endcell/jobs/gemma2_residual_preflight.sbatch
# then, when it finishes:
tail -40 logs/gemma2_residual_preflight_<jobid>.out
```

**Proves it worked:** `RESULTS/gemma2_residual/RESIDUAL_DATA_PASSED.json`, and three numbers
printed at the end. Record all three:

- `max_response_tokens` — replaces the standard target's 1699.
- `residual_generation_cap` — replaces 1800. **This is the number Stage 6 must use.** It is
  the next multiple of 50 *strictly* greater than the measured maximum; strict, because the
  response already contains `[END_CELL]` and a cap equal to the maximum leaves no room for the
  terminator.
- `max_total_sequence_tokens` — must be `< 8192`. If it is not, `--max_length` has to change,
  which changes the contract and hence the run.

**If it refuses:**

- Submitted to the wrong partition — it hard-refuses anything but `defq`, so a GPU allocation
  is never spent on a CPU job.
- Missing `RESOLVED_TGT.txt` or `row_audit.json` — run Stages 1 and 2.
- Model-provenance failure from `verify` — that is the standard arm's certificate, not the
  residual data. Re-run the standard preflight; do not edit constants.
- A non-zero UNK count, a truncation, or a missing sentinel — the probe raises. This is the
  refusal you want, and it costs 40 CPU-minutes instead of an H200 allocation. Under
  `--strict_token_contract` the same failure would otherwise surface inside
  `C2SDataset.__getitem__` at the offending **row**, hours into training.

A failed run never leaves a stale authorization: an existing certificate is moved aside to
`RESIDUAL_DATA_PASSED.previous.<UTC>.json` before the new one is attempted.

**Cost:** 30–50 minutes on 8 CPUs, 32G. It tokenizes **every row** of both residual files
(`--max_examples 0`) under the run's own 256002-length tokenizer.

### Freeze the data now

Once Stage 3 passes, the two JSONLs are load-bearing bytes. `build_residual_targets.py` does
**not** refuse an existing `out_dir` — it `makedirs(exist_ok=True)` and `os.replace`s over both
files — and the trainer hashes `train_sha256` / `eval_sha256` into every checkpoint's resume
contract. Re-running the builder "just to regenerate report.json" strands every published
residual checkpoint permanently.

```bash
TGT="$(cat RESULTS/gemma2_residual/RESOLVED_TGT.txt)"
chmod a-w "$TGT/residual.jsonl" "$TGT/residual_val.jsonl"
```

---

## STAGE 4 — smoke: 150 real optimizer steps on an H200

Buys five things that no amount of code reading buys: that the **unforked** trainer accepts the
residual directory; that the loss is finite and decreasing; that zero sequences truncate inside
the real `C2SDataset` on the real device; that a resumable checkpoint publishes and reloads
within the 600 s signal margin; and a **measured seconds-per-optimizer-step**, which replaces
the ~20 GPU-hour inference with a number before any 24-hour allocation is committed.

```bash
cd ~/tahoe
sbatch endcell/jobs/gemma2_residual_smoke.sbatch
tail -60 logs/gemma2_residual_smoke_<jobid>.out
```

**Proves it worked:** `RESULTS/gemma2_residual_smoke/SMOKE_PASSED.json`, and these lines:

```
[PASS] loss finite and decreasing; zero truncated response tokens; zero sentinel losses
[PASS] atomic residual smoke authorization: .../SMOKE_PASSED.json
[PASS] residual smoke complete.
```

Then read the `MEASURED COST` block, not just the verdict:
`measured_seconds_per_step`, `residual_total_steps`, `projected_total_gpu_hours`,
`projected_allocations at --time=23:50:00`. **Decide whether to commit Stage 5 on that
projection.**

The loss gate is deliberately weak and says so: step 50 is early in warmup at
`warmup_ratio 0.03`, so a strict monotone requirement would fail healthy runs and burn the
allocation. The gate is (hard) every loss finite and positive, and (hard, weak)
`min(last two window losses) < first window loss`. The full trajectory prints either way —
read it.

**If it refuses:**

- `exit 2` — wrong partition, non-H200 GPU, missing/rejected residual certificate, or the
  throwaway output path already exists. The certificate rejection means one of the two JSONLs
  changed since Stage 3: something wrote to the target dir. Investigate before anything else.
- `exit 3` — trainer interface missing a flag, or the 64-row eval shard is wrong.
- `exit 4` — checkpoint publish exceeded `MAX_CHECKPOINT_SECONDS` (420 s). That is a real
  finding: the production job runs on a 600 s signal margin, so a slow publish means segmented
  training will lose a segment. Do not raise the constant to make it pass.
- `exit 5` — no evidence of restoring post-step-50 state, or the resumed segment never reached
  step 100. Resume is broken; Stage 5 is not safe.
- `exit 12` — the loss/truncation gate failed. The failures are listed by name on stderr.

The throwaway checkpoint tree `gemma2b_residual_smoke_<jobid>` is removed on success only,
behind four independent guards. The production tree is never created or opened here, and the
live standard tree is never touched. Set `RESIDUAL_SMOKE_CLEANUP=0` to keep the throwaway tree
for inspection.

**Note one stale line:** the smoke's closing message says "Stage 5 is NOT authored by this
change." It is — `gemma2_residual_train.sbatch` exists. Ignore that line.

**Cost:** 1–2 GPU-hours on one H200, inside a 3-hour wall request. Two segments of 50 steps
each, plus two full model-load + 150k-row dataset constructions.

---

## STAGE 5 — the production run, segmented and resumable

This is `gemma2_standard_train.sbatch` with four retargetings and nothing else: the residual
`--train_file` / `--eval_file`, a NEW `--output_dir`, `--max_length` resolved from the
certificate, and the CPU gates running to completion before any GPU work, any lock, any output
tree, or any trainer launch. **The shared trainer is not forked and not edited** — the standard
arm is live at ~41,301 of 42,198 steps and `_validate_resume_fingerprints` compares
`trainer_sha256` out of every checkpoint's contract, so an edit there permanently strands its
remaining ~900 steps.

The CLI is byte-identical to the standard arm apart from those four flags, `--seed 42` very much
included: `DeterministicEpochSampler` rebuilds the order from `torch.randperm(seed + epoch)`,
so a changed seed silently reorders the remaining data on resume.

**Allocation #1 — cold start:**

```bash
cd ~/tahoe
sbatch endcell/jobs/gemma2_residual_train.sbatch
```

**Allocations #2..N — after each rc-99 exit, resubmitted identically until it exits 0:**

```bash
cd ~/tahoe
sbatch endcell/jobs/gemma2_residual_train.sbatch --resume
```

`--resume` is *permission to continue*, not a hint. Without it the job refuses when the tree
holds a published checkpoint; with it the job refuses when the tree holds none. There is no
branch in which the flag is ignored and no branch in which existing state is overwritten.
`RESIDUAL_RESUME=1` in the environment is equivalent.

**Proves it worked:**

- Per allocation: `exit 99` and
  `[signal] resumable checkpoint published; continue with: sbatch ... --resume`.
- Before the trainer ever launches, on every allocation:
  `[PASS] residual data authorization`, `[PASS] row-count gate: train=… val=… -> N optimizer
  steps`, `[PASS] production disk gate`, `[PASS] CPU preflight complete; proceeding to GPU
  gates`.
- Final allocation: `exit 0` and
  ```
  [PASS] final manifest/content and terminal state: completed=true, step=<N>, epoch=1, microbatch=0, accumulation=0
  [PASS] one-epoch RESIDUAL-target SFT completed -> .../gemma2b_sft_endcell_residual/final
  ```

**If it refuses:**

- `exit 2` — gate or provenance refusal: output path collides with or sits inside the live
  standard arm, `GEMMA_RESIDUAL_OUT` not absolute, missing `RESOLVED_TGT.txt`, residual
  certificate rejected, train file is the standard training file, or the row-count gate failed.
  A rejected certificate means a JSONL changed since Stage 3 — see the freeze rule.
- `exit 3` — lock unavailable (`flock`) or the trainer's `--help` is missing an expected flag.
  A held lock means another allocation of this run is alive; check `squeue` rather than forcing.
- `exit 4` — output-tree refusal: clobber without `--resume`, `--resume` with nothing to
  resume, or an unpublished tmp checkpoint. **Never clear the tree.** If a previous allocation
  died before its first publish, the tree holds at most `run_provenance.json`, which carries no
  progress, and submitting *without* `--resume` is the supported cold-start recovery. Anything
  else in there is an audit, not a retry: inspect it, then archive it or start over with
  `GEMMA_RESIDUAL_OUT=<new absolute path>` set on **both** Stage 1 and Stage 5.
- `exit 5` — trainer post-condition failure: rc other than 99 after the checkpoint signal, or
  success without a complete `final/` artifact. Do not resubmit blindly; read the trainer log.

**Do not change any of these mid-run:** `residual.jsonl`, `residual_val.jsonl`,
`train_c2s_tahoe_endcell.py`, `--max_length`, `--seed`, `--batch_size`, `--grad_accum`. All are
hashed into every checkpoint's contract and `_validate_resume_fingerprints` refuses any
mismatch. There is no recovery short of restarting from step 0. Allocation #1 pins the resolved
parameters into `$OUT/.residual_run_pin.json` and every later allocation re-checks them.

If you deliberately want a different sequence budget, `RESIDUAL_MAX_LENGTH=<int>` overrides it —
but you must re-run Stage 4 with `SMOKE_MAX_LENGTH=<int>` set to the same value, or the smoke
gate refuses, correctly: `max_length` is in the trainer's contract dict, so a production run at
a budget the smoke never exercised is untested. Exporting `RESIDUAL_MAX_LENGTH` once before
submitting both jobs moves both together.

---

## STAGE 6 — evaluation (NOT BUILT)

There is no Gemma residual eval job script yet. `endcell/analysis/residual_eval.py` is the
residual-space evaluator (signed-rank NIR with a scramble arm; `model − scramble` is the
decisive drug-use number), but it is currently driven only for the Pythia arm. When you write
the Gemma Stage 6, two parameters must come from this pipeline and not from defaults:

- `--max_new_tokens <residual_generation_cap>` — from Stage 3. **The default is 1400 and the
  standard arm's 1800 does not apply.**
- `--prompt_order <prompt_order_in_data>` — from Stage 2's `row_audit.json`, copied into the
  certificate. The `residual_eval.py` default is `drug_first`; do not assume it, read it.

Stage 5's closing lines print both values, so capture the tail of the final allocation's log.

Read the leakage note at the top of `residual_eval.py` before quoting any absolute number: the
residual targets were built from the same cache, so sampled conditions may have been trained on.
`model − scramble` remains a valid test of whether the model *uses* the drug token;
a high absolute NIR may be memorisation.

---

## THE HONEST COST

The standard Gemma SFT needed **several 24-hour allocations to reach step 41,301 of 42,198 —
97.9% of one epoch — and it is still not finished.** Assume the residual arm is comparable
unless Stage 4's preflight-measured numbers say otherwise. The one thing that could make it
materially cheaper is shorter sequences: the residual response is ~200 gene symbols against a
one-sentence control prompt, versus ~946–1500 gene symbols for the standard target. If
`max_total_sequence_tokens` from Stage 3 and `measured_seconds_per_step` from Stage 4 come in
well below the standard arm's, the projection in `cost_projection.json` is the number to plan
against — it is measured, not inferred. If they do not, budget several full allocations.

Note the contention explicitly: the standard arm still needs at least one more allocation for
its last ~900 steps, and both checkpoint trees want ≥ 150 GiB on the same filesystem. Stage 1
and Stage 5 both refuse below that threshold rather than filling the disk out from under the
live run.
