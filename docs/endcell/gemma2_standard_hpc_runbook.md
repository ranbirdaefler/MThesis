# Gemma-2 standard-target SFT: audited HPC runbook

This launches the first Gemma arm only: full SFT of the pinned C2S-Gemma checkpoint on the unchanged
standard `[END_CELL]` cell-sentence target. It never rebuilds or edits Tahoe data. Scheduler inspection
is allowed on the login node; downloads, hashing, tokenization, testing, training and evaluation run only
through `srun` or `sbatch`.

## Frozen protocol

- Model: `vandijklab/C2S-Scale-Gemma-2-2B`
- Revision: `5ddf28b8f1c81b7ab7a9be192924da82b6c5d512`
- Train: `/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big/train.jsonl`
- Development loss: adjacent `eval_tier1_seen_conditions.jsonl`
- Primary test: canonical same-plate Tier 2 on exactly 606 rows, 35 drugs, 40 cell lines and 80 groups
- Recipe: one epoch; batch 1; accumulation 16; BF16; length 8192; AdamW LR `1e-5`, weight decay
  `0.01`, warm-up `0.03`, cosine schedule; seed 42; 42,198 optimizer updates
- Output: `/data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell`

This is example- and fixed-recipe-matched to Pythia. It is not token-, compute-, architecture- or
pretraining-matched. Cross-tokenizer losses are not compared as biological performance.

## 0. Put the reviewed implementation on the cluster

### Preferred after review: commit, push and pull the exact reviewed branch

Do this only after the complete repair series is reviewed and committed. On local PowerShell:

```powershell
Set-Location C:\Users\avsd8\OneDrive\Desktop\tahoe
git status --short
git add endcell/train/train_c2s_tahoe_endcell.py endcell/train/gemma_tokenizer_probe.py
git add endcell/analysis/freeze_nir_manifest.py endcell/analysis/nir_benchmark.py
git add endcell/analysis/compare_backbones.py endcell/analysis/residual_eval.py
git add endcell/eval/evaluate_endcell.py tests/test_gemma_phase1a_training_contract.py
git add tests/test_gemma_eval_contract.py endcell/jobs/gemma2_standard_checkpoint_fingerprint.py
git add endcell/jobs/gemma2_standard_provenance.py
git add endcell/jobs/gemma2_standard_cli_contract.py endcell/jobs/gemma2_standard_preflight_contract.py
git add endcell/jobs/gemma2_standard_tests.sh endcell/jobs/gemma2_standard_tests_contract.py
git add endcell/jobs/gemma2_standard_protobuf_env.sh
git add endcell/jobs/gemma2_standard_preflight.sh endcell/jobs/gemma2_standard_smoke.sbatch
git add endcell/jobs/gemma2_standard_train.sbatch endcell/jobs/gemma2_standard_eval.sbatch
git add docs/endcell/gemma2_standard_hpc_runbook.md
git diff --cached --check
git commit -m "Add audited Gemma-2 standard-target SFT pipeline"
git push origin codex/gemma2-standard-sft
```

On the cluster, use scheduler-only Git/file management on the login node; do not run Python there:

```bash
cd ~/tahoe
git fetch origin codex/gemma2-standard-sft
if git show-ref --verify --quiet refs/heads/codex/gemma2-standard-sft; then
  git switch codex/gemma2-standard-sft
else
  git switch --track -c codex/gemma2-standard-sft origin/codex/gemma2-standard-sft
fi
git pull --ff-only origin codex/gemma2-standard-sft
```

### Uncommitted fallback: copy every changed implementation and test file

Run each command separately in PowerShell. This fallback is complete; do not copy only the jobs.

```powershell
Set-Location C:\Users\avsd8\OneDrive\Desktop\tahoe
scp .\endcell\train\train_c2s_tahoe_endcell.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/train/
scp .\endcell\train\gemma_tokenizer_probe.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/train/
scp .\endcell\analysis\freeze_nir_manifest.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/analysis/
scp .\endcell\analysis\nir_benchmark.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/analysis/
scp .\endcell\analysis\compare_backbones.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/analysis/
scp .\endcell\analysis\residual_eval.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/analysis/
scp .\endcell\eval\evaluate_endcell.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/eval/
scp .\tests\test_gemma_phase1a_training_contract.py 3180408@login.hpc.unibocconi.it:~/tahoe/tests/
scp .\tests\test_gemma_eval_contract.py 3180408@login.hpc.unibocconi.it:~/tahoe/tests/
scp .\endcell\jobs\gemma2_standard_checkpoint_fingerprint.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_provenance.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_cli_contract.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_preflight_contract.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_tests_contract.py 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_tests.sh 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_protobuf_env.sh 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_preflight.sh 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_smoke.sbatch 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_train.sbatch 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\endcell\jobs\gemma2_standard_eval.sbatch 3180408@login.hpc.unibocconi.it:~/tahoe/endcell/jobs/
scp .\docs\endcell\gemma2_standard_hpc_runbook.md 3180408@login.hpc.unibocconi.it:~/tahoe/docs/endcell/
```

Record the exact local source hashes in the transfer evidence:

```powershell
$files = Get-ChildItem .\endcell\jobs\gemma2_standard_* -File
$files += Get-Item .\endcell\train\train_c2s_tahoe_endcell.py, .\endcell\train\gemma_tokenizer_probe.py
$files += Get-Item .\endcell\analysis\freeze_nir_manifest.py, .\endcell\analysis\nir_benchmark.py
$files += Get-Item .\endcell\analysis\compare_backbones.py, .\endcell\analysis\residual_eval.py
$files += Get-Item .\endcell\eval\evaluate_endcell.py
$files += Get-Item .\tests\test_gemma_phase1a_training_contract.py, .\tests\test_gemma_eval_contract.py
$files += Get-Item .\docs\endcell\gemma2_standard_hpc_runbook.md
$files | Get-FileHash -Algorithm SHA256 | Format-Table Path, Hash -AutoSize
```

The CPU preflight re-hashes the executed cluster copies and binds them into `PREFLIGHT_PASSED.json`, so
an uncommitted Git SHA is never presented as sufficient provenance. A source-bundle installation need
not contain `.git`: preflight and production record Git metadata when available and otherwise explicitly
record that provenance comes from the certificate-bound SHA-256 inventory.

## 1. Run the certificate-producing test gate on a CPU worker

The canonical `c2s` environment is intentionally not modified. Gemma's tokenizer is SentencePiece and
requires Python Protobuf for Transformers to convert `tokenizer.model`. Without it, Transformers 5.12.1
was observed to silently construct a five-token tokenizer (four prompt tokens and two response tokens)
from a valid 256,000-piece model. Do **not** install TikToken as a workaround and do not install anything
inside `envs/c2s`. Install the pinned runtime dependency into its isolated additive directory on a CPU
worker:

```bash
srun --account=3180408 --partition=defq --cpus-per-task=2 --mem=4G --time=00:15:00 bash -lc '
set -euo pipefail
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
TARGET=/data/BuffaF-Projetcs/florian_c2s/test_deps/protobuf-5.29.5
unset PYTHONHOME
mkdir -p "$TARGET"
if [[ ! -f "$TARGET/google/protobuf/__init__.py" ]]; then
  "$PY" -m pip install --no-cache-dir --target "$TARGET" "protobuf==5.29.5"
fi
PYTHONPATH="$TARGET" "$PY" -c "import google.protobuf; assert google.protobuf.__version__ == \"5.29.5\"; print(google.protobuf.__file__)"
'
```

Every preflight, smoke, training and evaluation job sources
`gemma2_standard_protobuf_env.sh`. That helper replaces `PYTHONPATH` with exactly this directory (it
does not preserve inherited entries), unsets `PYTHONHOME`, then fails unless version 5.29.5 imports from
beneath the exact isolated path. Its digest inventory is built from every path in
`protobuf-5.29.5.dist-info/RECORD`, safely confined to the isolated root; this explicitly includes the
executed `google/_upb/_message.abi3.so` payload as well as Python and metadata files. Preflight records
that digest and every smoke, training and evaluation verification recomputes it. The trainer detects Gemma from
`AutoConfig.model_type`, requires the two Gemma flags and validates the same helper-bound payload before
`AutoTokenizer`. In addition, canonical Gemma training requires `--model_load_path` to equal the exact
cache path implied by the logical model ID and revision, so direct invocation cannot fall through to a
collapsed model-ID tokenizer.

Next install the pinned test runner into a separate isolated directory on a CPU worker:

```bash
srun --account=3180408 --partition=defq --cpus-per-task=2 --mem=4G --time=00:15:00 bash -lc '
set -euo pipefail
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
ROOT=/data/BuffaF-Projetcs/florian_c2s/test_deps
TARGET="$ROOT/pytest-8.3.5"
unset PYTHONHOME
mkdir -p "$ROOT"
if [[ ! -f "$TARGET/pytest/__init__.py" ]]; then
  TMP="$ROOT/.pytest-8.3.5.tmp.$SLURM_JOB_ID"
  rm -rf -- "$TMP"
  "$PY" -m pip install --no-cache-dir --target "$TMP" "pytest==8.3.5"
  mv -- "$TMP" "$TARGET"
fi
PYTHONPATH="$TARGET" "$PY" -c "import pytest; assert pytest.__version__ == \"8.3.5\"; print(\"[PASS] isolated pytest\", pytest.__version__)"
'
```

The temporary removal is narrowly confined to this job's newly-created test-dependency directory. The
test gate exposes that directory only to the `pytest` process, then creates its environment certificate
from the unchanged canonical `c2s` environment.

After that passes, submit this `srun`:

```bash
srun --account=3180408 --partition=defq --cpus-per-task=4 --mem=24G --time=01:00:00 bash -lc '
set -euo pipefail
cd ~/tahoe
bash endcell/jobs/gemma2_standard_tests.sh
'
```

Do not continue unless every test and self-test passes and the final line confirms the atomic artifact:

```text
~/tahoe/RESULTS/gemma2_standard_tests/TESTS_PASSED.json
```

That certificate binds the exact Section-1 command, the exact closed set of 20 intended source keys
(including `protobuf_env` and the shared provenance module), the Python executable and version, and the
normalized package environment.
Missing or extra source entries are rejected. Preflight re-verifies all hashes and refuses to run if the
command, source or environment changed after testing.

## 2. Verify `gpuh200`, download Gemma and create the atomic preflight certificate

Scheduler commands only on the login node:

```bash
cd ~/tahoe
GPU_PARTITION=gpuh200
bash endcell/jobs/gemma2_standard_preflight.sh "$GPU_PARTITION"
```

The script first verifies that the regular, sub-24-hour partition advertises H200 resources. It then opens
a CPU `srun` on `defq` and performs the only Gemma network download in the protocol:

```text
snapshot_download(
  repo_id="vandijklab/C2S-Scale-Gemma-2-2B",
  revision="5ddf28b8f1c81b7ab7a9be192924da82b6c5d512",
  cache_dir="/data/BuffaF-Projetcs/florian_c2s/hf_cache/hub"
)
```

It exports `HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache` and
`HF_HUB_CACHE=$HF_HOME/hub` consistently for download, smoke, training and evaluation. It verifies the
Hub-resolved commit plus `config.json`, tokenizer files, model index and exactly the two expected model
shards. The certificate binds the exact `snapshots/<revision>` directory, its complete relative file
inventory, every symlink target and every resolved file-content hash. Independently of that generated
inventory, preflight verifies eight load-bearing files against immutable upstream pins: Hugging Face LFS
SHA-256 values for `tokenizer.model` and both model shards, and SHA-256 values independently computed from
the official raw immutable-revision URLs for the five small configuration/index files. It also requires
the exact eleven-file top-level tree published at that revision; additional Transformers-recognized files
such as `model.safetensors`, `tokenizer.json`, or adapter sidecars are rejected rather than allowed to
override the pinned sharded model or SentencePiece tokenizer. A changed cache
therefore cannot be blessed merely by regenerating a certificate. After the download it switches to
`HF_HUB_OFFLINE=1`. The logical model ID and revision remain the scientific provenance, but every
Transformers cold-parent `AutoConfig`, fast/slow `AutoTokenizer`, and model load uses the exact local
snapshot path with `local_files_only=True` and no revision argument; resume loads only the immutable
checkpoint after proving that checkpoint's exact parent ancestry. This distinction is load-bearing: on this
cluster, Transformers 5.12.1 was observed to produce `vocab_size=5` and `vocab_file=None` from the cached
model-ID route, while the same pinned bytes loaded by exact snapshot produce `vocab_size=256000` and the
correct `tokenizer.model`. Preflight then runs the complete five-JSONL tokenizer audit under
`TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` and binds the report's load source to that certified
snapshot. Every config, tokenizer and model load from the parent is bracketed by complete snapshot-digest
checks. This detects persistent shared-cache mutation during a load. The explicit trust boundary is a
non-adversarial shared cache: the protocol does not claim to defeat a same-account process that can alter
and restore bytes between checks, and it does not copy approximately 5.2 GB into each job's scratch space.

The CPU tokenizer gate is fail-closed: the pinned model configuration and raw base tokenizer must expose
256,000 entries before sentinel registration; `[END_CELL]` and `[DOWN]` must then occupy IDs 256000 and
256001 and produce a tokenizer length of 256002. All five uniquely identified, hash-bound JSONLs must
have noncollapsed prompt/response lengths and zero unknown tokens, and 512 rows must agree under the
slow and fast tokenizers. Both reports must identify the certified snapshot's `tokenizer.model` as their
opened vocabulary file; a missing or different `vocab_file` fails authorization. CPU preflight
deliberately does not load the multi-gigabyte model. The real
256,000-row parent embedding table is first proved by the mandatory H200 smoke/trainer immediately
before any resize; resume proves that model and tokenizer are already 256002.

The only launch authorization is the schema 8 certificate:

```text
~/tahoe/RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json
```

It is published atomically only after every check succeeds. Schema 8 invalidates every earlier
certificate: schema 7 independently pinned the official model files but did not reject overriding
snapshot filenames or compare the five Tahoe inputs against immutable canonical hashes. The schema 8
creator and verifier both require the pre-registered train/Tier-1/Tier-2/Tier-3/Tier-4 SHA-256 values, so
regenerating preflight cannot bless altered data. It binds the
Section-1 test certificate, tokenizer report, trainer, evaluators, manifest/comparison code, every Gemma
job, environment, package freeze, the complete Protobuf `RECORD` payload inventory, both focused tests,
this runbook and the complete pinned snapshot.
Smoke, training and evaluation
re-hash those inputs. A failed
preflight therefore cannot authorize a job through an intermediate `data_sha256.txt`.

The 1,800-token generation cap is not an assumption: the certificate authorizes it only when the full
audit proves it exceeds the maximum observed target-response token length and no semantic or sentinel
truncation occurs. If that gate fails, stop and revise the cap/protocol before generating anything.

## 3. Submit the deliberate interruption/resume smoke test

```bash
cd ~/tahoe
sbatch --partition=gpuh200 endcell/jobs/gemma2_standard_smoke.sbatch
squeue -u 3180408
```

The smoke requires an actual NVIDIA H200 with at least 135,000 MiB HBM. It uses the production
42,198-step schedule, interrupts after step 50, verifies a published `checkpoint-N/training_state.pt`,
resumes and interrupts after step 100. It measures signal-to-checkpoint completion twice and fails when
either save exceeds 420 seconds, leaving three minutes of safety inside the ten-minute production
warning. It also records GPU memory, throughput, utilization, temperature and power. Only after all
checks pass does it atomically publish
`RESULTS/gemma2_standard_smoke/SMOKE_PASSED.json`; production verifies that this marker belongs to the
current preflight certificate, trainer and smoke job.
The smoke resolves the exact local snapshot from the verified certificate and passes it separately as
`--model_load_path`; `--model_name` and `--model_revision` remain the logical provenance. Its first segment
cold-starts from that snapshot. Its second segment passes the same parent identity but loads model and
tokenizer exclusively from the published resume checkpoint. The schema-4 smoke certificate binds the
parent snapshot path, complete inventory digest and preflight certificate.

Do not submit training unless the log ends in `[PASS]`.

## 4. Submit the canonical 47.5-hour job

Recheck free space on a CPU worker after the model download and smoke checkpoints. Production requires
at least 150 GiB free on the checkpoint filesystem and repeats this gate inside the batch job:

```bash
srun --account=3180408 --partition=defq --cpus-per-task=1 --mem=1G --time=00:05:00 bash -lc '
set -euo pipefail
TARGET=/data/BuffaF-Projetcs/florian_c2s/checkpoints
AVAILABLE=$(df -B1 --output=avail "$TARGET" | tail -1 | tr -d " ")
REQUIRED=$((150 * 1024 * 1024 * 1024))
test "$AVAILABLE" -ge "$REQUIRED"
echo "[PASS] production disk gate: available_bytes=$AVAILABLE >= $REQUIRED (150 GiB)"
'
```

Only after that command passes:

```bash
cd ~/tahoe
GPU_PARTITION=gpuh200
sbatch --partition="$GPU_PARTITION" \
  endcell/jobs/gemma2_standard_train.sbatch
squeue -u 3180408
```

At runtime the job compares `SLURM_JOB_PARTITION` directly with the certified literal `gpuh200`;
there is no user-overridable expected-partition variable. It then verifies the certificate, exact data,
actual H200 name and required HBM
plus the exact passing interruption/resume smoke before loading weights. It acquires `flock` on the
canonical output directory and stores the Slurm job ID
in the lock, so concurrent production writers are rejected.

Only published `checkpoint-N/training_state.pt` files count as resume state. Temporary checkpoints,
partial published directories and a nonfresh output without a valid checkpoint fail closed.
The production job independently resolves the same exact local snapshot from the verified preflight
certificate. A cold start loads that path; a resume loads only the selected checkpoint while retaining
the logical parent ID/revision, certificate SHA-256, certified snapshot inventory digest and official
revision-manifest digest in the
training contract, training state and provenance. The trainer recomputes that complete inventory
before and after configuration, tokenizer and model loading; this closes persistent substitutions under
the non-adversarial shared-cache boundary stated above.
Ten minutes
before timeout Slurm sends one `USR1`; the trainer publishes an optimizer-boundary state and exits 99.
Resubmit the identical command to continue. A successful final artifact must contain
`final/checkpoint_manifest.json` whose complete recursive inventory and SHA-256 hashes validate, plus
`final/training_state.pt` with `completed=true`, `global_step=42198`, `epoch=1`,
`microbatch_position=0` and `accumulation_position=0`. Whether the final checkpoint already exists or is
created by the current job, production also verifies provenance schema 4 through the same checkpoint
validator used by evaluation. Its model block and training contract must bind the logical model
ID/revision, certificate hash, exact parent path, complete snapshot inventory digest and official
revision-manifest digest.

## 5. Freeze Tier-2 support and all four checkpoint declarations

Do this only after the Gemma SFT final checkpoint exists and before any Tier-2 generation. It proves
Tier-2 drug disjointness against `train.jsonl`, requires the exact canonical support, and publishes a
hash-named manifest that cannot overwrite an earlier one.

```bash
srun --account=3180408 --partition=defq --cpus-per-task=2 --mem=16G --time=02:00:00 bash -lc '
set -euo pipefail
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd ~/tahoe
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
source endcell/jobs/gemma2_standard_protobuf_env.sh
FP=endcell/jobs/gemma2_standard_checkpoint_fingerprint.py
CERT=RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json
CONTRACT=endcell/jobs/gemma2_standard_preflight_contract.py
RUNTIME=RESULTS/gemma2_standard_preflight/PREFLIGHT_RUNTIME_CONTRACT.json
$PY $CONTRACT verify --certificate $CERT \
  --model-id vandijklab/C2S-Scale-Gemma-2-2B \
  --revision 5ddf28b8f1c81b7ab7a9be192924da82b6c5d512 --runtime-contract-out "$RUNTIME"
mapfile -t GEMMA_BINDING < <($PY - "$RUNTIME" <<"PY"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("snapshot_path", "preflight_certificate_sha256", "snapshot_inventory_sha256",
            "authoritative_revision_files_sha256"):
    print(d[key])
PY
)
test "${#GEMMA_BINDING[@]}" -eq 4
GEMMA_PARENT="${GEMMA_BINDING[0]}"
PREFLIGHT_SHA256="${GEMMA_BINDING[1]}"
SNAPSHOT_SHA256="${GEMMA_BINDING[2]}"
AUTHORITATIVE_SHA256="${GEMMA_BINDING[3]}"
PYTHIA_PARENT=$($PY - <<"PY"
import os
from huggingface_hub import snapshot_download
print(snapshot_download("vandijklab/C2S-Scale-Pythia-1b-pt",
    revision="830e4689d4238bbb5e2ec9a89b76e6a6d48061db",
    cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True))
PY
)
GEMMA_FP=$($PY $FP \
  --checkpoint /data/BuffaF-Projetcs/florian_c2s/checkpoints/gemma2b_sft_endcell/final \
  --require_complete_sft --expected_global_step 42198 --expected_epoch 1 \
  --expected_microbatch_position 0 --expected_accumulation_position 0 \
  --require_gemma_ancestry --expected_model_id vandijklab/C2S-Scale-Gemma-2-2B \
  --expected_revision 5ddf28b8f1c81b7ab7a9be192924da82b6c5d512 \
  --expected_parent_snapshot "$GEMMA_PARENT" \
  --expected_preflight_certificate_sha256 "$PREFLIGHT_SHA256" \
  --expected_snapshot_inventory_sha256 "$SNAPSHOT_SHA256" \
  --expected_authoritative_files_sha256 "$AUTHORITATIVE_SHA256" --digest_only)
PYTHIA_FP=$($PY $FP --checkpoint /data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final --digest_only)
GEMMA_PARENT_FP=$($PY $FP --checkpoint "$GEMMA_PARENT" --require_gemma_parent \
  --expected_model_id vandijklab/C2S-Scale-Gemma-2-2B \
  --expected_revision 5ddf28b8f1c81b7ab7a9be192924da82b6c5d512 \
  --expected_parent_snapshot "$GEMMA_PARENT" \
  --expected_preflight_certificate_sha256 "$PREFLIGHT_SHA256" \
  --expected_snapshot_inventory_sha256 "$SNAPSHOT_SHA256" \
  --expected_authoritative_files_sha256 "$AUTHORITATIVE_SHA256" --digest_only)
PYTHIA_PARENT_FP=$($PY $FP --checkpoint "$PYTHIA_PARENT" --digest_only)
$PY endcell/analysis/freeze_nir_manifest.py \
  --eval_dir /data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big \
  --scram_dir /data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big_scram \
  --train_file /data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big/train.jsonl \
  --tier tier2_unseen_drugs --k_samples 8 --min_cells 8 \
  --min_drugs_per_group 3 --max_groups 80 --same_plate_only --seed 42 \
  --generation_contract_version nir-generation-v1 \
  --expected_rows 606 --expected_drugs 35 --expected_cell_lines 40 --expected_groups 80 \
  --model_fingerprint "$GEMMA_FP" --model_fingerprint "$PYTHIA_FP" \
  --validity_only_parent "gemma_parent=$GEMMA_PARENT_FP" \
  --validity_only_parent "pythia_parent=$PYTHIA_PARENT_FP" \
  --hash_named --out RESULTS/gemma2_standard_tier2_manifest.json \
  | tee RESULTS/gemma2_standard_manifest_freeze.log
'
```

The final log prints the immutable hash-named manifest path. Use that exact path below; do not create or
overwrite an alias after seeing outputs.

## 6. Run validity and Tier-2 NIR

Set `MANIFEST` to the exact hash-named path printed in step 5:

```bash
cd ~/tahoe
MANIFEST=$(sed -nE 's/^wrote ([^:]+):.*/\1/p' RESULTS/gemma2_standard_manifest_freeze.log | tail -1)
test -f "$MANIFEST"
```

Fine-tuned Gemma:

```bash
sbatch --partition=gpuh200 \
  --export=ALL,CHECKPOINT_ROLE=gemma_sft,TIER2_MANIFEST="$MANIFEST" \
  endcell/jobs/gemma2_standard_eval.sbatch
```

Fine-tuned Pythia, with its explicit legacy training-state exemption:

```bash
sbatch --partition=gpuh200 \
  --export=ALL,CHECKPOINT_ROLE=pythia_sft_legacy,ALLOW_LEGACY_PYTHIA_CHECKPOINT=1,MODEL_PATH=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final,TIER2_MANIFEST="$MANIFEST" \
  endcell/jobs/gemma2_standard_eval.sbatch
```

Resolve and persist the two parent paths on a CPU worker:

```bash
srun --account=3180408 --partition=defq --cpus-per-task=1 --mem=8G --time=00:30:00 bash -lc '
set -euo pipefail
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd ~/tahoe
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
source endcell/jobs/gemma2_standard_protobuf_env.sh
RUNTIME=RESULTS/gemma2_standard_preflight/PREFLIGHT_RUNTIME_CONTRACT.json
$PY endcell/jobs/gemma2_standard_preflight_contract.py verify \
  --certificate RESULTS/gemma2_standard_preflight/PREFLIGHT_PASSED.json \
  --model-id vandijklab/C2S-Scale-Gemma-2-2B \
  --revision 5ddf28b8f1c81b7ab7a9be192924da82b6c5d512 --runtime-contract-out "$RUNTIME"
GEMMA_PARENT=$($PY - "$RUNTIME" <<"PY"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["snapshot_path"])
PY
)
PYTHIA_PARENT=$($PY - <<"PY"
import os
from huggingface_hub import snapshot_download
print(snapshot_download("vandijklab/C2S-Scale-Pythia-1b-pt",
    revision="830e4689d4238bbb5e2ec9a89b76e6a6d48061db",
    cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True))
PY
)
printf "GEMMA_PARENT=%q\nPYTHIA_PARENT=%q\n" "$GEMMA_PARENT" "$PYTHIA_PARENT" \
  > RESULTS/gemma2_standard_parent_paths.env
'
source RESULTS/gemma2_standard_parent_paths.env
```

Frozen Gemma parent:

```bash
sbatch --partition=gpuh200 \
  --export=ALL,CHECKPOINT_ROLE=gemma_parent,MODEL_PATH="$GEMMA_PARENT",TIER2_MANIFEST="$MANIFEST" \
  endcell/jobs/gemma2_standard_eval.sbatch
```

Frozen Pythia parent:

```bash
sbatch --partition=gpuh200 \
  --export=ALL,CHECKPOINT_ROLE=pythia_parent,MODEL_PATH="$PYTHIA_PARENT",TIER2_MANIFEST="$MANIFEST" \
  endcell/jobs/gemma2_standard_eval.sbatch
```

Parent jobs are validity-first and are frozen as validity-only controls. If a parent lacks an atomic
`[END_CELL]` or emits nonsense—as frozen Pythia did in the original exploratory run—the evaluator writes
`status=validity_failure` and exits without NIR. That is the result; it is not repaired or scored.
The evaluation launcher derives each output label from `CHECKPOINT_ROLE` and rejects a conflicting
`MODEL_LABEL`; it also reads `config.json` and enforces `gemma2` for both Gemma roles and `gpt_neox` for
both Pythia roles. This preserves the Pythia comparators without allowing a Gemma checkpoint to bypass
ancestry validation under a Pythia label. Frozen-parent path comparison is filesystem-identity aware, so
the cluster's `/data` and `/mnt/beegfsnew` spellings may identify the same certified bytes while the
certificate retains its canonical lexical `/data` path.

Every inference job is offline, uses the eager attention implementation and restores KV caching through
the repaired evaluator. Production validity and NIR entrypoints are fixed to the canonical repo files;
environment-variable substitution is disabled, and each selected resolved path and SHA-256 must match
the corresponding source in the preflight certificate before execution. The job parses both the greedy
validity artifact and sampled NIR artifact; it does not print success for `validity_failure`.

## 7. Produce the paired Gemma–Pythia comparison

After both fine-tuned NIR artifacts report `status=ok`, run on a CPU worker:

```bash
srun --account=3180408 --partition=defq --cpus-per-task=4 --mem=32G --time=01:00:00 bash -lc '
set -euo pipefail
cd ~/tahoe
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python
$PY endcell/analysis/compare_backbones.py \
  --left RESULTS/gemma2_standard_eval/gemma/gemma_tier2_nir.json \
  --right RESULTS/gemma2_standard_eval/pythia/pythia_tier2_nir.json \
  --left_name gemma --right_name pythia --tier tier2_unseen_drugs \
  --per_drug_bh --out RESULTS/gemma2_standard_eval/gemma_vs_pythia_tier2.json
$PY - <<"PY"
import json
path = "RESULTS/gemma2_standard_eval/gemma_vs_pythia_tier2.json"
document = json.load(open(path, encoding="utf-8"))
required = {
    "gemma_minus_chance", "gemma_minus_control", "gemma_minus_linear",
    "gemma_minus_mean", "gemma_minus_wrong_condition", "gemma_minus_pythia",
}
estimands = document.get("estimands", {})
if document.get("status") != "ok" or set(estimands) != required:
    raise SystemExit(f"[FATAL] incomplete comparison contract: {estimands.keys()}")
for name, estimand in document["estimands"].items():
    expected = {"primary_drug_cell_line", "sensitivity_drug_well", "sensitivity_cell_line_well"}
    if set(estimand) != expected:
        raise SystemExit(f"[FATAL] {name} lacks declared interval set: {estimand.keys()}")
print("[PASS] all six preregistered contrasts and all three dependence analyses are present")
PY
'
```

The comparator refuses status failures, manifest/config/support mismatches and duplicate rows. Negative
multiway variance is not allowed to become a zero-width confidence interval.

## 8. Copy evidence back in PowerShell

```powershell
Set-Location C:\Users\avsd8\OneDrive\Desktop\tahoe
New-Item -ItemType Directory -Force .\gemma2_hpc_return | Out-Null
scp "3180408@login.hpc.unibocconi.it:~/tahoe/logs/gemma2_standard_*.out" .\gemma2_hpc_return\
scp -r "3180408@login.hpc.unibocconi.it:~/tahoe/RESULTS/gemma2_standard_*" .\gemma2_hpc_return\
```

Do not copy multi-gigabyte model weights unless a later diagnosis specifically requires them.

## Stop conditions

- No current `TESTS_PASSED.json`: do not run preflight.
- No `PREFLIGHT_PASSED.json`: do not smoke, train or evaluate.
- Isolated Protobuf 5.29.5 is absent, resolves outside its pinned directory, or any job mentions a
  TikToken fallback: do not load the Gemma tokenizer.
- Gemma base vocabulary/config/embedding rows are not exactly 256,000, post-sentinel length is not
  256,002, sentinel IDs are not 256000/256001, any canonical row contains an unknown token, or minimum
  prompt/response length is below 20 tokens: do not authorize preflight or resize the model.
- Token audit does not authorize 1,800 tokens: do not generate.
- Long partition is not H200-backed or runtime GPU/HBM gate fails: do not train.
- Either checkpoint save exceeds 420 seconds: do not rely on the 600-second warning.
- Resume test fails, output lock is held or output contains unpublished/partial state: do not train.
- Manifest is not exactly 606/35/40/80 or any Tier-2 drug occurs in training: do not evaluate.
- Gemma final manifest inventory/content does not validate, or its terminal state is not exactly
  completed at step 42,198, epoch 1, microbatch 0, accumulation 0: do not accept or evaluate it.
- Parent validity fails: preserve the failure artifact and do not calculate parent NIR.
