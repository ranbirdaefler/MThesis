#!/bin/bash
#SBATCH --job-name=drug_geom_rsa
#SBATCH --account=3180408
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/drug_geometry_rsa_%j.out
# Is the drug region of the representation ORGANISED, or merely LABELLED?
# (representational geometry alignment — internal distances vs real drug-effect distances)
#
# Separability of drug identity is already established and is nearly free: the drug name is in the
# prompt (mechanistic_drug_probe). Output-space geometry against biology is already established and
# is ~zero (drug_stratify_geometry test3). This job fills the missing cell: whether the METRIC
# STRUCTURE of the activations matches the metric structure of real drug effects, BEYOND what the
# prompt string hands the model for free.
#
# Headline = partial Spearman of activation distance on biological distance (MRM + Freedman-Lane
# over DRUG-LABEL permutations), controlling for name-token Jaccard, name char-3gram Jaccard,
# mechanism-token Jaccard, shared-known-mechanism, shared-'unclear'-mechanism, name-length,
# prompt-length and near-duplicate base name. The raw coefficient is reported beside it; the PAIR
# (r_raw, r_partial) is the finding and their GAP is how much of the internal geometry is the string.
#
# Guards that make either answer interpretable (not an unfalsifiable null):
#   * SELFTEST GATE (below): the job aborts before requesting a single forward pass unless the
#     planted worlds pass — in particular the NEGATIVE/NAME-ONLY world, where activations and
#     biology are BOTH generated from the name-token control and nothing else. That world is the
#     only evidence the confound control adjusts anything.
#   * CPU PRECHECK, STAGE [1/5] BELOW: n_drugs, B, rel_B and the gate depend only on the eval
#     jsonl — no model, no forward pass, no GPU. They are now computed FIRST, and a stratum that
#     refuses never reaches a checkpoint. The precheck path never imports torch or transformers; it
#     is run inside this same allocation only for simplicity of bookkeeping, and it needs no GPU and
#     takes seconds. Run `--precheck --precheck_sweep` as a plain CPU job BEFORE submitting this
#     one as well; the sweep prints the (n_drugs, rel_B) frontier over min_cells_per_drug x scope,
#     which is the trade-off this file names below and used to resolve only after the GPU was spent.
#   * rel_B CEILING GATE: the script computes rel_B from repeated cell half-splits and refuses the
#     headline when the bootstrap CI LOWER BOUND <= 0.20 (status no_measurable_biological_geometry),
#     and restricts the reading to the disattenuated value while that lower bound is under 0.40.
#     BOTH thresholds now read the lower bound; previously the 0.40 one read the point estimate.
#     The gate is now an EARLY RETURN: the script writes its JSON and exits before loading any
#     checkpoint, instead of warning and continuing into the full sweep. --force_through_gate
#     overrides it for diagnostics only and stamps every cell it produces.
#   * CONFOUND-RESIDUE FLOOR: per arm, per run, measured on this roster's own control matrices —
#     the 95th percentile of |r_partial| under a zero-pharmacology confounded null. Any cell below
#     its arm's floor is tagged within_confound_residue_envelope and is NOT a headline, whatever its
#     p. p is not a licensing statistic here: measured type-I under a confounded null runs 0.16-0.71.
#   * CELL-LINE POSITIVE CONTROL (--positive_control): the identical code path with the drug and
#     dose held fixed and the CELL LINE NAME varying in the same prompt slot. If it is also null,
#     the drug-side null is VOID and is reported as an instrument failure.
#   * DIRECT DECODABILITY ARM: leave-one-drug-out dual-form ridge from the same activations to the
#     same residual profiles, at the headline layers, both arms, all three positions, both
#     checkpoints, with a names-only ridge as the increment control. RSA is one-sided — a null
#     licenses nothing about organisation — and this is the arm that addresses whether a better
#     decoding head could recover drug effects.
#   * n REFUSAL RULE: n_drugs < 25 -> status untestable_n and no coefficient at all; 25..39 ->
#     computed but tagged exploratory; the headline requires n_drugs >= 40. The n<25 refusal fires
#     before any model load, so a thin stratum costs no GPU time.
#   * BASE-CHECKPOINT CONTRAST: byte-identical prompts through the un-finetuned backbone, with ONE
#     shared drug-relabeling sequence applied to both checkpoints inside each draw (a genuinely
#     PAIRED null). This is why both checkpoints go into a SINGLE invocation via --model_paths:
#     splitting them into two jobs would destroy the pairing and the max-statistic family.
#   * BLIND mechanism arm is PRIMARY: `Mechanism: unclear.` for every drug (in-distribution — about
#     half of atlas rows carry it), so the largest confound is removed by construction rather than
#     by a block-structured regressor whose type-I behaviour is exactly what is unsafe here. The
#     ANNOTATED arm runs beside it as a labelled secondary.
#   * DEGENERACY is RAISED, not scored: (layer 0, last position) is degenerate by construction (the
#     last token is ':' for every prompt) and is recorded as such, never returned as 0.0.
#   * Dose: the PROMPT is dose-fixed (one modal dose string per group) but B is a DOSE MIXTURE —
#     build_biology pools every cell of a drug regardless of dose. The earlier claim here that dose
#     was "held CONSTANT by construction" was true of the prompt and FALSE of B. The mismatch is now
#     ADJUSTED dyadically via x6 on each drug's median log10 dose (refused outright, with the parse
#     rate reported, if fewer than 90% of drugs parse), and each drug's dose composition is reported
#     in biology.dose_frame.per_drug. It is adjusted, NOT eliminated.
#   * Balanced control-cell design: the SAME 16 control sentences for EVERY drug, centered within
#     draw, so the control main effect cancels exactly and cannot manufacture between-drug geometry.
#   * Activations are hooked on BLOCK OUTPUTS (workspace_probe's pattern), never
#     output_hidden_states: GPT-NeoX applies final_layer_norm to hidden_states[-1], so the last
#     layer would live in a different frame — that silently broke workspace_probe v3.0.
#
# THREE PRE-REGISTERED STRATA, one invocation each (the script analyses ONE stratum per run and raw
# distances are never pooled across strata). G1 is the primary; G2 shares its cell line and changes
# plate; G3 changes cell line. plate4 strata are EXCLUDED — all their perm_p are None (no DMSO
# controls were collected there), so any activity criterion there is undefined, not merely
# restrictive.
#
# WHY THE PLATE IS PINNED, corrected. The earlier justification here — "the residual is defined
# against that plate's DMSO pool" — is FALSE and must not be repeated: the DMSO term cancels
# identically in the frame this job uses (see the EXPRESSION-frame note below, and
# PRE_REGISTRATION['plate_scope']['why_x9_exists']). What pinning the plate actually buys is the
# ELIMINATION OF PLATE BATCH BY STRATIFICATION: with one plate per run, x9 (the same-plate dyadic
# indicator) is constant and is dropped as zero-variance, which is recorded per cell in
# controls_dropped_zero_variance. That is a real benefit and it is the only one being claimed.
#
# The cost is n. tier-2 is 50 held-out drugs of 287, so a pinned (cell_line, plate) stratum plausibly
# holds only ~10-12 of them, against a refusal floor of 25 and a headline threshold of 40 — which is
# exactly why stage [1/5] measures n_drugs on CPU before any checkpoint is loaded, and why the
# precheck sweep reports BOTH scopes. --scope is therefore passed EXPLICITLY below rather than left
# to the script's default: cell_line_plate is the pre-registered starting configuration, and moving
# to --scope cell_line (larger n, plates pooled, plate batch ADJUSTED dyadically via x9 instead of
# eliminated) is a configuration_selection decision to be made FROM THE SWEEP under
# PRE_REGISTRATION['configuration_selection'], before any activation exists — never an ad-hoc edit
# made after a coefficient has been seen. The two scopes are different readings and are never pooled
# or swapped after the fact.
#
# THE PRE-REGISTERED READING RULE ACROSS STRATA (PRE_REGISTRATION['multi_stratum_rule'], written
# before submission and stamped into every JSON): G1 is the PRIMARY and is named in --primary_stratum
# before the run. G2 and G3 are REPLICATION, never a second chance at a headline. The claim is
# written only if the primary clears BOTH the n>=40 threshold and the rel_B gate AND every
# gate-clearing stratum agrees in sign; any sign disagreement is reported as heterogeneity and the
# claim is downgraded to \notestablished. Agreement is not a combined coefficient, and it carries the
# caveat that a treated well spans many cell lines, so replication reproduces well-level artefact in
# B as well as biology.
#
# NOT AVAILABLE IN THIS SCRIPT, and therefore NOT claimed anywhere in the reading of this job:
#   * an explicit NC-1 pass/fail. The (layer 0, position 'name') cell IS a pure tokenizer null and
#     should be read as one, but the script does not gate the run on it — check it by hand.
#   * an EXPRESSION frame. B is always the residual signed-rank frame — which is ALGEBRAICALLY
#     `pseudobulk(drug) - roster mean`, then signed_rank k=100, then cosine. The DMSO term cancels
#     identically (verified: replacing the DMSO vector with a different one moves the residual by
#     5.6e-15), so this job carries NO plate-matched DMSO control and must never be described as
#     carrying one. There is no frame-sensitivity arm, so "the verdict is frame-robust" may NOT be
#     written; a full-panel (k=P) reliability companion is reported so the truncation's cost is at
#     least measured.
#   * a per-drug ACTIVITY gate (keep only drugs with a significant DMSO-referenced effect). The only
#     roster filter is the per-drug cell floor below. Activity magnitude is instead REPORTED
#     (biology.activity_magnitude) and carried as a labelled secondary partial
#     (r_partial_activity_adjusted), which over-adjusts by construction and is never the headline.
#   * a WELL-DISJOINT rel_B as the gate quantity. The gate's rel_B is a cell half-split, which is
#     optimistic because drug nests in well; a well-split companion is REPORTED beside it with its
#     drug coverage, but it is not allowed to make the refusal decision, being confounded with dose.
#   * a revision pin on the base backbone. The base is loaded by repo id at whatever revision the HF
#     cache holds; the resolved commit is not recorded in the output JSON.
# All of these are real gaps, not stylistic ones, and are flagged rather than silently dropped.
#
# NOT RUN HERE: the output-side comparator — the matched-support generation pass. Until that runs,
# the "internal tracks biology, output does not" contrast MUST NOT be written; the existing
# RESULTS_cluster/stratify_geometry.json test3 is a different roster, grouping, frame and metric
# and is not a legitimate comparator. The script writes that refusal into its own JSON.
#
# NOTE: paths are FLAT (~/tahoe/<script>.py) to match the cluster layout, not the nested repo.

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME=/data/BuffaF-Projetcs/florian_c2s/hf_cache
export HF_TOKEN=$(cat ~/.hf_token)
export HF_HUB_DISABLE_XET=1
PY=/data/BuffaF-Projetcs/florian_c2s/envs/c2s/bin/python

cd ~/tahoe
mkdir -p RESULTS logs

# ---- paths ---------------------------------------------------------------------------------
DATA=/data/BuffaF-Projetcs/florian_c2s/data_diverse2_endcell_big
PANEL="$DATA/l1000_panel.json"
TIER=tier2_unseen_drugs
# Fine-tuned single-cell model — the drug-blind one the whole story is about.
# CONFIRMED: identical path in workspace_probe / drug_stratify / drug_lookup_baseline / scramble_sweep.
CKPT=/data/BuffaF-Projetcs/florian_c2s/checkpoints/pythia_sft_endcell/final
# Base backbone for the fine-tuning contrast. CONFIRMED as the repo id used by every training sbatch
# (arm1a_consensus_train.sbatch, ot/ot_train.sbatch, arm1_consensus.sbatch).
BASE=vandijklab/C2S-Scale-Pythia-1b-pt

# ---- outputs (new stem; nothing in RESULTS_cluster/ uses it) --------------------------------
OUT_G1=RESULTS/drug_geometry_rsa_CVCL_0546_plate5.json
OUT_G2=RESULTS/drug_geometry_rsa_CVCL_0546_plate6.json
OUT_G3=RESULTS/drug_geometry_rsa_CVCL_0459_plate6.json
# The CPU precheck writes one JSON per stratum. They are results in their own right — a refusal on
# CPU-computable grounds IS the finding for that stratum — so they are protected by GUARD 2 too.
PRE_G1=RESULTS/precheck_drug_geometry_rsa_G1.json
PRE_G2=RESULTS/precheck_drug_geometry_rsa_G2.json
PRE_G3=RESULTS/precheck_drug_geometry_rsa_G3.json

# ============================================================================================
# GUARD 0 — never run on the login node. Compute goes through SLURM, without exception.
# ============================================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "[FATAL] no SLURM_JOB_ID — this script is being run on the login node. Use sbatch."
    exit 1
fi
echo "[env] job=$SLURM_JOB_ID host=$(hostname) node=${SLURM_NODELIST:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ============================================================================================
# GUARD 1 — inputs must exist. The script builds the biological side ITSELF from the eval jsonl
# (pseudobulk of the response sentences, then ROSTER-MEAN CENTERING — the code writes the frame as
# "minus the plate DMSO pool carried in the prompts", but that term cancels identically and is
# algebraically inert here; see the note at the top and PRE_REGISTRATION['plate_scope']), so there
# is no separate biology artefact to depend on and no stage-1 job to wait for.
# ============================================================================================
for f in drug_geometry_rsa.py "$DATA/eval_${TIER}.jsonl" "$PANEL"; do
    if [ ! -e "$f" ]; then
        echo "[FATAL] required input missing: $f"
        exit 1
    fi
done
test -d "$CKPT" || { echo "[FATAL] fine-tuned checkpoint not found: $CKPT"; exit 1; }
test -d "$DATA" || { echo "[FATAL] data dir not found: $DATA"; exit 1; }

# ============================================================================================
# GUARD 2 — never overwrite an existing result. Results are append-only by filename; a re-run
# that would clobber a prior number is refused rather than silently destroying it.
# ============================================================================================
for f in "$OUT_G1" "$OUT_G2" "$OUT_G3" "$PRE_G1" "$PRE_G2" "$PRE_G3"; do
    if [ -e "$f" ]; then
        echo "[FATAL] refusing to overwrite an existing result: $f"
        echo "        move it aside or change the output stem in this sbatch."
        echo "        (the precheck JSONs are covered by this rule as well: a refusal recorded on"
        echo "         CPU-computable grounds is a result, and re-running until a stratum passes is"
        echo "         exactly the forking path the pre-registration exists to close.)"
        exit 1
    fi
done

# ============================================================================================
# GUARD 3 — SELFTEST GATE. README: "Run the self-test before any cluster job." No GPU, no real
# data, no network, ~5 seconds. It plants worlds where the answer is known and checks the
# estimator returns it. The thresholds below are the ones the script actually enforces:
#
#   (a)  POSITIVE (activations = linear(biology) + noise, names/moa random)
#                                 r_partial >= +0.350 at the p floor 1/(n_perm+1);
#                                 co-primary matched-triplet accuracy > 0.550
#   (a2) MIXED (real biology AND a correlated name confound)
#                                 r_partial > +0.300, p < 0.010, raw-minus-partial gap > 0.005
#   (b)  NEGATIVE / NAME-ONLY — THE LOAD-BEARING ONE. Activations and biology are BOTH generated
#        from the name-token control; there is no pharmacology in the world at all:
#                                 r_raw > +0.300, |r_partial| < 0.080,
#                                 |r_partial|/|r_raw| < 0.250, p > 0.010
#   (b3) MULTI-CONFOUND (tokens + char n-grams + shared mechanism)
#                                 r_raw > +0.450, ratio < 0.250
#   (b2) NAME-DRIVEN via the vector path (Jaccard is not the generator's functional form, so the
#        annihilation is only partial by construction)
#                                 r_raw > +0.100, |r_partial| < 0.55 * |r_raw|
#   (c)  MECHANISM-BLOCK          r_raw > +0.300, |r_partial| < 0.080
#   (d)  DEGENERATE-MECHANISM (70% 'unclear'): the x3/x4 SPLIT annihilates (|r| < 0.060) where a
#        single collapsed 'shared mechanism' indicator manufactures |r| > 0.090 with the wrong sign
#   (e)  DEGENERACY GUARD: identical activations recorded as degenerate, no coefficient emitted
#   (f)  DYADIC interval >= 4.00x the iid half-width on the same influence values (the analytic
#        truth for that planted world is ~7.5 and the estimator measures ~6.9-7.6, so the gate is
#        tight enough to catch a regression to near-iid and still leaves ~1.7x margin)
#   (g)  NULL CALIBRATION over 120 replicates: mean |r_partial| < 0.060, mean p in [0.30, 0.70],
#        frac(p <= 0.05) <= 0.140
#   (h)  DISATTENUATION recovers a planted 0.50 within 0.050
#   (i)/(j) rel_B gate thresholds fire in order, and rel_B separates a reliable target from noise
#   (k)  COLLINEAR CONTROLS: the rank-deficient design is REPORTED, signal kept, confound still dies
#   (l)  INPUT GUARDS: asymmetric or non-finite matrices refused, all-zero encodings counted, and
#        the dose-string parser that feeds x6 checked on nM / uM / micro-sign / mM / unparseable
#   (m)  TRIPLET TOLERANCE: under the (b3) zero-pharmacology confound, matched-triplet accuracy at
#        tol_frac = 0.10 must sit within 0.020 of the 0.500 null (mean over 16 worlds), a WIDE
#        tolerance on the identical code path must read above it, and the (a) POSITIVE world must
#        keep accuracy > 0.550 with > 200 admitted triples
#   (n)  NAME-CONTROL GUARD: on a roster where 1 of 1770 pairs shares a token, name_control_inert
#        must be TRUE (the old trigger could not fire there); on the stem roster it must be FALSE
#   (o)  DISATTENUATED EQUIVALENCE: a planted TRUE r of 0.50 with rel_A 0.70 / rel_B 0.35 must come
#        back not_equivalent on the disattenuated scale while the attenuated scale calls it
#        equivalent, and a wide interval must return underpowered_for_equivalence
#   (p)  rel_A BY CONSTRUCTION: draw-invariant activations must give spearman_brown = 1.0 with
#        source control_draw_invariant_by_causal_structure, not None; varying draws still estimated
#   (q)  SPECTRUM: participation ratio and top-eigenvalue share recovered within 5% of the analytic
#        truth on two worlds with prescribed singular values
#   (r)  CONFOUND-RESIDUE CALIBRATOR: p95 |r_partial| > 0.050 on the (b3) recipe, <= 0.075 and
#        < 0.75x that on an all-noise control recipe whose p stays calibrated (frac(p<=.05) <= 0.14),
#        and the blind arm's own residue must exceed the chance control — which is why the floor is
#        per-arm rather than annotated-only
#   (s)  ACTIVITY MAGNITUDE: in a magnitude-only world with iid drug directions, activity covariates
#        must explain > 0.600 of B's dyad-rank variance (< 0.050 at equal amplitudes); the headline
#        must read > +0.200 and the activity-adjusted secondary must collapse below 0.060 in the
#        identified case and below both 0.5x the headline and the 0.120 residue envelope in the
#        monotone case. Means over 8 replicate worlds
#   (t)  DIRECT ARM: (t1) the POSITIVE world gives retrieval > 0.200 against a 1/n null of 0.020,
#        mean held-out r > +0.800 and increment_licensed True; (t2) the NAME-DRIVEN world gives an
#        increment <= +0.050 and increment_licensed FALSE (p_increment alone is 0.01 there and would
#        have licensed a representation that adds nothing); (t3) THE CASE THE ARM EXISTS FOR — at participation
#        ratio ~3 the RSA reads a null (|r_partial| < 0.150) while the direct arm still reads
#        mean r > +0.800 and perfect retrieval
#   (u)  CELL-LINE CONTROL B PATH: a planted per-line geometry must be recovered at Spearman > 0.700
#        against the planted matrix, with rel_B > 0.400 clearing the gate
#   (v)  x9 SAME-PLATE: a two-plate world where plate drives both matrices must read > +0.300
#        without x9 and |r| < 0.080 with it, and x9 must be DROPPED as zero-variance on a
#        single-plate roster (which is why cell_line_plate scope is unaffected)
#
# What the selftest does NOT cover, and what therefore rests on inspection alone: the PAIRED
# base-checkpoint contrast has no planted world, and there is no control-cell false-positive world.
# `set -e` would already abort, but the gate is explicit so the failure is unambiguous.
# ============================================================================================
echo "=== [0/5] selftest (no GPU, no real data, no network) ==="
if ! $PY drug_geometry_rsa.py --selftest; then
    echo "[FATAL] selftest FAILED — aborting the whole job before any forward pass."
    echo "        A failing NEGATIVE/NAME-ONLY world in particular means the confound control does"
    echo "        not adjust anything and no r_partial from this run would be interpretable."
    exit 1
fi
echo "[ok] selftest passed."

# Base backbone must resolve WITHOUT the network: it is loaded second, after the fine-tuned pass has
# already been computed, and the script writes its JSON only at the very end — so a mid-run fetch
# failure would throw away the whole invocation. Warning, not fatal, because the node may have net.
$PY - "$BASE" << 'PYEOF' || true
import sys
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)
    print(f"[ok] base backbone {sys.argv[1]} resolves from the local HF cache")
except Exception as e:
    print(f"[warn] base backbone {sys.argv[1]} is NOT in the local HF cache "
          f"({e.__class__.__name__}); the run will try to fetch it mid-job, and a network failure "
          f"would discard the fine-tuned half of that invocation too.")
PYEOF

# --------------------------------------------------------------------------------------------
# Shared configuration, identical across the three strata. Anything not named here does not run.
#   BLIND (primary) + ANNOTATED (secondary) mechanism arms; both checkpoints in ONE invocation so
#   the base contrast is paired; 16 balanced control sentences; layers 0..16 hooked on block
#   outputs; positions name (drug-name span, a tokenizer null), instr (end of the instruction),
#   last (the ':' of 'Response cell:', where the readout reads); headline cells L12 and L16 at
#   'last' on the blind arm of the fine-tuned checkpoint; 4000 Freedman-Lane permutations and 4000
#   rel_B bootstrap resamples (p floor = 1/4001); 1000 permutations for the triplet co-primary.
#
# Both distance metrics (centered and z-scored cosine) are computed for every cell unconditionally,
# the max-statistic FWER correction runs over one shared permutation sequence, and the n<25 / n<40
# and rel_B 0.20 / 0.40 thresholds are pre-registered CONSTANTS inside the script — none of them is
# a flag, and none of them can be loosened from here. That is deliberate.
#
# --min_cells_per_drug is the ONLY roster filter and it is a real trade-off: raising it lifts rel_B
# (more cells per drug -> a more reliable biological target) but shrinks n_drugs toward the n>=40
# headline threshold and the n>=25 refusal. 10 keeps the roster wide; if rel_B comes back below
# 0.40, the sensitivity re-run to make is this flag raised, not any threshold lowered.
# --------------------------------------------------------------------------------------------
COMMON=(
    --eval_dir "$DATA"
    --panel_file "$PANEL"
    --tier "$TIER"
    --scope cell_line_plate
    --model_paths "$CKPT,$BASE"
    --model_names finetuned,base
    --primary_stratum G1
    --positive_control
    --arms blind,annotated
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    --positions name,instr,last
    --headline_layers 12,16
    --n_drugs 60
    --n_controls 16
    --min_cells_per_drug 10
    --n_perm 4000
    --n_perm_triplet 1000
    --n_rel_splits 25
    --n_boot 4000
    --bf16
    --seed 42
)

# ============================================================================================
# STAGE [1/5] — CPU PRECHECK, BEFORE ANY CHECKPOINT IS LOADED.
#
# n_drugs, the biological matrix B, rel_B and the gate depend ONLY on the eval jsonl. The precheck
# path never imports torch or transformers — it is CPU-only and takes seconds, and it is run inside
# this same allocation purely for bookkeeping, not because it needs the GPU. It exits 0 when the
# stratum is runnable and 2 when it refuses (n_drugs < 25, or the rel_B gate withholding
# headline_permitted), so a stratum that cannot be tested never reaches a forward pass.
#
# WHAT A REFUSAL DOES HERE, precisely:
#   * a REFUSING stratum is SKIPPED — its GPU invocation is never issued and the reason is printed.
#   * if the PRIMARY stratum G1 refuses, the whole job aborts before any checkpoint is loaded.
#     PRE_REGISTRATION['multi_stratum_rule'] makes G2 and G3 replication and never a second chance
#     at a headline, so with G1 gone there is no headline any amount of GPU time can produce, and
#     spending it would only manufacture two numbers that the pre-registration forbids reading.
#   * if all three refuse, the job exits non-zero before the model is ever loaded.
#   * if any stratum was skipped, the job still exits non-zero at the end, so a partial run is never
#     mistaken for a complete one.
#
# The repo's own numbers say refusal is a live possibility: RESULTS_cluster/stratify_geometry.json
# test3 reports tier-2 n_drugs = 46 for CVCL_0546 and 35 for CVCL_0459 counted across ALL 11 plates,
# and these strata are pinned per plate, so the per-plate counts are lower again — against a
# headline threshold of 40 and a refusal floor of 25.
#
# RUN THIS AS A SEPARATE CPU JOB FIRST, and read the frontier before submitting this GPU job (note
# that GUARD 2 protects RESULTS/precheck_drug_geometry_rsa_G*.json, so use a different --out here):
#   $PY drug_geometry_rsa.py --eval_dir "$DATA" --panel_file "$PANEL" --tier "$TIER" \
#       --scope cell_line_plate --cell_line CVCL_0546 --plate plate5 \
#       --precheck --precheck_sweep --out RESULTS/precheck_scan_G1.json
# The sweep writes one row per (scope, min_cells_per_drug) with n_drugs and rel_B [lo, hi]. The
# selection rule is pre-registered in PRE_REGISTRATION['configuration_selection']: take the
# configuration maximising rel_B subject to n_drugs >= 40; if none reaches 40, take the best subject
# to n_drugs >= 25 and mark the run exploratory; if no configuration gives a rel_B lower bound above
# 0.20, DO NOT SUBMIT this job — that refusal is itself the result.
# ============================================================================================
# The precheck runs at the SAME configuration the GPU stages would use, so its verdict is a verdict
# about this job and not about a neighbouring one. --precheck_sweep then reports the whole
# (scope x min_cells_per_drug) frontier beside it, which is the table configuration_selection reads.
PRECHECK_COMMON=(
    --eval_dir "$DATA" --panel_file "$PANEL" --tier "$TIER"
    --scope cell_line_plate
    --n_drugs 60 --min_cells_per_drug 10 --n_rel_splits 25 --n_boot 4000 --seed 42
)
precheck_stratum () {                # precheck_stratum <label> <cell_line> <plate> <out>
    local label="$1" cl="$2" pl="$3" out="$4"
    echo "--- precheck $label : cell_line $cl, plate $pl ---"
    $PY drug_geometry_rsa.py "${PRECHECK_COMMON[@]}" --cell_line "$cl" --plate "$pl" \
        --stratum_label "$label" --primary_stratum G1 --precheck --precheck_sweep \
        --out "$out"
}

# Print the precheck's own verdict, n_drugs and rel_B, plus the frontier the sweep measured. This is
# what the configuration decision is made on, and it is printed whether the stratum runs or refuses.
report_precheck () {                 # report_precheck <label> <out>
    $PY - "$1" "$2" << 'PYEOF' || true
import json, os, sys
label, path = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print(f"  [{label}] precheck JSON not written — treat as a refusal."); raise SystemExit(0)
d = json.load(open(path))
bio = d.get("biology") or {}
relb = bio.get("rel_B") or {}
gate = bio.get("gate") or {}
print(f"  [{label}] verdict={d.get('precheck_verdict')}  n_drugs={d.get('n_drugs')}  "
      f"exploratory={d.get('precheck_exploratory')}  status={d.get('status')}")
print(f"  [{label}] rel_B={relb.get('rel_B')} [{relb.get('lo')}, {relb.get('hi')}]  "
      f"gate={gate.get('status')}  headline_permitted={gate.get('headline_permitted')}")
rows = d.get("precheck_sweep") or []
if rows:
    print(f"  [{label}] (n_drugs, rel_B) frontier — PRE_REGISTRATION['configuration_selection'] "
          f"picks the row MAXIMISING rel_B subject to n_drugs >= 40:")
    for r in (rows if isinstance(rows, list) else rows.get("rows", [])):
        print(f"      scope={r.get('scope'):<16} min_cells={str(r.get('min_cells_per_drug')):<4} "
              f"n_drugs={str(r.get('n_drugs')):<4} rel_B={r.get('rel_B')} "
              f"[{r.get('rel_B_lo')}, {r.get('rel_B_hi')}] "
              f"median_cells={r.get('median_cells_per_drug')} status={r.get('status')}")
PYEOF
}

echo "=== [1/5] CPU precheck (no torch, no transformers, no GPU) ==="
RUN_G1=0; RUN_G2=0; RUN_G3=0
SKIPPED=0
N_RUNNABLE=0
for spec in "G1 CVCL_0546 plate5 $PRE_G1" "G2 CVCL_0546 plate6 $PRE_G2" "G3 CVCL_0459 plate6 $PRE_G3"; do
    set -- $spec
    if precheck_stratum "$1" "$2" "$3" "$4"; then
        echo "[ok] precheck $1 says RUN"
        eval "RUN_$1=1"
        N_RUNNABLE=$((N_RUNNABLE + 1))
    else
        echo "[REFUSE] precheck $1 refused on CPU-computable grounds (n_drugs < 25, or the rel_B"
        echo "         gate withheld headline_permitted). Its GPU invocation will NOT be issued."
        SKIPPED=1
    fi
    report_precheck "$1" "$4"
done

if [ "$N_RUNNABLE" -eq 0 ]; then
    echo "[FATAL] all three strata refuse on CPU-computable grounds. Aborting BEFORE any checkpoint"
    echo "        is loaded — no model has been read and no forward pass has been requested."
    echo "        This refusal IS the result for this configuration: at this cell budget and this"
    echo "        scope the biological target carries no reliable off-diagonal structure at drug"
    echo "        resolution, so NO geometry-matching test — internal or output — can be run."
    echo "        Read the precheck JSONs and their sweep rows, and pick the configuration by"
    echo "        PRE_REGISTRATION['configuration_selection'] — a function of B and the roster only,"
    echo "        never of any activation. Do NOT lower a threshold to obtain a runnable stratum."
    exit 1
fi

if [ "$RUN_G1" -ne 1 ]; then
    echo "[FATAL] the PRIMARY stratum G1 refuses. G2 and G3 are REPLICATION and never a second"
    echo "        chance at a headline (PRE_REGISTRATION['multi_stratum_rule']), so no amount of GPU"
    echo "        time can now produce a licensed headline. Aborting BEFORE any checkpoint is"
    echo "        loaded rather than manufacturing two coefficients the pre-registration forbids"
    echo "        reading as a result."
    exit 1
fi

if [ "$SKIPPED" -ne 0 ]; then
    echo "[warn] at least one replication stratum refused and will be SKIPPED. The job will exit"
    echo "       non-zero so a partial run is never mistaken for a complete one."
fi
echo "[ok] $N_RUNNABLE of 3 strata are runnable on the pre-registered configuration; G1 (primary) is"
echo "     among them."

RC=$SKIPPED
run_stratum () {                     # run_stratum <label> <cell_line> <plate> <out>
    local label="$1" cl="$2" pl="$3" out="$4"
    echo "=== $label : cell_line $cl, plate $pl -> $out ==="
    if $PY drug_geometry_rsa.py "${COMMON[@]}" --cell_line "$cl" --plate "$pl" \
            --stratum_label "${label%% *}" --out "$out"; then
        echo "[ok] $label complete"
    else
        echo "[FAIL] $label did not complete; the remaining strata still run, and this job will"
        echo "       exit non-zero so the failure is not mistaken for a null."
        RC=1
    fi
}

# ============================================================================================
# G1 — PRIMARY stratum. G2 changes only the plate (same cell line); G3 changes the cell line.
# The three are separate analyses and raw distances are never pooled across them. The reason is NOT
# "the residual is defined against each plate's own DMSO pool" — that claim was false and is
# corrected at the top of this file: the DMSO term cancels identically. The reason is that each
# stratum has its own roster, its own roster mean, its own rel_B and its own batch structure, so the
# coefficients are not commensurable. Agreement across them is a robustness statement, not a
# combined coefficient, and PRE_REGISTRATION['multi_stratum_rule'] governs how it is read.
# ============================================================================================
echo "=== [2/5] G1 (primary) ==="
if [ "$RUN_G1" -eq 1 ]; then
    run_stratum "G1 PRIMARY" CVCL_0546 plate5 "$OUT_G1"
else
    echo "[SKIP] G1 — precheck refused it; see $PRE_G1. No forward pass issued."
fi

echo "=== [3/5] G2 (same cell line, different plate) ==="
if [ "$RUN_G2" -eq 1 ]; then
    run_stratum "G2 replication" CVCL_0546 plate6 "$OUT_G2"
else
    echo "[SKIP] G2 — precheck refused it on CPU-computable grounds; see $PRE_G2. No checkpoint"
    echo "       was loaded for this stratum and no forward pass was issued. The refusal, not a"
    echo "       missing number, is what this stratum reports."
fi

echo "=== [4/5] G3 (different cell line) ==="
if [ "$RUN_G3" -eq 1 ]; then
    run_stratum "G3 replication" CVCL_0459 plate6 "$OUT_G3"
else
    echo "[SKIP] G3 — precheck refused it on CPU-computable grounds; see $PRE_G3. No checkpoint"
    echo "       was loaded for this stratum and no forward pass was issued. The refusal, not a"
    echo "       missing number, is what this stratum reports."
fi

# ============================================================================================
# Read the results in this order, and in no other:
#   1. status and the rel_B gate. If the gate fails, or n < 25, NOTHING is licensed in either
#      direction and the cell is \notestablished. The reader STOPS there.
#   2. n_drugs and whether the run is exploratory (n < 40 is below the pre-registered headline
#      threshold, whatever the coefficient says).
#   3. the (L0, name) tokenizer null — a pure tokenizer cell that must not be a positive.
#   4. THE CELL-LINE POSITIVE CONTROL. If the identical code path cannot see a geometry the model
#      provably encodes, this job measured nothing about drugs and the reader is shown an
#      INSTRUMENT FAILURE instead of a headline. The summary below does not merely annotate that
#      case — it declines to print the headline at all.
#   5. only then the headline cells, and each coefficient only ever beside its arm's measured
#      confound-residue floor, the disattenuated equivalence verdict, the spectrum, the triplet
#      co-primary with its REALISED tolerances, and the direct decodability arm.
# The order is the point. Every quantity above can void the one after it.
# ============================================================================================
echo "=== [5/5] gate summary (status/gate -> n/exploratory -> L0/name null -> POSITIVE CONTROL -> headline) ==="
$PY - "$OUT_G1" "$OUT_G2" "$OUT_G3" << 'PYEOF' || { echo "[warn] the gate summary reader itself failed; the result JSONs are still on disk, read them by hand"; RC=1; }
import json, os, sys
for path in sys.argv[1:]:
    if not os.path.exists(path):
        print(f"--- {path} --- (not written)")
        continue
    try:
        d = json.load(open(path))
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        continue
    bio = d.get("biology") or {}
    relb = (bio.get("rel_B") or {})
    gate = (bio.get("gate") or {})
    print(f"--- {path} ---")
    n_top = d.get("n_drugs")
    expl = (None if n_top is None else bool(n_top < 40))
    print(f"  [1] status={d.get('status')}  gate={gate.get('status')}  "
          f"headline_permitted={gate.get('headline_permitted')}  "
          f"rel_B={relb.get('rel_B')} [{relb.get('lo')}, {relb.get('hi')}]")
    print(f"  [2] n_drugs={n_top}  exploratory={expl} "
          f"(n < 40 is below the pre-registered headline threshold, whatever the coefficient says; "
          f"n < 25 is a refusal)")
    prov = d.get("provenance") or {}
    print(f"  stratum={prov.get('stratum_label')} primary={prov.get('primary_stratum')} "
          f"is_primary={prov.get('is_primary_stratum')}  scope={(prov.get('args') or {}).get('scope')}")
    floors = bio.get("null_confound_residue_real_roster") or {}
    print("  confound-residue floors (p95 |r_partial| with NO pharmacology, this roster): "
          + ", ".join(f"{k}={(v or {}).get('p95_abs_r_partial')}" for k, v in floors.items()))
    comp = bio.get("rel_B_companions") or {}
    print(f"  rel_B companions (REPORT ONLY, not the gate): "
          f"full_panel={((comp.get('rel_B_full_panel') or {}).get('rel_B'))}  "
          f"well_split={((comp.get('rel_B_wellsplit') or {}).get('rel_B'))} over "
          f"{comp.get('rel_B_wellsplit_n_drugs_covered')} drugs")
    act = bio.get("activity_magnitude") or {}
    dose = bio.get("dose_frame") or {}
    print(f"  B is made of: R2(dyad ranks | activity magnitude alone)="
          f"{act.get('r2_dyad_ranks_on_activity_only')}  leading_eig_share="
          f"{act.get('leading_eigenvalue_share_of_B')}  |  dose x6 built={dose.get('x6_built')} "
          f"(parse rate {dose.get('drug_parse_rate')})")
    if d.get("status") in ("untestable_n", "no_measurable_biological_geometry", "rel_B_unavailable"):
        print(f"  {d.get('note')}")
        print("  NOTHING IS LICENSED IN EITHER DIRECTION. Do not read past this line.")
        continue
    # [3] the tokenizer null: layer 0 at the drug-name span, before any block has run
    for c in d.get("cells", []):
        if (c.get("layer") == 0 and c.get("position") == "name" and c.get("metric") == "centered"
                and c.get("model") == "finetuned" and c.get("arm") == "blind"):
            print(f"  [3] NULL-CHECK L0/name  status={c.get('status')}  r_raw={c.get('r_raw')}  "
                  f"r_partial={c.get('r_partial')} (p={c.get('p')})  <- must not be a positive")

    # ---- [4] THE CELL-LINE POSITIVE CONTROL, READ BEFORE ANY HEADLINE -----------------------
    # PRE_REGISTRATION['positive_control']['the_rule']: if the identical code path is ALSO null on a
    # geometry the model provably encodes, the drug-side null is VOID and is an INSTRUMENT FAILURE,
    # not a finding about drugs. So it is read FIRST and it can suppress the headline entirely.
    # The criterion is stated here rather than inferred later: the instrument is DEMONSTRATED if any
    # finetuned control cell at a headline layer (layer 0 excluded — it is a tokenizer cell) at an
    # internal position has status ok, r_partial > 0 and p <= 0.05.
    pc = d.get("positive_control") or {}
    pc_cells = [c for c in (pc.get("cells") or [])
                if c.get("status") == "ok" and c.get("model") == "finetuned"
                and c.get("layer") not in (0, None) and c.get("position") in ("instr", "last")]
    pc_hits = [c for c in pc_cells
               if (c.get("r_partial") or 0.0) > 0.0 and (c.get("p") is not None)
               and c.get("p") <= 0.05]
    pc_demonstrated = bool(pc_hits)
    print(f"  [4] POSITIVE CONTROL (cell-line geometry, identical code path): "
          f"status={pc.get('status')} n_cell_lines={pc.get('n_cell_lines')} "
          f"rel_B={((pc.get('rel_B') or {}).get('rel_B'))} "
          f"n_refusal_applied={pc.get('n_refusal_applied')} "
          f"drug_held_fixed={pc.get('drug_held_fixed')} dose={pc.get('dose_held_fixed')}")
    for c in sorted(pc_cells, key=lambda c: (c.get("layer"), c.get("position"))):
        print(f"      PC L{c.get('layer')}/{c.get('position')} r_raw={c.get('r_raw')} "
              f"r_partial={c.get('r_partial')} (p={c.get('p')})")
    if pc.get("status") != "ok" or not pc_cells:
        print("  [4] the positive control did not run or produced no usable cell "
              f"(status={pc.get('status')}). The drug-side result below is therefore bounded by "
              "MDE80 and rel_B ALONE and by no demonstrated positive: a null from it says 'we saw "
              "no correlation', never 'the correlation is not there'.")
    elif not pc_demonstrated:
        best = max(pc_cells, key=lambda c: (c.get("r_partial") or -9.9))
        print("  ============================================================================")
        print("  INSTRUMENT FAILURE. The cell-line positive control is NULL on the identical code")
        print(f"  path (best control cell: L{best.get('layer')}/{best.get('position')} "
              f"r_partial={best.get('r_partial')}, p={best.get('p')}).")
        print("  Cell-line identity is a geometry this model provably encodes. If this estimator")
        print("  cannot see it here, it measured nothing about drugs either. THE DRUG-SIDE RESULT")
        print("  IS VOID IN BOTH DIRECTIONS and must be reported as an instrument failure, not as a")
        print("  finding about drugs — neither 'organised' nor 'merely labelled' is licensed.")
        print("  The headline is NOT printed. Reading it out of the JSON by hand does not make it a")
        print("  result; PRE_REGISTRATION['positive_control']['the_rule'] was written before the run.")
        print("  ============================================================================")
        continue
    else:
        print("  [4] the instrument is DEMONSTRATED on this code path. A drug-side null below is a"
              " bounded null, not a blind one; proceed to the headline.")

    # ---- [5] the headline cells ------------------------------------------------------------
    for c in d.get("cells", []):
        if not c.get("is_headline"):
            continue
        tri = c.get("triplet") or {}
        ci = c.get("ci") or {}
        print(f"  HEADLINE {c.get('model')}/{c.get('arm')} L{c.get('layer')}/{c.get('position')} "
              f"{c.get('metric')}  n={c.get('n_drugs')} exploratory={c.get('exploratory')}")
        # r_partial NEVER appears without its arm's measured floor on the same line. p is not the
        # licensing statistic here: measured type-I under a confounded null runs 0.16-0.71.
        print(f"     r_raw={c.get('r_raw')}  r_partial={c.get('r_partial')} "
              f"vs CONFOUND-RESIDUE FLOOR {c.get('confound_residue_floor')} "
              f"-> within_envelope={c.get('within_confound_residue_envelope')}")
        print(f"       (p={c.get('p')}, fwer={c.get('p_fwer_max')}, "
              f"ci=[{ci.get('lo')}, {ci.get('hi')}])  "
              f"headline_permitted_for_this_cell={c.get('headline_permitted_for_this_cell')}")
        cid = c.get("ci_disattenuated") or {}
        print(f"     disatt={c.get('r_disattenuated')} [{cid.get('lo')}, {cid.get('hi')}] "
              f"status={c.get('r_disattenuated_status')}  "
              f"equivalence={(c.get('tost_margin_0.30_disattenuated') or {}).get('status')} "
              f"(disattenuated scale; the attenuated one is NOT the licensing statistic)")
        print(f"     MDE80={c.get('MDE80')}  effective_n_implied={c.get('effective_n_implied')} "
              f"(n_dyads={c.get('n_dyads')}, n_drugs={c.get('n_drugs')})  "
              f"triplet_acc={tri.get('triplet_acc')} (null exactly 0.50, "
              f"multiway ci={(tri.get('triplet_ci_multiway') or {}).get('lo')}-"
              f"{(tri.get('triplet_ci_multiway') or {}).get('hi')}, "
              f"p={tri.get('triplet_p')}, n_triples={tri.get('n_triples')})")
        mt = tri.get("matching") or {}
        print(f"     TRIPLET MATCHING REALISED: tol_frac={mt.get('tol_frac')} -> "
              f"tol_x1={mt.get('tol_x1')} (sd offdiag {mt.get('sd_offdiag_x1')}), "
              f"tol_x1p={mt.get('tol_x1p')} (sd offdiag {mt.get('sd_offdiag_x1p')})  |  "
              f"{mt.get('n_matching_keys_with_variance')}/7 matching keys have off-diagonal "
              f"variance  <- a cell matched on two keys is NOT 'matched on seven'; tol_frac is "
              f"never loosened to recover triples")
        spec = (c.get("activation_diag") or {}).get("spectrum") or {}
        print(f"     SPECTRUM participation_ratio={spec.get('participation_ratio')} "
              f"top_eig_share={spec.get('top_eig_share')}  <- a null with PR < 5 or share > 0.35 is "
              f"'not organised in the variance-dominant subspace', NOT 'not organised'")
        print(f"     name_control_inert={c.get('name_control_inert')} "
              f"(frac of dyads sharing any name token = {c.get('x1_frac_sharing')})  "
              f"activity_adjusted r_partial="
              f"{(c.get('r_partial_activity_adjusted') or {}).get('r_partial')} (over-adjusts by "
              f"construction; never the headline)")
        da = c.get("direct_arm") or {}
        if da.get("status") == "ok":
            print(f"     DIRECT ARM retrieval={da['activations'].get('retrieval_acc')} "
                  f"(null {da.get('retrieval_null')}, p={da.get('p_retrieval')})  "
                  f"mean_held_out_r={da['activations'].get('mean_held_out_r')}  "
                  f"names_only={da['names_only'].get('mean_held_out_r')}  "
                  f"increment={da.get('increment_mean_r')} "
                  f"licensed={da.get('increment_licensed')} "
                  f"(fwer={da.get('p_fwer_max_increment')})")
    for b in d.get("base_contrast", []):
        if b.get("arm") == "blind" and b.get("position") == "last" and b.get("metric") == "centered" \
                and b.get("layer") in (12, 16):
            print(f"  BASE-CONTRAST L{b.get('layer')}  "
                  f"delta_r_partial={b.get('delta_r_partial')} (paired p={b.get('p_paired')})")
    for b in d.get("position_contrast", []):
        if b.get("model") == "finetuned" and b.get("arm") == "blind" \
                and b.get("metric") == "centered" and b.get("layer") in (12, 16):
            print(f"  POSITION-CONTRAST L{b.get('layer')} instr-minus-last="
                  f"{b.get('delta_r_partial')} (paired p={b.get('p_paired')}, "
                  f"{b.get('scale')} scale; rel_A instr={b.get('rel_A_instr')} vs "
                  f"last={b.get('rel_A_last')})")
    for b in d.get("direct_base_contrast", []):
        if b.get("arm") == "blind" and b.get("position") == "last" and b.get("layer") in (12, 16):
            print(f"  DIRECT BASE-CONTRAST L{b.get('layer')}  "
                  f"delta_increment={b.get('delta_increment_mean_r')} (paired p={b.get('p_paired')})")
    # (the cell-line positive control is NOT printed here — it is read at [4], BEFORE the headline,
    #  because it can void everything above and a caveat printed underneath a coefficient is read
    #  after the coefficient has already been believed.)
PYEOF

echo "done -> $OUT_G1 / $OUT_G2 / $OUT_G3"
echo "        prechecks -> $PRE_G1 / $PRE_G2 / $PRE_G3"
if [ "$SKIPPED" -ne 0 ]; then
    echo "NOTE: at least one stratum was SKIPPED by the CPU precheck and has no result JSON. That is"
    echo "      a refusal on CPU-computable grounds, recorded in its precheck JSON, and it is the"
    echo "      finding for that stratum — not a missing number to be filled in by re-running with a"
    echo "      lowered threshold. This job exits non-zero for that reason alone."
fi
echo "REMINDER: the internal-vs-output contrast may NOT be written until the matched-support"
echo "          output-side generation pass has been run. Until then the comparator is \\notestablished."
echo "REMINDER: a positive r_partial licenses 'the representation is biologically ordered beyond what"
echo "          the prompt string supplies'. It does NOT license 'a better decoding head would"
echo "          recover drug effects' — that claim belongs to the direct arm, read against the"
echo "          ledger baselines, and to nothing else. See PRE_REGISTRATION['licences']."
echo "REMINDER: a null licenses nothing on its own either. The CELL-LINE POSITIVE CONTROL is read"
echo "          FIRST — the summary above declines to print a headline at all when it is null,"
echo "          because that is an INSTRUMENT FAILURE and not a finding about drugs. Only past it"
echo "          do the SPECTRUM (participation ratio below 5 means 'not organised in the"
echo "          variance-dominant subspace', NOT 'not organised') and the DIRECT ARM apply."
echo "REMINDER: a small positive inside its arm's confound-residue envelope is not a finding,"
echo "          however small its p. The envelope was measured on this roster, with no"
echo "          pharmacology in the world at all."
exit $RC
