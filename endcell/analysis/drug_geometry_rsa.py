#!/usr/bin/env python
r"""
drug_geometry_rsa.py
====================
**Is the drug region of the representation ORGANISED, or merely LABELLED?**

The thesis already establishes two things and this script re-does neither:

  1. *Separability.* Drug identity is linearly decodable from the activations at every layer
     (`mechanistic_drug_probe.py`). The thesis itself discounts it: the drug name is literally in
     the prompt, so a positive is nearly free.
  2. *Output-space geometry.* The model's predicted drug-drug distances do not track the real ones
     (`drug_stratify_geometry.py` test3; Mantel mean +0.011, inconsistent in sign per cell line).

The missing cell is the INTERNAL one: does the **metric structure** of the activation space match
the metric structure of real drug effects? Not "do drugs separate" -- that is free -- but "are the
drugs that behave alike biologically also the drugs that sit close together in the residual stream".

Either answer is a result:

  * organised internally, discarded at the readout  -> the sharpest form of "represented but not
    consulted": the information is not merely present, it is correctly *arranged*, and the readout
    still throws it away;
  * not organised internally                        -> "represented" only ever meant "labelled",
    the readout has nothing biologically useful to consult, and the thesis's framing needs revising.

THE CONFOUND THAT DECIDES WHETHER ANY OF THIS IS WORTH ANYTHING
---------------------------------------------------------------
The drug NAME and (in the deployed prompt) the MECHANISM string are both literally in the text.
Drugs whose names share BPE tokens, or that carry the same mechanism annotation, would cluster for
reasons that have nothing to do with pharmacology. So the headline is **not** a raw correlation. It
is a PARTIAL Spearman of activation distance on biological distance, controlling for name-token
overlap, mechanism annotation, prompt-length differences and near-duplicate naming, estimated by
multiple regression on distance matrices (MRM) with a **Freedman-Lane** permutation null in which
the DRUG LABELS are permuted (permuting matrix entries elementwise is invalid -- distance matrices
are not elementwise exchangeable).

WHAT IT DOES
------------
  * rebuilds every prompt from scratch with `format_prompt` -- it does NOT reuse
    `mechanistic_drug_probe.build_fixed_prompts`, which is broken: it computes a template and never
    uses it, handing each drug its own control-cell sentence, so the resulting geometry is a
    geometry of CONTROL CELLS, not of drugs;
  * uses a BALANCED design: the SAME K control sentences for EVERY drug, one fixed dose string, and
    (primary arm) `Mechanism: unclear.` for every drug -- so the only varying text in the whole
    prompt is the drug name, and the control-cell main effect cancels exactly under the within-draw
    centering;
  * captures the residual stream with FORWARD HOOKS on block outputs (`workspace_probe`'s pattern),
    never `output_hidden_states`: in GPT-NeoX `hidden_states[-1]` has `final_layer_norm` applied and
    lives in a different frame from the block output, which silently broke workspace_probe v3.0;
  * reads three positions from the same forward pass: the drug-name span (a tokenizer null), the end
    of the instruction, and the last prompt position (where the readout actually reads);
  * builds the biological matrix in the residual signed-rank frame the residual chapter grades in.
    The frame is written in the code as `pseudobulk(drug) - pseudobulk(plate DMSO) -
    generic_shift(cell line)`, but ALGEBRAICALLY that is exactly `res = pseudobulk(drug) - roster
    mean`: the generic is computed as `mean_d(pb(d) - dmso)`, so the DMSO term cancels identically
    and the plate-DMSO referencing is INERT here (verified: replacing the DMSO vector with a
    completely different one moves `res` by 5.6e-15, and `max|res - (pb - pb.mean(0))| = 4.4e-16`).
    What IS real credit, and is verified rather than assumed: leave-one-out and leave-one-in
    generics differ by the positive scalar m/(m-1) applied to every drug alike, so they give
    BYTE-IDENTICAL B under cosine (max diff 0.0) and the -1/(m-1) self-inclusion artefact does not
    apply. Then `signed_rank_from_vector`, then cosine distance;
  * reports reliability ceilings for BOTH matrices (`rel_A` from the K control draws, `rel_B` from
    repeated cell half-splits) and disattenuates two-sidedly, because a null against an unreliable
    target licenses nothing;
  * runs every arm on BOTH checkpoints on byte-identical prompts and reports the PAIRED contrast,
    which separates "fine-tuning organised the representation" from "drug names already meant
    something to a language model".

USAGE
-----
  python drug_geometry_rsa.py \
     --eval_dir DATA_endcell_big \
     --model_paths CKPT_endcell/final,vandijklab/C2S-Scale-Pythia-1b-pt \
     --model_names finetuned,base \
     --tier tier2_unseen_drugs --layers all --n_drugs 60 \
     --out RESULTS/drug_geometry_rsa.json --bf16 --seed 42

SELFTEST (no GPU, no data, no network, seconds) -- run this before any sbatch:
  python drug_geometry_rsa.py --selftest

The selftest is the point of the file. It plants worlds where the answer is known and checks that
the estimator gives it:
  (a) POSITIVE  -- activations are a linear image of the biological latent plus noise, names and
      mechanisms assigned at random. The PARTIAL statistic must recover a strong positive.
  (b) NEGATIVE / CONFOUND -- activations and biology are BOTH driven only by name-token overlap,
      with no pharmacology anywhere. The RAW statistic must come out clearly positive and the
      PARTIAL statistic must collapse to ~0. Without this line the whole experiment is worthless:
      it is the only evidence that the control does what it claims.
plus mechanism-block collapse, the x3/x4 split-coding test, dyadic-vs-iid interval width,
permutation calibration, the degeneracy guard, disattenuation recovery, a RANK-DEFICIENT control
design (the real one is: x2 is an exact affine function of x3, and x5b equals x5a under this
balanced design), and the input guards that refuse an asymmetric or non-finite distance matrix.
Exits non-zero on any failure so a job script can gate on it.

WRITES NOTHING except the single JSON at --out.
"""
# --- repo path bootstrap: works in BOTH the reorganized repo AND the flat cluster layout ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
_cands = [_HERE, os.path.join(_HERE, "src")]
if os.path.isdir(os.path.join(_ROOT, "shared")):
    _cands += [os.path.join(_ROOT, "shared")] + sorted(glob.glob(os.path.join(_PIPE, "*")))
for _p in _cands:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, logging, math, re, hashlib, platform
from collections import defaultdict, Counter
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-12

# ---------------------------------------------------------------------------------------------
# Pre-registration block. Copied verbatim into the output JSON so the reading is auditable against
# what was declared before the run.
# ---------------------------------------------------------------------------------------------
PRE_REGISTRATION = {
    "question": ("Does the metric structure of the model's internal representation of drugs match "
                 "the metric structure of real drug effects, beyond what the prompt string hands it "
                 "for free?"),
    "headline_cell": {
        "arm": "blind (Mechanism: unclear for every drug)",
        "model": "the first entry of --model_names (the fine-tuned checkpoint)",
        "layer": 12,
        "position": "last (the ':' of 'Response cell:', where the readout reads)",
        "metric": "centered cosine",
        "statistic": "partial Spearman of offdiag(A) on offdiag(B) given the controls, via MRM",
        "null": ("Freedman-Lane, permuting DRUG LABELS, --n_perm draws (4000 in the pre-registered "
                 "configuration; the realised count is in each cell's n_perm and the p floor is "
                 "1/(n_perm+1))"),
    },
    "co_primary": [
        "layer 16 (readout boundary; block output, pre-final-LayerNorm -- one LN plus unembed "
        "from logits, reported as its own labelled cell, not folded into 'internal')",
        "position 'instr' (end of the instruction), free from the same forward pass; the "
        "instr-minus-last gap is a named secondary result, reported in `position_contrast` with a "
        "PAIRED permutation p (one drug relabeling applied to both positions inside each draw) and "
        "computed on the DISATTENUATED scale whenever both rel_A values are available, because "
        "'instr' carries rel_A = 1 by causal construction while 'last' does not, and an attenuated "
        "comparison is biased toward 'instr' for purely measurement reasons",
        "matched-triplet retrieval, whose null is exactly 0.50 and which holds the confounds "
        "constant by matching rather than by adjustment",
    ],
    "matched_triplet_tolerance": {
        "rule": ("the x1 and x1p matching tolerances are tol_frac * std(offdiag(control)) computed "
                 "on THIS run's roster, with tol_frac = 0.10, not fixed constants. The realised "
                 "tolerances and the sds are recorded per cell in triplet.matching."),
        "why": ("the shipped constant 0.05 was 0.43-0.59 SD of the confound on realistic rosters, "
                "and because a triple is admitted only when B separates near from far, a wide "
                "tolerance composes with the admission rule to HARVEST confound-driven triples. "
                "Measured under a zero-pharmacology confounded null: accuracy 0.6297 with "
                "triplet_p <= 0.05 in 40/40 replicates at the constant, versus 0.5058 and 0/40 at "
                "0.10 sd. On a clean positive the same tightening cost nothing (0.5953 -> 0.5948, "
                "power 1.000 unchanged)."),
        "yield_rule": ("if the tightened tolerance leaves fewer than 200 admitted triples, the cell "
                       "reports insufficient_matched_triples. tol_frac is NEVER loosened to recover "
                       "yield: it is a matching standard, not a dial."),
        "keys_actually_binding": ("the number of the seven matching keys with nonzero off-diagonal "
                                  "variance is reported per cell as "
                                  "triplet.matching.n_matching_keys_with_variance, because on a "
                                  "blind roster x5a and x8 are commonly identically zero and x5b is "
                                  "collinear with x5a."),
    },
    # PER ARM, because build_control_matrices constructs x2/x3/x4 ONLY when arm == 'annotated'.
    # A flat list of eight described a design that never runs in the primary arm.
    "controls_partialled": {
        "blind": ["x1 name-token Jaccard", "x1p name char-3gram Jaccard",
                  "x5a name length difference", "x5b prompt length difference",
                  "x8 near-duplicate base name"],
        "annotated": ["x1 name-token Jaccard", "x1p name char-3gram Jaccard",
                      "x5a name length difference", "x5b prompt length difference",
                      "x8 near-duplicate base name",
                      "x2 mechanism-token Jaccard", "x3 shared known mechanism",
                      "x4 shared 'unclear' mechanism"],
        "x6 dose": ("added to BOTH arms when >= 90% of drugs' dose strings parse to a numeric "
                    "micromolar value; see dose_frame"),
        "x9 same-plate": ("added to BOTH arms only when the roster spans more than one plate, i.e. "
                          "under cell_line scope; dropped as zero-variance under cell_line_plate "
                          "scope, where plate is eliminated by stratification instead"),
        "note_how_many_actually_apply": (
            "the DECLARED set is not the APPLIED set and the two must never be conflated. On a "
            "balanced blind roster x5b is EXACTLY x5a (the prompt varies only through the drug "
            "name, so the design is rank deficient by construction and one column is absorbed), x8 "
            "is identically zero on any roster without salt/hydrate variants, and x1 is typically "
            "inert because few pairs of distinct compounds share a BPE token. The number of "
            "controls ACTUALLY applied is reported per cell in controls_in_design, "
            "controls_dropped_zero_variance, design_rank / design_rank_deficient and "
            "name_control_inert. This result may NOT be quoted as 'adjusted for eight controls'; "
            "quote the per-cell controls_in_design instead."),
    },
    # WHAT A POSITIVE BUYS, AND WHAT IT DOES NOT. Written before the run, copied verbatim into
    # every output JSON, because the inference in `not_licensed` is the stated motivation for the
    # experiment and it does not follow from the instrument.
    "licences": {
        "licensed": ("a positive r_partial (or matched-triplet accuracy above its null) licenses "
                     "the claim that the representation is biologically ordered beyond what the "
                     "prompt string supplies, in this stratum, at this layer and read position."),
        "not_licensed": (
            "it does NOT license the claim that a better decoding head on this representation "
            "would recover drug effects. A partial Spearman of pairwise distances, and a "
            "matched-triplet ordering accuracy, are statements about RANK ORDER; neither "
            "establishes that the biological profile is decodable to useful precision. A "
            "representation with r_partial = +0.25 against a target of reliability 0.3 can be "
            "genuinely ordered and still carry nothing a head could turn into a competitive "
            "prediction. That claim is licensed, if at all, only by the direct decodability arm "
            "(direct_arm) scored against the ledger baselines."),
        "and_the_converse": (
            "a NULL r_partial does not license 'not organised' either, on its own: RSA is a "
            "variance-weighted second-order instrument, so structure living in a low-variance "
            "subspace can be invisible to it while a linear head reads it perfectly. See "
            "spectrum_reading_rule, positive_control and direct_arm for the conditions under "
            "which a null becomes attributable."),
    },
    "never_collapse_x3_x4": ("about half the drugs carry moa == 'unclear'; a single 'shared "
                             "mechanism' indicator would code ~40% of dyads as mechanistically "
                             "matched on shared IGNORANCE"),
    # THE READING RULE ACROSS STRATA, written before submission. Three strata x two arms x two
    # headline layers x two co-primaries with no declared rule is a forking path sitting on the
    # thesis headline, and sign disagreement is not hypothetical: the analogous per-cell-line
    # Mantels in stratify_geometry test3 run from -0.455 to +0.221 with a mean of +0.011.
    "multi_stratum_rule": {
        "primary_named_before_the_run": (
            "the PRIMARY stratum is named in the sbatch via --primary_stratum and stamped into "
            "every output JSON as provenance.stratum_label / provenance.primary_stratum / "
            "provenance.is_primary_stratum BEFORE any number exists. Default G1."),
        "replication_not_a_second_chance": (
            "G2 and G3 are REPLICATION. They are never a second chance at a headline: a positive "
            "in G2 or G3 while the primary is null is reported as a non-replicating primary null, "
            "not as a finding."),
        "claim_condition": (
            "the organised-representation claim is written only if the PRIMARY clears BOTH the "
            "n >= 40 threshold and the rel_B gate, AND every stratum that clears its own gate "
            "agrees in sign with it."),
        "disagreement": (
            "any sign disagreement among gate-clearing strata is reported as HETEROGENEITY and the "
            "claim is downgraded to \\notestablished. It is not resolved by picking the "
            "significant stratum, by pooling, or by declaring the disagreeing stratum an outlier."),
        "agreement_is_not_a_coefficient": (
            "cross-stratum agreement is a robustness statement, NOT a combined coefficient, and it "
            "carries this caveat: a treated well spans many cell lines, so replication across "
            "strata reproduces well-level technical artefact in B as well as biology. Agreement is "
            "therefore weaker evidence than it looks."),
    },
    "refusal_rule": {"n_drugs < 25": "statistic not computed, status = untestable_n",
                     "25 <= n_drugs < 40": "computed but tagged exploratory",
                     "headline requires": "n_drugs >= 40"},
    "reliability_gate_on_rel_B": {
        "quantity_read": ("the BOOTSTRAP CI LOWER BOUND of rel_B at BOTH thresholds. Previously the "
                          "0.20 threshold read the lower bound while the 0.40 threshold read the "
                          "point estimate; conservatism is now consistent in both directions."),
        "lower_bound <= 0.20": "no_measurable_biological_geometry; nothing licensed in either "
                               "direction",
        "0.20 < lower_bound < 0.40": "attenuation_limited; quote only the disattenuated value, "
                                     "with the ceiling beside it",
        "lower_bound >= 0.40": "headline permitted",
        "behaviour_on_failure": ("the run RETURNS at the gate, writing the JSON with status = the "
                                 "gate status, BEFORE any checkpoint is loaded. It does not warn "
                                 "and continue into the model sweep. --force_through_gate overrides "
                                 "this for diagnostics only; it is recorded in provenance and every "
                                 "cell it produces is stamped produced_under_forced_gate = true, "
                                 "and such cells are not interpretable in either direction."),
    },
    # HOW THE CONFIGURATION IS CHOSEN, declared before the sweep is read. Every quantity below is a
    # function of the eval jsonl alone -- roster and B -- and never of any activation, so the choice
    # cannot bias the headline.
    "configuration_selection": {
        "search_space": ("--min_cells_per_drug in {10, 25, 50, 100} x --scope in "
                         "{cell_line_plate, cell_line}, read from --precheck --precheck_sweep on "
                         "CPU before any GPU job is submitted"),
        "rule": ("take the configuration MAXIMISING rel_B subject to n_drugs >= 40. If no "
                 "configuration reaches n_drugs >= 40, take the one maximising rel_B subject to "
                 "n_drugs >= 25 and mark the run EXPLORATORY. If no configuration gives a rel_B "
                 "bootstrap lower bound above 0.20, DO NOT SUBMIT the GPU job: the target carries "
                 "no reliable off-diagonal structure at any available cell budget, which is itself "
                 "the result."),
        "why_it_cannot_bias": ("selection is a function of B and the roster only. No activation, "
                               "no checkpoint and no forward pass enters the precheck, so the "
                               "configuration is fixed before the quantity of interest exists."),
        "what_it_does_not_license": ("choosing the configuration with the best rel_B is NOT "
                                     "choosing the configuration with the best r_partial, and the "
                                     "sweep table is published in the JSON so the choice is "
                                     "auditable."),
    },
    "activity_magnitude": {
        "reported_always": ("biology.activity_magnitude carries the R^2 of B's dyad ranks on the "
                            "activity covariates {x7 = |rank a(d) - rank a(e)|, x7b = min(rank a(d), "
                            "rank a(e))} ALONE, B's leading-eigenvalue share, and the a(d) "
                            "distribution of the selected roster against the full eligible pool -- "
                            "so the top-n-by-cell-count truncation is visible rather than assumed "
                            "away. a(d) = ||res(d)||_2, taken BEFORE the signed-rank encoding."),
        "secondary_arm": ("r_partial_activity_adjusted, at headline cells only, is the identical "
                          "partial with {x7, x7b} appended to the control design. It is a LABELLED "
                          "SECONDARY and never the headline."),
        "why_it_over_adjusts": ("activity class is genuinely correlated with mechanism, so this arm "
                                "removes real pharmacology along with the artefact and reads LOW BY "
                                "CONSTRUCTION. Its use is asymmetric: a headline that SURVIVES it "
                                "excludes the activity reading; a headline that collapses under it "
                                "is not thereby shown to be artefact, only shown to be "
                                "unseparated by this design."),
        "why_it_is_needed": ("res = pb - roster mean, so every weak drug converges on -(mean drug "
                             "programme) and all inert drugs collapse onto one another in B. In a "
                             "planted world where drugs differ ONLY in effect magnitude, with iid "
                             "drug-specific directions and no pharmacology anywhere, activity "
                             "covariates explain ~0.85 of B's dyad-rank variance (0.000 at equal "
                             "amplitudes). With the atlas reporting median SNR 0.75 and 27% of "
                             "conditions not significantly active, the real B carries this axis "
                             "heavily, and 'the model knows which compounds do something' is one "
                             "scalar that licenses no readout-rescue claim."),
    },
    "plate_scope": {
        "cell_line_plate": ("plate batch is ELIMINATED by stratification; x9 is constant across the "
                            "roster and is dropped as zero-variance by the design machinery, which "
                            "is recorded per cell in controls_dropped_zero_variance."),
        "cell_line": ("the roster spans plates, so plate batch enters B through the pseudobulk "
                      "itself and is ADJUSTED dyadically via x9, the same-plate indicator on each "
                      "drug's modal plate."),
        "not_interchangeable": ("the two scopes are different readings and are never pooled or "
                                "swapped after the fact. Which one runs is fixed by the CPU "
                                "precheck under configuration_selection, before any activation "
                                "exists."),
        "why_x9_exists": ("the design's stated justification for pinning the plate -- 'the residual "
                          "is defined against that plate's DMSO pool' -- is FALSE: the DMSO term "
                          "cancels identically (see biology.frame). What survives is batch structure "
                          "in the pseudobulk, which is what x9 addresses. So if the precheck shows "
                          "the pinned per-plate strata are untestable, cell_line scope is the only "
                          "way the question gets asked at all, and it must arrive with plate "
                          "adjusted rather than ignored."),
    },
    "positive_control": {
        "what_it_is": ("cell-line geometry on the IDENTICAL code path: drug and dose held fixed at "
                       "the roster's modal compound, the CELL LINE NAME varying in the same prompt "
                       "slot, the same K balanced control sentences, the same hooks, the same "
                       "centering, the same estimator, and B built from each line's own control-cell "
                       "pseudobulk in the same residual signed-rank cosine frame."),
        "why_cell_line": ("the model provably encodes it -- the thesis measures a cell-line subspace "
                          "fraction of ~0.6 against ~0.03 for the purified drug slab -- and it is a "
                          "MATCHED anchor: it enters through 1-3 tokens in the same prompt slot "
                          "against the same ~400-token context, so it exercises the identical "
                          "variance-domination risk."),
        "the_rule": ("IF THE CELL-LINE ARM IS ALSO NULL, THE DRUG-SIDE NULL IS VOID and must be "
                     "reported as an INSTRUMENT FAILURE, not as a finding about drugs. A null "
                     "bounded by MDE80 alone says 'we saw no correlation'; only a null beside a "
                     "demonstrated positive on the same code path says 'the correlation is not "
                     "there'."),
        "n_rule": ("the n < 25 refusal is NOT applied to this arm and the waiver is stamped in the "
                   "output. The refusal is a rule about claims; this arm makes no claim about "
                   "cell-line geometry, it measures whether the instrument can see anything at all. "
                   "Its rel_B and gate are computed and REPORTED, never applied."),
        "never": "this block is a control. It is never a headline and never enters the RSA families.",
    },
    "direct_arm": {
        "what_it_is": ("leave-one-drug-out ridge, in the DUAL form, from X = the K-averaged, "
                       "across-drug-centered activations to Y = the drug's residual profile, at the "
                       "headline cells only. alpha is chosen by an inner LOO on each training fold "
                       "over a fixed grid, never fixed: a fixed alpha shrinks low-variance "
                       "directions and reimports the exact pathology this arm exists to detect."),
        "scores": ("(i) held-out RETRIEVAL -- encode the predicted residual with "
                   "signed_rank_from_vector and ask whether its nearest true encoding is the "
                   "held-out drug's; null exactly 1/n_drugs with no distributional assumption. "
                   "(ii) held-out mean correlation between predicted and true residual."),
        "confound_control": ("BY INCREMENT, not by partialling: the identical LOO ridge is fit from "
                             "NAME features alone (character-3grams plus this model's own BPE token "
                             "indicators) and the activations-over-names increment is reported."),
        "increment_licensing": (
            "the licensing field is increment_licensed = (increment > 0 AND p_increment <= 0.05), "
            "and p_increment must NEVER be read alone. The increment's permutation null is not "
            "centred at zero: relabeling Y destroys both relationships at once and the two kernels "
            "do not collapse to the same floor, so a name kernel that interpolates its training "
            "drugs scores far worse under a relabeling than a smoother activation kernel does. "
            "Measured in a planted world where the names are a SUPERSET of what the activations "
            "carry: observed increment -0.41 against a null q95 of -0.93, giving p = 0.01 for a "
            "representation that adds nothing whatever."),
        "null": ("the first --n_perm_direct draws of the SHARED permutation sequence, permuting the "
                 "drug labels of Y. The arm carries its OWN max-statistic family and its OWN paired "
                 "base-checkpoint contrast; it is never folded into the RSA's family, because a max "
                 "taken across two different statistics is neither."),
        "why_it_is_needed": ("RSA is a one-sided instrument. A positive licenses 'organised'; a null "
                             "licenses NOTHING about organisation, because RSA on centered cosine is "
                             "variance-weighted and structure in a low-variance subspace is "
                             "invisible to it. Verified: with drug-varying nuisance at participation "
                             "ratio 2.93, RSA r_partial falls to +0.038 while LOO ridge from the "
                             "same activations to the same target holds at r = +0.992. The branch "
                             "this thesis needs is the null branch."),
        "why_it_does_not_replace_the_RSA": ("the RSA fits nothing, so it cannot overfit; its "
                                            "Freedman-Lane null is exact at n = 60; and it is "
                                            "invariant to how a head is parameterised. A positive "
                                            "RSA is the stronger and more general claim. The ridge "
                                            "has a real regularisation choice and a real overfitting "
                                            "surface at n = 60. They ACCOMPANY each other."),
        "joint_reading_rule": {
            "RSA+ / direct+": ("organised, and organised in the variance-dominant geometry. The "
                               "strongest form of the readout-problem claim."),
            "RSA0 / direct+": ("organised in a subspace the cosine geometry does not weight. STILL "
                               "a readout problem, and a concrete next experiment. Read the "
                               "spectrum: a low participation ratio is the expected signature."),
            "RSA+ / direct0": ("suspect the direct arm's regularisation and claim NOTHING from the "
                               "direct arm; the RSA positive stands on its own terms."),
            "RSA0 / direct0": ("licenses 'represented only ever meant labelled' ONLY when the "
                               "cell-line positive control is POSITIVE on the identical code path "
                               "AND rel_B >= 0.40. Without both, a double null is an instrument "
                               "report, not a finding about drugs."),
        },
        "what_a_positive_direct_arm_still_does_not_license": (
            "a held-out correlation is not a competitive prediction. Compare it to the ledger "
            "baselines before writing anything about rescuing the readout."),
    },
    "dose_frame": {
        "the_mismatch": ("A is DOSE-FIXED: run() puts ONE modal dose string in every prompt. B is a "
                         "DOSE MIXTURE: build_biology pools every cell of a drug in the stratum "
                         "regardless of dose. The sbatch's claim that dose is 'held CONSTANT by "
                         "construction' is true of the prompt and FALSE of B."),
        "resolution": ("option (b), ADJUST rather than filter. Each drug's median log10 dose enters "
                       "the control design as x6 = |median log10 dose(d) - median log10 dose(e)| in "
                       "both arms. x6 was already implemented and was never passed; it is now."),
        "why_not_filter": ("option (a) -- filtering each drug's cells to its modal dose -- is "
                           "cleaner in principle and costs cells per drug, and n and rel_B are the "
                           "binding constraints this whole build order exists to protect. Option "
                           "(b) is free."),
        "refusal": ("if fewer than 90% of drugs yield a parseable dose, x6 is NOT built at all and "
                    "dose_parse_failed is written with the parse rate. A partially parsed covariate "
                    "is worse than none."),
        "partial_parse": ("at or above 0.90 but below 1.00, x6 is built and the unparsed drugs are "
                          "IMPUTED at the modal dose string the prompt itself fixes (or, if that "
                          "string does not parse either, at the roster median of the parsed "
                          "per-drug medians). The count, the drug names and the source are written "
                          "to biology.dose_frame as n_drugs_dose_imputed / drugs_dose_imputed / "
                          "dose_imputation_source. This is declared as NOT conservative: an imputed "
                          "drug's x6 entries understate its dose distance, so dose is UNDER-adjusted "
                          "for those pairs. A dyadic design has no missing-value semantics, so the "
                          "covariate is total over the roster or absent -- there is no third option, "
                          "and leaving a None in the vector was a crash, not a policy."),
        "what_remains": ("the mismatch is adjusted, NOT eliminated, and it is not eliminable "
                         "without filtering cells. biology.dose_frame.per_drug reports each drug's "
                         "number of distinct dose strings and the fraction of its cells at the "
                         "roster's modal dose string, so the residual mismatch is visible per drug "
                         "rather than asserted away. Note that dose composition and activity "
                         "magnitude are the same axis viewed twice; see activity_magnitude."),
    },
    "spectrum_reading_rule": (
        "every analysed cell reports activation_diag.spectrum: the top-10 singular values of Hbar, "
        "top_eig_share = s1^2 / sum(s^2), and participation_ratio = (sum s^2)^2 / sum(s^4). A NULL "
        "r_partial accompanied by a participation ratio below 5 or a top-eigenvalue share above "
        "0.35 may NOT be reported as absence of organisation, only as ABSENCE OF ORGANISATION IN "
        "THE VARIANCE-DOMINANT SUBSPACE. Reproduced independently: holding the biology perfectly "
        "decodable, RSA r_partial falls +0.946 -> +0.038 as the participation ratio falls 9.41 -> "
        "2.93, while leave-one-out ridge from the same activations holds at r = +0.992. At the "
        "'last' position the drug enters through 1-4 name tokens out of ~400, so the dominant "
        "drug-varying directions are very likely lexical rather than pharmacological, and the "
        "design must be able to SAY whether that happened rather than assume it did not."),
    "degeneracy_rule": ("(layer 0, last position) is degenerate by construction -- the last token "
                        "is ':' for every prompt -- and must be recorded as 'degenerate', never "
                        "returned as 0.0 or NaN into an aggregate"),
    "p_floor": "p = (1 + #{|perm| >= |obs|}) / (n_perm + 1); flagged p_at_floor when it sits there",
    "confound_residue_floor": {
        "value": ("MEASURED PER RUN AND PER ARM; the realised numbers are in "
                  "biology.null_confound_residue_real_roster and are copied into each cell as "
                  "confound_residue_floor"),
        "provenance": ("the 95th percentile of |r_partial| under a zero-pharmacology confounded "
                       "null generated from THIS run's own control matrices -- built from the "
                       "actual drug names and the model's own tokenizer -- at this n and this "
                       "control set, over --n_residue_rep replicates."),
        "recipe_annotated": "0.55*x1 + 0.30*x1p + 0.15*(1 - x3), the selftest's world (b3)",
        "recipe_blind": ("0.65*x1 + 0.35*x1p; the mechanism term cannot exist in the blind arm, but "
                         "the blind arm has its own lexical residue, which is why the floor is "
                         "per-arm rather than annotated-only"),
        "control": ("a noise_only recipe is calibrated alongside and must return the chance "
                    "envelope; it is what separates 'a real confound residue' from 'sampling'"),
        "enforcement": ("any cell whose |r_partial| falls below its arm's floor is tagged "
                        "within_confound_residue_envelope = true and "
                        "headline_permitted_for_this_cell = false, whatever its p or p_fwer_max."),
        "why_p_is_not_enough": ("Freedman-Lane is a correct test of 'A-residual independent of "
                                "B-residual after linear-in-ranks adjustment', which is not the "
                                "scientific null. Measured type-I under a zero-pharmacology "
                                "confounded null: 0.163 with a rank-linear confound, 0.707 with a "
                                "dominant exp-link one. Expanding the control basis does not repair "
                                "it (0.707 -> 0.607 with quantile bins), so this is an "
                                "identification failure, not an inference failure, and magnitude "
                                "against a measured envelope is the only licensing statistic left."),
        "not_a_weakening": ("this rule only ever REFUSES a headline; it can never permit one that "
                            "the n rule or the rel_B gate has refused."),
    },
    "equivalence_margin": {
        "margin": 0.30,
        "licensing_statistic": (
            "tost_margin_0.30_disattenuated, computed in analyse_cell by testing the RAW interval "
            "against margin_eff = 0.30 * sqrt(rel_A * rel_B). That is algebraically identical to "
            "dividing the interval by sqrt(rel_A * rel_B) and testing 0.30, but it never divides by "
            "a near-zero reliability. Statuses: equivalent (interval entirely inside the effective "
            "margin), not_equivalent (interval entirely outside it), underpowered_for_equivalence "
            "(interval straddles it, or a reliability is unavailable)."),
        "also_reported_not_licensing": (
            "tost_margin_0.30_attenuated, the old attenuated-scale boolean, kept beside it and "
            "explicitly labelled NOT the licensing statistic. At a dyadic half-width of ~0.045 it "
            "fires for any |r_partial| < 0.255 and is close to unfalsifiable."),
        "direction_of_the_change": ("equivalence becomes HARDER to earn, and at a low rel_B it will "
                                    "usually return underpowered_for_equivalence rather than a "
                                    "verdict. That is the correct behaviour: the null branch is "
                                    "then reported as underpowered, not as a result."),
    },
    "disattenuation": {
        "point": ("two-sided, r / sqrt(rel_A * rel_B); rel_A is measurable because the K control "
                  "draws give the activation matrix within-drug variance"),
        "interval": ("ci_disattenuated widens the raw interval over rel_B's own bootstrap "
                     "endpoints, because the multiplier is not a constant: at rel_A 0.90 and rel_B "
                     "0.30 CI [0.22, 0.40] it alone spans 1.67x to 2.25x. It is CONDITIONAL on "
                     "rel_A, which carries no interval by its own declaration."),
        "status": ("r_disattenuated_status distinguishes 'ok', 'inputs_unavailable' and "
                   "'correction_exceeded_unity'. A blown-up correction is a loud signal that the "
                   "reliabilities cannot support the adjustment; it must not read as missing data."),
    },
    "clustering": ("dyadic cluster-robust on the drug pair, from the OLS influence function; "
                   "df = n_drugs - 1. Cell-line clustering is UNAVAILABLE BY CONSTRUCTION, not "
                   "'unavailable at this n' and not 'fewer than 5 usable clusters': this script "
                   "analyses ONE stratum per invocation, so a run contains exactly ONE cell line "
                   "and a cluster-robust variance with G = 1 is undefined rather than imprecise. "
                   "The true count is written to cellline_cluster_ci.n_cell_line_clusters. "
                   "Cross-cell-line variation is addressed only by running separate strata and "
                   "reading them under multi_stratum_rule."),
    "forbidden": ["reusing mechanistic_drug_probe.build_fixed_prompts (it never applies its own "
                  "template, so each drug gets its own control cell and the geometry becomes a "
                  "geometry of control cells)",
                  "reading out.hidden_states[L] (final_layer_norm frame mismatch at the last layer)",
                  "permuting distance-matrix entries elementwise",
                  "comparing this number to RESULTS_cluster/stratify_geometry.json test3, which is "
                  "a different roster, grouping, frame and metric, with no p and no interval"],
}


# =============================================================================================
# Optional repo imports, each with a local fallback so --selftest runs anywhere, on any layout,
# with nothing installed but numpy. Which source was used is recorded in the output JSON.
# =============================================================================================
_SOURCES = {}

try:
    import inference as inf
    _SOURCES["inference"] = "shared/inference.py"
except Exception as _e:                                                       # pragma: no cover
    inf = None
    _SOURCES["inference"] = f"fallback (import failed: {_e})"

try:
    import residual_eval as _re_mod
    signed_rank_from_vector = _re_mod.signed_rank_from_vector
    _SOURCES["signed_rank_from_vector"] = "residual_eval.signed_rank_from_vector"
except Exception as _e:
    _re_mod = None
    _SOURCES["signed_rank_from_vector"] = f"local copy (import failed: {_e})"

    def signed_rank_from_vector(res, P, k=100):
        """Byte-for-byte the body of residual_eval.signed_rank_from_vector."""
        v = np.zeros(P, dtype=np.float32)
        up = [j for j in np.argsort(-res)[:k].tolist() if res[j] > 0]
        dn = [j for j in np.argsort(res)[:k].tolist() if res[j] < 0]
        for blk, sgn in ((up, 1.0), (dn, -1.0)):
            for r, j in enumerate(blk, 1):
                v[j] = sgn / np.log2(r + 1.0)
        return v


def _format_prompt_local(cell_line_name, drug_name, dose_str, moa, control_cell_sentence=None):
    """Local copy of tahoe_c2s_preprocess_endcell_v2.format_prompt (line 460), including its
    coercion of missing / 'unknown' / 'nan' / 'None' moa to the literal string 'unclear'. Used only
    when the preprocess module cannot be imported (it pulls in `datasets` at module scope). When
    the real one IS importable we use it and assert the two agree byte for byte."""
    if not moa or moa == "unknown" or moa == "nan" or moa == "None":
        moa = "unclear"
    prompt = (f"Predict the response of {cell_line_name} to {drug_name} "
              f"at {dose_str}. Mechanism: {moa}.")
    if control_cell_sentence:
        prompt += f"\nControl cell: {control_cell_sentence}"
    prompt += "\n\nResponse cell:"
    return prompt


def get_format_prompt():
    """Prefer the real preprocessor's formatter; fall back to the local copy. Never edits either."""
    try:
        import tahoe_c2s_preprocess_endcell_v2 as pp
        f = pp.format_prompt
        probe = ("A549", "Vorinostat", "0.5 uM", None, "GENE1 GENE2 [END_CELL]")
        if f(*probe) != _format_prompt_local(*probe):
            raise RuntimeError("format_prompt drifted from the local copy; refusing to guess")
        _SOURCES["format_prompt"] = "tahoe_c2s_preprocess_endcell_v2.format_prompt"
        return f
    except Exception as e:
        _SOURCES["format_prompt"] = f"local copy (import failed: {e})"
        return _format_prompt_local


def control_from_prompt(prompt):
    """Copied verbatim from mechanistic_drug_probe.control_from_prompt (that function is correct)."""
    marker = "Control cell:"
    i = prompt.find(marker)
    if i == -1:
        return ""
    rest = prompt[i + len(marker):]
    j = rest.find("\n")
    return (rest if j == -1 else rest[:j]).strip()


# ---------------------------------------------------------------------------- inference fallbacks
def _spearman_brown(r_half):
    if inf is not None:
        return inf.spearman_brown(r_half)
    if r_half is None or not np.isfinite(r_half) or r_half <= 0 or r_half >= 1:
        return None
    return float(2 * r_half / (1 + r_half))


def _disattenuate(observed, rel_a, rel_b=None):
    if inf is not None:
        return inf.disattenuate(observed, rel_a, rel_b)
    rb = rel_a if rel_b is None else rel_b
    if observed is None or rel_a is None or rb is None or rel_a <= 0 or rb <= 0:
        return None
    d = observed / math.sqrt(rel_a * rb)
    return None if not np.isfinite(d) or abs(d) > 1.0 else float(d)


def _tost(ci, margin):
    if inf is not None:
        return inf.tost(ci, margin)
    if ci is None or not np.isfinite(ci.get("lo", np.nan)):
        return None
    lo, hi = ci["lo"], ci["hi"]
    return {"margin": float(margin), "equivalent": bool(lo > -margin and hi < margin),
            "contains_zero": bool(lo <= 0 <= hi)}


def _disattenuation_status(observed, rel_a, rel_b):
    """Why `r_disattenuated` is None, as a separate field rather than as the same token.

    `_disattenuate` returns None both when a reliability was never measured and when the correction
    blew past 1.0 in absolute value. Those are opposite situations -- missing data versus a loud
    signal that the reliabilities are too small to support the correction at all -- and in the JSON
    they were indistinguishable and read as missing data. The return type of `_disattenuate` is
    deliberately NOT changed (selftest (h) and the inference.py delegation both depend on it); this
    is a sibling field.
    """
    if observed is None or rel_a is None or rel_b is None or rel_a <= 0 or rel_b <= 0:
        return "inputs_unavailable"
    d = observed / math.sqrt(rel_a * rel_b)
    if not np.isfinite(d) or abs(d) > 1.0:
        return "correction_exceeded_unity"
    return "ok"


def _ci_disattenuated(ci, rel_a, relb_lo, relb_hi):
    """The disattenuated interval, folding in rel_B's own bootstrap uncertainty.

    The reported multiplier is not a constant: with rel_A = 0.90 and rel_B = 0.30 CI [0.22, 0.40],
    1/sqrt(rel_A*rel_B) alone spans 1.67x to 2.25x, and the pre-registration mandates quoting the
    disattenuated value precisely in the 0.20-0.40 band where it is least stable. Quoting it as a
    bare point estimate to three decimals there is false precision.

    Implemented as interval arithmetic over the box [ci.lo, ci.hi] x [relb_lo, relb_hi], which is
    conservative in every sign configuration and reduces to [lo/sqrt(rel_A*hi), hi/sqrt(rel_A*lo)]
    when the interval is entirely positive. It can only make a claim look WEAKER. It is conditional
    on rel_A, which carries no interval by its own declaration ("no interval is claimed").
    """
    if ci is None or rel_a is None or rel_a <= 0:
        return None
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    rls = [r for r in (relb_lo, relb_hi) if r is not None and r > 0]
    if not rls:
        return None
    cand = [e / math.sqrt(rel_a * r) for e in (float(lo), float(hi)) for r in rls]
    return {"lo": float(min(cand)), "hi": float(max(cand)),
            "rel_A_used": float(rel_a), "rel_B_lo": min(rls), "rel_B_hi": max(rls),
            "unit": "disattenuated interval; conditional on rel_A, which carries no interval",
            "note": ("widened over rel_B's own bootstrap endpoints, so it is conservative by "
                     "construction and cannot make the claim look stronger")}


def _tost_disattenuated(ci, rel_a, rel_b, margin=0.30):
    """Equivalence at `margin` ON THE DISATTENUATED SCALE, without ever dividing by a near-zero
    reliability.

    Testing ci/sqrt(rel_A*rel_B) against `margin` and testing the RAW ci against
    margin_eff = margin * sqrt(rel_A * rel_B) are algebraically the same statement, but the second
    never forms the ratio, so a reliability at the floor degrades the VERDICT rather than exploding
    the arithmetic.

    Why this had to move off the attenuated scale: at n = 60 the dyadic CI half-width is ~0.045, so
    `equivalent: true` on the raw interval fires for any |r_partial| < 0.255 -- which at
    rel_A 0.7 / rel_B 0.35 is compatible with a TRUE |r| of 0.61, and at 0.5 / 0.25 with 0.85. The
    design was scrupulous about disattenuating the point estimate and then tested equivalence on
    the undisattenuated interval, contradicting its own logic.

    Three outcomes, and the third is not a failure of the world but of the sample:
      equivalent                  -- the interval lies entirely inside +/- margin_eff;
      not_equivalent              -- the interval lies entirely OUTSIDE it, so the true correlation
                                     is demonstrably larger than the margin;
      underpowered_for_equivalence-- the interval straddles the boundary, or a reliability is
                                     unavailable, so neither statement can be made.
    """
    out = {"margin_declared": float(margin), "scale": "disattenuated",
           "rel_A_used": None if rel_a is None else float(rel_a),
           "rel_B_used": None if rel_b is None else float(rel_b)}
    if (ci is None or rel_a is None or rel_b is None or rel_a <= 0 or rel_b <= 0
            or not np.isfinite(ci.get("lo", np.nan)) or not np.isfinite(ci.get("hi", np.nan))):
        out.update({"status": "underpowered_for_equivalence", "margin_effective": None,
                    "note": "rel_A or rel_B unavailable, or no usable interval"})
        return out
    m_eff = float(margin) * math.sqrt(float(rel_a) * float(rel_b))
    lo, hi = float(ci["lo"]), float(ci["hi"])
    out["margin_effective"] = m_eff
    out["ci_tested"] = [lo, hi]
    if lo > -m_eff and hi < m_eff:
        out["status"] = "equivalent"
    elif lo >= m_eff or hi <= -m_eff:
        out["status"] = "not_equivalent"
    else:
        out["status"] = "underpowered_for_equivalence"
        out["note"] = ("the interval straddles the effective margin: it neither excludes a true "
                       "correlation of 0.30 nor demonstrates one")
    return out


def _crit(alpha=0.05, df=None):
    if inf is not None:
        return inf.crit(alpha, df)
    if df is None or df > 5000:
        return 1.959963984540054
    # crude Student-t two-sided 95% via Cornish-Fisher; only ever used in the fallback path
    z = 1.959963984540054
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    return float(z + g1 / df + g2 / df ** 2)


def _dyadic_cluster_ci(values, member_a, member_b, alpha=0.05, df_override=None):
    """Dyadic cluster-robust interval for the MEAN of `values`, where observation i belongs to the
    two nodes {member_a[i], member_b[i]} and two observations are dependent if their node sets
    intersect. Uses shared/inference.dyadic_cluster_ci when available; the `df_override` re-derives
    the critical value on the binding dimension (drugs), because the default n_nodes - 1 counts the
    wrong thing when the pooled object binds on groups.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 3:
        return None
    if inf is not None:
        out = inf.dyadic_cluster_ci(v, list(member_a), list(member_b), alpha)
        if out is None or not np.isfinite(out.get("lo", np.nan)):
            return out
        if df_override is not None and "se" in out and np.isfinite(out["se"]):
            k = _crit(alpha, df_override)
            out = dict(out)
            out["lo"] = float(out["point"] - k * out["se"])
            out["hi"] = float(out["point"] + k * out["se"])
            out["df"] = int(df_override)
            out["df_override_note"] = ("critical value recomputed on the binding dimension; "
                                       "inference.py was not modified")
        return out
    # -- fallback: the Fafchamps-Gubert form, same formula as inference.multiway_cluster_ci --
    e = v - v.mean()
    holds = defaultdict(list)
    for i, (a, b) in enumerate(zip(member_a, member_b)):
        holds[a].append(i); holds[b].append(i)
    if any(len(ix) >= n for ix in holds.values()):
        return {"point": float(v.mean()), "lo": float("nan"), "hi": float("nan"),
                "unit": "UNDEFINED: one node touches every observation"}
    total = 0.0
    for i, (a, b) in enumerate(zip(member_a, member_b)):
        nb = set(holds[a]) | set(holds[b])
        total += e[i] * float(e[list(nb)].sum())
    var = total / (n ** 2)
    if var <= 0:
        return None
    se = math.sqrt(var)
    df = df_override if df_override is not None else (len(holds) - 1)
    k = _crit(alpha, df)
    return {"point": float(v.mean()), "lo": float(v.mean() - k * se), "hi": float(v.mean() + k * se),
            "se": float(se), "n": n, "df": int(df), "unit": "dyadic cluster (fallback)"}


# =============================================================================================
# Numeric helpers
# =============================================================================================
def rankdata(x):
    """Average ranks, ties handled. (No scipy in this stack.)"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return None
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return None
    return float(a @ b / (na * nb))


def spearman(a, b):
    if len(a) < 3:
        return None
    return pearson(rankdata(a), rankdata(b))


def offdiag(M):
    """Upper triangle i<j, in row-major order. The canonical dyad vectorisation of this file."""
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return np.asarray(M)[iu]


def dyad_index(n):
    return np.triu_indices(n, 1)


def sym_from_offdiag(v, n):
    M = np.zeros((n, n), dtype=float)
    iu = np.triu_indices(n, 1)
    M[iu] = v
    return M + M.T


def _assert_finite_symmetric(M, name, tol=1e-9):
    """Only the UPPER TRIANGLE of every distance matrix is ever read (`offdiag`). An asymmetric
    matrix would therefore be silently half-ignored rather than rejected, and a NaN would travel
    into a comparison where it evaluates False and reads as a legitimate answer. Both are refused
    here, loudly, at the door of every estimator."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
    if not np.isfinite(M).all():
        bad = int(np.sum(~np.isfinite(M)))
        raise AssertionError(f"{name} contains {bad} non-finite entries; a NaN compared with < "
                             f"evaluates False and would be scored as a wrong answer rather than "
                             f"as missing data")
    if not np.allclose(M, M.T, atol=tol, rtol=0.0):
        worst = float(np.max(np.abs(M - M.T)))
        raise AssertionError(f"{name} is not symmetric (max |M - M.T| = {worst:.3e}); only the "
                             f"upper triangle is read, so the lower triangle would be discarded "
                             f"without warning")
    return M


def cosine_distance_matrix(X):
    """1 - cosine, rows of X are the objects. Zero diagonal, exactly symmetric."""
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < EPS] = 1.0
    Xn = X / nrm
    C = Xn @ Xn.T
    np.clip(C, -1.0, 1.0, out=C)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def cv(v):
    v = np.asarray(v, float)
    m = float(np.mean(v))
    return float(np.std(v) / m) if abs(m) > EPS else None


def _hash_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# =============================================================================================
# Control (confound) matrices
# =============================================================================================
_SALT_WORDS = {"hydrochloride", "dihydrochloride", "hcl", "citrate", "sulfate", "sulphate",
               "mesylate", "maleate", "tartrate", "sodium", "potassium", "calcium", "acetate",
               "phosphate", "besylate", "fumarate", "succinate", "hydrate", "dihydrate",
               "monohydrate", "trifluoroacetate", "tosylate", "bromide", "chloride", "oxalate",
               "malate", "lactate", "nitrate", "salt", "free", "base", "racemic"}


def normalized_base_name(name):
    """Strip parentheticals and salt/hydrate suffixes so Tofacitinib and 'Tofacitinib (citrate)'
    collapse. This is x8's key -- it exists because near-duplicate NAMES are near-duplicate
    MOLECULES, and a positive driven by those is not a discovery."""
    s = re.sub(r"\([^)]*\)", " ", str(name))
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s).lower().strip()
    toks = [t for t in s.split() if t and t not in _SALT_WORDS]
    return " ".join(toks) if toks else s


def char_ngrams(s, n=3):
    s = str(s).lower()
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def jaccard_distance(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 1.0 - (len(a & b) / u if u else 0.0)


def _pair_matrix(items, fn):
    n = len(items)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = float(fn(items[i], items[j]))
    return M


def build_control_matrices(drugs, name_token_sets, moas, name_tok_counts, prompt_tok_counts,
                           arm, dose_log10=None, plate_of=None):
    """All the dyadic covariates the partial has to annihilate.

    x3 and x4 are NEVER collapsed into one 'shared mechanism' indicator: with about half the roster
    annotated 'unclear', a single indicator codes ~40% of dyads as mechanistically matched on shared
    IGNORANCE, and the selftest's DEGENERATE-MECHANISM world shows that miscoding manufactures a
    spurious partial correlation with the wrong sign.
    """
    n = len(drugs)
    C = {}
    C["x1"] = _pair_matrix(name_token_sets, jaccard_distance)
    C["x1p"] = _pair_matrix([char_ngrams(d) for d in drugs], jaccard_distance)
    if arm == "annotated":
        moa_tok = [set(re.split(r"[^0-9a-zA-Z]+", str(m).lower())) - {""} for m in moas]
        C["x2"] = _pair_matrix(moa_tok, jaccard_distance)
        C["x3"] = _pair_matrix(list(moas), lambda a, b: 1.0 if (a == b and a != "unclear") else 0.0)
        C["x4"] = _pair_matrix(list(moas), lambda a, b: 1.0 if (a == b == "unclear") else 0.0)
    C["x5a"] = _pair_matrix(list(name_tok_counts), lambda a, b: abs(a - b))
    C["x5b"] = _pair_matrix(list(prompt_tok_counts), lambda a, b: abs(a - b))
    C["x8"] = _pair_matrix([normalized_base_name(d) for d in drugs],
                           lambda a, b: 1.0 if a == b else 0.0)
    if dose_log10 is not None:
        C["x6"] = _pair_matrix(list(dose_log10), lambda a, b: abs(a - b))
    if plate_of is not None:
        # x9: same-plate indicator. Under cell_line_plate scope every drug shares one plate, so x9
        # is constant and `_design` drops it as zero-variance -- plate batch is eliminated by
        # stratification there. Under cell_line scope the roster spans plates, B carries plate batch
        # structure in the pseudobulk itself, and x9 is the dyadic adjustment for it. Note that item
        # 1 removed the OTHER stated reason for pinning the plate: the residual is not defined
        # against that plate's DMSO pool, because the DMSO term cancels identically.
        C["x9"] = _pair_matrix(list(plate_of), lambda a, b: 1.0 if str(a) == str(b) else 0.0)
    for k, M in C.items():
        assert M.shape == (n, n), k
    return C


def control_diagnostics(controls, design_names, X):
    """Variance, fraction of nonzero dyads, and VIF in the actual design. The reported item that
    matters most: if x1 has nonzero variance on fewer than 2% of dyads it did not adjust anything,
    and r_partial must not be described as name-adjusted."""
    out = {}
    for k, M in controls.items():
        v = offdiag(M)
        out[k] = {"variance": float(np.var(v)),
                  "frac_nonzero": float(np.mean(np.abs(v) > EPS)),
                  "in_design": bool(k in design_names),
                  "vif": None}
    # X's column 0 is the INTERCEPT and design_names does not name it, so design_names[pos] is
    # X[:, pos + 1]. Indexing X[:, pos] (as this did) shifted every VIF by one control: x1's slot
    # got the intercept (zero variance -> skipped -> reported as None), each remaining control was
    # handed its PREDECESSOR's VIF, and the last control never got one at all. Verified on the
    # selftest roster, where a control with a true VIF of 1e6 was reported at 2.3.
    for pos, k in enumerate(design_names):
        col = pos + 1
        others = [p for p in range(X.shape[1]) if p != col]
        y = X[:, col]
        if len(others) == 0 or np.std(y) < EPS:
            continue
        Z = X[:, others]
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        r = y - Z @ beta
        ss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum(r ** 2)) / ss if ss > EPS else 0.0
        out[k]["vif"] = float(1.0 / max(1e-6, 1.0 - r2))
    return out


# =============================================================================================
# Activation distance matrix A
# =============================================================================================
def kaveraged_centered(H, mode="centered"):
    """The (n_drugs, hidden) matrix every downstream statistic is built from: within-draw centering
    across drugs, then the average over the K draws, then the metric's own rescaling. Factored out
    so the RSA and the direct decodability arm provably consume the SAME object."""
    H = np.asarray(H, dtype=np.float64)
    Hbar = (H - H.mean(axis=0, keepdims=True)).mean(axis=1)
    if mode == "zscored":
        sd = Hbar.std(axis=0, keepdims=True)
        sd[sd < EPS] = 1.0
        Hbar = Hbar / sd
    elif mode != "centered":
        raise ValueError(f"unknown mode {mode}")
    return Hbar


def activation_matrix(H, mode="centered", spectrum=False):
    """H: (n_drugs, K, hidden) float32 activations, K balanced control draws per drug.

    1. WITHIN-DRAW centering across drugs: h~[d,k] = h[d,k] - mean_d h[d,k]. This removes the shared
       prompt context and the fixed control sentence k EXACTLY (it is the same sentence for every
       drug, by construction of the balanced design), and it removes the massive-activation offset.
       It is NOT correlation distance, which centers each vector across its own hidden dimensions
       and is arbitrary.
    2. average over the K draws;
    3. A = 1 - cos.

    Returns (A, diagnostics). `diagnostics["status"] == "degenerate_by_construction"` when the
    off-diagonal has no spread at all -- e.g. layer 0 at the last position, where the token is ':'
    for every prompt and rotary position encoding lives inside attention, so the vectors are
    bit-identical. That cell is recorded, never folded into an aggregate as 0.0 or NaN.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 3:
        raise ValueError(f"activation_matrix expects (n_drugs, K, hidden), got {H.shape}")
    n, K, _ = H.shape
    Hm = H.mean(axis=1)
    A_unc = cosine_distance_matrix(Hm)
    unc = offdiag(1.0 - A_unc)
    diag = {"mode": mode, "n_draws": int(K),
            "uncentered_mean_cos": float(np.mean(unc)),
            "uncentered_cv": cv(1.0 - unc)}
    Hbar = kaveraged_centered(H, mode)
    if spectrum:
        # THE DIAGNOSTIC THAT MAKES A NULL ATTRIBUTABLE. RSA on centered cosine is a
        # variance-weighted second-order statistic: it sees whatever directions carry the most
        # across-drug variance. Reproduced independently: with the biology held perfectly
        # decodable, r_partial falls +0.946 -> +0.255 -> +0.093 -> +0.038 as the participation
        # ratio falls 9.41 -> 5.13 -> 3.36 -> 2.93, while leave-one-out ridge from the SAME
        # activations holds at r = +0.992 throughout. Without the spectrum in the JSON a reader
        # cannot tell "no structure present" from "structure crushed by a handful of dominant
        # lexical directions", and those license opposite conclusions about whether the failure is
        # fixable. One SVD of a (n_drugs x hidden) matrix -- milliseconds.
        s = np.linalg.svd(Hbar, compute_uv=False)
        s2 = s ** 2
        tot = float(np.sum(s2))
        diag["spectrum"] = {
            "singular_values_top10": [float(x) for x in s[:10]],
            "top_eig_share": float(s2[0] / tot) if tot > EPS else None,
            "participation_ratio": (float(tot ** 2 / float(np.sum(s2 ** 2)))
                                    if float(np.sum(s2 ** 2)) > EPS else None),
            "n_singular_values": int(s.size),
            "computed_on": f"Hbar after within-draw centering and K-averaging, mode={mode}",
            "reading_rule": ("a null r_partial accompanied by participation_ratio < 5 or "
                             "top_eig_share > 0.35 may NOT be reported as absence of organisation, "
                             "only as absence of organisation in the variance-dominant subspace"),
        }
    A = cosine_distance_matrix(Hbar)
    od = offdiag(A)
    if not np.isfinite(od).all() or float(np.std(od)) <= 1e-12:
        diag["status"] = "degenerate_by_construction"
        diag["offdiag_std"] = float(np.std(od)) if np.isfinite(od).all() else None
        return A, diag
    diag["status"] = "ok"
    diag["offdiag_std"] = float(np.std(od))
    diag["cv_A"] = cv(od)
    if diag["uncentered_cv"] is not None and diag["uncentered_cv"] < 0.02:
        diag["note_uncentered"] = ("the uncentered geometry is degenerate; the centered matrix is "
                                   "the only usable object")
    return A, diag


def reliability_A(H, mode="centered", n_splits=25, seed=11):
    """Split the K control draws into disjoint halves, build A from each, correlate off-diagonals,
    Spearman-Brown up to the full K. Averaged over `n_splits` random halvings. This is the number
    that makes the two-sided disattenuation legitimate: the activation matrix is NOT noiseless once
    the control sentence varies."""
    H = np.asarray(H, dtype=np.float64)
    n, K, _ = H.shape
    if K < 4:
        return None
    rng = np.random.RandomState(seed)
    vals = []
    half = K // 2
    for _ in range(n_splits):
        p = rng.permutation(K)
        A1, d1 = activation_matrix(H[:, p[:half], :], mode)
        A2, d2 = activation_matrix(H[:, p[half:2 * half], :], mode)
        if d1.get("status") != "ok" or d2.get("status") != "ok":
            continue
        r = spearman(offdiag(A1), offdiag(A2))
        if r is not None:
            vals.append(r)
    if not vals:
        return None
    sb = _spearman_brown(float(np.mean(vals)))
    return {"half_split_spearman": float(np.mean(vals)), "spearman_brown": sb,
            "n_splits": len(vals), "clustered_on": "control-sentence draw (K half-splits)",
            "unit": "point estimate, Spearman-Brown corrected; no interval is claimed"}


def control_draw_invariant(H, tol=1e-10):
    """Are the K control draws IDENTICAL for every drug, to machine precision?

    They are, by causal structure, at every read position that sits BEFORE the control sentence in
    a decoder-only model: 'name' and 'instr' are upstream of '\\nControl cell: ...', so varying k
    cannot change those activations at all. Detected empirically rather than by hardcoding position
    names, so a template change cannot silently invalidate it.
    """
    H = np.asarray(H, dtype=np.float64)
    dev = float(np.max(np.abs(H - H.mean(axis=1, keepdims=True)))) if H.size else 0.0
    scale = max(1.0, float(np.max(np.abs(H))) if H.size else 1.0)
    return bool(dev <= tol * scale), float(dev / scale)


def rel_A_for(H, mode="centered", seed=11, n_splits=25):
    """rel_A, with the by-construction case handled instead of mis-estimated.

    On control-draw-invariant activations the half-split correlation is 1.0 EXACTLY (verified by
    execution: 1.0000000000000002), and `spearman_brown` rejects r_half >= 1 and returns None -- so
    r_disattenuated came back None at 'name' and 'instr' but a number at 'last', which silently
    made the pre-registered instr-minus-last gap uncomputable on the disattenuated scale. The
    behaviour was also bit-fragile: a half-split of 0.9999999999999998 would have returned ~1.0
    instead, so the field's value depended on floating-point luck.

    Where the K draws are identical by construction the reliability is 1.0, not unmeasurable.
    """
    inv, rel_dev = control_draw_invariant(H)
    if inv:
        return {"spearman_brown": 1.0, "half_split_spearman": 1.0,
                "source": "control_draw_invariant_by_causal_structure",
                "max_rel_draw_deviation": rel_dev,
                "note": ("the control sentence is causally downstream of this read position in a "
                         "decoder-only model, so the K draws are identical by construction; "
                         "reliability is 1.0, not unmeasurable. NOTE that this makes the read "
                         "positions differently attenuated: 'last' carries rel_A < 1 from "
                         "control-sentence sensitivity while 'instr' and 'name' carry rel_A = 1, "
                         "so any raw across-position comparison is biased toward the upstream "
                         "position for purely measurement reasons."),
                "unit": "exact by causal structure; no interval is claimed"}
    out = reliability_A(H, mode, n_splits=n_splits, seed=seed)
    if out is not None:
        out["source"] = "half_split_over_control_draws"
        out["max_rel_draw_deviation"] = rel_dev
        # The same artefact one step out: when the draws VARY but the half-split correlation still
        # lands at 1.0, spearman_brown rejects r_half >= 1 and returns None -- so a MORE reliable
        # activation matrix yields a MISSING reliability and silently drops the disattenuated scale.
        # A half-split at unity means reliability 1.0; it is not missing data.
        if (out.get("spearman_brown") is None
                and out.get("half_split_spearman") is not None
                and out["half_split_spearman"] >= 1.0 - 1e-9):
            out["spearman_brown"] = 1.0
            out["source"] = "half_split_at_unity"
            out["note"] = ("the half-split correlation over the K control draws reached 1.0, so "
                           "Spearman-Brown is undefined; the reliability is 1.0, not unmeasurable. "
                           "Returning None here would drop the disattenuated scale precisely where "
                           "the activation matrix is MOST reliable.")
    return out


# =============================================================================================
# Biological distance matrix B
# =============================================================================================
def biology_matrix(residuals, P, k=100, diag=False):
    """residuals: (n_drugs, P) continuous residual profiles -> signed-rank encoding -> cosine.

    The residual signed-rank frame is the frame the residual chapter grades in. Euclidean on raw
    pseudobulk is deliberately NOT an arm: mixing frames is an error the thesis names elsewhere.

    Two silent-collapse paths are guarded here. A non-finite residual entry sorts to the end of
    `argsort` and fails the `> 0` test, so it does not raise -- it just quietly drops out of the
    encoding. And a drug whose encoded profile is ALL ZERO (no gene clears the sign test) gets its
    norm replaced by 1.0 inside `cosine_distance_matrix`, giving it cosine 0 and therefore distance
    exactly 1.0 to every other drug: a fabricated, perfectly uninformative row that no downstream
    check would flag. Both are counted and, for non-finite input, refused outright.
    """
    R = np.asarray(residuals, dtype=np.float64)
    if not np.isfinite(R).all():
        bad = int(np.sum(~np.isfinite(R)))
        raise AssertionError(f"residual profiles contain {bad} non-finite entries; the signed-rank "
                             f"encoding would drop them silently rather than fail")
    S = np.stack([signed_rank_from_vector(R[i].astype(np.float32), P, k) for i in range(R.shape[0])])
    n_empty = int(np.sum(np.linalg.norm(S, axis=1) < EPS))
    if n_empty:
        logger.warning(f"biology_matrix: {n_empty} drug(s) encode to an ALL-ZERO signed-rank "
                       f"profile; each is assigned distance exactly 1.0 to every other drug, which "
                       f"is a fabricated row, not a measurement")
    B = cosine_distance_matrix(S)
    if diag:
        return B, S, {"n_empty_profiles": n_empty, "n_drugs": int(S.shape[0])}
    return B, S


def residual_profiles(pb, dmso, generic):
    """res(d) = pseudobulk(d) - plate DMSO pseudobulk - cell-line generic shift.

    generic scope is the CELL LINE, matching residual_eval's generic_scope='cell_line' truth config.

    THE DMSO TERM CANCELS IDENTICALLY, and nothing downstream may claim otherwise. `build_biology`
    forms `generic = mean_d(pb(d) - dmso)`, so

        res = pb - dmso - (mean_d pb - dmso) = pb - mean_d pb

    i.e. the frame is ROSTER-MEAN CENTERING and the plate-DMSO referencing is a no-op. Verified
    numerically: replacing `dmso` with an entirely different vector moves `res` by 5.6e-15, and
    max|res - (pb - pb.mean(0))| = 4.4e-16. The subtraction is retained because it documents the
    frame residual_eval uses, but it is NOT a plate-matched control and must not be reported as one.

    What the frame DOES buy, and this part is real: a leave-one-out generic differs from this
    leave-one-in generic by the factor m/(m-1) applied identically to every drug, so B is
    byte-identical under cosine (verified max |B_LOO - B_LOI| = 0.0) and the -1/(m-1)
    self-inclusion artefact that 04c warns about does not arise here.
    """
    return np.asarray(pb, float) - np.asarray(dmso, float)[None, :] - np.asarray(generic, float)[None, :]


def activity_covariates(res_full):
    """a(d) = ||res(d)||_2 BEFORE the signed-rank encoding, and its two dyadic covariates.

    THE MECHANISM, which is exact and not a worry about a worry: res = pb - roster mean, so a drug
    with little effect has res ~= -(mean drug programme), and ALL weak drugs therefore collapse onto
    one another in B. In a planted world where drugs differ ONLY in effect magnitude and their
    drug-specific directions are iid random -- no pharmacology anywhere -- dyadic covariates built
    solely from activity magnitude explain ~0.85 of B's dyad-rank variance; at equal amplitudes they
    explain 0.000. With the atlas reporting median SNR 0.75 and 27% of conditions not significantly
    active, the real B carries this axis heavily.

    A referee's mundane reading of a positive is therefore "the model knows which compounds do
    something" -- one scalar, no drug-specific direction, no readout-rescue claim -- and nothing in
    the output otherwise distinguishes that from organised pharmacology.

      x7  = |rank a(d) - rank a(e)|   (do the two drugs differ in how much they did)
      x7b = min(rank a(d), rank a(e)) (is either of them inert)
    """
    a = np.linalg.norm(np.asarray(res_full, dtype=float), axis=1)
    ra = rankdata(a)
    return a, {"x7": _pair_matrix(list(ra), lambda p, q: abs(p - q)),
               "x7b": _pair_matrix(list(ra), lambda p, q: min(p, q))}


def activity_structure_of_B(B, C7):
    """How much of B is just activity magnitude, and how one-dimensional B is.

    R^2 is of B's DYAD RANKS on the ranks of {x7, x7b} alone -- the same rank frame the estimator
    works in. The leading-eigenvalue share comes from the classical-MDS Gram of B, which is the
    standard "how close to one-dimensional is this distance matrix" summary.
    """
    y = rankdata(offdiag(B))
    X = np.column_stack([np.ones(len(y)), rankdata(offdiag(C7["x7"])), rankdata(offdiag(C7["x7b"]))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / sst) if sst > EPS else None
    n = B.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    Gm = -0.5 * J @ (np.asarray(B, float) ** 2) @ J
    ev = np.linalg.eigvalsh(0.5 * (Gm + Gm.T))
    pos = ev[ev > 0]
    share = float(np.max(pos) / np.sum(pos)) if pos.size else None
    return {"r2_dyad_ranks_on_activity_only": r2,
            "leading_eigenvalue_share_of_B": share,
            "note": ("r2 is B's dyad ranks regressed on the ranks of {x7, x7b} ALONE. A high value "
                     "means a positive headline is compatible with 'the model separates active "
                     "compounds from inert ones', which is one scalar and licenses no readout claim. "
                     "In a planted magnitude-only world with iid drug directions this reads ~0.85; "
                     "at equal amplitudes, 0.000.")}


def reliability_B(half_profiles, P, k=100, n_boot=4000, seed=17):
    """half_profiles: list of (resA, resB) residual arrays, each (n_drugs, P), from disjoint random
    halves of each drug's cells. Spearman between the two half matrices, Spearman-Browned, averaged
    over the splits; interval by CLUSTER BOOTSTRAP OVER DRUGS (splits are not dyads, so a dyadic
    interval on the per-split values would be answering a different question).

    The bootstrap resamples drugs and indexes into the precomputed cosine matrices, which is exact:
    a distance depends only on the pair, so a resampled drug set selects a submatrix. Pairs where
    both positions are the same original drug are dropped (they are identically zero in both halves
    and would inflate the correlation).
    """
    mats = []
    for resA, resB in half_profiles:
        BA, _ = biology_matrix(resA, P, k)
        BB, _ = biology_matrix(resB, P, k)
        mats.append((BA, BB))
    if not mats:
        return None
    n = mats[0][0].shape[0]

    def stat(idx, want_half=False):
        idx = np.asarray(idx, dtype=int)
        m = len(idx)
        if m < 4:
            return None
        ii, jj = np.triu_indices(m, 1)
        keep = idx[ii] != idx[jj]
        if keep.sum() < 6:
            return None
        ii, jj = ii[keep], jj[keep]
        vals = []
        for BA, BB in mats:
            r = spearman(BA[idx[ii], idx[jj]], BB[idx[ii], idx[jj]])
            if r is not None:
                vals.append(r)
        if not vals:
            return None
        half = float(np.mean(vals))
        if want_half:
            return half
        sb = _spearman_brown(half)
        # A non-positive half-split correlation means the two halves of the same drugs' cells
        # disagree about the geometry. Spearman-Brown is undefined there; the honest report is a
        # reliability of ZERO, which trips the gate, not a `None` that reads as "not measured".
        return float(sb) if sb is not None else 0.0

    point = stat(np.arange(n))
    if point is None:
        return None
    half_point = stat(np.arange(n), want_half=True)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        v = stat(rng.randint(0, n, n))
        if v is not None:
            boot.append(v)
    out = {"rel_B": float(point), "half_split_spearman": float(half_point),
           "n_splits": len(mats), "n_boot_ok": len(boot),
           "floored_at_zero": bool(point == 0.0),
           "clustered_on": "drug",
           "unit": "cluster bootstrap over drugs, 95%"}
    if len(boot) < max(50, n_boot // 20):
        out.update({"lo": None, "hi": None,
                    "unit": "cluster bootstrap over drugs (too few valid draws for an interval)"})
        return out
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out.update({"lo": float(lo), "hi": float(hi), "boot_sd": float(np.std(boot))})
    return out


def rel_B_gate(relb):
    """The gate that is read BEFORE any headline. A null against an unreliable target licenses
    nothing, in either direction."""
    if relb is None or relb.get("rel_B") is None:
        return {"status": "rel_B_unavailable", "headline_permitted": False}
    # BOTH thresholds read the SAME quantity -- the bootstrap CI lower bound. Previously the 0.20
    # threshold read the lower bound (the conservative choice) while the 0.40 threshold read the
    # point estimate, so the gate was conservative about refusing and permissive about permitting.
    # Reading the lower bound throughout makes the headline HARDER to permit, never easier.
    lo = relb.get("lo")
    lo = relb["rel_B"] if lo is None else lo
    if lo <= 0.20:
        return {"status": "no_measurable_biological_geometry", "headline_permitted": False,
                "note": ("the drug-level biological distance matrix carries no reliable off-diagonal "
                         "structure at this cell budget, so NO geometry-matching test -- internal or "
                         "output -- can be run at drug resolution. This is itself a result, and the "
                         "existing output-side Mantels inherit the same caveat.")}
    if lo < 0.40:
        return {"status": "attenuation_limited", "headline_permitted": True,
                "note": ("quote only the disattenuated value, and only with the ceiling beside it. "
                         "The threshold is read on the bootstrap CI LOWER BOUND, the same quantity "
                         "the 0.20 refusal reads.")}
    return {"status": "ok", "headline_permitted": True,
            "note": "rel_B's bootstrap lower bound clears 0.40"}


# =============================================================================================
# THE ESTIMATOR: partial Spearman by MRM with a Freedman-Lane drug-label permutation null
# =============================================================================================
def _design(n_dyads, control_vecs):
    """[1, ranked controls], zero-variance columns dropped (and named, so the drop is reportable)."""
    cols, names, dropped = [np.ones(n_dyads)], [], []
    for k, v in control_vecs:
        if np.std(v) < EPS:
            dropped.append(k)
            continue
        cols.append(rankdata(v))
        names.append(k)
    return np.column_stack(cols), names, dropped


def _resid_maker(X, return_rank=False):
    """Residualiser onto the orthogonal complement of col(X): r(y) = y - U_k U_k' y, where U_k are
    the left singular vectors of X whose singular value clears the numerical rank tolerance.

    This was `Q, _ = np.linalg.qr(X); y - Q Q' y`, which is WRONG on a rank-deficient design.
    Unpivoted Householder QR still returns a full set of orthonormal columns, so the columns that
    correspond to the dependent regressors are arbitrary directions lying OUTSIDE col(X), and the
    projector strips genuine signal along them -- it over-adjusts, which biases the partial toward
    zero and therefore toward a false null.

    The condition is not hypothetical in this design. x2 (mechanism-token Jaccard) and x3 (shared
    known mechanism) are exactly affinely dependent whenever the annotated mechanisms are single
    tokens, and x5b (prompt-length difference) is exactly x5a (name-length difference) whenever the
    prompt varies only through the drug name -- which is precisely what the balanced design
    enforces. Measured on the selftest's own POSITIVE world the QR projector removed a direction
    lying 99.94% outside col(X). The coefficient moved by 2e-4 there, so this is a correctness
    repair rather than a result-changing one, but with the fix the deficiency is REPORTED
    (`design_rank_deficient`) instead of silently absorbed.
    """
    X = np.asarray(X, dtype=float)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    tol = max(X.shape) * np.finfo(float).eps * (float(s[0]) if s.size else 0.0)
    k = int(np.sum(s > tol))
    if k == 0:
        f = lambda y: np.array(y, dtype=float, copy=True)
    else:
        Uk = np.ascontiguousarray(U[:, :k])
        f = lambda y: y - Uk @ (Uk.T @ y)
    return (f, k) if return_rank else f


def mrm_partial(A, B, controls, n_perm=4000, seed=0, perms=None, drugs=None,
                collapse_x3_x4=False):
    """Partial Spearman of offdiag(A) on offdiag(B), controlling for the dyadic covariates.

    * everything is rank-transformed within the group first, which is what makes the coefficient a
      partial SPEARMAN and puts it on the same scale as the output-side Mantel numbers;
    * the coefficient is the correlation of the two residualised vectors, which is identical to the
      partial correlation implied by the full OLS but numerically better behaved;
    * the null is FREEDMAN-LANE: fit the reduced model rank(A) ~ [1, controls], keep the fitted part
      F and the residual R, re-form R as a SYMMETRIC MATRIX, permute its rows and columns by a DRUG
      permutation, and refit. Permuting the entries of a distance matrix elementwise is invalid --
      the entries are not exchangeable -- and is not done anywhere in this file;
    * `perms` may be supplied so that every cell in the run shares one permutation sequence. That is
      what makes the max-statistic family-wise correction and the PAIRED base-checkpoint contrast
      legitimate: the same relabeling is applied everywhere inside a draw.

    Returns everything a figure needs without a rerun, including the null distribution summary.
    """
    A = _assert_finite_symmetric(A, "activation matrix A")
    B = _assert_finite_symmetric(B, "biological matrix B")
    if A.shape != B.shape:
        raise ValueError(f"A {A.shape} and B {B.shape} disagree on the drug roster")
    n = A.shape[0]
    a = offdiag(A); b = offdiag(B)
    N = len(a)
    out = {"n_drugs": int(n), "n_dyads": int(N)}
    if n < 4 or np.std(a) < EPS or np.std(b) < EPS:
        out["status"] = "degenerate"
        return out

    ctrl_items = sorted(controls.items())
    if collapse_x3_x4 and "x3" in controls and "x4" in controls:
        merged = np.maximum(controls["x3"], controls["x4"])
        ctrl_items = [(k, v) for k, v in ctrl_items if k not in ("x3", "x4")]
        ctrl_items.append(("x34_collapsed", merged))
        ctrl_items = sorted(ctrl_items)
    ctrl_vecs = [(k, offdiag(M)) for k, M in ctrl_items]

    X, design_names, dropped = _design(N, ctrl_vecs)
    ar, br = rankdata(a), rankdata(b)
    res, design_rank = _resid_maker(X, return_rank=True)
    a_res, b_res = res(ar), res(br)

    r_raw = spearman(a, b)
    r_partial = pearson(a_res, b_res)
    out.update({"r_raw": r_raw, "r_partial": r_partial,
                "controls_in_design": design_names,
                "controls_dropped_zero_variance": dropped,
                "design_n_columns": int(X.shape[1]),
                "design_rank": int(design_rank),
                "design_rank_deficient": bool(design_rank < X.shape[1]),
                "x3_x4_collapsed": bool(collapse_x3_x4)})
    if design_rank < X.shape[1]:
        out["note_design_rank"] = (
            f"the control design is rank deficient ({X.shape[1]} columns, rank {design_rank}): at "
            f"least one control is an exact linear combination of the others (x2 vs x3 when the "
            f"mechanism strings are single tokens; x5b vs x5a when prompt length varies only "
            f"through the drug name). The projection is taken onto col(X) itself, so the redundant "
            f"columns cost degrees of freedom but do NOT over-adjust; the per-control VIFs report "
            f"which ones are redundant.")
    if dropped:
        out["note_dropped"] = (f"zero-variance regressors dropped, not silently retained: {dropped}")
    if "x1" in controls:
        # x1 is a Jaccard DISTANCE: a pair sharing NO name token scores 1.0, not 0.0. The previous
        # trigger, frac(|x1| > EPS) < 0.02, therefore fired only on the reverse pathology -- a
        # roster where almost every pair shares everything -- and could not fire on the realistic
        # case. Verified: on a roster where exactly 1 of 1770 pairs shares a token, frac_nonzero =
        # 1.0000 and var = 6.3e-5 > EPS, so the guard stayed silent and reported the name control
        # as ACTIVE while frac_sharing was 0.00056. The trigger now reads the quantity the note
        # describes: how many dyads carry any shared-token information at all.
        x1v = offdiag(controls["x1"])
        fz = float(np.mean(np.abs(x1v) > EPS))
        frac_sharing = float(np.mean(x1v < 1.0 - EPS))
        n_unique = int(np.unique(x1v).size)
        out["x1_frac_sharing"] = frac_sharing
        out["x1_frac_nonzero"] = fz
        out["x1_n_unique_offdiag"] = n_unique
        if "x1" in dropped or frac_sharing < 0.05 or float(np.var(x1v)) < EPS:
            out["name_control_inert"] = True
            out["note_x1"] = ("THE NAME CONTROL DID NOT ADJUST ANYTHING -- only "
                              f"{frac_sharing:.4f} of dyads share ANY name token (x1 < 1.0), the "
                              f"off-diagonal takes {n_unique} distinct value(s), and x1 has "
                              f"variance {float(np.var(x1v)):.3e} "
                              f"({'dropped from the design' if 'x1' in dropped else 'retained'}). "
                              "Essentially no drug pair in this roster shares a name token, so "
                              "r_partial must NOT be described as name-adjusted; the x1p "
                              "character-n-gram control is the only lexical adjustment actually "
                              "applied. Trigger: frac_sharing < 0.05, x1 dropped, or var < EPS.")
        else:
            out["name_control_inert"] = False
    if r_partial is None:
        out["status"] = "degenerate"
        return out

    # ---- influence-function dyadic interval (the raw values are not exchangeable; the influence
    # function is what carries the dyadic dependence into the variance) ----
    sa, sb = np.std(a_res), np.std(b_res)
    ci = None
    if sa > EPS and sb > EPS:
        ax, bx = a_res / sa, b_res / sb
        e = ax - r_partial * bx
        denom = float(np.mean(bx ** 2))
        psi = (bx * e) / denom if denom > EPS else None
        if psi is not None:
            ii, jj = dyad_index(n)
            vals = psi - float(np.mean(psi)) + float(r_partial)
            ci = _dyadic_cluster_ci(vals, [f"d{x}" for x in ii], [f"d{x}" for x in jj],
                                    df_override=n - 1)
            if ci is not None:
                ci["unit"] = "dyadic cluster on the drug pair, from the OLS influence function"
                ci["clustered_on"] = "drug"
                ci["df"] = int(n - 1)
                out["psi_sd"] = float(np.std(psi))
    out["ci"] = ci
    # NOT the licensing statistic. Kept, relabelled, so nothing is lost and nothing is confused:
    # the equivalence verdict that licenses the null branch is the DISATTENUATED one, computed in
    # analyse_cell where rel_A and rel_B are in scope.
    out["tost_margin_0.30_attenuated"] = _tost(ci, 0.30)
    out["note_tost_attenuated"] = (
        "this tests the ATTENUATED interval against 0.30 and is NOT the licensing statistic. With a "
        "dyadic half-width of ~0.045 at n = 60 it fires for any |r_partial| < 0.255, which at "
        "rel_A 0.7 / rel_B 0.35 is compatible with a true |r| of 0.61. Read "
        "tost_margin_0.30_disattenuated instead.")

    # ---- Freedman-Lane permutation null ----
    F = ar - a_res
    R_mat = sym_from_offdiag(a_res, n)
    iu = np.triu_indices(n, 1)
    bres_c = b_res - b_res.mean()
    bres_n = float(np.linalg.norm(bres_c))
    if perms is None:
        rng = np.random.RandomState(seed)
        perms = [rng.permutation(n) for _ in range(n_perm)]
    stats = np.empty(len(perms), dtype=float)
    n_perm_degenerate = 0
    for t, pi in enumerate(perms):
        rp = R_mat[np.ix_(pi, pi)][iu]
        rr = res(rp)
        rr = rr - rr.mean()
        d = float(np.linalg.norm(rr)) * bres_n
        val = (float(rr @ bres_c) / d) if d > EPS else np.nan
        # A draw whose residual vector has no spread has an UNDEFINED correlation. Writing 0.0 for
        # it (as this did, silently) narrows the null and makes p anti-conservative. The value is
        # still recorded as 0.0 -- dropping draws would change the denominator of the p-value -- but
        # the count is now reported, so a null built substantially out of undefined draws is
        # visible instead of invisible.
        if not np.isfinite(val):
            val = 0.0
            n_perm_degenerate += 1
        stats[t] = val
    n_ge = int(np.sum(np.abs(stats) >= abs(r_partial) - 1e-15))
    p = (1.0 + n_ge) / (len(perms) + 1.0)
    sd_null = float(np.std(stats))
    out.update({
        "p": float(p),
        "p_at_floor": bool(n_ge == 0),
        "n_perm": int(len(perms)),
        "n_perm_degenerate": int(n_perm_degenerate),
        "frac_perm_degenerate": float(n_perm_degenerate / max(1, len(perms))),
        "sd_null": sd_null,
        "MDE80": float(2.80 * sd_null),
        # The dyadic dependence is priced by the drug-label permutation itself, and this reads the
        # price off the output: a correlation estimated on N independent pairs has sd 1/sqrt(N-3),
        # so inverting sd_null says how many independent observations this null behaves like. The
        # ~1770 dyads from 60 drugs do NOT collapse to an effective n of 60 -- a dyad-noise world
        # measures 1766 -- so "underpowered by construction" is answerable from the JSON.
        "effective_n_implied": (float(1.0 / sd_null ** 2 + 3.0) if sd_null > EPS else None),
        "effective_n_implied_note": ("1/sd_null^2 + 3, the number of independent observations whose "
                                     "sampling sd would match this permutation null. Compare it to "
                                     "n_dyads (the upper bound if dyads were independent) and to "
                                     "n_drugs (the lower bound if each drug were one observation)."),
        "null": {"mean": float(np.mean(stats)), "sd": sd_null,
                 "q02.5": float(np.percentile(stats, 2.5)),
                 "q05": float(np.percentile(stats, 5)),
                 "q50": float(np.percentile(stats, 50)),
                 "q95": float(np.percentile(stats, 95)),
                 "q97.5": float(np.percentile(stats, 97.5)),
                 "min": float(np.min(stats)), "max": float(np.max(stats)),
                 "max_abs": float(np.max(np.abs(stats)))},
        "reduced_model_fit_sd": float(np.std(F)),
        "status": "ok",
    })
    out["_perm_stats"] = stats                 # stripped before serialisation; used for FWER/pairing
    out["_control_diag_design"] = (dict(ctrl_items), design_names, X)
    return out


def confound_residue_floor(controls, arm, n_rep=200, n_perm=199, seed=0, noise=0.06):
    """The magnitude below which a positive is inside an envelope this file has already measured.

    The Freedman-Lane p is a valid test of "residual of A after LINEAR-IN-RANKS control adjustment
    is independent of residual of B". That is NOT the null "there is no pharmacological
    organisation", and the gap between the two nulls is a real residual association that no
    permutation scheme repairs: a confound entering monotonically but not rank-linearly survives
    the adjustment. Measured type-I of the shipped estimator under a zero-pharmacology confounded
    null runs from 0.163 (rank-linear confound) to 0.707 (dominant exp-link confound), and expanding
    the control basis with quantile bins moved the latter only to 0.607. So p is not a licensing
    statistic here; magnitude against a measured envelope is the only one available.

    The file already knew this and said so in a `logger.info` inside selftest world (b3) -- "across
    generating seeds this world leaves a residue of up to |r| ~ 0.12" -- where no reader of the
    output JSON would ever see it. This computes the same envelope ON THE RUN'S OWN ROSTER, from
    the control matrices built out of the actual drug names and the model's own tokenizer, and
    writes it into the JSON.

    PER ARM, because the two arms have different control sets and therefore different envelopes:
    the annotated recipe carries a mechanism term that cannot exist in the blind arm, and the blind
    arm still has its own lexical residue. A fixed constant (0.12) would be a fact about a synthetic
    roster rather than about this one.
    """
    x1 = controls.get("x1")
    x1p = controls.get("x1p")
    x3 = controls.get("x3")
    if x1 is None or x1p is None:
        return {"status": "unavailable: x1 or x1p missing"}
    n = x1.shape[0]
    if arm == "annotated" and x3 is not None:
        gen = 0.55 * x1 + 0.30 * x1p + 0.15 * (1.0 - x3)
        recipe = "0.55*x1 + 0.30*x1p + 0.15*(1 - x3)   [selftest world (b3)]"
    elif arm == "blind":
        gen = 0.65 * x1 + 0.35 * x1p
        recipe = ("0.65*x1 + 0.35*x1p   [(b3) with the mechanism term dropped and the remaining "
                  "weights renormalised; x3 does not exist in the blind arm]")
    elif arm == "noise_only":
        gen = np.zeros((n, n), dtype=float)
        recipe = "no confound at all -- the calibration control, which must return the chance envelope"
    else:
        return {"status": f"unavailable: no recipe for arm {arm!r}"}
    np.fill_diagonal(gen, 0.0)
    prng = np.random.RandomState(seed + 313)
    perms = [prng.permutation(n) for _ in range(n_perm)]
    vals, ps = [], []
    for rep in range(n_rep):
        r = np.random.RandomState(seed + 7919 * rep + 101)
        A = gen + _sym_noise(n, r, noise); np.fill_diagonal(A, 0.0)
        B = gen + _sym_noise(n, r, noise); np.fill_diagonal(B, 0.0)
        m = mrm_partial(A, B, controls, perms=perms)
        if m.get("status") != "ok" or m.get("r_partial") is None:
            continue
        vals.append(abs(float(m["r_partial"]))); ps.append(float(m["p"]))
    if len(vals) < 10:
        return {"status": "unavailable: too few valid replicates", "n_ok": len(vals)}
    v = np.asarray(vals)
    return {"status": "ok", "arm": arm, "recipe": recipe,
            "mean_abs_r_partial": float(np.mean(v)),
            "p95_abs_r_partial": float(np.percentile(v, 95)),
            "max_abs_r_partial": float(np.max(v)),
            "frac_p_le_0.05": float(np.mean(np.asarray(ps) <= 0.05)),
            "n_replicates": int(len(vals)), "n_perm_per_replicate": int(n_perm),
            "noise_scale": float(noise), "n_drugs": int(n),
            "floor": float(np.percentile(v, 95)),
            "provenance": ("the 95th percentile of |r_partial| under a ZERO-PHARMACOLOGY confounded "
                           "null generated from THIS run's own control matrices, at this n and this "
                           "control set. A cell whose |r_partial| falls below it is inside an "
                           "envelope reachable with no pharmacology in the world, however small its "
                           "p and however it fares under the FWER correction."),
            "unit": "absolute partial Spearman"}


# =============================================================================================
# Co-primary: matched-triplet retrieval. Null exactly 0.50.
# =============================================================================================
def matched_triplets(B, controls, delta_mult=0.5, tol_frac=0.10, tol_x5a=2.0,
                     tol_x5b=8.0, max_triples=200000, seed=3):
    """Admit ordered triples (anchor, near, far) where biology separates near from far by at least
    delta = delta_mult * std(offdiag(B)), AND the two candidates are matched to the anchor on every
    confound: identical x3/x4 status, identical x8 near-duplicate status, name-token overlap,
    character-n-gram overlap, name length and prompt length all within tol. Matching holds the
    confound constant instead of adjusting it away, so it degrades gracefully at small n and its
    null is exactly 0.50 with no distributional assumption.

    x1p and x8 were previously partialled out of the MRM but NOT matched on here, which left the
    co-primary open to exactly the confound the primary was built to exclude: a triple whose `near`
    is a salt/hydrate variant of the anchor ('Tofacitinib' vs 'Tofacitinib (citrate)') would be
    admitted and scored as a pharmacology hit, when the docstring of `normalized_base_name` already
    says a positive driven by those "is not a discovery". Both are now matching keys. On the
    selftest roster the added constraints keep 13024 of 22201 triples; x8 costs nothing there and
    everything on a real roster that contains salt variants.

    THE TOLERANCES ARE A FRACTION OF THE OBSERVED SPREAD, NOT A CONSTANT. The previous defaults
    tol_x1 = tol_x1p = 0.05 were not matching: sd(offdiag(x1)) is ~0.085 on the selftest roster and
    sd(offdiag(x1p)) ~0.116 on a realistic 60-drug name roster, so 0.05 admitted candidates
    differing by 0.43-0.59 SD of the confound -- and because a triple is admitted only when B
    separates near from far, the two constraints COMPOSE into a rule that actively harvests
    confound-driven triples. Measured under a zero-pharmacology confounded null: accuracy 0.6297
    with triplet_p <= 0.05 in 40/40 replicates at tol 0.05, collapsing to 0.5058 and 0/40 at 0.01,
    which is 0.09-0.12 sd on these rosters -- i.e. exactly tol_frac = 0.10. On a clean positive the
    same tightening moved accuracy 0.5953 -> 0.5948 with power unchanged at 1.000. The fraction
    form is preferred over the constant because it adapts to the actual roster and is RECORDED.

    If the tightened tolerance drops admitted triples below the n_triples >= 200 floor, the honest
    report is `insufficient_matched_triples` for that cell. tol_frac is NOT loosened to recover it.
    """
    B = _assert_finite_symmetric(B, "biological matrix B (matched_triplets)")
    n = B.shape[0]
    delta = delta_mult * float(np.std(offdiag(B)))
    x1 = controls.get("x1")
    x1p = controls.get("x1p")

    def _sd(M):
        return None if M is None else float(np.std(offdiag(M)))

    sd_x1, sd_x1p = _sd(x1), _sd(x1p)
    tol_x1 = None if sd_x1 is None else max(tol_frac * sd_x1, 1e-12)
    tol_x1p = None if sd_x1p is None else max(tol_frac * sd_x1p, 1e-12)
    # How many of the seven matching keys actually CONSTRAIN anything on this roster. A cell matched
    # on effectively two keys must be readable as such rather than described as matched on seven.
    key_var = {}
    for kk in ("x1", "x1p", "x5a", "x5b", "x3", "x4", "x8"):
        M = controls.get(kk)
        key_var[kk] = None if M is None else float(np.var(offdiag(M)))
    n_keys_with_variance = int(sum(1 for v in key_var.values() if v is not None and v > EPS))
    matching_diag = {
        "tol_frac": float(tol_frac),
        "tol_x1": tol_x1, "sd_offdiag_x1": sd_x1,
        "tol_x1p": tol_x1p, "sd_offdiag_x1p": sd_x1p,
        "tol_x5a": float(tol_x5a), "tol_x5b": float(tol_x5b),
        "key_offdiag_variance": key_var,
        "n_matching_keys_with_variance": n_keys_with_variance,
        "note": ("tolerances on x1 and x1p are tol_frac * sd(offdiag) on THIS roster, not fixed "
                 "constants; keys with zero off-diagonal variance constrain nothing and are counted "
                 "here so a cell matched on two keys is not described as matched on seven."),
    }
    x5a = controls.get("x5a")
    x5b = controls.get("x5b")
    x3 = controls.get("x3")
    x4 = controls.get("x4")
    x8 = controls.get("x8")
    A_idx, N_idx, F_idx = [], [], []
    for a in range(n):
        gap = B[a][None, :] - B[a][:, None]                  # gap[i, j] = B[a,j] - B[a,i]
        ok = gap >= delta
        ok[a, :] = False; ok[:, a] = False
        np.fill_diagonal(ok, False)
        if x1 is not None:
            ok &= np.abs(x1[a][:, None] - x1[a][None, :]) <= tol_x1
        if x1p is not None:
            ok &= np.abs(x1p[a][:, None] - x1p[a][None, :]) <= tol_x1p
        if x5a is not None:
            ok &= np.abs(x5a[a][:, None] - x5a[a][None, :]) <= tol_x5a
        if x5b is not None:
            ok &= np.abs(x5b[a][:, None] - x5b[a][None, :]) <= tol_x5b
        if x3 is not None:
            ok &= (x3[a][:, None] == x3[a][None, :])
        if x4 is not None:
            ok &= (x4[a][:, None] == x4[a][None, :])
        if x8 is not None:
            ok &= (x8[a][:, None] == x8[a][None, :])
        ii, jj = np.nonzero(ok)
        if len(ii):
            A_idx.append(np.full(len(ii), a)); N_idx.append(ii); F_idx.append(jj)
    if not A_idx:
        return {"anchor": np.zeros(0, dtype=int), "near": np.zeros(0, dtype=int),
                "far": np.zeros(0, dtype=int), "delta": float(delta), "matching": matching_diag}
    a_i = np.concatenate(A_idx); n_i = np.concatenate(N_idx); f_i = np.concatenate(F_idx)
    if len(a_i) > max_triples:
        sel = np.random.RandomState(seed).choice(len(a_i), max_triples, replace=False)
        a_i, n_i, f_i = a_i[sel], n_i[sel], f_i[sel]
    return {"anchor": a_i, "near": n_i, "far": f_i, "delta": float(delta),
            "matching": matching_diag}


def triplet_stat(A, trip, n_boot=4000, n_perm=0, seed=5, multiway_cap=4000):
    """Fraction of admitted triples where the activation distance also puts `near` nearer than
    `far`. Interval: cluster bootstrap over ANCHOR DRUGS (the triples sharing an anchor are not
    independent). Null: drug-label permutation of A, same (1+r)/(1+n) floor convention."""
    if trip is None:
        return {"status": "insufficient_matched_triples", "n_triples": 0}
    A = _assert_finite_symmetric(A, "activation matrix A (triplet_stat)")
    a_i, n_i, f_i = trip["anchor"], trip["near"], trip["far"]
    T = len(a_i)
    if T < 200:
        return {"status": "insufficient_matched_triples", "n_triples": int(T),
                "matching": trip.get("matching"),
                "note": ("fewer than 200 admitted triples; the co-primary is DROPPED for this "
                         "cell. This is reported, never repaired by loosening tol_frac -- the "
                         "tolerance is a matching standard, not a yield dial.")}
    dn, df = A[a_i, n_i], A[a_i, f_i]
    correct = (dn < df).astype(float)
    acc = float(np.mean(correct))
    # Exact ties are scored as MISSES by the strict `<`, which biases the accuracy down (toward the
    # null) rather than up, so it is the safe direction -- but a cell whose accuracy is depressed by
    # ties rather than by geometry must be readable as such, so the tie rate is reported.
    frac_ties = float(np.mean(dn == df))
    anchors = np.unique(a_i)
    per = {a: (float(correct[a_i == a].sum()), float((a_i == a).sum())) for a in anchors}
    rng = np.random.RandomState(seed)
    boot = []
    keys = list(anchors)
    for _ in range(n_boot):
        take = rng.randint(0, len(keys), len(keys))
        num = sum(per[keys[t]][0] for t in take)
        den = sum(per[keys[t]][1] for t in take)
        if den > 0:
            boot.append(num / den)
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (None, None)

    # ---- multiway interval on {anchor, near, far} ----------------------------------------------
    # A triple involves THREE drugs, and each drug appears as near or far across many anchors, so
    # resampling anchors alone captures a strict subset of the dependence and the interval is
    # falsely narrow. inference.py's own docstring names this failure mode: "a dependence model that
    # captures a superset of another's dependence cannot be narrower. That is the diagnostic."
    #
    # The estimator is O(T * neighbourhood), which at T up to 200000 is ~10^9 set operations, so it
    # is computed on a random SUBSAMPLE of at most `multiway_cap` triples. Fewer observations give a
    # LARGER standard error, so the subsample is conservative; the interval is then recentred on the
    # full-sample accuracy, which is the reported statistic.
    multiway, method = None, "anchor cluster bootstrap (fallback)"
    if inf is not None and hasattr(inf, "multiway_cluster_ci") and T >= 3:
        try:
            if T > multiway_cap:
                sel = np.random.RandomState(seed + 7).choice(T, multiway_cap, replace=False)
            else:
                sel = np.arange(T)
            node_sets = [(f"d{a_i[i]}", f"d{n_i[i]}", f"d{f_i[i]}") for i in sel]
            mw = inf.multiway_cluster_ci(correct[sel], node_sets)
            if mw is not None and np.isfinite(mw.get("lo", np.nan)):
                shift = acc - float(mw["point"])
                multiway = {"point": acc, "lo": float(mw["lo"] + shift),
                            "hi": float(mw["hi"] + shift),
                            "se": mw.get("se"), "n_nodes": mw.get("n_nodes"),
                            "n_triples_used": int(len(sel)),
                            "clustered_on": "{anchor, near, far} -- all three drugs in the triple",
                            "unit": mw.get("unit"),
                            "note": ("computed on a random subsample when the triple count exceeds "
                                     "the cap (the estimator is quadratic in T); fewer observations "
                                     "widen the standard error, so this is conservative. Recentred "
                                     "on the full-sample accuracy.")}
                method = "inference.multiway_cluster_ci on {anchor, near, far}"
            else:
                method = "anchor cluster bootstrap (multiway returned no usable interval)"
        except Exception as e:                                                # pragma: no cover
            method = f"anchor cluster bootstrap (multiway failed: {e.__class__.__name__})"
    out = {"status": "ok", "triplet_acc": acc, "triplet_excess": acc - 0.5,
           "triplet_ci_multiway": multiway, "interval_method": method,
           "frac_exact_ties": frac_ties, "ties_scored_as": "miss (biases accuracy toward 0.50)",
           "n_triples": int(T), "n_anchors": int(len(anchors)), "delta": trip["delta"],
           "matching": trip.get("matching"),
           "triplet_ci": {"point": acc, "lo": lo, "hi": hi, "unit": "cluster bootstrap over anchors",
                          "clustered_on": "anchor drug", "n_boot": len(boot),
                          "note": ("ANCHOR-ONLY dependence: it captures a strict subset of the "
                                   "triple's dependence and is therefore too narrow. Read "
                                   "triplet_ci_multiway when it is available.")}}
    if n_perm > 0:
        nn = A.shape[0]
        prng = np.random.RandomState(seed + 1)
        ge = 0
        for _ in range(n_perm):
            pi = prng.permutation(nn)
            Ap = A[np.ix_(pi, pi)]
            v = float(np.mean(Ap[a_i, n_i] < Ap[a_i, f_i]))
            ge += abs(v - 0.5) >= abs(acc - 0.5) - 1e-15
        out["triplet_p"] = float((1.0 + ge) / (n_perm + 1.0))
        out["triplet_p_at_floor"] = bool(ge == 0)
        out["triplet_n_perm"] = int(n_perm)
    return out


# =============================================================================================
# THE DIRECT DECODABILITY ARM. Accompanies the RSA; does not replace it.
# =============================================================================================
_ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def _loo_dual_ridge_operator(Kmat, Yw, alphas=_ALPHA_GRID):
    """The leave-one-drug-out ridge as a LINEAR OPERATOR L, with Yhat_loo = L @ Y.

    Dual form throughout: the kernel K = X X^T is n x n, so nothing ever touches the hidden
    dimension and there is no new dependency beyond numpy. For each held-out drug i, alpha is chosen
    by an INNER leave-one-out on the remaining n-1 drugs over a fixed grid -- never fixed, because a
    fixed alpha shrinks low-variance directions and reimports precisely the pathology this arm
    exists to detect. The inner LOO uses the smoother shortcut
    loo_j = (Hy_j - H_jj y_j) / (1 - H_jj) with H = U diag(d/(d+alpha)) U', so one eigendecomposition
    per fold serves the whole grid.

    Yw must be Y in an orthonormal-row basis (Y = Yw @ Vt with Vt Vt' = I). Working there is exact
    and drops the cost from O(P) to O(n) per operation, which is what makes the permutation null
    affordable.
    """
    n = Kmat.shape[0]
    L = np.zeros((n, n), dtype=float)
    chosen = []
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        Ktr = Kmat[np.ix_(idx, idx)]
        Ytr = Yw[idx]
        d, U = np.linalg.eigh(0.5 * (Ktr + Ktr.T))
        d = np.clip(d, 0.0, None)
        G = U.T @ Ytr
        best_mse, best_a = None, alphas[0]
        for a in alphas:
            f = d / (d + a)
            Hd = (U * f) @ U.T
            hii = np.clip(np.diag(Hd), -np.inf, 1.0 - 1e-9)
            pred = Hd @ Ytr
            loo = (pred - hii[:, None] * Ytr) / (1.0 - hii)[:, None]
            mse = float(np.mean((loo - Ytr) ** 2))
            if best_mse is None or mse < best_mse:
                best_mse, best_a = mse, a
        Kinv = (U * (1.0 / (d + best_a))) @ U.T
        L[i, idx] = Kmat[i, idx] @ Kinv
        chosen.append(float(best_a))
    return L, chosen


def _pearson_rows_in_basis(Pw, Yw, m_v, P):
    """Row-wise Pearson between Pw @ Vt and Yw @ Vt, computed exactly in the reduced basis.

    Vt has orthonormal rows, so inner products and norms are preserved; the only thing the basis
    does not carry for free is the per-row mean over the P genes, and that is recovered exactly from
    m_v = Vt.mean(axis=1) since mean_p(u) = Pw . m_v.
    """
    mu = Pw @ m_v
    my = Yw @ m_v
    dot = np.sum(Pw * Yw, axis=1) - P * mu * my
    nu = np.sqrt(np.maximum(np.sum(Pw ** 2, axis=1) - P * mu ** 2, 0.0))
    ny = np.sqrt(np.maximum(np.sum(Yw ** 2, axis=1) - P * my ** 2, 0.0))
    den = nu * ny
    out = np.zeros(len(Pw))
    good = den > EPS
    out[good] = dot[good] / den[good]
    return out


def _retrieval_from_predictions(pred_full, S_true, P, k=100):
    """Encode each predicted residual in the SAME signed-rank frame B is built in, and ask whether
    its nearest true encoding is the held-out drug's. Null is exactly 1/n_drugs, with no
    distributional assumption -- the same virtue the matched-triplet co-primary has."""
    n = pred_full.shape[0]
    Sp = np.stack([signed_rank_from_vector(pred_full[i].astype(np.float32), P, k) for i in range(n)])
    nrm_p = np.linalg.norm(Sp, axis=1, keepdims=True); nrm_p[nrm_p < EPS] = 1.0
    nrm_t = np.linalg.norm(S_true, axis=1, keepdims=True); nrm_t[nrm_t < EPS] = 1.0
    C = (Sp / nrm_p) @ (S_true / nrm_t).T
    return float(np.mean(np.argmax(C, axis=1) == np.arange(n)))


def _name_feature_matrix(drugs, name_token_sets):
    """Character-3gram indicators plus this model's own BPE token indicators, across-drug centered.

    This is the confound control for the direct arm, and it is applied BY INCREMENT rather than by
    partialling: fit the identical LOO ridge from names alone and report how much the activations
    add over it. That is the direct-arm analogue of the (r_raw, r_partial) pair.
    """
    grams = sorted({g for d in drugs for g in char_ngrams(d)})
    gi = {g: i for i, g in enumerate(grams)}
    toks = sorted({t for s in name_token_sets for t in s})
    ti = {t: i for i, t in enumerate(toks)}
    Xn = np.zeros((len(drugs), len(grams) + len(toks)), dtype=float)
    for i, d in enumerate(drugs):
        for g in char_ngrams(d):
            Xn[i, gi[g]] = 1.0
        for t in name_token_sets[i]:
            Xn[i, len(grams) + ti[t]] = 1.0
    return Xn - Xn.mean(axis=0, keepdims=True)


def direct_decodability(Hbar, res_full, S_true, drugs, name_token_sets, P,
                        perms=None, n_perm=1000, k=100, alphas=_ALPHA_GRID):
    """Can a head fit on THIS representation predict the drug's transcriptional residual?

    RSA is a ONE-SIDED instrument: a positive licenses "organised", a null licenses nothing about
    organisation, because RSA on centered cosine is variance-weighted and structure in a low-variance
    subspace is invisible to it. Reproduced independently: with drug-varying nuisance raising the
    top-eigenvalue share to 0.42 and dropping the participation ratio to 2.93, RSA r_partial falls to
    +0.038 while leave-one-out ridge from the SAME activations to the SAME target holds at r = +0.992.
    The branch this thesis needs is the null branch, so the null branch needs an instrument that can
    support it.

    Held-out predictive accuracy of a head fit on the representation IS the estimand the discriminator
    names ("a better decoding head might rescue it"), rather than a proxy for it. Zero additional
    forward passes; it consumes activations the run already holds; numpy only.

    It ACCOMPANIES the RSA and does not replace it: the RSA fits nothing so it cannot overfit, its
    Freedman-Lane null is exact at n = 60, and it is invariant to how a head is parameterised, so a
    positive RSA remains the stronger and more general claim.
    """
    X = np.asarray(Hbar, dtype=np.float64)
    Y = np.asarray(res_full, dtype=np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n < 8 or Y.shape[0] != n:
        return {"status": "untestable_n_direct", "n_drugs": int(n)}
    # exact reduced basis for Y: Y = Yw @ Vt with Vt Vt' = I
    Uy, sy, Vt = np.linalg.svd(Y, full_matrices=False)
    keep = sy > (max(Y.shape) * np.finfo(float).eps * (sy[0] if sy.size else 0.0))
    Vt = Vt[keep]; Yw = (Uy[:, keep] * sy[keep])
    m_v = Vt.mean(axis=1)

    Xn = _name_feature_matrix(drugs, name_token_sets)
    kernels = {"activations": X @ X.T, "names": Xn @ Xn.T}
    out = {"status": "ok", "n_drugs": int(n), "retrieval_null": float(1.0 / n),
           "alpha_grid": [float(a) for a in alphas]}
    ops, scores = {}, {}
    for key, Km in kernels.items():
        sc = float(np.mean(np.diag(Km)))
        Km = Km / sc if sc > EPS else Km
        L, chosen = _loo_dual_ridge_operator(Km, Yw, alphas)
        Pw = L @ Yw
        corr = _pearson_rows_in_basis(Pw, Yw, m_v, P)
        ret = _retrieval_from_predictions(Pw @ Vt, S_true, P, k)
        ops[key] = (L, Km)
        scores[key] = {"mean_held_out_r": float(np.mean(corr)),
                       "median_held_out_r": float(np.median(corr)),
                       "retrieval_acc": ret,
                       "alphas_chosen_median": float(np.median(chosen)),
                       "alphas_chosen_min": float(np.min(chosen)),
                       "alphas_chosen_max": float(np.max(chosen))}
    out["activations"] = scores["activations"]
    out["names_only"] = scores["names"]
    out["increment_mean_r"] = float(scores["activations"]["mean_held_out_r"]
                                    - scores["names"]["mean_held_out_r"])
    out["increment_retrieval"] = float(scores["activations"]["retrieval_acc"]
                                       - scores["names"]["retrieval_acc"])
    out["note_centering"] = (
        "the leave-one-out is over the RIDGE FIT ONLY. X (across-drug centering, upstream in "
        "kaveraged_centered) and Y (centered here) are both centred over ALL n drugs including the "
        "held-out one, and the kernel is rescaled by the mean of its own full diagonal -- so the "
        "fold is transductive in its centering, not fully held out. At n = 60 the held-out drug "
        "contributes 1/60 of each mean, so the inflation is second order; and the permutation null "
        "runs through the IDENTICAL pipeline, so p_retrieval / p_mean_r / p_increment are valid "
        "against it. It is recorded rather than assumed negligible.")
    out["note_increment"] = (
        "the confound is controlled BY INCREMENT, not by partialling: the identical LOO ridge is fit "
        "from name features alone (character-3grams plus this model's own BPE token indicators) and "
        "the activations-over-names increment is the reported quantity. This is the direct-arm "
        "analogue of the (r_raw, r_partial) pair.")

    # ---- permutation null: relabel the drugs of Y. The LOO operator depends only on the kernel and
    # the chosen alphas, so each draw is one small matmul plus one re-encoding.
    if perms:
        use = perms[:int(n_perm)]
        L_act = ops["activations"][0]
        L_nam = ops["names"][0]
        stats_ret, stats_corr, stats_inc = [], [], []
        for pi in use:
            Yp = Yw[pi]
            Sp_true = S_true[pi]
            Pw_a = L_act @ Yp
            r_a = float(np.mean(_pearson_rows_in_basis(Pw_a, Yp, m_v, P)))
            ret_a = _retrieval_from_predictions(Pw_a @ Vt, Sp_true, P, k)
            Pw_n = L_nam @ Yp
            r_n = float(np.mean(_pearson_rows_in_basis(Pw_n, Yp, m_v, P)))
            stats_ret.append(ret_a); stats_corr.append(r_a); stats_inc.append(r_a - r_n)
        sr = np.asarray(stats_ret); sc_ = np.asarray(stats_corr); si = np.asarray(stats_inc)
        obs_ret = out["activations"]["retrieval_acc"]
        obs_corr = out["activations"]["mean_held_out_r"]
        obs_inc = out["increment_mean_r"]
        out["null"] = {
            "n_perm": int(len(use)),
            "retrieval": {"mean": float(sr.mean()), "q95": float(np.percentile(sr, 95)),
                          "max": float(sr.max())},
            "mean_r": {"mean": float(sc_.mean()), "q95": float(np.percentile(sc_, 95)),
                       "sd": float(sc_.std())},
            "increment": {"mean": float(si.mean()), "q95": float(np.percentile(si, 95)),
                          "sd": float(si.std())},
        }
        out["p_retrieval"] = float((1.0 + int(np.sum(sr >= obs_ret - 1e-15))) / (len(sr) + 1.0))
        out["p_mean_r"] = float((1.0 + int(np.sum(sc_ >= obs_corr - 1e-15))) / (len(sc_) + 1.0))
        out["p_increment"] = float((1.0 + int(np.sum(si >= obs_inc - 1e-15))) / (len(si) + 1.0))
        out["_perm_increment"] = si
        out["note_null"] = ("one-sided against the SHARED permutation sequence, permuting the drug "
                            "labels of Y; the same relabeling the RSA uses, so the two arms are "
                            "answerable to the same null.")
        # THE INCREMENT'S NULL IS NOT CENTRED AT ZERO, and reading p_increment alone would be a
        # mistake. Relabeling Y destroys BOTH relationships at once, and the two kernels do not
        # collapse to the same place: a name kernel that interpolates the training drugs perfectly
        # scores far WORSE under a relabeling than a smoother activation kernel does, which pushes
        # the null increment strongly negative. Measured in the planted name-only world: null
        # increment q95 = -0.93 while the observed increment is -0.41 -- so p_increment = 0.0099
        # even though the activations add nothing whatever over the names. The licensing condition
        # is therefore the CONJUNCTION: the increment must be positive AND beat its own null.
        out["increment_licensed"] = bool(obs_inc > 0.0 and out["p_increment"] <= 0.05)
        out["note_increment_licensing"] = (
            "increment_licensed = (increment > 0 AND p_increment <= 0.05). p_increment alone does "
            "NOT license 'the activations add over the names': the increment's permutation null is "
            "not centred at zero, because relabeling Y destroys both relationships and the two "
            "kernels do not collapse to the same floor. In a planted world where the names are a "
            "superset of what the activations carry, the observed increment is -0.41 against a null "
            "q95 of -0.93, giving p = 0.01 for a representation that adds nothing.")
    return out


# =============================================================================================
# Real data: load a tier, choose a (cell line, plate) group, build the biology side
# =============================================================================================
def sentence_to_expression_proxy(sentence, panel_index, P):
    """Cell sentence -> a length-P monotone-in-rank expression proxy, v[g] = 1/log2(rank+1), 0 for
    genes the sentence does not express. The SAME decay the signed-rank encoding uses, so the
    pseudobulk difference and its encoding live in one frame."""
    v = np.zeros(P, dtype=np.float32)
    r = 0
    for g in str(sentence).replace("[END_CELL]", " ").replace("[DOWN]", " ").split():
        gi = panel_index.get(g)
        if gi is None or v[gi] != 0:
            continue
        r += 1
        v[gi] = 1.0 / np.log2(r + 1.0)
    return v


_DOSE_RE = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(nm|um|mm|m)\b")
_DOSE_MULT_UM = {"nm": 1e-3, "um": 1.0, "mm": 1e3, "m": 1e6}


def parse_dose_um(s):
    """Dose string -> micromolar, or None. Handles nM, uM, the micro sign (U+00B5 and U+03BC), mM
    and plain M. Returns None rather than guessing, so the parse RATE is measurable."""
    if s is None:
        return None
    t = str(s).strip().lower().replace("µ", "u").replace("μ", "u")
    m = _DOSE_RE.search(t)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:                                                        # pragma: no cover
        return None
    if not np.isfinite(val) or val <= 0:
        return None
    return val * _DOSE_MULT_UM[m.group(2)]


def dose_frame_diagnostics(group_examples, drugs, modal_dose_str):
    """The dose frame mismatch, measured per drug instead of asserted away.

    A is DOSE-FIXED -- run() puts one modal dose string in every prompt -- while B pools every cell
    of a drug in the stratum REGARDLESS of dose. Dose is the largest single driver of transcriptional
    response magnitude in this atlas, so dose composition that varies across drugs puts a component
    in B that A cannot track. That is not eliminable without filtering cells, which costs exactly
    the n and rel_B this build order exists to protect; it is ADJUSTED dyadically instead, via x6 on
    each drug's median log10 dose.
    """
    per_drug, med_log10 = {}, []
    for d in drugs:
        exs = group_examples[d]
        raw = [e.get("metadata", {}).get("dose") for e in exs]
        vals = [parse_dose_um(x) for x in raw]
        okv = [v for v in vals if v is not None]
        n_at_modal = sum(1 for x in raw if x is not None and str(x) == str(modal_dose_str))
        per_drug[str(d)] = {
            "n_cells": len(exs),
            "n_distinct_dose_strings": int(len({str(x) for x in raw if x is not None})),
            "frac_cells_at_modal_dose_string": (float(n_at_modal / len(exs)) if exs else None),
            "median_dose_uM": float(np.median(okv)) if okv else None,
            "parse_rate": float(len(okv) / len(raw)) if raw else 0.0,
        }
        med_log10.append(math.log10(float(np.median(okv))) if okv else None)
    n_parsed = sum(1 for v in med_log10 if v is not None)
    rate = float(n_parsed / max(1, len(drugs)))
    out = {"drug_parse_rate": rate, "per_drug": per_drug,
           "modal_dose_string_in_prompt": str(modal_dose_str),
           "frac_cells_at_modal_dose_string_median": float(np.median(
               [v["frac_cells_at_modal_dose_string"] for v in per_drug.values()
                if v["frac_cells_at_modal_dose_string"] is not None])) if per_drug else None}
    if rate < 0.90:
        out.update({"dose_parse_failed": True, "x6_built": False,
                    "note": (f"only {rate:.3f} of drugs yielded a parseable dose; below the 0.90 "
                             f"threshold x6 is NOT built at all. A control constructed from a "
                             f"partially parsed covariate is worse than no control, because the "
                             f"unparsed drugs would enter as an arbitrary constant.")})
        return out, None
    # THE PARTIAL-PARSE PATH, which used to be a crash. At 0.90 <= rate < 1.0 this function returned
    # a list containing None, and `build_control_matrices` then evaluated abs(a - b) on it and raised
    # TypeError -- AFTER the checkpoint had been loaded and the whole forward sweep paid for.
    # Verified before the repair: rate 0.95 -> "TypeError: unsupported operand type(s) for -: 'float'
    # and 'NoneType'". The covariate must be TOTAL over the roster or absent; there is no third
    # option, because a dyadic design has no missing-value semantics.
    missing = [str(d) for d, v in zip(drugs, med_log10) if v is None]
    fill_src = "none_needed"
    if missing:
        modal_um = parse_dose_um(modal_dose_str)
        if modal_um is not None:
            fill = math.log10(float(modal_um))
            fill_src = "the modal dose string that the prompt itself fixes"
        else:
            fill = float(np.median([v for v in med_log10 if v is not None]))
            fill_src = "the roster median of the parsed per-drug median log10 doses"
        med_log10 = [fill if v is None else v for v in med_log10]
    out.update({"dose_parse_failed": False, "x6_built": True,
                "n_drugs_dose_imputed": len(missing),
                "drugs_dose_imputed": missing,
                "dose_imputation_source": fill_src,
                "note": ("x6 = |median log10 dose(d) - median log10 dose(e)| enters the control "
                         "design in BOTH arms. The mismatch is ADJUSTED, not eliminated.")})
    if missing:
        out["note_imputation"] = (
            f"{len(missing)} of {len(drugs)} drugs carry no parseable dose and were IMPUTED at "
            f"{fill_src} (log10 uM = {fill:.4f}). This is recorded rather than hidden because it is "
            f"NOT a conservative substitution: an imputed drug's x6 entries understate its true dose "
            f"distance to the rest of the roster, so dose is UNDER-adjusted for those pairs and any "
            f"residual dose confound survives there. The alternative -- dropping x6 entirely -- "
            f"under-adjusts for the whole roster, which is worse. The affected drugs are named in "
            f"drugs_dose_imputed so the pairs can be excluded by hand if the count is material.")
    return out, med_log10


def load_examples(eval_dir, tier):
    path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_panel(eval_dir, panel_file=None):
    cands = [panel_file, os.path.join(eval_dir, "l1000_panel.json"),
             os.path.join(_ROOT, "shared", "l1000_panel.json"),
             os.path.join(_HERE, "l1000_panel.json"), "l1000_panel.json"]
    for c in cands:
        if c and os.path.exists(c):
            panel = json.load(open(c))
            return panel, {g: i for i, g in enumerate(panel)}, c
    raise FileNotFoundError("l1000_panel.json not found in any known location")


def choose_group(examples, min_cells_per_drug, cell_line=None, plate=None,
                 scope="cell_line_plate"):
    """Groups are STRATA. Picks the group with the most drugs clearing the per-drug cell floor.

    Two scopes, and they are NOT interchangeable readings:

      * `cell_line_plate` (default) -- one (cell_line_id, plate) stratum. Plate batch is eliminated
        by stratification. Note that the frame's stated justification for pinning the plate is
        weaker than it looks: the residual is NOT defined against that plate's DMSO pool, because
        the DMSO term cancels identically (see `residual_profiles`). What survives is batch
        structure in the pseudobulk itself.
      * `cell_line` -- one cell line, pooling its plates, which is also what residual_eval's truth
        config uses (generic_scope='cell_line', loo=False). n_drugs is materially larger here, but
        B now spans plates, so plate batch enters the biological matrix and must be ADJUSTED
        dyadically via the x9 same-plate indicator rather than eliminated.
    """
    if scope not in ("cell_line_plate", "cell_line"):
        raise ValueError(f"unknown scope {scope!r}; expected cell_line_plate or cell_line")
    if scope == "cell_line" and plate is not None:
        raise ValueError("--plate is meaningless at --scope cell_line: the scope pools plates. "
                         "Pin one or the other, never both.")
    by = defaultdict(lambda: defaultdict(list))
    for e in examples:
        m = e.get("metadata", {})
        cl, pl = m.get("cell_line_id"), m.get("plate")
        d = m.get("drug")
        if cl is None or d is None:
            continue
        key = (cl, pl) if scope == "cell_line_plate" else (cl, "ALL_PLATES")
        by[key][d].append(e)
    scored = []
    for g, dd in by.items():
        if cell_line is not None and g[0] != cell_line:
            continue
        if plate is not None and str(g[1]) != str(plate):
            continue
        n_ok = sum(1 for d, v in dd.items() if len(v) >= min_cells_per_drug)
        scored.append((n_ok, len(dd), g))
    if not scored:
        return None, {}, {}
    scored.sort(reverse=True, key=lambda t: (t[0], t[1], str(t[2])))
    n_ok, _, g = scored[0]
    ranking = [{"group": [str(x) for x in gg], "n_drugs_over_floor": int(a), "n_drugs_total": int(b)}
               for a, b, gg in scored[:8]]
    return g, by[g], {"considered": len(scored), "ranking": ranking, "n_drugs_over_floor": int(n_ok)}


def wellsplit_profiles(group_examples, drugs, panel_index, P, seed, n_splits=25, min_wells=2):
    """Half-split profiles built from DISJOINT WELLS instead of disjoint cells of the same well.

    Drug and well do not cross in this atlas -- one drug is assigned per well, so wells NEST within
    drugs -- and therefore every between-drug distance in B is a between-well contrast. Any
    well-level technical structure is reproduced perfectly across two cell halves of the same well,
    and the cell-split rel_B counts that as signal. The thesis makes exactly this objection about
    its own transfer coefficient.

    This is REPORT ONLY and deliberately does not drive the gate: it covers only the ~62% of drugs
    that are multi-dose (one drug at one dose is one well), and a well split for a multi-dose drug
    also changes dose, whose transfer is measured elsewhere at 0.707. Gating on it could refuse a
    runnable experiment on a contaminated lower bound. Reporting it makes the inflation visible
    without letting a confounded number make the refusal decision.
    """
    by_well = {}
    for d in drugs:
        w = defaultdict(list)
        for e in group_examples[d]:
            s = e.get("metadata", {}).get("sample")
            if s is None:
                continue
            w[s].append(sentence_to_expression_proxy(e.get("response", ""), panel_index, P))
        if len(w) >= min_wells:
            by_well[d] = [np.stack(v) for v in w.values()]
    covered = sorted(by_well.keys(), key=str)
    if len(covered) < 6:
        return None, covered
    halves = []
    for s in range(n_splits):
        rng = np.random.RandomState(3000 + s * 7919 + seed)
        pa, pbb, ok = [], [], True
        for d in covered:
            wells = by_well[d]
            idx = rng.permutation(len(wells))
            h = max(1, len(wells) // 2)
            A_ = np.concatenate([wells[i] for i in idx[:h]])
            B_ = np.concatenate([wells[i] for i in idx[h:]])
            if A_.size == 0 or B_.size == 0:
                ok = False
                break
            pa.append(A_.mean(axis=0)); pbb.append(B_.mean(axis=0))
        if not ok:
            break
        pa, pbb = np.stack(pa), np.stack(pbb)
        z = np.zeros(P, dtype=float)
        halves.append((residual_profiles(pa, z, (pa - z[None, :]).mean(axis=0)),
                       residual_profiles(pbb, z, (pbb - z[None, :]).mean(axis=0))))
    return (halves or None), covered


def collect_cellline_controls(examples):
    """Per cell line: the control-cell sentences already carried in its prompts, and its name.

    No new data is needed for the positive control -- a treated well's prompt carries a control cell
    from the same line, and the eval jsonl covers every line in the tier.
    """
    by, names = defaultdict(list), {}
    for e in examples:
        m = e.get("metadata", {})
        cl = m.get("cell_line_id")
        if cl is None:
            continue
        c = control_from_prompt(e.get("prompt", ""))
        if c:
            by[cl].append(c)
            names.setdefault(cl, m.get("cell_line_name") or str(cl))
    return by, names


def build_cellline_biology(examples, panel_index, P, seed, min_cells=10, max_lines=60,
                           n_rel_splits=25):
    """THE POSITIVE CONTROL'S B, built through the IDENTICAL frame as the drug side.

    Each cell line's own control cells are pooled into a pseudobulk, then `residual_profiles` is
    called with a ZERO dmso vector and the matching generic -- which is not a shortcut but the exact
    algebra of the drug side, where the DMSO term cancels identically anyway (see
    `residual_profiles`). Then the same signed-rank cosine, the same disjoint-cell half-split
    reliability, the same everything.

    Why this control and not another: the sbatch names its absence as gap #1 and is explicit that
    without it "a null from this job is bounded by MDE80 and by rel_B, not by a demonstrated
    positive". rel_B bounds the B side; MDE80 bounds noise; NOTHING currently bounds whether centered
    cosine on this residual stream can see a geometry that is definitely there. Cell line is the
    ideal anchor and a genuinely MATCHED one: it enters through 1-3 tokens in the same prompt slot,
    against the same ~400-token context, so it exercises the identical variance-domination risk with
    a factor the thesis has independently shown the model encodes.
    """
    by, names = collect_cellline_controls(examples)
    elig = [(cl, len(v)) for cl, v in by.items() if len(v) >= min_cells]
    elig.sort(key=lambda t: (-t[1], str(t[0])))
    lines = sorted([cl for cl, _ in elig[:max_lines]], key=str)
    if len(lines) < 4:
        return None
    per_line = {cl: np.stack([sentence_to_expression_proxy(s, panel_index, P) for s in by[cl]])
                for cl in lines}
    pb = np.stack([per_line[cl].mean(axis=0) for cl in lines])
    zero = np.zeros(P, dtype=float)
    generic = (pb - zero[None, :]).mean(axis=0)
    res_full = residual_profiles(pb, zero, generic)
    B, S, bdiag = biology_matrix(res_full, P, diag=True)
    half_profiles = []
    for s in range(n_rel_splits):
        rng = np.random.RandomState(2000 + s * 7919 + seed)
        pa, pbb, ok = [], [], True
        for cl in lines:
            X = per_line[cl]
            if X.shape[0] < 2:
                ok = False
                break
            idx = rng.permutation(X.shape[0]); h = X.shape[0] // 2
            pa.append(X[idx[:h]].mean(axis=0)); pbb.append(X[idx[h:2 * h]].mean(axis=0))
        if not ok:
            break
        pa, pbb = np.stack(pa), np.stack(pbb)
        half_profiles.append((residual_profiles(pa, zero, (pa - zero[None, :]).mean(axis=0)),
                              residual_profiles(pbb, zero, (pbb - zero[None, :]).mean(axis=0))))
    return {"B": B, "S": S, "res_full": res_full, "half_profiles": half_profiles,
            "lines": lines, "line_names": [names[cl] for cl in lines],
            "n_cells": {str(cl): int(per_line[cl].shape[0]) for cl in lines},
            "encoding_diag": bdiag}


def build_biology(group_examples, drugs, panel_index, P, seed, n_rel_splits=25):
    """Per drug: pseudobulk of its response cells; per group: pseudobulk of the plate-matched DMSO
    control cells carried in the prompts; per cell line: the generic shift. Then the residual
    signed-rank cosine matrix, plus `n_rel_splits` disjoint half-splits of each drug's cells for the
    reliability ceiling."""
    ctrl_vecs, per_drug = [], {}
    for d in drugs:
        mats = []
        for e in group_examples[d]:
            resp = e.get("response", "")
            mats.append(sentence_to_expression_proxy(resp, panel_index, P))
            c = control_from_prompt(e.get("prompt", ""))
            if c:
                ctrl_vecs.append(sentence_to_expression_proxy(c, panel_index, P))
        per_drug[d] = np.stack(mats) if mats else None
    if not ctrl_vecs:
        raise RuntimeError("no control-cell sentences found in this group's prompts")
    dmso = np.stack(ctrl_vecs).mean(axis=0)
    pb = np.stack([per_drug[d].mean(axis=0) for d in drugs])
    generic = (pb - dmso[None, :]).mean(axis=0)
    res_full = residual_profiles(pb, dmso, generic)
    B, S, bdiag = biology_matrix(res_full, P, diag=True)

    half_profiles = []
    for s in range(n_rel_splits):
        rng = np.random.RandomState(1000 + s * 7919 + seed)
        pa, pbb = [], []
        ok = True
        for d in drugs:
            X = per_drug[d]
            if X.shape[0] < 2:
                ok = False
                break
            idx = rng.permutation(X.shape[0])
            h = X.shape[0] // 2
            pa.append(X[idx[:h]].mean(axis=0))
            pbb.append(X[idx[h:2 * h]].mean(axis=0))
        if not ok:
            break
        pa, pbb = np.stack(pa), np.stack(pbb)
        ga = (pa - dmso[None, :]).mean(axis=0)
        gb = (pbb - dmso[None, :]).mean(axis=0)
        half_profiles.append((residual_profiles(pa, dmso, ga), residual_profiles(pbb, dmso, gb)))

    counts = {d: int(per_drug[d].shape[0]) for d in drugs}
    return {"B": B, "S": S, "res_full": res_full, "dmso": dmso, "generic": generic,
            "half_profiles": half_profiles, "n_cells": counts, "encoding_diag": bdiag}


# =============================================================================================
# Prompts and activations
# =============================================================================================
def pick_control_sentences(group_examples, K, seed=7):
    """The SAME K control sentences for EVERY drug -- the balanced design. Because control sentence
    k is identical across drugs, its main effect is removed EXACTLY by the within-draw centering, so
    it cannot manufacture between-drug geometry; and K draws still give the activation matrix
    within-drug variance, which is what makes rel_A measurable."""
    sents = set()
    for d, exs in group_examples.items():
        for e in exs:
            c = control_from_prompt(e.get("prompt", ""))
            if c:
                sents.add(c)
    sents = sorted(sents)
    if not sents:
        raise RuntimeError("no control sentences available in this group")
    rng = np.random.RandomState(seed)
    if len(sents) >= K:
        idx = rng.choice(len(sents), K, replace=False)
        return [sents[i] for i in sorted(idx)], "plate_matched_exact_K"
    return sents, f"only_{len(sents)}_available"


def build_prompts(fmt, cell_line_name, drugs, dose_str, moa_of, controls_sents, arm):
    """Rebuilt from scratch with format_prompt. Nothing is reused from an existing example's prompt
    string -- which is exactly the defect in build_fixed_prompts."""
    out = []
    for di, d in enumerate(drugs):
        moa = "unclear" if arm == "blind" else (moa_of.get(d) or "unclear")
        for k, c in enumerate(controls_sents):
            out.append({"drug_idx": di, "drug": d, "k": k,
                        "prompt": fmt(cell_line_name, d, dose_str, moa, c)})
    return out


def locate_positions(tok, prompt, drug, span_prefix=" to ", span_suffix=" at "):
    """Token indices for the three read positions, from one encoding with offsets.

      P1 'name'  : the drug-name span (INCLUDING its leading space -- ' Vorinostat' and 'Vorinostat'
                   are different BPE sequences). A tokenizer null, not a claim position.
      P2 'instr' : the final '.' of 'Mechanism: {moa}.', immediately before '\\nControl cell:'.
                   Structurally independent of the control sentence.
      P3 'last'  : the ':' of 'Response cell:' -- where the readout actually reads.

    Hard failures, not warnings: truncation would delete 'Response cell:' and silently turn the
    'last prompt position' into an arbitrary gene token.
    """
    enc = tok(prompt, return_offsets_mapping=True, return_tensors=None)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    if len(ids) > 4096:
        raise AssertionError(f"prompt is {len(ids)} tokens, over the 4096 cap; truncation would "
                             f"delete 'Response cell:'")
    tail = tok.decode(ids[-3:])
    if not tail.rstrip().endswith("Response cell:"):
        raise AssertionError(f"prompt does not end with 'Response cell:' (tail={tail!r})")

    # `span_prefix`/`span_suffix` bracket the ONE varying name in the prompt. They default to the
    # drug's slot (' to {drug} at '); the cell-line positive control passes ('response of ', ' to ')
    # so the identical code path reads the cell-line span instead. Nothing else differs.
    marker = f"{span_prefix}{drug}{span_suffix}"
    ci = prompt.find(marker)
    if ci < 0:
        raise AssertionError(f"varying name {drug!r} not found in its own prompt")
    # the name's first character sits at ci + len(span_prefix). Selecting tokens that
    # intersect the NAME characters (not the preceding space) picks up the leading-space token when
    # the tokenizer folds the space into it -- ' Vorinostat' and 'Vorinostat' are different BPE
    # sequences and the span must be the one the model actually saw -- while never swallowing 'to'.
    n_start, n_end = ci + len(span_prefix), ci + len(span_prefix) + len(drug)
    name_toks = [t for t, (s, e) in enumerate(offs) if e > n_start and s < n_end and e > s]
    if not name_toks:
        raise AssertionError("could not recover the drug-name token span")
    span = prompt[offs[name_toks[0]][0]:offs[name_toks[-1]][1]]
    if span.strip() != drug.strip():
        raise AssertionError(f"name span {span!r} does not decode back to {drug!r}")

    ctrl_i = prompt.rfind("\nControl cell:")
    if ctrl_i < 0:
        ctrl_i = prompt.rfind("\n\nResponse cell:")
    instr_char = ctrl_i - 1                                   # the '.' closing the Mechanism clause
    instr_tok = None
    for t, (s, e) in enumerate(offs):
        if s <= instr_char < e:
            instr_tok = t
            break
    if instr_tok is None:
        instr_tok = max(0, name_toks[-1])
    return {"name": list(name_toks), "instr": [instr_tok], "last": [len(ids) - 1]}, len(ids)


def get_layers(model):
    """Copied from workspace_probe.get_layers."""
    for attr in ("gpt_neox", "model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None:
            if hasattr(base, "layers"):
                return base.layers
            if hasattr(base, "h"):
                return base.h
    raise RuntimeError("could not locate transformer layer list on this model")


def get_embedding(model):
    for attr in ("gpt_neox", "model", "transformer"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        for e in ("embed_in", "embed_tokens", "wte"):
            if hasattr(base, e):
                return getattr(base, e)
    raise RuntimeError("could not locate the embedding module on this model")


def extract_activations(model, tok, prompt_records, layers, positions, device, n_drugs, K, hidden):
    """acts[L][pos] = (n_drugs, K, hidden) float32, read FROM THE SAME POINT THE HOOK WRITES TO.

    Hooks on block outputs, never `output_hidden_states`: in GPT-NeoX `hidden_states[-1]` has
    `final_layer_norm` applied while the block's own output does not, so the last layer would be
    estimated in one coordinate frame and compared in another. That mismatch silently broke
    workspace_probe v3.0 and is documented at workspace_probe.py:355-387. L=0 is the embedding
    module output. Every capture is cast to float32 immediately, so the bf16 load buys speed without
    putting bf16 quantisation into distances whose CV is of order 0.03.

    One prompt at a time (batch_size=1). If this is ever batched, index with
    attention_mask.sum(-1)-1 and assert the gathered token id -- never hs[:, -1, :].
    """
    import torch
    acts = {L: {p: np.zeros((n_drugs, K, hidden), dtype=np.float32) for p in positions}
            for L in layers}
    grab = {}
    idx_holder = {}

    def capture(L):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            grab[L] = {p: hs[0, idx_holder[p], :].detach().float().cpu().numpy().mean(axis=0)
                       for p in idx_holder}
        return hook

    def _check_fresh(rec):
        """`grab` used to persist between prompts. If any hook failed to fire -- a layer list that
        does not match the module list, a module skipped by a config branch, an exception swallowed
        upstream -- the PREVIOUS prompt's activation was still sitting in `grab` and was written
        into this drug's row as though it were this drug's. Silent, and it would have manufactured
        geometry out of prompt ORDER. `grab` is now cleared before every forward pass and its
        contents verified after."""
        missing = [(L, p) for L in layers for p in positions
                   if L not in grab or p not in grab[L]]
        if missing:
            raise RuntimeError(f"forward hooks did not fire for {missing[:8]} on drug "
                               f"{rec['drug']!r} (control draw {rec['k']}); refusing to write a "
                               f"stale or zero activation into the matrix")

    layer_mods = get_layers(model)
    emb = get_embedding(model)
    n_tok_seen = []
    for rec in prompt_records:
        pos_map, n_ids = locate_positions(tok, rec["prompt"], rec.get("span_target", rec["drug"]),
                                          rec.get("span_prefix", " to "),
                                          rec.get("span_suffix", " at "))
        n_tok_seen.append(n_ids)
        idx_holder.clear()
        for p in positions:
            idx_holder[p] = pos_map[p]
        enc = tok(rec["prompt"], return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        # `locate_positions` encodes separately; if that encoding and this one ever disagree on
        # length, 'last' = len(ids) - 1 indexes the wrong token. Both use the tokenizer's default
        # add_special_tokens, so they agree -- but agreement is asserted rather than assumed.
        n_fwd = int(enc["input_ids"].shape[-1])
        if n_fwd != n_ids:
            raise AssertionError(f"position encoding has {n_ids} tokens but the forward-pass "
                                 f"encoding has {n_fwd}; the 'last' index would be off by "
                                 f"{n_fwd - n_ids}")
        grab.clear()
        handles = []
        for L in layers:
            mod = emb if L == 0 else layer_mods[L - 1]
            handles.append(mod.register_forward_hook(capture(L)))
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in handles:
                h.remove()
        _check_fresh(rec)
        for L in layers:
            for p in positions:
                v = grab[L][p]
                if not np.isfinite(v).all():
                    raise AssertionError(f"non-finite activation captured at layer {L}, position "
                                         f"{p}, drug {rec['drug']!r}")
                acts[L][p][rec["drug_idx"], rec["k"], :] = v
    return acts, {"n_tokens_min": int(min(n_tok_seen)), "n_tokens_max": int(max(n_tok_seen)),
                  "n_prompts": len(prompt_records)}


# =============================================================================================
# Assembling a cell
# =============================================================================================
def _strip_private(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def analyse_cell(A, B, controls, drugs, perms, rel_a, relb, gate, n_perm_triplet, seed,
                 label, diag_A, trip="recompute", confound_floor=None, activity_controls=None,
                 enforce_n_floor=True):
    """One (model, arm, layer, position, metric) cell: raw and partial coefficients, the null, the
    interval, the ceilings, the disattenuated value, and the matched-triplet co-primary."""
    n = len(drugs)
    cell = dict(label)
    cell.update({"n_drugs": int(n), "n_dyads": int(n * (n - 1) // 2)})
    cell["activation_diag"] = {k: v for k, v in diag_A.items()}
    if diag_A.get("status") == "degenerate_by_construction":
        cell["status"] = "degenerate_by_construction"
        cell["note"] = ("the activation distance matrix has no off-diagonal spread; this cell is "
                        "recorded, never returned as 0.0 or NaN into an aggregate")
        return cell, None
    if n < 25:
        if enforce_n_floor:
            cell["status"] = "untestable_n"
            return cell, None
        # the n refusal is a rule about CLAIMS. The cell-line positive control makes no claim: it
        # is an instrument check, and refusing to compute it because the tier carries fewer than 25
        # cell lines would remove the only thing that bounds the drug-side null.
        cell["n_floor_waived"] = True
        cell["n_floor_waived_note"] = (
            f"n = {n} < 25 and the pre-registered refusal was NOT applied, because this is a "
            f"labelled CONTROL rather than a claim. It may never be quoted as a result about "
            f"geometry at this n; it may only be read as 'the instrument does / does not see a "
            f"geometry that is known to be there'.")

    m = mrm_partial(A, B, controls, seed=seed, perms=perms, drugs=drugs)
    perm_stats = m.pop("_perm_stats", None)
    ctrl_map, design_names, X = m.pop("_control_diag_design", ({}, [], np.zeros((1, 1))))
    cell.update(_strip_private(m))
    cell["controls"] = control_diagnostics(ctrl_map, design_names, X)
    cell["exploratory"] = bool(n < 40)
    if cell["exploratory"]:
        cell["exploratory_note"] = "25 <= n_drugs < 40: below the pre-registered headline threshold"
    cell["rel_A"] = rel_a
    cell["rel_B"] = relb
    cell["rel_B_gate"] = gate
    ra = (rel_a or {}).get("spearman_brown")
    rb = (relb or {}).get("rel_B")
    cell["r_disattenuated"] = _disattenuate(m.get("r_partial"), ra, rb)
    cell["r_disattenuated_status"] = _disattenuation_status(m.get("r_partial"), ra, rb)
    cell["ci_disattenuated"] = _ci_disattenuated(m.get("ci"), ra,
                                                 (relb or {}).get("lo"), (relb or {}).get("hi"))
    # the equivalence verdict that licenses the null branch, on the scale the pre-registration
    # already mandates for the point estimate
    cell["tost_margin_0.30_disattenuated"] = _tost_disattenuated(m.get("ci"), ra, rb, 0.30)
    # ---- the confound-residue envelope: magnitude, not p, is the licensing statistic here ----
    if confound_floor is not None and m.get("r_partial") is not None:
        inside = bool(abs(m["r_partial"]) < float(confound_floor))
        cell["confound_residue_floor"] = float(confound_floor)
        cell["within_confound_residue_envelope"] = inside
        gate_ok = bool((gate or {}).get("headline_permitted", False))
        cell["headline_permitted_for_this_cell"] = bool(gate_ok and not inside and n >= 40)
        if inside:
            cell["note_confound_residue"] = (
                f"|r_partial| = {abs(m['r_partial']):.4f} sits BELOW this arm's measured "
                f"confound-residue floor of {float(confound_floor):.4f} -- the 95th percentile of "
                f"|r_partial| under a zero-pharmacology confounded null on this very roster. It may "
                f"be significant by p and survive the FWER correction and still be reachable with "
                f"no pharmacology anywhere in the world. NOT a headline.")
    # ---- LABELLED SECONDARY: the same partial with activity magnitude in the control design ----
    if activity_controls:
        ctrl_act = dict(controls); ctrl_act.update(activity_controls)
        m_act = mrm_partial(A, B, ctrl_act, seed=seed, perms=perms, drugs=drugs)
        m_act.pop("_perm_stats", None); m_act.pop("_control_diag_design", None)
        cell["r_partial_activity_adjusted"] = {
            "r_partial": m_act.get("r_partial"), "p": m_act.get("p"),
            "ci": m_act.get("ci"), "sd_null": m_act.get("sd_null"),
            "controls_in_design": m_act.get("controls_in_design"),
            "status": m_act.get("status"),
            "note": ("SECONDARY, NEVER THE HEADLINE, and it reads LOW BY CONSTRUCTION. Activity "
                     "class is genuinely correlated with mechanism -- cytotoxics really do act more "
                     "than most cardiovascular agents -- so partialling out activity magnitude "
                     "removes real pharmacology along with the artefact. Report it BESIDE the "
                     "headline, never instead of it. Its use is asymmetric: if the headline "
                     "SURVIVES this over-adjustment the activity reading is excluded; if it "
                     "collapses, the two readings are not separated by this design and neither is "
                     "licensed."),
        }
    cell["cv_A"] = diag_A.get("cv_A")
    cell["cv_real_reference"] = [0.070, 0.097]
    cell["cv_model_reference"] = [0.032, 0.049]
    # the admitted triples depend only on B and the controls, so they are computed once per arm and
    # passed in; "recompute" is the standalone path used by the selftest
    if isinstance(trip, str) and trip == "recompute":
        trip = matched_triplets(B, controls, seed=seed)
    cell["triplet"] = triplet_stat(A, trip, n_perm=n_perm_triplet, seed=seed)
    return cell, perm_stats


# =============================================================================================
# Main real-data run
# =============================================================================================
def _cpu_stage(args):
    """Everything that depends ONLY on the eval jsonl: panel, roster, biology, rel_B, the gate.

    NO torch, NO transformers, NO checkpoint on this path. It is the whole of --precheck and the
    prefix of a real run, so the two can never disagree about which roster was analysed. Returns
    (result, state); state is None when the run refuses at the n floor.
    """
    panel, panel_index, panel_path = load_panel(args.eval_dir, args.panel_file)
    P = len(panel)
    logger.info(f"panel {P} from {panel_path}")

    examples = load_examples(args.eval_dir, args.tier)
    logger.info(f"loaded {len(examples)} examples from tier {args.tier}")

    g, group_examples, gsel = choose_group(examples, args.min_cells_per_drug,
                                           args.cell_line, args.plate, scope=args.scope)
    if g is None:
        raise RuntimeError(f"no group found at scope {args.scope} for the requested cell line/plate")
    cl, plate = g
    logger.info(f"group = (cell_line {cl}, plate {plate}) at scope {args.scope}; "
                f"{gsel['n_drugs_over_floor']} drugs over the {args.min_cells_per_drug}-cell floor")

    eligible = [(d, len(v)) for d, v in group_examples.items()
                if len(v) >= args.min_cells_per_drug and str(d).upper() not in ("DMSO", "DMSO_TF")]
    eligible.sort(key=lambda t: (-t[1], str(t[0])))
    drugs = sorted([d for d, _ in eligible[:args.n_drugs]], key=str)
    n = len(drugs)
    logger.info(f"{n} drugs after the cell floor and the --n_drugs cap")

    result = {
        # a shallow COPY, so a per-run measurement can be written beside the declaration in the
        # output JSON without ever mutating the module-level pre-registration constant
        "pre_registration": dict(PRE_REGISTRATION),
        "provenance": {
            "script": os.path.basename(__file__),
            "argv": sys.argv,
            "args": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
                     for k, v in vars(args).items()},
            "sources": _SOURCES,
            "panel_file": panel_path, "panel_P": P,
            "python": platform.python_version(), "numpy": np.__version__,
            "tier": args.tier, "eval_dir": args.eval_dir,
            "group": {"cell_line_id": str(cl), "plate": str(plate)},
            "group_selection": gsel,
            # declared BEFORE any coefficient exists; see PRE_REGISTRATION['multi_stratum_rule']
            "stratum_label": str(args.stratum_label),
            "primary_stratum": str(args.primary_stratum),
            "is_primary_stratum": bool(str(args.stratum_label) == str(args.primary_stratum)),
        },
        "n_drugs": n, "drugs": [str(d) for d in drugs],
    }

    if n < 25:
        result["status"] = "untestable_n"
        result["note"] = (f"only {n} drugs clear the gate in the chosen group; the pre-registered "
                          f"refusal rule forbids computing the coefficient below 25")
        return result, None

    # ---------------- biology ----------------
    logger.info("building the biological matrix (residual signed-rank frame)")
    bio = build_biology(group_examples, drugs, panel_index, P, args.seed,
                        n_rel_splits=args.n_rel_splits)
    B = bio["B"]
    relb = reliability_B(bio["half_profiles"], P, n_boot=args.n_boot, seed=args.seed + 17)
    gate = rel_B_gate(relb)
    result["biology"] = {
        "rel_B": relb, "gate": gate,
        "encoding_diag": bio["encoding_diag"],
        "n_cells_per_drug": bio["n_cells"],
        "median_cells_per_drug": float(np.median(list(bio["n_cells"].values()))),
        "cv_B": cv(offdiag(B)),
        "frame": (
            "res = pseudobulk(drug) - roster mean, then signed_rank_from_vector(k=100), then "
            "1 - cosine. The code writes this as 'pb - plate DMSO - cell-line generic shift', but "
            "the generic is mean_d(pb(d) - dmso), so the DMSO term CANCELS IDENTICALLY and the "
            "plate-DMSO referencing is algebraically inert: replacing the DMSO vector with a "
            "completely different one moves res by 5.6e-15, and max|res - (pb - pb.mean(0))| = "
            "4.4e-16. This output must NOT be read as carrying a plate-matched DMSO control."),
        "frame_credit_that_is_real": (
            "leave-one-out and leave-one-in generics differ only by the positive scalar m/(m-1) "
            "applied identically to every drug, so B is BYTE-IDENTICAL under cosine (verified max "
            "|B_LOO - B_LOI| = 0.0). The -1/(m-1) self-inclusion artefact does not apply here."),
    }
    # ---- REPORT-ONLY reliability companions. Neither drives any gate. ----
    if getattr(args, "rel_companions", True):
        comp = {"gate_quantity": "rel_B above (cell half-splits, signed-rank k=100)"}
        rb_full = reliability_B(bio["half_profiles"], P, k=P, n_boot=max(200, args.n_boot // 4),
                               seed=args.seed + 41)
        comp["rel_B_full_panel"] = rb_full
        comp["rel_B_full_panel_note"] = (
            f"the identical pipeline with k = P = {P} instead of k = 100, so the cost of the "
            f"top-100 signed-rank truncation on the panel is a MEASURED number rather than an "
            f"argument. NOT the gate quantity, and not a selectable frame: the gate stays on the "
            f"k=100 frame the residual chapter grades in.")
        wsp, covered = wellsplit_profiles(group_examples, drugs, panel_index, P, args.seed,
                                          n_splits=min(args.n_rel_splits, 10))
        comp["rel_B_wellsplit"] = (reliability_B(wsp, P, n_boot=max(200, args.n_boot // 4),
                                                 seed=args.seed + 43) if wsp else None)
        comp["rel_B_wellsplit_n_drugs_covered"] = len(covered)
        comp["rel_B_wellsplit_drugs"] = [str(d) for d in covered]
        comp["bias_directions"] = {
            "cell_split_rel_B (the gate)": ("OPTIMISTIC. Drug nests in well, so every between-drug "
                                            "distance is a between-well contrast and well-level "
                                            "technical structure is perfectly reproducible across "
                                            "cell halves of the same well; rel_B counts it as "
                                            "signal."),
            "well_split_rel_B": ("A LOWER BOUND, and confounded. It covers only the drugs with >= 2 "
                                 "wells, and a well split for a multi-dose drug also changes dose, "
                                 "whose transfer is measured elsewhere at 0.707. It is reported so "
                                 "the inflation is visible; it is NOT allowed to make the refusal "
                                 "decision on a contaminated number."),
        }
        result["biology"]["rel_B_companions"] = comp
        logger.info(f"  rel_B companions (REPORT ONLY): full-panel="
                    f"{(rb_full or {}).get('rel_B')}, well-split="
                    f"{(comp['rel_B_wellsplit'] or {}).get('rel_B')} over {len(covered)} drugs")

    # ---- activity magnitude: what B is made of, reported whether or not anyone asks ----
    a_sel, C7 = activity_covariates(bio["res_full"])
    act = activity_structure_of_B(B, C7)
    # the same scalar over the FULL eligible pool, so the top-n-by-cell-count truncation is visible
    # rather than assumed away: cell recovery is a function of cytotoxicity, so selecting the drugs
    # with the most cells biases the roster toward inert compounds and truncates B's dynamic range.
    all_elig = [d for d, _ in eligible]
    try:
        pb_all = np.stack([np.stack([sentence_to_expression_proxy(e.get("response", ""),
                                                                  panel_index, P)
                                     for e in group_examples[d]]).mean(axis=0) for d in all_elig])
        a_all = np.linalg.norm(pb_all - pb_all.mean(axis=0, keepdims=True), axis=1)
        sel_mask = np.array([d in set(drugs) for d in all_elig])
        act["a_distribution"] = {
            "selected_roster": {"n": int(sel_mask.sum()),
                                "quantiles_10_50_90": [float(x) for x in np.percentile(
                                    a_all[sel_mask], [10, 50, 90])] if sel_mask.any() else None},
            "full_eligible_pool": {"n": int(len(all_elig)),
                                   "quantiles_10_50_90": [float(x) for x in np.percentile(
                                       a_all, [10, 50, 90])]},
            "note": ("a(d) computed against the FULL eligible pool's mean in both rows so the two "
                     "are comparable. The roster is selected as the top n by cell count, and cell "
                     "recovery is a function of cytotoxicity, so a selected roster shifted low "
                     "relative to the pool means the truncation is biasing toward inert compounds."),
        }
    except Exception as e:                                                    # pragma: no cover
        act["a_distribution"] = {"status": f"unavailable: {e.__class__.__name__}: {e}"}
    result["biology"]["activity_magnitude"] = act

    logger.info(f"  rel_B = {relb}")
    logger.info(f"  activity: R2(B ranks | activity only) = "
                f"{act['r2_dyad_ranks_on_activity_only']}, leading eig share of B = "
                f"{act['leading_eigenvalue_share_of_B']}")
    logger.info(f"  GATE  : {gate['status']} (headline permitted: {gate['headline_permitted']})")
    return result, {"panel_index": panel_index, "P": P, "examples": examples,
                    "group_examples": group_examples, "drugs": drugs, "n": n,
                    "cl": cl, "plate": plate, "bio": bio, "B": B, "relb": relb, "gate": gate,
                    "activity_controls": C7, "activity": a_sel}


def precheck(args):
    """--precheck: run the CPU stage and STOP, before any checkpoint exists in memory.

    n_drugs, B, rel_B and the gate depend only on the eval jsonl. Discovering that a stratum is
    untestable after an H200 has been allocated is indefensible when it is a minute of CPU. This
    writes the same JSON shape as a real run, with `mode = precheck`.
    """
    result, st = _cpu_stage(args)
    result["mode"] = "precheck"
    if st is None:
        result["precheck_verdict"] = "REFUSE: untestable_n"
        write_json(result, args.out)
        logger.warning(f"PRECHECK_VERDICT REFUSE untestable_n n_drugs={result.get('n_drugs')}")
        return result, False
    gate = st["gate"]
    n = st["n"]
    relb = st["relb"] or {}
    result["status"] = "ok" if gate["headline_permitted"] else gate["status"]
    runnable = bool(gate["headline_permitted"] and n >= 25)
    result["precheck_verdict"] = ("RUN" if runnable else f"REFUSE: {gate['status']}")
    result["precheck_exploratory"] = bool(n < 40)
    result["precheck_note"] = (
        "this JSON contains NO activation-derived quantity. Selection among configurations on the "
        "basis of these numbers is a function of B and the roster only -- never of any activation "
        "-- so it cannot bias the headline. See PRE_REGISTRATION['configuration_selection'].")
    if args.precheck_sweep is not None:
        result["precheck_sweep"] = _precheck_sweep(args)
    write_json(result, args.out)
    logger.warning(f"PRECHECK_VERDICT {result['precheck_verdict']}  n_drugs={n}  "
                   f"rel_B={relb.get('rel_B')} [{relb.get('lo')}, {relb.get('hi')}]  "
                   f"exploratory={result['precheck_exploratory']}")
    return result, runnable


def _precheck_sweep(args):
    """One row per (scope, min_cells_per_drug): n_drugs and rel_B with its interval.

    The sbatch names --min_cells_per_drug as THE sensitivity dial but currently resolves the
    trade-off only after the GPU has been spent. This reads the (n_drugs, rel_B) frontier directly,
    on CPU, from B and the roster alone.
    """
    vals = [int(x) for x in str(args.precheck_sweep).split(",") if str(x).strip()]
    rows = []
    for scope in ("cell_line_plate", "cell_line"):
        for mc in vals:
            sub = argparse.Namespace(**vars(args))
            sub.min_cells_per_drug = mc
            sub.scope = scope
            sub.rel_companions = False        # the sweep reads the frontier, not the companions
            if scope == "cell_line":
                sub.plate = None
            row = {"scope": scope, "min_cells_per_drug": mc}
            try:
                _r, _st = _cpu_stage(sub)
                row["n_drugs"] = int(_r.get("n_drugs", 0))
                if _st is None:
                    row["status"] = "untestable_n"
                else:
                    rb = _st["relb"] or {}
                    row.update({"status": _st["gate"]["status"],
                                "rel_B": rb.get("rel_B"),
                                "rel_B_lo": rb.get("lo"), "rel_B_hi": rb.get("hi"),
                                "median_cells_per_drug": float(np.median(
                                    list(_st["bio"]["n_cells"].values()))),
                                "headline_permitted": bool(_st["gate"]["headline_permitted"])})
            except Exception as e:                                          # pragma: no cover
                row["status"] = f"error: {e.__class__.__name__}: {e}"
            logger.info(f"  sweep {scope:16s} min_cells={mc:4d} -> {row}")
            rows.append(row)
    return {"rows": rows,
            "selection_rule": PRE_REGISTRATION["configuration_selection"],
            "note": ("every quantity here is a function of the eval jsonl alone. No activation, no "
                     "checkpoint, no forward pass entered this table.")}


def run(args):
    fmt = get_format_prompt()
    result, st = _cpu_stage(args)
    if st is None:
        write_json(result, args.out)
        return result
    panel_index, P, examples = st["panel_index"], st["P"], st["examples"]
    group_examples, drugs, n = st["group_examples"], st["drugs"], st["n"]
    cl, plate = st["cl"], st["plate"]
    bio, B, relb, gate = st["bio"], st["B"], st["relb"], st["gate"]

    if not gate["headline_permitted"]:
        logger.warning("  the reliability gate FAILED.")
        if not args.force_through_gate:
            result["status"] = gate["status"]
            result["note"] = (
                "the reliability gate refused BEFORE any checkpoint was loaded. No coefficient was "
                "computed, because a coefficient against a target with no measurable off-diagonal "
                "structure is not interpretable in either direction. Re-run with "
                "--force_through_gate only for diagnostics, and read PRE_REGISTRATION["
                "'reliability_gate_on_rel_B'] before doing so.")
            write_json(result, args.out)
            logger.warning(f"  EARLY RETURN: {gate['status']} -- no model will be loaded.")
            return result
        logger.warning("  --force_through_gate was passed: the sweep will run, and its results are "
                       "NOT interpretable in either direction. Every cell is stamped "
                       "produced_under_forced_gate.")
    result["provenance"]["force_through_gate"] = bool(args.force_through_gate)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---------------- prompts ----------------
    meta0 = group_examples[drugs[0]][0].get("metadata", {})
    cl_name = meta0.get("cell_line_name") or str(cl)
    doses = Counter()
    moa_of = {}
    for d in drugs:
        for e in group_examples[d]:
            m = e.get("metadata", {})
            if m.get("dose"):
                doses[m["dose"]] += 1
            if d not in moa_of and m.get("moa"):
                moa_of[d] = m["moa"]
    dose_str = args.dose_str or (doses.most_common(1)[0][0] if doses else "0.5 uM")
    # each drug's modal plate, for x9. Constant (and therefore dropped) under cell_line_plate scope.
    plate_of, plate_purity = [], []
    for d in drugs:
        pls = Counter(str(e.get("metadata", {}).get("plate")) for e in group_examples[d])
        top, cnt = pls.most_common(1)[0]
        plate_of.append(top)
        plate_purity.append(float(cnt / max(1, sum(pls.values()))))
    result["biology"]["plate_structure"] = {
        "scope": args.scope,
        "n_distinct_modal_plates": int(len(set(plate_of))),
        "median_frac_cells_on_modal_plate": float(np.median(plate_purity)),
        "x9_will_have_variance": bool(len(set(plate_of)) > 1),
        "note": ("plate batch is ELIMINATED by stratification under cell_line_plate scope (x9 is "
                 "constant and dropped as zero-variance) and ADJUSTED dyadically via x9 under "
                 "cell_line scope. The two scopes are NOT interchangeable readings."),
    }
    dose_diag, dose_log10 = dose_frame_diagnostics(group_examples, drugs, dose_str)
    result["biology"]["dose_frame"] = dose_diag
    logger.info(f"  dose frame: parse rate {dose_diag['drug_parse_rate']:.3f}, x6 built = "
                f"{dose_diag['x6_built']}, median frac of cells at the modal dose string = "
                f"{dose_diag['frac_cells_at_modal_dose_string_median']}")
    ctrl_sents, ctrl_src = pick_control_sentences(group_examples, args.n_controls, seed=7)
    K = len(ctrl_sents)
    logger.info(f"  cell line name '{cl_name}', fixed dose '{dose_str}', K={K} shared control "
                f"sentences ({ctrl_src})")
    result["provenance"].update({
        "cell_line_name": cl_name, "dose_str_fixed": dose_str, "n_controls": K,
        "control_source": ctrl_src,
        "control_sentence_hashes": [_hash_text(s) for s in ctrl_sents],
        "moa_of": {str(k): str(v) for k, v in moa_of.items()},
    })

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    bad = [a for a in arms if a not in ("blind", "annotated")]
    if bad:
        raise ValueError(f"unknown arm(s) {bad}; expected 'blind' and/or 'annotated'")
    bad = [p for p in positions if p not in ("name", "instr", "last")]
    if bad:
        raise ValueError(f"unknown position(s) {bad}; expected 'name', 'instr', 'last'")
    names = [x.strip() for x in args.model_names.split(",")]
    paths = [x.strip() for x in args.model_paths.split(",")]
    if len(names) != len(paths):
        raise ValueError("--model_names and --model_paths must have the same number of entries")
    # ONE permutation sequence, shared by every cell: that is what makes the max-statistic
    # family-wise correction and the PAIRED base-checkpoint contrast legitimate.
    prng = np.random.RandomState(args.seed + 991)
    perms = [prng.permutation(n) for _ in range(args.n_perm)]
    headline_layers = [int(x) for x in args.headline_layers.split(",") if x.strip()]

    cells, perm_bank, direct_bank = [], {}, {}
    # per-ARM confound-residue envelope, measured on this roster's own control matrices; the dict
    # is attached now and filled inside the arm loop, where the tokenizer-derived controls exist
    residue_floors = {}
    result["biology"]["null_confound_residue_real_roster"] = residue_floors

    # ---- cell-line positive control: B, rel_B and the roster, all on the identical frame ----
    pc_cells, pc_bio, pc_relb, pc_gate = [], None, None, None
    modal_drug = max(drugs, key=lambda d: bio["n_cells"].get(d, 0))
    if args.positive_control:
        pc_bio = build_cellline_biology(examples, panel_index, P, args.seed,
                                        min_cells=args.min_cells_per_drug,
                                        max_lines=args.n_drugs, n_rel_splits=args.n_rel_splits)
        if pc_bio is None:
            logger.warning("  positive control: fewer than 4 usable cell lines; not run")
        else:
            pc_relb = reliability_B(pc_bio["half_profiles"], P, n_boot=args.n_boot,
                                    seed=args.seed + 29)
            pc_gate = rel_B_gate(pc_relb)
            logger.info(f"  positive control: {len(pc_bio['lines'])} cell lines, "
                        f"rel_B = {(pc_relb or {}).get('rel_B')}, gate = {pc_gate['status']} "
                        f"(REPORTED, never applied -- this is a control, not a claim)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device = {device}")

    for name, mpath in zip(names, paths):
        logger.info(f"=== {name} ({mpath}) ===")
        tok = AutoTokenizer.from_pretrained(mpath)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            mpath, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
        model.eval()
        n_layers = model.config.num_hidden_layers
        hidden = model.config.hidden_size
        layers = (list(range(n_layers + 1)) if args.layers.strip() == "all"
                  else [int(x) for x in args.layers.split(",") if x.strip()])
        layers = [L for L in layers if 0 <= L <= n_layers]
        logger.info(f"  {n_layers} layers, hidden {hidden}; reading layers {layers} "
                    f"at positions {positions}")

        # name-token sets from THIS model's own tokenizer, on the span as it appears in the prompt
        name_tok_sets = [set(tok(f" {d}", add_special_tokens=False)["input_ids"]) for d in drugs]
        name_tok_counts = [len(tok(f" {d}", add_special_tokens=False)["input_ids"]) for d in drugs]

        for arm in arms:
            recs = build_prompts(fmt, cl_name, drugs, dose_str, moa_of, ctrl_sents, arm)
            prompt_tok_counts = []
            for di in range(n):
                r0 = recs[di * K]
                prompt_tok_counts.append(len(tok(r0["prompt"], add_special_tokens=False)["input_ids"]))
            moas = [("unclear" if arm == "blind" else (moa_of.get(d) or "unclear")) for d in drugs]
            controls = build_control_matrices(drugs, name_tok_sets, moas, name_tok_counts,
                                              prompt_tok_counts, arm, dose_log10=dose_log10,
                                              plate_of=plate_of)
            trip_arm = matched_triplets(B, controls, seed=args.seed)
            logger.info(f"  arm={arm}: {len(recs)} prompts, "
                        f"{0 if trip_arm is None else len(trip_arm['anchor'])} matched triples")
            if arm not in residue_floors:
                logger.info(f"  calibrating the confound-residue envelope for arm={arm} on this "
                            f"roster's own controls ({args.n_residue_rep} replicates, CPU)")
                # STAMPED WITH THE CHECKPOINT WHOSE TOKENIZER BUILT x1. The floor is calibrated once
                # per arm, on the first checkpoint's controls, and then applied to BOTH checkpoints'
                # cells -- which is correct only while the two share a tokenizer (they do here: both
                # are Pythia). If they ever do not, x1 differs between them and this field is what
                # makes the mismatch visible instead of silent.
                f_arm = confound_residue_floor(controls, arm, n_rep=args.n_residue_rep,
                                               n_perm=199, seed=args.seed)
                f_arm["calibrated_on_model"] = str(name)
                f_arm["calibrated_on_model_note"] = (
                    "x1 is built from THIS checkpoint's tokenizer; the floor is reused for every "
                    "other checkpoint in the run, which is only valid while they share a tokenizer")
                residue_floors[arm] = f_arm
                # `setdefault(k, expensive())` evaluated `expensive()` on every arm and threw the
                # result away after the first, paying the whole calibration twice for nothing.
                if "noise_only_control" not in residue_floors:
                    f_noi = confound_residue_floor(controls, "noise_only",
                                                   n_rep=args.n_residue_rep, n_perm=199,
                                                   seed=args.seed)
                    f_noi["calibrated_on_model"] = str(name)
                    residue_floors["noise_only_control"] = f_noi
                logger.info(f"    floor(|r_partial|, p95) = "
                            f"{residue_floors[arm].get('p95_abs_r_partial')} "
                            f"(chance-only control: "
                            f"{residue_floors['noise_only_control'].get('p95_abs_r_partial')})")
            arm_floor = residue_floors.get(arm, {}).get("floor")
            acts, ainfo = extract_activations(model, tok, recs, layers, positions, device,
                                              n, K, hidden)
            logger.info(f"    prompts {ainfo['n_tokens_min']}-{ainfo['n_tokens_max']} tokens")
            for L in layers:
                for pos in positions:
                    Hm = acts[L][pos]
                    for metric in ("centered", "zscored"):
                        # spectrum=True ONLY here: reliability_A calls activation_matrix 50 times
                        # per cell, and computing the SVD there would multiply the count by 50 for
                        # no benefit.
                        A, dA = activation_matrix(Hm, metric, spectrum=True)
                        rel_a = (rel_A_for(Hm, metric, seed=args.seed + 11)
                                 if dA.get("status") == "ok" else None)
                        label = {"model": name, "model_path": mpath, "arm": arm, "layer": int(L),
                                 "position": pos, "metric": metric,
                                 "group": [str(cl), str(plate)],
                                 "is_headline": bool(name == names[0] and arm == "blind"
                                                     and pos == "last" and metric == "centered"
                                                     and L in headline_layers)}
                        ntp = args.n_perm_triplet if label["is_headline"] else 0
                        cell, ps = analyse_cell(
                            A, B, controls, drugs, perms, rel_a, relb, gate, ntp, args.seed,
                            label, dA, trip=trip_arm, confound_floor=arm_floor,
                            # headline cells only: the second MRM costs another permutation sweep
                            activity_controls=(st["activity_controls"] if label["is_headline"]
                                               else None))
                        if args.force_through_gate:
                            cell["produced_under_forced_gate"] = True
                            cell["produced_under_forced_gate_note"] = (
                                "the rel_B gate REFUSED and the run was forced through it for "
                                "diagnostics; this number is not interpretable in either direction")
                        # ---- DIRECT DECODABILITY ARM ----
                        # the HEADLINE CELLS in the direct arm's own sense: both arms, both
                        # checkpoints, the headline layers, all three positions, centered metric.
                        # NOT `is_headline`, which is the single RSA headline cell (finetuned,
                        # blind, last) -- gating on that would leave the direct arm with no base
                        # checkpoint to pair against and no family to correct over.
                        is_direct = bool(L in headline_layers and metric == "centered")
                        if is_direct and cell.get("status") == "ok":
                            dd = direct_decodability(
                                kaveraged_centered(Hm, metric), bio["res_full"], bio["S"], drugs,
                                name_tok_sets, P, perms=perms, n_perm=args.n_perm_direct)
                            dinc = dd.pop("_perm_increment", None)
                            cell["direct_arm"] = dd
                            if dinc is not None:
                                direct_bank[(name, arm, L, pos, metric)] = dinc
                            if dd.get("status") == "ok":
                                logger.info(
                                    f"      DIRECT L{L} {pos} {metric}: retrieval="
                                    f"{dd['activations']['retrieval_acc']:.3f} "
                                    f"(null {dd['retrieval_null']:.3f}, p={dd.get('p_retrieval')}) "
                                    f"mean_r={dd['activations']['mean_held_out_r']:+.3f} "
                                    f"names_only={dd['names_only']['mean_held_out_r']:+.3f} "
                                    f"increment={dd['increment_mean_r']:+.3f} "
                                    f"(p={dd.get('p_increment')})")
                        cells.append(cell)
                        if ps is not None:
                            perm_bank[(name, arm, L, pos, metric)] = ps
                        if cell.get("status") == "ok":
                            logger.info(f"    L{L:2d} {pos:5s} {metric:8s} "
                                        f"r_raw={cell['r_raw']:+.3f} "
                                        f"r_part={cell['r_partial']:+.3f} "
                                        f"p={cell['p']:.4f} MDE80={cell['MDE80']:.3f}")
        # ---------------- CELL-LINE POSITIVE CONTROL, identical code path ----------------
        # Drug and dose held fixed at the roster's modal compound; the CELL LINE NAME varies in the
        # same prompt slot. Same K balanced control sentences, same hooks, same within-draw
        # centering, same activation_matrix / mrm_partial / matched_triplets / Freedman-Lane. A
        # labelled control block, never a headline: it is what converts "we saw no correlation" into
        # "the instrument works on this code path and saw no correlation".
        if args.positive_control and pc_bio is not None:
            pc_lines, pc_names_txt = pc_bio["lines"], pc_bio["line_names"]
            n_pc = len(pc_lines)
            pc_recs = []
            for li, lname in enumerate(pc_names_txt):
                for k, c in enumerate(ctrl_sents):
                    pc_recs.append({"drug_idx": li, "drug": modal_drug, "k": k,
                                    "span_target": lname, "span_prefix": "response of ",
                                    "span_suffix": " to ",
                                    "prompt": fmt(lname, modal_drug, dose_str, "unclear", c)})
            pc_tok_sets = [set(tok(f" {x}", add_special_tokens=False)["input_ids"])
                           for x in pc_names_txt]
            pc_tok_counts = [len(t) for t in pc_tok_sets]
            pc_prompt_counts = [len(tok(pc_recs[i * K]["prompt"],
                                        add_special_tokens=False)["input_ids"])
                                for i in range(n_pc)]
            pc_controls = build_control_matrices(pc_names_txt, pc_tok_sets,
                                                 ["unclear"] * n_pc, pc_tok_counts,
                                                 pc_prompt_counts, "blind")
            pc_trip = matched_triplets(pc_bio["B"], pc_controls, seed=args.seed)
            logger.info(f"  POSITIVE CONTROL: {n_pc} cell lines, {len(pc_recs)} prompts, "
                        f"drug fixed at {modal_drug!r}")
            pc_acts, _pi = extract_activations(model, tok, pc_recs, layers, positions, device,
                                               n_pc, K, hidden)
            pc_perms = [np.random.RandomState(args.seed + 4242 + t).permutation(n_pc)
                        for t in range(min(args.n_perm, 2000))]
            for L in [x for x in layers if x in headline_layers + [0]]:
                for pos in positions:
                    A_pc, d_pc = activation_matrix(pc_acts[L][pos], "centered", spectrum=True)
                    rel_pc = (rel_A_for(pc_acts[L][pos], "centered", seed=args.seed + 11)
                              if d_pc.get("status") == "ok" else None)
                    lab = {"model": name, "arm": "cellline_positive_control", "layer": int(L),
                           "position": pos, "metric": "centered", "is_headline": False}
                    c_pc, _ps = analyse_cell(A_pc, pc_bio["B"], pc_controls, pc_lines, pc_perms,
                                             rel_pc, pc_relb, pc_gate,
                                             args.n_perm_triplet if L in headline_layers else 0,
                                             args.seed, lab, d_pc, trip=pc_trip,
                                             enforce_n_floor=False)
                    c_pc["is_control_not_a_claim"] = True
                    pc_cells.append(c_pc)
                    if c_pc.get("status") == "ok":
                        logger.info(f"    PC L{L:2d} {pos:5s} r_raw={c_pc['r_raw']:+.3f} "
                                    f"r_part={c_pc['r_partial']:+.3f} p={c_pc['p']:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    result["cells"] = cells
    result["positive_control"] = {
        "status": ("not_run" if not args.positive_control else
                   ("unavailable_too_few_cell_lines" if pc_bio is None else "ok")),
        "n_cell_lines": (0 if pc_bio is None else len(pc_bio["lines"])),
        "cell_lines": ([] if pc_bio is None else [str(x) for x in pc_bio["lines"]]),
        "drug_held_fixed": str(modal_drug), "dose_held_fixed": str(dose_str),
        "rel_B": pc_relb, "rel_B_gate_reported_not_applied": pc_gate,
        "n_cells_per_line": (None if pc_bio is None else pc_bio["n_cells"]),
        "encoding_diag": (None if pc_bio is None else pc_bio["encoding_diag"]),
        "n_refusal_applied": bool(pc_bio is not None and len(pc_bio["lines"]) >= 25),
        "cells": pc_cells,
        "reading_rule": PRE_REGISTRATION["positive_control"],
        "note": ("a LABELLED CONTROL, never a headline. Identical code path: same prompts template, "
                 "same K balanced control sentences, same hooks, same within-draw centering, same "
                 "activation_matrix / mrm_partial / matched_triplets / Freedman-Lane, same residual "
                 "signed-rank cosine frame for B, same disjoint-cell half-split for rel_B. Only the "
                 "varying token changes, from the drug name to the cell-line name."),
    }
    # the measured floor travels WITH the declaration, not only in the biology block
    result["pre_registration"]["confound_residue_floor"] = dict(
        PRE_REGISTRATION["confound_residue_floor"],
        measured_this_run={a: f.get("p95_abs_r_partial") for a, f in residue_floors.items()})

    # ---------------- family-wise max statistic, over one shared permutation sequence ----------
    fam = {}
    for arm in arms:
        for name in names:
            keys = [k for k in perm_bank if k[0] == name and k[1] == arm]
            if not keys:
                continue
            M = np.stack([np.abs(perm_bank[k]) for k in keys])          # (n_cells, n_perm)
            maxnull = M.max(axis=0)
            for c in cells:
                if c.get("status") != "ok" or c["model"] != name or c["arm"] != arm:
                    continue
                obs = abs(c["r_partial"])
                ge = int(np.sum(maxnull >= obs - 1e-15))
                c["p_fwer_max"] = float((1.0 + ge) / (len(maxnull) + 1.0))
            fam[f"{name}|{arm}"] = {
                "n_cells_in_family": len(keys),
                "cells": [f"L{k[2]}|{k[3]}|{k[4]}" for k in keys],
                "max_null_q95": float(np.percentile(maxnull, 95)),
                "note": ("one drug relabeling applied jointly to every (layer, position, metric) "
                         "cell; the profile claim is tested against max|r_partial| under it. The "
                         "family spans both distance metrics as well as (layer, position), which "
                         "is deliberately conservative."),
            }
    result["fwer"] = fam

    # ---------------- paired base contrast ----------------
    deltas = []
    if len(names) >= 2:
        for arm in arms:
            for L in sorted({c["layer"] for c in cells}):
                for pos in positions:
                    for metric in ("centered", "zscored"):
                        ka = (names[0], arm, L, pos, metric)
                        kb = (names[1], arm, L, pos, metric)
                        ca = next((c for c in cells if c.get("status") == "ok"
                                   and (c["model"], c["arm"], c["layer"], c["position"],
                                        c["metric"]) == ka), None)
                        cb = next((c for c in cells if c.get("status") == "ok"
                                   and (c["model"], c["arm"], c["layer"], c["position"],
                                        c["metric"]) == kb), None)
                        if ca is None or cb is None or ka not in perm_bank or kb not in perm_bank:
                            continue
                        d = ca["r_partial"] - cb["r_partial"]
                        dn = perm_bank[ka] - perm_bank[kb]       # PAIRED: same relabeling in both
                        ge = int(np.sum(np.abs(dn) >= abs(d) - 1e-15))
                        deltas.append({
                            "arm": arm, "layer": int(L), "position": pos, "metric": metric,
                            "model_a": names[0], "model_b": names[1],
                            "delta_r_partial": float(d),
                            "p_paired": float((1.0 + ge) / (len(dn) + 1.0)),
                            "sd_null": float(np.std(dn)),
                            "triplet_delta": (ca["triplet"].get("triplet_acc", np.nan)
                                              - cb["triplet"].get("triplet_acc", np.nan))
                            if ca["triplet"].get("status") == "ok"
                               and cb["triplet"].get("status") == "ok" else None,
                            "note": ("one drug relabeling applied to BOTH matrices inside each "
                                     "draw; separates 'fine-tuning organised the representation' "
                                     "from 'drug names already meant something to a language "
                                     "model'"),
                        })
    result["base_contrast"] = deltas

    # ---------------- paired POSITION contrast: instr minus last ----------------
    # PRE_REGISTRATION['co_primary'][1] names this gap as a secondary result and the design
    # previously supplied neither an interval nor a p for it, while base_contrast supplied both
    # using machinery that transfers unchanged at zero extra compute. The third outcome the study
    # can return -- organised at 'instr', destroyed by 'last', i.e. organised but not ROUTED to the
    # read position -- lives entirely in this gap.
    pos_deltas = []
    if "instr" in positions and "last" in positions:
        _cell_at = {(c["model"], c["arm"], c["layer"], c["position"], c["metric"]): c
                    for c in cells if c.get("status") == "ok"}
        for name in names:
            for arm in arms:
                for L in sorted({c["layer"] for c in cells}):
                    for metric in ("centered", "zscored"):
                        ki = (name, arm, L, "instr", metric)
                        kl = (name, arm, L, "last", metric)
                        ci_, cl_ = _cell_at.get(ki), _cell_at.get(kl)
                        if ci_ is None or cl_ is None or ki not in perm_bank or kl not in perm_bank:
                            continue
                        ra_i = (ci_.get("rel_A") or {}).get("spearman_brown")
                        ra_l = (cl_.get("rel_A") or {}).get("spearman_brown")
                        rb_ = (relb or {}).get("rel_B")
                        use_dis = (ra_i is not None and ra_l is not None and rb_ is not None
                                   and ra_i > 0 and ra_l > 0 and rb_ > 0)
                        if use_dis:
                            si = math.sqrt(ra_i * rb_); sl = math.sqrt(ra_l * rb_)
                            d = ci_["r_partial"] / si - cl_["r_partial"] / sl
                            dn = perm_bank[ki] / si - perm_bank[kl] / sl
                            scale = "disattenuated"
                        else:
                            d = ci_["r_partial"] - cl_["r_partial"]
                            dn = perm_bank[ki] - perm_bank[kl]
                            scale = "attenuated"
                        ge = int(np.sum(np.abs(dn) >= abs(d) - 1e-15))
                        pos_deltas.append({
                            "model": name, "arm": arm, "layer": int(L), "metric": metric,
                            "position_a": "instr", "position_b": "last",
                            "scale": scale,
                            "delta_r_partial": float(d),
                            "p_paired": float((1.0 + ge) / (len(dn) + 1.0)),
                            "sd_null": float(np.std(dn)),
                            "rel_A_instr": ra_i, "rel_A_last": ra_l, "rel_B": rb_,
                            "r_partial_instr": ci_["r_partial"],
                            "r_partial_last": cl_["r_partial"],
                            "note": ("one drug relabeling applied to BOTH positions inside each "
                                     "draw, so the null is genuinely paired. THE TWO POSITIONS "
                                     "CARRY DIFFERENT rel_A: the control sentence is causally "
                                     "downstream of 'instr', so rel_A = 1 there by construction "
                                     "while 'last' is attenuated by control-sentence sensitivity. "
                                     "On the attenuated scale that biases the gap toward 'instr' "
                                     f"for purely measurement reasons; this row used the {scale} "
                                     "scale."),
                        })
    result["position_contrast"] = pos_deltas

    # ---------------- the DIRECT arm's OWN max-statistic family and base contrast --------------
    # It gets its own family over its own cells, on the same shared permutation prefix, rather than
    # being folded into the RSA family: the two arms test different statistics and a max taken
    # across both would be neither.
    dfam = {}
    for arm in arms:
        for name in names:
            keys = [k for k in direct_bank if k[0] == name and k[1] == arm]
            if not keys:
                continue
            M = np.stack([np.abs(direct_bank[k]) for k in keys])
            maxnull = M.max(axis=0)
            for c in cells:
                da = c.get("direct_arm")
                if (not da or da.get("status") != "ok" or c["model"] != name or c["arm"] != arm
                        or (name, arm, c["layer"], c["position"], c["metric"]) not in direct_bank):
                    continue
                obs = abs(da["increment_mean_r"])
                ge = int(np.sum(maxnull >= obs - 1e-15))
                da["p_fwer_max_increment"] = float((1.0 + ge) / (len(maxnull) + 1.0))
            dfam[f"{name}|{arm}"] = {
                "n_cells_in_family": len(keys),
                "cells": [f"L{k[2]}|{k[3]}|{k[4]}" for k in keys],
                "max_null_q95": float(np.percentile(maxnull, 95)),
                "statistic": "activations-over-names increment in held-out mean correlation",
                "note": ("one drug relabeling of Y applied jointly to every direct-arm cell; this "
                         "family is SEPARATE from the RSA family because it tests a different "
                         "statistic."),
            }
    result["direct_fwer"] = dfam

    direct_deltas = []
    if len(names) >= 2:
        _da_at = {(c["model"], c["arm"], c["layer"], c["position"], c["metric"]): c.get("direct_arm")
                  for c in cells if c.get("direct_arm")}
        for arm in arms:
            for L in sorted({c["layer"] for c in cells}):
                for pos in positions:
                    for metric in ("centered", "zscored"):
                        ka = (names[0], arm, L, pos, metric)
                        kb = (names[1], arm, L, pos, metric)
                        da, db = _da_at.get(ka), _da_at.get(kb)
                        if (da is None or db is None or da.get("status") != "ok"
                                or db.get("status") != "ok"
                                or ka not in direct_bank or kb not in direct_bank):
                            continue
                        d = da["increment_mean_r"] - db["increment_mean_r"]
                        dn = direct_bank[ka] - direct_bank[kb]   # PAIRED: same relabeling in both
                        ge = int(np.sum(np.abs(dn) >= abs(d) - 1e-15))
                        direct_deltas.append({
                            "arm": arm, "layer": int(L), "position": pos, "metric": metric,
                            "model_a": names[0], "model_b": names[1],
                            "delta_increment_mean_r": float(d),
                            "delta_retrieval": float(da["activations"]["retrieval_acc"]
                                                     - db["activations"]["retrieval_acc"]),
                            "p_paired": float((1.0 + ge) / (len(dn) + 1.0)),
                            "sd_null": float(np.std(dn)),
                            "note": ("the direct arm's own paired base-checkpoint contrast, on the "
                                     "same shared permutation prefix as the RSA's"),
                        })
    result["direct_base_contrast"] = direct_deltas

    # ---------------- figure-ready layer profiles (so a plot needs no rerun) ----------------
    prof = {}
    for c in cells:
        key = f"{c['model']}|{c['arm']}|{c['position']}|{c.get('metric', 'centered')}"
        p = prof.setdefault(key, {"layer": [], "r_raw": [], "r_partial": [], "p": [],
                                  "p_fwer_max": [], "sd_null": [], "MDE80": [],
                                  "r_disattenuated": [], "triplet_acc": [], "status": []})
        p["layer"].append(int(c["layer"]))
        p["status"].append(c.get("status"))
        for k in ("r_raw", "r_partial", "p", "p_fwer_max", "sd_null", "MDE80", "r_disattenuated"):
            p[k].append(c.get(k))
        p["triplet_acc"].append((c.get("triplet") or {}).get("triplet_acc"))
    result["profiles"] = prof
    result["cellline_cluster_ci"] = {
        "status": "unavailable_by_construction",
        "n_cell_line_clusters": 1,
        "cell_line_id": str(cl),
        "note": ("this script analyses ONE (cell_line, plate) stratum per invocation, so the run "
                 "contains exactly 1 cell line and cell-line clustering has 1 cluster, not 'fewer "
                 "than 5 usable clusters'. A cluster-robust variance with G = 1 is undefined, not "
                 "merely imprecise. Cross-cell-line variation is addressed only by running "
                 "separate strata and comparing them under multi_stratum_rule."),
    }
    result["output_space_comparator"] = (
        "notestablished: the internal number may NOT be contrasted with "
        "RESULTS_cluster/stratify_geometry.json test3 (different roster, grouping, frame and "
        "metric, no permutation and no interval). A matched companion generation pass is required "
        "before the contrast is written.")
    result["status"] = "ok" if gate["headline_permitted"] else gate["status"]
    write_json(result, args.out)
    return result


def write_json(result, out):
    """Writes exactly one file, at --out, and nothing anywhere else."""
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    out_abs = os.path.abspath(os.path.expanduser(str(out)))
    if os.path.isdir(out_abs):
        raise IsADirectoryError(f"--out points at an existing DIRECTORY ({out_abs}); refusing, "
                                f"because the only thing this script may create is that one file")
    if not os.path.basename(out_abs):
        raise ValueError(f"--out has no filename component: {out!r}")
    d = os.path.dirname(out_abs)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as fh:
        json.dump(clean(result), fh, indent=2)
    logger.info(f"-> {out_abs}   (the ONLY path this script writes)")


# =============================================================================================
# SELFTEST — the most important part of the file. No GPU, no data, no network, seconds.
# =============================================================================================
def _synth_names(n, rng, n_families=10):
    """Names with real token structure: a family stem plus a suffix, so name-token overlap has
    genuine block structure to confound with. The 'tokenizer' for the synthetic worlds splits on
    '-', which stands in for BPE without needing a model."""
    stems = ["vora", "gefi", "dasa", "erlo", "sunit", "imat", "nilo", "pazo", "cabo", "lena",
             "trame", "vemu", "rux", "tofa", "bort"]
    suff = ["tinib", "nib", "zomib", "stat", "mide", "cin", "mycin", "sartan", "olol", "pine"]
    names, tok_sets = [], []
    for i in range(n):
        f = i % n_families
        s = stems[f % len(stems)]
        t = suff[rng.randint(len(suff))]
        u = f"x{rng.randint(1000)}"
        nm = f"{s}-{t}-{u}"
        names.append(nm)
        tok_sets.append(set(nm.split("-")))
    return names, tok_sets


def _sym_noise(n, rng, scale):
    M = rng.randn(n, n) * scale
    M = np.abs(M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return M


def _controls_for(names, tok_sets, moas, arm, rng):
    n = len(names)
    return build_control_matrices(names, tok_sets, moas,
                                  [len(t) for t in tok_sets],
                                  [40 + rng.randint(6) for _ in range(n)], arm)


def _report(label, ok, detail):
    logger.info(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return bool(ok)


def selftest(n_perm=1499, seed=0):
    logger.info("=" * 92)
    logger.info("drug_geometry_rsa SELFTEST — planted worlds where the answer is known")
    logger.info("=" * 92)
    ok_all = True
    P, nd, K, HD = 946, 50, 16, 64

    # -------------------------------------------------------------------------------------
    # (a) POSITIVE / ORACLE. Activations are a linear image of the biological latent plus noise.
    #     Names and mechanisms are assigned AT RANDOM, so no confound can explain the result.
    #     Routed through the real activation_matrix and biology_matrix code paths, control-cell
    #     draw offsets included, so the centering and the K-draw averaging are exercised too.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(a) POSITIVE world — activations = linear(biology) + noise, names/moa random")
    rng = np.random.RandomState(seed)
    names, toks = _synth_names(nd, rng)
    # break any name<->latent alignment. The shuffle must move the NAME and its TOKEN SET together:
    # shuffling `names` alone (as this did) left x1/x5a keyed to one name and x1p/x8 keyed to
    # another, so the control block was internally inconsistent and no longer described any
    # roster. Harmless to the verdict in this world -- everything is random with respect to V --
    # but it made the controls untrustworthy exactly where they are being validated.
    _pi = rng.permutation(nd)
    names = [names[i] for i in _pi]
    toks = [toks[i] for i in _pi]
    # kept under their own names because `names`/`toks` are REBOUND by worlds (b), (b2) and (c)
    # below, and world (t) consumes world (a)'s and world (b2)'s activations. Passing whichever
    # roster happened to be bound last would hand the direct arm's names-only control a roster that
    # did not generate the data it is controlling for -- which is exactly the defect that arm exists
    # to detect elsewhere.
    names_a, toks_a = list(names), list(toks)
    moas = [f"moa{rng.randint(8)}" for _ in range(nd)]
    V = rng.randn(nd, 12)
    res = V @ rng.randn(12, P) * 1.0 + rng.randn(nd, P) * 0.35
    B_pos, _ = biology_matrix(res, P)
    Wh = rng.randn(12, HD)
    ctx = rng.randn(K, HD) * 6.0                        # the shared control-cell main effect
    H = (ctx[None, :, :] + (V @ Wh)[:, None, :] * 1.0 + rng.randn(nd, K, HD) * 0.30)
    A_pos, dA = activation_matrix(H, "centered")
    ctrls = _controls_for(names, toks, moas, "annotated", rng)
    prng = np.random.RandomState(1234)
    perms = [prng.permutation(nd) for _ in range(n_perm)]
    m = mrm_partial(A_pos, B_pos, ctrls, perms=perms)
    m.pop("_perm_stats", None); m.pop("_control_diag_design", None)
    rel_a = reliability_A(H, "centered", n_splits=8)
    ok = (m["r_partial"] is not None and m["r_partial"] >= 0.35
          and m["p"] <= 1.0 / (n_perm + 1) + 1e-12)
    ok_all &= _report("POSITIVE recovers a strong positive partial", ok,
                      f"r_raw={m['r_raw']:+.3f}  r_partial={m['r_partial']:+.3f} "
                      f"(threshold >= +0.350)  p={m['p']:.5f} (threshold <= {1.0/(n_perm+1):.5f})  "
                      f"sd_null={m['sd_null']:.3f}  rel_A={rel_a['spearman_brown']:.3f}")
    trip = matched_triplets(B_pos, ctrls)
    ts = triplet_stat(A_pos, trip, n_boot=800, n_perm=0)
    ok_all &= _report("POSITIVE co-primary triplet accuracy above 0.50",
                      ts.get("status") == "ok" and ts["triplet_acc"] > 0.55,
                      f"acc={ts.get('triplet_acc')} over {ts.get('n_triples')} matched triples "
                      f"(threshold > 0.550)")

    # -------------------------------------------------------------------------------------
    # (a2) MIXED world -- the case that actually occurs in the data: a REAL biological signal AND
    #      a name-token confound, both present. The partial must keep the biology (over-stripping
    #      would turn every real positive into a false null) while coming in clearly below the raw.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(a2) MIXED world — real biology PLUS a name confound that is CORRELATED with it; "
                "the partial must keep the biology and shed the inflation")
    rng = np.random.RandomState(seed + 21)
    names_m, toks_m = _synth_names(nd, rng)               # name family = index % 10
    moas_m = [f"moa{i % 8}" for i in range(nd)]
    ctrls_m = _controls_for(names_m, toks_m, moas_m, "annotated", rng)
    clus = np.arange(nd) % 10                             # biology shares the name families, so
    cent = rng.randn(10, 12)                              # the confound genuinely INFLATES the raw
    Vm = cent[clus] + rng.randn(nd, 12) * 0.55
    res_m = Vm @ rng.randn(12, P) + rng.randn(nd, P) * 0.35
    B_mix, _ = biology_matrix(res_m, P)
    vocab_m = sorted({t for s in toks_m for t in s})
    vim = {t: i for i, t in enumerate(vocab_m)}
    Um = np.zeros((nd, len(vocab_m)))
    for i, s in enumerate(toks_m):
        for t in s:
            Um[i, vim[t]] = 1.0
    ctxm = rng.randn(K, HD) * 6.0
    H_mix = (ctxm[None, :, :] + (Vm @ rng.randn(12, HD))[:, None, :]
             + (Um @ rng.randn(len(vocab_m), HD))[:, None, :] * 2.5
             + rng.randn(nd, K, HD) * 0.30)
    A_mix, _ = activation_matrix(H_mix, "centered")
    mm_mix = mrm_partial(A_mix, B_mix, ctrls_m, perms=perms)
    mm_mix.pop("_perm_stats", None); mm_mix.pop("_control_diag_design", None)
    gap = mm_mix["r_raw"] - mm_mix["r_partial"]
    ok = (mm_mix["r_partial"] > 0.30 and mm_mix["p"] < 0.01 and gap > 0.005)
    ok_all &= _report("MIXED: biology survives the controls, the inflation does not", ok,
                      f"r_raw={mm_mix['r_raw']:+.3f}  r_partial={mm_mix['r_partial']:+.3f}  "
                      f"raw-partial gap={gap:+.4f} "
                      f"(thresholds: partial > +0.300, p < 0.010, gap > 0.005)  "
                      f"p={mm_mix['p']:.4f}")
    logger.info("      This is the pair the chapter reports: (r_raw, r_partial) together, because "
                "their GAP is exactly how much of the internal geometry is the prompt string.")

    # -------------------------------------------------------------------------------------
    # (b) NEGATIVE / CONFOUND. The single most important line in the file.
    #     Activations and biology are BOTH generated from the name-token control and nothing else.
    #     There is no pharmacology anywhere in this world. The RAW statistic must therefore come
    #     out clearly positive, and the PARTIAL must collapse to ~0 -- if it does not, the control
    #     is not adjusting anything and every r_partial this script ever prints is void.
    #     The generator is the control matrix itself, so the annihilation has to be exact rather
    #     than approximate; the approximate version (activations built from name-token VECTORS) is
    #     world (b2) below.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(b) NEGATIVE/CONFOUND world — activations AND biology driven only by name tokens")
    rng = np.random.RandomState(seed + 1)
    names, toks = _synth_names(nd, rng)
    moas = [f"moa{rng.randint(8)}" for _ in range(nd)]
    ctrls_b = _controls_for(names, toks, moas, "annotated", rng)
    gen = ctrls_b["x1"]
    A_neg = gen + _sym_noise(nd, rng, 0.12)
    B_neg = gen + _sym_noise(nd, rng, 0.12)
    np.fill_diagonal(A_neg, 0.0); np.fill_diagonal(B_neg, 0.0)
    mn = mrm_partial(A_neg, B_neg, ctrls_b, perms=perms)
    mn.pop("_perm_stats", None); mn.pop("_control_diag_design", None)
    # Thresholds are the empirical envelope over 24 generating seeds, not aspirations:
    # r_raw >= 0.372, |r_partial| <= 0.060, |r_partial| / r_raw <= 0.159.
    ratio = abs(mn["r_partial"]) / max(EPS, abs(mn["r_raw"]))
    ok = (mn["r_raw"] > 0.30 and abs(mn["r_partial"]) < 0.08 and ratio < 0.25 and mn["p"] > 0.01)
    ok_all &= _report("NEGATIVE: raw positive, partial annihilated", ok,
                      f"r_raw={mn['r_raw']:+.3f} (threshold > +0.300)  "
                      f"r_partial={mn['r_partial']:+.3f} (threshold |r| < 0.080)  "
                      f"ratio={ratio:.3f} (threshold < 0.250)  "
                      f"p={mn['p']:.3f} (threshold > 0.010)")

    logger.info("")
    logger.info("(b3) MULTI-CONFOUND world — name tokens AND char n-grams AND shared mechanism")
    rng = np.random.RandomState(seed + 11)
    names_b3, toks_b3 = _synth_names(nd, rng)
    moas_b3 = ["unclear" if rng.rand() < 0.5 else f"moa{rng.randint(5)}" for _ in range(nd)]
    ctrls_b3 = _controls_for(names_b3, toks_b3, moas_b3, "annotated", rng)
    gen3 = (0.55 * ctrls_b3["x1"] + 0.30 * ctrls_b3["x1p"] + 0.15 * (1.0 - ctrls_b3["x3"]))
    A_b3 = gen3 + _sym_noise(nd, rng, 0.06); np.fill_diagonal(A_b3, 0.0)
    B_b3 = gen3 + _sym_noise(nd, rng, 0.06); np.fill_diagonal(B_b3, 0.0)
    m3 = mrm_partial(A_b3, B_b3, ctrls_b3, perms=perms)
    m3.pop("_perm_stats", None); m3.pop("_control_diag_design", None)
    ratio3 = abs(m3["r_partial"]) / max(EPS, abs(m3["r_raw"]))
    ok = (m3["r_raw"] > 0.45 and ratio3 < 0.25)
    ok_all &= _report("MULTI-CONFOUND: controls remove the great majority of the raw association",
                      ok,
                      f"r_raw={m3['r_raw']:+.3f} (threshold > +0.450)  "
                      f"r_partial={m3['r_partial']:+.3f}  ratio={ratio3:.3f} (threshold < 0.250)")
    logger.info("      NOTE, and it matters for reading the real run: the design partials out the "
                "RANKS of each control, while a several-component confound enters A through a "
                "monotone-but-not-rank-linear composition. Annihilation is therefore not exact "
                "here -- across generating seeds this world leaves a residue of up to |r| ~ 0.12. "
                "A small positive r_partial in the real run is NOT by itself evidence of "
                "pharmacological organisation.")

    logger.info("")
    logger.info("(b2) NAME-DRIVEN ACTIVATIONS — the same confound routed through the vector path")
    rng = np.random.RandomState(seed + 2)
    names, toks = _synth_names(nd, rng)
    names_b2, toks_b2 = list(names), list(toks)     # world (t2) consumes THIS roster; see world (a)
    moas = [f"moa{rng.randint(8)}" for _ in range(nd)]
    vocab = sorted({t for s in toks for t in s})
    vi = {t: i for i, t in enumerate(vocab)}
    U = np.zeros((nd, len(vocab)))
    for i, s in enumerate(toks):
        for t in s:
            U[i, vi[t]] = 1.0
    ctrls_b2 = _controls_for(names, toks, moas, "annotated", rng)
    Wt = rng.randn(len(vocab), HD)
    ctx = rng.randn(K, HD) * 6.0
    H2 = ctx[None, :, :] + (U @ Wt)[:, None, :] * 2.0 + rng.randn(nd, K, HD) * 0.2
    A_b2, _ = activation_matrix(H2, "centered")
    res2 = U @ rng.randn(len(vocab), P) * 1.0 + rng.randn(nd, P) * 0.30
    B_b2, _ = biology_matrix(res2, P)
    m2 = mrm_partial(A_b2, B_b2, ctrls_b2, perms=perms)
    m2.pop("_perm_stats", None); m2.pop("_control_diag_design", None)
    # Jaccard is not the exact functional form the generator used, so the annihilation here is
    # partial by construction. The honest, robust claim is that the control removes most of it.
    ok = (m2["r_raw"] > 0.10 and abs(m2["r_partial"]) < 0.55 * abs(m2["r_raw"]))
    ok_all &= _report("NAME-DRIVEN: controls remove most of the raw association", ok,
                      f"r_raw={m2['r_raw']:+.3f}  r_partial={m2['r_partial']:+.3f}  "
                      f"(threshold |r_partial| < 0.55*|r_raw| = {0.55*abs(m2['r_raw']):.3f})")

    # -------------------------------------------------------------------------------------
    # (c) MECHANISM-BLOCK world: both matrices are pure shared-mechanism block structure.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(c) MECHANISM-BLOCK world — both matrices are shared-mechanism block structure")
    rng = np.random.RandomState(seed + 3)
    names, toks = _synth_names(nd, rng)
    moas = [f"moa{i % 6}" for i in range(nd)]
    rng.shuffle(moas)
    ctrls_c = _controls_for(names, toks, moas, "annotated", rng)
    genc = 1.0 - ctrls_c["x3"]
    A_c = genc + _sym_noise(nd, rng, 0.15); np.fill_diagonal(A_c, 0.0)
    B_c = genc + _sym_noise(nd, rng, 0.15); np.fill_diagonal(B_c, 0.0)
    mc = mrm_partial(A_c, B_c, ctrls_c, perms=perms)
    mc.pop("_perm_stats", None); mc.pop("_control_diag_design", None)
    ok = (mc["r_raw"] > 0.30 and abs(mc["r_partial"]) < 0.08)
    ok_all &= _report("MECHANISM-BLOCK: raw positive, partial annihilated", ok,
                      f"r_raw={mc['r_raw']:+.3f} (threshold > +0.300)  "
                      f"r_partial={mc['r_partial']:+.3f} (threshold |r| < 0.080)")

    # -------------------------------------------------------------------------------------
    # (d) DEGENERATE-MECHANISM world: 70% of drugs carry 'unclear'. Activations are driven by
    #     shared KNOWN mechanism, biology by shared IGNORANCE. Splitting x3 and x4 annihilates it;
    #     collapsing them into one 'shared mechanism' indicator manufactures a spurious partial.
    #     This is the test that justifies never collapsing them.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(d) DEGENERATE-MECHANISM world — 70% 'unclear'; x3/x4 split vs collapsed coding")
    rng = np.random.RandomState(seed + 4)
    nd_d = 70
    names_d, toks_d = _synth_names(nd_d, rng)
    moas = ["unclear" if rng.rand() < 0.70 else f"moa{rng.randint(2)}" for _ in range(nd_d)]
    ctrls_d = _controls_for(names_d, toks_d, moas, "annotated", rng)
    A_d = ctrls_d["x3"] + _sym_noise(nd_d, rng, 0.20); np.fill_diagonal(A_d, 0.0)
    B_d = ctrls_d["x4"] + _sym_noise(nd_d, rng, 0.20); np.fill_diagonal(B_d, 0.0)
    prd = np.random.RandomState(77)
    perms_d = [prd.permutation(nd_d) for _ in range(400)]
    m_split = mrm_partial(A_d, B_d, ctrls_d, perms=perms_d)
    m_coll = mrm_partial(A_d, B_d, ctrls_d, perms=perms_d, collapse_x3_x4=True)
    for mm in (m_split, m_coll):
        mm.pop("_perm_stats", None); mm.pop("_control_diag_design", None)
    # Residualising on the UNION of the two indicators leaves x3 and x4 exactly anti-proportional
    # on the dyads where the union fires, so the collapsed coding does not merely lose power -- it
    # manufactures a correlation with the wrong sign out of shared ignorance.
    ok = (abs(m_split["r_partial"]) < 0.06 and abs(m_coll["r_partial"]) > 0.09)
    ok_all &= _report("x3/x4 SPLIT annihilates where the collapsed indicator does not", ok,
                      f"split r_partial={m_split['r_partial']:+.3f} (threshold |r| < 0.060)  vs  "
                      f"collapsed r_partial={m_coll['r_partial']:+.3f} (threshold |r| > 0.090, "
                      f"and note its sign)")

    # -------------------------------------------------------------------------------------
    # (e) DEGENERACY GUARD: identical activation vectors must be RECORDED as degenerate, never
    #     returned as 0.0 into an aggregate. (Layer 0 at the last position is exactly this case:
    #     the token is ':' for every prompt.)
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(e) DEGENERACY GUARD — identical activations must be recorded, not scored 0.0")
    Hd = np.tile(np.random.RandomState(9).randn(1, 1, HD), (nd, K, 1))
    A_deg, d_deg = activation_matrix(Hd, "centered")
    cell, ps = analyse_cell(A_deg, B_pos, ctrls, names, perms[:10], None, None, None, 0, 0,
                            {"model": "synthetic", "layer": 0, "position": "last"}, d_deg)
    ok = (d_deg.get("status") == "degenerate_by_construction"
          and cell.get("status") == "degenerate_by_construction"
          and "r_partial" not in cell and ps is None)
    ok_all &= _report("DEGENERACY recorded, no coefficient emitted", ok,
                      f"activation status={d_deg.get('status')}  cell status={cell.get('status')}")

    # -------------------------------------------------------------------------------------
    # (f) DYADIC DEPENDENCE: the influence-function interval must be materially wider than an iid
    #     interval on the same values. Dyads sharing a drug are not independent, and an interval
    #     that ignores that is false precision.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(f) DYADIC DEPENDENCE — cluster-robust interval vs iid on the same influence values")
    rng = np.random.RandomState(seed + 6)
    node = rng.randn(nd) * 1.0
    ii, jj = dyad_index(nd)
    vals = node[ii] + node[jj] + rng.randn(len(ii)) * 0.3
    ci_dy = _dyadic_cluster_ci(vals, [f"d{x}" for x in ii], [f"d{x}" for x in jj],
                              df_override=nd - 1)
    iid_hw = 1.96 * float(np.std(vals, ddof=1)) / math.sqrt(len(vals))
    dy_hw = (ci_dy["hi"] - ci_dy["lo"]) / 2.0 if ci_dy and np.isfinite(ci_dy.get("hi", np.nan)) else None
    # Threshold raised 1.30 -> 4.00. In this world the planted values are node_i + node_j + noise,
    # whose true variance ratio is known analytically at 7.51; the shipped estimator measures 7.56
    # in one independent replication and 6.88 at this seed. At 1.30 a regression that silently
    # degraded the Fafchamps-Gubert variance to near-iid would still pass, and every interval in the
    # file would be ~7x too narrow with no alarm. 4.00 still leaves ~1.7x margin, and the plain
    # constant is preferred over an analytic-ratio-within-25% form, which would be brittle against
    # the ~10% seed-to-seed spread.
    ok = dy_hw is not None and dy_hw >= 4.0 * iid_hw
    ok_all &= _report("dyadic interval materially wider than iid", ok,
                      f"dyadic half-width={dy_hw:.4f}  iid half-width={iid_hw:.4f}  "
                      f"ratio={dy_hw/iid_hw:.2f} (threshold >= 4.00; analytic truth for this "
                      f"planted world is ~7.5)")

    # -------------------------------------------------------------------------------------
    # (g) NULL CALIBRATION: with A and B independent, the permutation p must behave like a p-value.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(g) NULL CALIBRATION — independent A and B over 120 replicates")
    ps_list, rs = [], []
    small = 25
    for rep in range(120):
        r2 = np.random.RandomState(9000 + rep)
        nm, tk = _synth_names(small, r2)
        mo = [f"moa{r2.randint(6)}" for _ in range(small)]
        cc = _controls_for(nm, tk, mo, "annotated", r2)
        Aq = _sym_noise(small, r2, 1.0)
        Bq = _sym_noise(small, r2, 1.0)
        pr_rng = np.random.RandomState(500 + rep)          # ONE stream: re-seeding inside the
        pr = [pr_rng.permutation(small) for _ in range(199)]  # comprehension gives 199 identical
        #                                                     permutations and a degenerate null
        mm = mrm_partial(Aq, Bq, cc, perms=pr)
        mm.pop("_perm_stats", None); mm.pop("_control_diag_design", None)
        ps_list.append(mm["p"]); rs.append(mm["r_partial"])
    ps_arr = np.array(ps_list)
    frac05 = float(np.mean(ps_arr <= 0.05))
    ok = (abs(float(np.mean(rs))) < 0.06 and 0.30 <= float(np.mean(ps_arr)) <= 0.70
          and frac05 <= 0.14)
    ok_all &= _report("permutation p calibrated under the null", ok,
                      f"mean r_partial={np.mean(rs):+.4f} (|.| < 0.060)  mean p={np.mean(ps_arr):.3f} "
                      f"(in [0.30, 0.70])  frac(p<=0.05)={frac05:.3f} (<= 0.140)")

    # -------------------------------------------------------------------------------------
    # (h) DISATTENUATION: a planted truth with known reliabilities must come back.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(h) DISATTENUATION — recover a planted true correlation from known reliabilities")
    true_r, ra, rb = 0.50, 0.64, 0.49
    obs = true_r * math.sqrt(ra * rb)
    rec = _disattenuate(obs, ra, rb)
    ok = rec is not None and abs(rec - true_r) < 0.05
    ok_all &= _report("two-sided disattenuation recovers the planted value", ok,
                      f"observed={obs:.4f} -> recovered={rec:.4f} vs planted {true_r:.2f} "
                      f"(tolerance 0.050)")

    # -------------------------------------------------------------------------------------
    # (i) rel_B GATE: an unreliable biological matrix must refuse, not return a null.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(i) rel_B GATE — an unreliable biology matrix must refuse rather than null out")
    g_bad = rel_B_gate({"rel_B": 0.10, "lo": 0.02, "hi": 0.20})
    g_mid = rel_B_gate({"rel_B": 0.30, "lo": 0.24, "hi": 0.38})
    g_ok = rel_B_gate({"rel_B": 0.55, "lo": 0.46, "hi": 0.64})
    # the case that pins the consistency repair: a POINT estimate above 0.40 whose lower bound is
    # not. Under the old code -- point estimate at 0.40, lower bound at 0.20 -- this returned 'ok'
    # and permitted an unqualified headline. Both thresholds now read the lower bound.
    g_edge = rel_B_gate({"rel_B": 0.45, "lo": 0.35, "hi": 0.55})
    ok = (g_bad["status"] == "no_measurable_biological_geometry" and not g_bad["headline_permitted"]
          and g_mid["status"] == "attenuation_limited" and g_ok["status"] == "ok"
          and g_edge["status"] == "attenuation_limited" and g_edge["headline_permitted"])
    ok_all &= _report("gate thresholds fire in the right order, on the lower bound in BOTH "
                      "directions", ok,
                      f"{{0.10, lo 0.02}} -> {g_bad['status']}  |  {{0.30, lo 0.24}} -> "
                      f"{g_mid['status']}  |  {{0.55, lo 0.46}} -> {g_ok['status']}  |  NEW "
                      f"{{0.45, lo 0.35}} -> {g_edge['status']} (was 'ok' when the 0.40 threshold "
                      f"read the point estimate)")

    # -------------------------------------------------------------------------------------
    # (j) rel_B pipeline end to end on planted data: a reliable world scores high, an all-noise
    #     world scores at the floor and trips the gate.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(j) rel_B PIPELINE — reliable world vs all-noise world")
    r3 = np.random.RandomState(seed + 10)
    Vt = r3.randn(20, 12) @ r3.randn(12, P)
    good = [(Vt + r3.randn(20, P) * 0.4, Vt + r3.randn(20, P) * 0.4) for _ in range(6)]
    bad = [(r3.randn(20, P), r3.randn(20, P)) for _ in range(6)]
    rb_good = reliability_B(good, P, n_boot=200, seed=1)
    rb_bad = reliability_B(bad, P, n_boot=200, seed=1)
    g_good, g_bad2 = rel_B_gate(rb_good), rel_B_gate(rb_bad)
    ok = (rb_good is not None and rb_good["rel_B"] > 0.40 and g_good["headline_permitted"]
          and rb_bad is not None and not g_bad2["headline_permitted"])
    ok_all &= _report("rel_B separates a reliable target from an unreliable one", ok,
                      f"reliable rel_B={rb_good['rel_B']:.3f} -> {g_good['status']} "
                      f"(threshold > 0.400, headline permitted)  |  "
                      f"noise rel_B={rb_bad['rel_B']:.3f} -> {g_bad2['status']} "
                      f"(headline refused)")

    # -------------------------------------------------------------------------------------
    # (k) COLLINEAR CONTROLS. The real design IS rank deficient: x2 (mechanism-token Jaccard) is an
    #     exact affine function of x3 whenever the mechanism strings are single tokens, and x5b
    #     (prompt length difference) equals x5a (name length difference) whenever the prompt varies
    #     only through the drug name -- which is exactly what the balanced design enforces. The
    #     residualiser must project onto col(X) ITSELF; an unpivoted-QR projector keeps orthonormal
    #     columns for the dependent regressors, which point OUTSIDE col(X), and strips real signal
    #     along them. Signal must survive, confound must still die, deficiency must be reported.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(k) COLLINEAR CONTROLS — a rank-deficient design must be reported, not absorbed")

    def _make_collinear(c):
        d = {kk: vv.copy() for kk, vv in c.items()}
        d["x5b"] = d["x5a"].copy()               # prompt length varies only through the name
        d["x2"] = 1.0 - d["x3"]                  # single-token mechanism strings
        np.fill_diagonal(d["x2"], 0.0)
        return d

    m_k = mrm_partial(A_pos, B_pos, _make_collinear(ctrls), perms=perms)
    m_kn = mrm_partial(A_neg, B_neg, _make_collinear(ctrls_b), perms=perms)
    for mm in (m_k, m_kn):
        mm.pop("_perm_stats", None); mm.pop("_control_diag_design", None)
    ok = (m_k.get("design_rank_deficient") is True
          and m_k.get("r_partial") is not None and m_k["r_partial"] >= 0.35
          and m_kn.get("r_partial") is not None and abs(m_kn["r_partial"]) < 0.08)
    ok_all &= _report("rank-deficient design reported; signal kept, confound still annihilated", ok,
                      f"rank {m_k.get('design_rank')}/{m_k.get('design_n_columns')} "
                      f"(deficient={m_k.get('design_rank_deficient')})  "
                      f"POSITIVE r_partial={m_k['r_partial']:+.3f} (threshold >= +0.350)  "
                      f"NEGATIVE r_partial={m_kn['r_partial']:+.3f} (threshold |r| < 0.080)")

    # -------------------------------------------------------------------------------------
    # (l) INPUT GUARDS. Only the upper triangle of a distance matrix is ever read and NaN compares
    #     False, so an asymmetric or non-finite matrix would be silently half-used rather than
    #     rejected; a drug encoding to an all-zero signed-rank profile gets distance exactly 1.0 to
    #     every other drug, which is a fabricated row. Refuse the first two, count the third.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(l) INPUT GUARDS — asymmetry and NaN refused, empty encodings counted")
    checks = []
    A_asym = A_pos.copy(); A_asym[0, 1] += 0.5
    try:
        mrm_partial(A_asym, B_pos, ctrls, perms=perms[:5])
        checks.append(("asymmetric A refused", False))
    except AssertionError:
        checks.append(("asymmetric A refused", True))
    B_nan = B_pos.copy(); B_nan[2, 3] = B_nan[3, 2] = np.nan
    try:
        mrm_partial(A_pos, B_nan, ctrls, perms=perms[:5])
        checks.append(("NaN in B refused", False))
    except AssertionError:
        checks.append(("NaN in B refused", True))
    try:
        biology_matrix(np.array([[np.nan, 1.0], [0.0, 1.0]]), 2)
        checks.append(("NaN residuals refused", False))
    except AssertionError:
        checks.append(("NaN residuals refused", True))
    R_empty = np.zeros((6, P)); R_empty[0] = np.arange(P) - P / 2.0
    _, _, dz = biology_matrix(R_empty, P, diag=True)
    checks.append(("empty encodings counted", dz["n_empty_profiles"] == 5))
    # the dose parser feeds x6; a silent mis-parse would put an arbitrary covariate in the design
    checks.append(("dose units parsed", all([
        parse_dose_um("0.5 uM") == 0.5, parse_dose_um("10 nM") == 0.01,
        abs(parse_dose_um("1 mM") - 1000.0) < 1e-9, parse_dose_um("2 µM") == 2.0,
        parse_dose_um("3 μM") == 3.0, parse_dose_um("unknown") is None,
        parse_dose_um(None) is None, parse_dose_um("0 uM") is None])))
    # THE PARTIAL-PARSE PATH. At 0.90 <= parse rate < 1.0 dose_frame_diagnostics used to hand back a
    # vector containing None, and build_control_matrices then evaluated abs(a - b) on it and raised
    # TypeError -- after the checkpoint was loaded and the whole forward sweep was already paid for.
    # A dyadic design has no missing-value semantics, so the covariate must be TOTAL or absent.
    #   world 1: 19/20 parse (rate 0.95, above the threshold) -> x6 BUILT, finite everywhere, exactly
    #            one drug recorded as imputed;
    #   world 2: 16/20 parse (rate 0.80, below it)            -> x6 NOT built and nothing to impute.
    def _dose_world(n_ok, n_tot=20):
        dd = [f"drug{i}" for i in range(n_tot)]
        ge = {d: [{"metadata": {"dose": ("0.5 uM" if i < n_ok else "unknown")}} for _ in range(4)]
              for i, d in enumerate(dd)}
        return dd, ge

    dd_hi, ge_hi = _dose_world(19)
    diag_hi, ml_hi = dose_frame_diagnostics(ge_hi, dd_hi, "0.5 uM")
    dd_lo, ge_lo = _dose_world(16)
    diag_lo, ml_lo = dose_frame_diagnostics(ge_lo, dd_lo, "0.5 uM")
    dose_partial_ok = False
    try:
        C_hi = build_control_matrices(dd_hi, [{i} for i in range(20)], ["unclear"] * 20,
                                      [1] * 20, [40] * 20, "blind", dose_log10=ml_hi)
        dose_partial_ok = (diag_hi["x6_built"] is True
                           and diag_hi["n_drugs_dose_imputed"] == 1
                           and diag_hi["drugs_dose_imputed"] == ["drug19"]
                           and ml_hi is not None and all(v is not None for v in ml_hi)
                           and "x6" in C_hi and np.isfinite(C_hi["x6"]).all()
                           and diag_lo["x6_built"] is False and ml_lo is None
                           and abs(diag_lo["drug_parse_rate"] - 0.80) < 1e-9)
    except Exception:
        dose_partial_ok = False
    checks.append(("partial dose parse imputed, not crashed", dose_partial_ok))
    ok = all(v for _, v in checks)
    ok_all &= _report("degenerate inputs refused or counted, never silently scored", ok,
                      "  ".join(f"{kk}={'ok' if v else 'MISSED'}" for kk, v in checks))

    # -------------------------------------------------------------------------------------
    # (m) TRIPLET TOLERANCE. The co-primary is nominated precisely because it holds the confounds
    #     constant BY MATCHING. A tolerance of 0.43-0.59 SD of the confound is not matching, and
    #     the delta >= 0.5*sd(B) admission rule then preferentially selects the triples the
    #     confound itself separates. Under the zero-pharmacology (b3) recipe the accuracy must sit
    #     at the 0.50 null with tol_frac = 0.10, while a wide tolerance on the IDENTICAL code path
    #     reads clearly above it -- and the (a) POSITIVE world must be undamaged.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(m) TRIPLET TOLERANCE — matched triples under the (b3) zero-pharmacology confound")
    acc_tight, acc_wide, n_tight = [], [], []
    for s in range(seed + 11, seed + 27):
        r_m = np.random.RandomState(s)
        nm_m, tk_m = _synth_names(nd, r_m)
        mo_m = ["unclear" if r_m.rand() < 0.5 else f"moa{r_m.randint(5)}" for _ in range(nd)]
        c_m = _controls_for(nm_m, tk_m, mo_m, "annotated", r_m)
        gen_m = 0.55 * c_m["x1"] + 0.30 * c_m["x1p"] + 0.15 * (1.0 - c_m["x3"])
        A_m = gen_m + _sym_noise(nd, r_m, 0.06); np.fill_diagonal(A_m, 0.0)
        B_m = gen_m + _sym_noise(nd, r_m, 0.06); np.fill_diagonal(B_m, 0.0)
        t_tight = triplet_stat(A_m, matched_triplets(B_m, c_m, tol_frac=0.10), n_boot=20, n_perm=0)
        t_wide = triplet_stat(A_m, matched_triplets(B_m, c_m, tol_frac=0.60), n_boot=20, n_perm=0)
        if t_tight.get("status") == "ok":
            acc_tight.append(t_tight["triplet_acc"]); n_tight.append(t_tight["n_triples"])
        if t_wide.get("status") == "ok":
            acc_wide.append(t_wide["triplet_acc"])
    mean_tight = float(np.mean(acc_tight)) if acc_tight else float("nan")
    mean_wide = float(np.mean(acc_wide)) if acc_wide else float("nan")
    trip_pos = triplet_stat(A_pos, matched_triplets(B_pos, ctrls, tol_frac=0.10),
                            n_boot=200, n_perm=0)
    ok = (abs(mean_tight - 0.50) <= 0.02 and mean_wide > mean_tight
          and trip_pos.get("status") == "ok" and trip_pos["triplet_acc"] > 0.55
          and trip_pos["n_triples"] > 200 and min(n_tight) > 200)
    ok_all &= _report("tol_frac=0.10 returns the 0.50 null under a pure confound; the POSITIVE "
                      "world is undamaged", ok,
                      f"confounded acc(tol_frac=0.10)={mean_tight:.4f} over {len(acc_tight)} worlds "
                      f"(threshold |acc - 0.500| <= 0.020)  vs wide tol_frac=0.60 "
                      f"acc={mean_wide:.4f} (must exceed it)  |  POSITIVE acc="
                      f"{trip_pos.get('triplet_acc'):.4f} (threshold > 0.550) over "
                      f"{trip_pos.get('n_triples')} triples  min confounded triples={min(n_tight)} "
                      f"(threshold > 200)  |  realised tolerances "
                      f"tol_x1={trip_pos['matching']['tol_x1']:.4f} "
                      f"(sd {trip_pos['matching']['sd_offdiag_x1']:.4f}), "
                      f"tol_x1p={trip_pos['matching']['tol_x1p']:.4f} "
                      f"(sd {trip_pos['matching']['sd_offdiag_x1p']:.4f}), keys with variance="
                      f"{trip_pos['matching']['n_matching_keys_with_variance']}/7")

    # -------------------------------------------------------------------------------------
    # (n) NAME-CONTROL GUARD. The one sentence the whole partial statistic exists to earn is
    #     "controlling for name-token overlap". On a curated roster of 60 distinct compounds
    #     essentially no pair shares a BPE token, and the flag designed to withhold that sentence
    #     must fire there -- while staying quiet on a roster with genuine shared stems.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(n) NAME-CONTROL GUARD — must fire on a roster where ~1 of 1770 pairs shares a token")
    nd_n = 60
    r_n = np.random.RandomState(seed + 31)
    # a roster of distinct compounds: every name its own token set, except ONE deliberately
    # planted overlapping pair, which is the realistic case
    toks_sparse = [{f"u{i}", f"v{i}"} for i in range(nd_n)]
    toks_sparse[1] = {"u0", f"v{1}"}                       # drugs 0 and 1 share exactly one token
    names_sparse = [f"cmpd{i}" for i in range(nd_n)]
    ctrls_sparse = build_control_matrices(names_sparse, toks_sparse,
                                          [f"moa{i % 5}" for i in range(nd_n)],
                                          [len(t) for t in toks_sparse],
                                          [40 + r_n.randint(6) for _ in range(nd_n)], "annotated")
    A_n = _sym_noise(nd_n, r_n, 1.0); B_n = _sym_noise(nd_n, r_n, 1.0)
    pr_n = np.random.RandomState(4242)
    perms_n = [pr_n.permutation(nd_n) for _ in range(199)]
    m_sparse = mrm_partial(A_n, B_n, ctrls_sparse, perms=perms_n)
    names_d2, toks_d2 = _synth_names(nd_n, np.random.RandomState(seed + 32))
    ctrls_dense = _controls_for(names_d2, toks_d2, [f"moa{i % 5}" for i in range(nd_n)],
                                "annotated", np.random.RandomState(seed + 33))
    m_dense = mrm_partial(A_n, B_n, ctrls_dense, perms=perms_n)
    for mm in (m_sparse, m_dense):
        mm.pop("_perm_stats", None); mm.pop("_control_diag_design", None)
    ok = (m_sparse.get("name_control_inert") is True
          and m_dense.get("name_control_inert") is False)
    ok_all &= _report("name_control_inert fires on the sparse roster and not on the stem roster", ok,
                      f"sparse: frac_sharing={m_sparse['x1_frac_sharing']:.5f} "
                      f"frac_nonzero={m_sparse['x1_frac_nonzero']:.4f} "
                      f"var={np.var(offdiag(ctrls_sparse['x1'])):.3e} "
                      f"n_unique={m_sparse['x1_n_unique_offdiag']} -> "
                      f"inert={m_sparse.get('name_control_inert')} (must be True; the OLD trigger "
                      f"frac_nonzero < 0.02 could not fire here)  |  stem roster: "
                      f"frac_sharing={m_dense['x1_frac_sharing']:.4f} -> "
                      f"inert={m_dense.get('name_control_inert')} (must be False)")

    # -------------------------------------------------------------------------------------
    # (o) DISATTENUATED EQUIVALENCE. A world where the TRUE correlation is 0.50 -- comfortably
    #     above the 0.30 margin -- but the attenuated interval sits inside +/-0.30 because the two
    #     reliabilities shrink it. The attenuated test must say "equivalent" (that is the defect);
    #     the disattenuated test must say "not_equivalent" (that is the repair).
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(o) DISATTENUATED EQUIVALENCE — a true r of 0.50 must not be declared equivalent to 0")
    true_r_o, ra_o, rb_o = 0.50, 0.70, 0.35
    att = true_r_o * math.sqrt(ra_o * rb_o)                       # 0.2475
    ci_o = {"lo": att - 0.045, "hi": att + 0.045, "point": att}   # the measured n=60 half-width
    t_att = _tost(ci_o, 0.30)
    t_dis = _tost_disattenuated(ci_o, ra_o, rb_o, 0.30)
    # and the honest third case: a genuinely small effect with a wide interval must be reported as
    # underpowered rather than as equivalence
    t_wide = _tost_disattenuated({"lo": -0.20, "hi": 0.20, "point": 0.0}, ra_o, rb_o, 0.30)
    ok = (t_att is not None and t_att["equivalent"] is True
          and t_dis["status"] == "not_equivalent"
          and t_wide["status"] == "underpowered_for_equivalence")
    ok_all &= _report("equivalence is judged on the disattenuated scale, and refuses when underpowered",
                      ok,
                      f"planted true r={true_r_o:.2f} (rel_A={ra_o}, rel_B={rb_o}) -> attenuated "
                      f"r={att:.4f}, ci=[{ci_o['lo']:.4f}, {ci_o['hi']:.4f}]  |  ATTENUATED test: "
                      f"equivalent={t_att['equivalent']} (this is the defect: it declares a true "
                      f"0.50 equivalent to zero)  |  DISATTENUATED test: status={t_dis['status']} "
                      f"against margin_eff={t_dis['margin_effective']:.4f} (required "
                      f"not_equivalent)  |  wide-interval case: {t_wide['status']} (required "
                      f"underpowered_for_equivalence)")

    # -------------------------------------------------------------------------------------
    # (p) rel_A BY CONSTRUCTION. At 'name' and 'instr' the K control sentences are causally
    #     downstream of the read token, so the draws are bit-identical and the half-split is 1.0
    #     exactly -- which spearman_brown rejects as >= 1, returning None and silently making
    #     r_disattenuated unavailable at exactly the positions the co-primary compares. The
    #     invariance must be DETECTED and priced at 1.0, and a genuinely varying H must still be
    #     estimated the normal way.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(p) rel_A BY CONSTRUCTION — draw-invariant activations must give 1.0, not None")
    r_p = np.random.RandomState(seed + 41)
    H_inv = np.repeat(r_p.randn(nd, 1, HD) * 3.0, K, axis=1)      # K identical draws per drug
    H_var = H_inv + r_p.randn(nd, K, HD) * 0.30                    # the same world, draws varying
    old_inv = reliability_A(H_inv, "centered", n_splits=8)
    new_inv = rel_A_for(H_inv, "centered", n_splits=8)
    new_var = rel_A_for(H_var, "centered", n_splits=8)
    ok = (old_inv is not None and old_inv["spearman_brown"] is None
          and new_inv["spearman_brown"] == 1.0
          and new_inv["source"] == "control_draw_invariant_by_causal_structure"
          and new_var is not None and new_var["spearman_brown"] is not None
          and 0.0 < new_var["spearman_brown"] <= 1.0
          and new_var["source"] == "half_split_over_control_draws")
    ok_all &= _report("draw invariance is detected and priced at 1.0, varying draws still estimated",
                      ok,
                      f"OLD path on invariant H: half_split={old_inv['half_split_spearman']:.16f} -> "
                      f"spearman_brown={old_inv['spearman_brown']} (this is the defect)  |  NEW "
                      f"path: spearman_brown={new_inv['spearman_brown']} "
                      f"source={new_inv['source']} max_rel_dev={new_inv['max_rel_draw_deviation']:.2e}"
                      f"  |  varying H: spearman_brown={new_var['spearman_brown']:.4f} "
                      f"source={new_var['source']}")

    # -------------------------------------------------------------------------------------
    # (q) SPECTRUM. Two worlds whose participation ratio and top-eigenvalue share are known
    #     ANALYTICALLY, because the planted Hbar is built with prescribed singular values. The
    #     reported diagnostic must recover both. Construction: H[d,k] = ctx[k] + M[d] with M
    #     column-centered across drugs, so the within-draw centering and K-averaging return
    #     exactly M and the planted spectrum is the measured one.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(q) SPECTRUM — participation ratio and top-eigenvalue share against analytic truth")

    def _plant_spectrum(svals, rng_):
        r_ = len(svals)
        Q, _ = np.linalg.qr(np.column_stack([np.ones(nd), rng_.randn(nd, r_)]))
        U = Q[:, 1:1 + r_]                                  # orthonormal, orthogonal to 1 -> mean 0
        Vt = np.linalg.qr(rng_.randn(HD, r_))[0].T
        M = U @ np.diag(svals) @ Vt
        ctx_ = rng_.randn(K, HD) * 5.0
        return ctx_[None, :, :] + M[:, None, :]

    r_q = np.random.RandomState(seed + 51)
    spec_checks = []
    for svals in ([10.0, 1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]):
        sv = np.array(svals, dtype=float)
        pr_true = float(np.sum(sv ** 2) ** 2 / np.sum(sv ** 4))
        share_true = float(sv[0] ** 2 / np.sum(sv ** 2))
        _, dq = activation_matrix(_plant_spectrum(sv, r_q), "centered", spectrum=True)
        pr_hat = dq["spectrum"]["participation_ratio"]
        share_hat = dq["spectrum"]["top_eig_share"]
        spec_checks.append((pr_true, pr_hat, share_true, share_hat,
                            abs(pr_hat - pr_true) <= 0.05 * pr_true
                            and abs(share_hat - share_true) <= 0.05 * share_true))
    ok = all(c[4] for c in spec_checks)
    ok_all &= _report("eigenspectrum recovers the analytic participation ratio within 5%", ok,
                      "  |  ".join(f"PR true={a:.4f} got={b:.4f}; top share true={c:.4f} got={d:.4f}"
                                   for a, b, c, d, _ in spec_checks))

    # -------------------------------------------------------------------------------------
    # (r) CONFOUND-RESIDUE CALIBRATOR. It must find a real residue where one exists (the (b3)
    #     recipe) and return only the CHANCE envelope where none does (the noise_only control),
    #     and the two arms must differ, since that is the justification for calibrating per arm.
    #
    #     NOTE ON THE THRESHOLD. The brief for this world asked for "p95 below 0.02 on an all-noise
    #     control recipe". That number is unreachable and I decline it: p95 of |r_partial| under a
    #     clean null IS the chance envelope, about 1.96 * sd_null = 0.057 at n = 50 and 0.047 at
    #     n = 60, so 0.02 would fail for a perfectly correct calibrator. The honest form of the same
    #     requirement -- the one enforced here -- is that the noise_only p95 must sit AT the chance
    #     envelope and well BELOW the confounded recipe's, and that its permutation p must stay
    #     calibrated.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(r) CONFOUND-RESIDUE CALIBRATOR — a real residue vs the chance envelope")
    rng_r = np.random.RandomState(seed + 11)
    names_r, toks_r = _synth_names(nd, rng_r)
    moas_r = ["unclear" if rng_r.rand() < 0.5 else f"moa{rng_r.randint(5)}" for _ in range(nd)]
    ctrls_r = _controls_for(names_r, toks_r, moas_r, "annotated", rng_r)
    f_ann = confound_residue_floor(ctrls_r, "annotated", n_rep=60, n_perm=99, seed=seed)
    f_bli = confound_residue_floor(ctrls_r, "blind", n_rep=60, n_perm=99, seed=seed)
    f_noi = confound_residue_floor(ctrls_r, "noise_only", n_rep=60, n_perm=99, seed=seed)
    chance_env = 1.96 / math.sqrt(max(1, nd * (nd - 1) // 2 - 3))     # loose analytic reference
    ok = (f_ann["p95_abs_r_partial"] > 0.05
          and f_noi["p95_abs_r_partial"] <= 0.075
          and f_noi["p95_abs_r_partial"] < 0.75 * f_ann["p95_abs_r_partial"]
          and f_noi["frac_p_le_0.05"] <= 0.14
          and f_bli["p95_abs_r_partial"] > f_noi["p95_abs_r_partial"])
    ok_all &= _report("the calibrator separates a real confound residue from the chance envelope",
                      ok,
                      f"annotated p95={f_ann['p95_abs_r_partial']:.4f} "
                      f"(threshold > 0.050; mean={f_ann['mean_abs_r_partial']:.4f}, "
                      f"max={f_ann['max_abs_r_partial']:.4f}, "
                      f"frac(p<=.05)={f_ann['frac_p_le_0.05']:.3f})  |  blind "
                      f"p95={f_bli['p95_abs_r_partial']:.4f} (must exceed the chance control, which "
                      f"is why the floor is per-arm)  |  noise_only "
                      f"p95={f_noi['p95_abs_r_partial']:.4f} (threshold <= 0.075 and < 0.75x the "
                      f"annotated; loose analytic chance envelope {chance_env:.4f}), "
                      f"frac(p<=.05)={f_noi['frac_p_le_0.05']:.3f} (threshold <= 0.140)")

    # -------------------------------------------------------------------------------------
    # (s) ACTIVITY-MAGNITUDE CONFOUND. Drugs differ ONLY in how much they did; every drug-specific
    #     direction is iid random, so there is no pharmacological geometry anywhere. The headline
    #     partial must come out clearly positive -- that is the point, it is the mundane reading a
    #     referee will offer -- and the activity-adjusted secondary must collapse. The equal-
    #     amplitude version of the same world must show the covariates explain nothing, so the
    #     diagnostic is not just fitting noise.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(s) ACTIVITY-MAGNITUDE world — magnitude only, no pharmacology anywhere")
    # Averaged over 8 replicate worlds. A single seed is not enough here: this world's headline
    # ranges 0.14-0.56 and its residue -0.09 to +0.21 across seeds, so a one-seed threshold would
    # be a coin flip rather than a regression test.
    prs = np.random.RandomState(9911)
    perms_s = [prs.permutation(nd) for _ in range(99)]        # only r_partial is thresholded here
    r2_act, r2_equal, core_h, core_a, mono_h, mono_a, eig_share = [], [], [], [], [], [], []
    for s_i in range(seed + 61, seed + 69):
        rng_s = np.random.RandomState(s_i)
        names_s, toks_s = _synth_names(nd, rng_s)
        moas_s = [f"moa{rng_s.randint(8)}" for _ in range(nd)]
        ctrls_s = _controls_for(names_s, toks_s, moas_s, "annotated", rng_s)
        amp = np.exp(rng_s.randn(nd) * 1.1)                  # wide, log-normal activity spread
        U_s = rng_s.randn(nd, P)                             # iid directions: NO shared structure
        pb_s = amp[:, None] * U_s
        res_s = pb_s - pb_s.mean(axis=0, keepdims=True)      # the real frame: roster-mean centering
        B_s, _ = biology_matrix(res_s, P)
        _, C7_s = activity_covariates(res_s)
        st_s = activity_structure_of_B(B_s, C7_s)
        r2_act.append(st_s["r2_dyad_ranks_on_activity_only"])
        eig_share.append(st_s["leading_eigenvalue_share_of_B"])
        # the SAME directions at EQUAL amplitude: the covariates must now explain nothing, so the
        # diagnostic is not merely fitting noise
        res_e = U_s - U_s.mean(axis=0, keepdims=True)
        B_e, _ = biology_matrix(res_e, P)
        _, C7_e = activity_covariates(res_e)
        r2_equal.append(activity_structure_of_B(B_e, C7_e)["r2_dyad_ranks_on_activity_only"])
        ctrl_act = dict(ctrls_s); ctrl_act.update(C7_s)
        # (i) IDENTIFIED case, the analogue of world (b): the activation distance is driven by the
        #     activity covariate itself, so the adjustment must annihilate it.
        gen_s = C7_s["x7"] / max(1e-9, float(C7_s["x7"].max()))
        A_id = gen_s + _sym_noise(nd, rng_s, 0.12); np.fill_diagonal(A_id, 0.0)
        core_h.append(mrm_partial(A_id, B_s, ctrls_s, perms=perms_s)["r_partial"])
        core_a.append(mrm_partial(A_id, B_s, ctrl_act, perms=perms_s)["r_partial"])
        # (ii) MONOTONE-BUT-NOT-RANK-LINEAR case, the analogue of world (b3): the activation
        #      carries activity through log-amplitude and the real activation_matrix path.
        ctx_s = rng_s.randn(K, HD) * 6.0
        w_s = rng_s.randn(HD)
        H_s = (ctx_s[None, :, :] + (np.log(amp)[:, None] * w_s[None, :])[:, None, :]
               + rng_s.randn(nd, K, HD) * 0.30)
        A_mo, _ = activation_matrix(H_s, "centered")
        mono_h.append(mrm_partial(A_mo, B_s, ctrls_s, perms=perms_s)["r_partial"])
        mono_a.append(mrm_partial(A_mo, B_s, ctrl_act, perms=perms_s)["r_partial"])
    mR2, mR2e = float(np.mean(r2_act)), float(np.mean(r2_equal))
    mCh, mCa = float(np.mean(core_h)), float(np.mean(np.abs(core_a)))
    mMh, mMa = float(np.mean(mono_h)), float(np.mean(np.abs(mono_a)))
    ok = (mR2 > 0.60 and mR2e < 0.05 and mCh > 0.20 and mCa < 0.06
          and mMh > 0.20 and mMa < 0.5 * mMh and mMa <= 0.12)
    ok_all &= _report("a magnitude-only world reads positive on the headline and collapses when "
                      "activity is adjusted", ok,
                      f"R2(B ranks | activity only)={mR2:.3f} (threshold > 0.600) vs {mR2e:.4f} at "
                      f"EQUAL amplitudes (threshold < 0.050)  |  leading eig share of "
                      f"B={float(np.mean(eig_share)):.3f}  |  IDENTIFIED: headline "
                      f"r_partial={mCh:+.3f} (threshold > +0.200) -> activity-adjusted "
                      f"|r|={mCa:.4f} (threshold < 0.060)  |  MONOTONE (the (b3) analogue): "
                      f"headline {mMh:+.3f} -> adjusted |r|={mMa:.4f} (thresholds < 0.5x the "
                      f"headline AND <= 0.120, the measured confound-residue envelope; annihilation "
                      f"is NOT exact when the confound enters through different monotone transforms "
                      f"on the two sides)  |  means over 8 replicate worlds")

    # -------------------------------------------------------------------------------------
    # (t) DIRECT DECODABILITY ARM. Three cases, and the third is the whole justification for the
    #     arm's existence: a representation from which a linear head reads the biology almost
    #     perfectly, on which the RSA returns a NULL because the biology does not live in the
    #     variance-dominant directions. If that case is not pinned, the arm is decoration.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(t) DIRECT ARM — positive world, name-only world, and variance domination")
    perms_t = perms[:200]

    # EACH SUB-WORLD IS HANDED ITS OWN ROSTER. `names`/`toks` are rebound four times above, so
    # passing the last binding gave the names-only control a roster that did not generate the
    # activations it is controlling for. (t2) is where that actually matters: its whole claim is
    # "the names are a SUPERSET of what these activations carry", which is only true of the roster
    # that built U, H2 and res2.
    # (t1) the (a) POSITIVE world: activations are a linear image of the biological latent
    S_pos = np.stack([signed_rank_from_vector(res[i].astype(np.float32), P, 100)
                      for i in range(nd)])
    d_pos = direct_decodability(kaveraged_centered(H, "centered"), res, S_pos, names_a, toks_a, P,
                                perms=perms_t, n_perm=100)

    # (t2) the (b2) NAME-DRIVEN world: activations and biology are both name-token images, so a
    #      head fit on activations must add ~nothing over a head fit on the names themselves
    S_b2 = np.stack([signed_rank_from_vector(res2[i].astype(np.float32), P, 100)
                     for i in range(nd)])
    d_nam = direct_decodability(kaveraged_centered(H2, "centered"), res2, S_b2, names_b2, toks_b2, P,
                                perms=perms_t, n_perm=100)

    # (t3) VARIANCE DOMINATION. Same decodable biology, plus drug-varying nuisance in a handful of
    #      high-variance directions -- exactly what is expected at 'last', where the drug enters
    #      through 1-4 name tokens out of ~400 and the dominant drug-varying directions are
    #      lexical rather than pharmacological.
    rng_t = np.random.RandomState(seed + 71)
    V_t = rng_t.randn(nd, 12)
    res_t = V_t @ rng_t.randn(12, P) + rng_t.randn(nd, P) * 0.35
    B_t, S_t = biology_matrix(res_t, P)
    W_t = rng_t.randn(12, HD)
    nuis_dirs = rng_t.randn(3, HD)
    nuis = rng_t.randn(nd, 3) @ nuis_dirs * 9.0                # the crushing lexical directions
    ctx_t = rng_t.randn(K, HD) * 6.0
    H_t = (ctx_t[None, :, :] + (V_t @ W_t + nuis)[:, None, :] + rng_t.randn(nd, K, HD) * 0.30)
    A_t, dA_t = activation_matrix(H_t, "centered", spectrum=True)
    names_t, toks_t = _synth_names(nd, rng_t)
    ctrls_t = _controls_for(names_t, toks_t, [f"moa{rng_t.randint(8)}" for _ in range(nd)],
                            "annotated", rng_t)
    m_t = mrm_partial(A_t, B_t, ctrls_t, perms=perms)
    m_t.pop("_perm_stats", None); m_t.pop("_control_diag_design", None)
    # the SAME roster the RSA's controls were built from, so the two arms are compared on one world
    d_t = direct_decodability(kaveraged_centered(H_t, "centered"), res_t, S_t, names_t, toks_t, P,
                              perms=perms_t, n_perm=100)
    pr_t = dA_t["spectrum"]["participation_ratio"]

    ok = (d_pos["activations"]["retrieval_acc"] > 10.0 / nd
          and d_pos["activations"]["mean_held_out_r"] > 0.80
          and d_pos.get("increment_licensed") is True
          # the name-only world: the activations must add NOTHING over the names. A strongly
          # negative increment is the correct answer there, because the name features are a
          # superset of what those activations carry -- so the threshold is one-sided, and the
          # licensing conjunction must refuse even though p_increment alone is small.
          and d_nam["increment_mean_r"] <= 0.05
          and d_nam.get("increment_licensed") is False
          and pr_t < 5.0
          and abs(m_t["r_partial"]) < 0.15
          and d_t["activations"]["mean_held_out_r"] > 0.80
          and d_t["activations"]["retrieval_acc"] > 10.0 / nd)
    ok_all &= _report("the direct arm sees what the RSA cannot", ok,
                      f"(t1) POSITIVE: retrieval={d_pos['activations']['retrieval_acc']:.3f} "
                      f"(null {1.0/nd:.3f}, threshold > {10.0/nd:.3f}) mean_r="
                      f"{d_pos['activations']['mean_held_out_r']:+.3f} (threshold > +0.800) "
                      f"p_retrieval={d_pos.get('p_retrieval')} "
                      f"increment_licensed={d_pos.get('increment_licensed')} (must be True)  |  "
                      f"(t2) NAME-ONLY: activations "
                      f"mean_r={d_nam['activations']['mean_held_out_r']:+.3f} vs names-only "
                      f"{d_nam['names_only']['mean_held_out_r']:+.3f}, increment="
                      f"{d_nam['increment_mean_r']:+.4f} (threshold <= +0.050), "
                      f"p_increment={d_nam.get('p_increment'):.4f} but "
                      f"increment_licensed={d_nam.get('increment_licensed')} (must be False -- the "
                      f"increment's null is not centred at zero, so p alone would have licensed a "
                      f"representation that adds nothing)  |  "
                      f"(t3) VARIANCE DOMINATION: participation_ratio={pr_t:.2f} "
                      f"(threshold < 5.00), top_eig_share="
                      f"{dA_t['spectrum']['top_eig_share']:.3f} -> RSA r_partial="
                      f"{m_t['r_partial']:+.3f} (threshold |r| < 0.150, i.e. a NULL) WHILE the "
                      f"direct arm holds mean_r={d_t['activations']['mean_held_out_r']:+.3f} "
                      f"(threshold > +0.800) and retrieval="
                      f"{d_t['activations']['retrieval_acc']:.3f}")

    # -------------------------------------------------------------------------------------
    # (u) CELL-LINE POSITIVE CONTROL's B PATH. Synthetic eval records whose control-cell sentences
    #     carry a PLANTED per-line expression geometry. build_cellline_biology must recover it --
    #     both as a distance matrix that tracks the planted latent distances, and as a reliability
    #     that clears the gate. If this path could not recover a geometry that is definitely there,
    #     a null in the positive control would mean nothing and the control would be worthless.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(u) CELL-LINE CONTROL B PATH — a planted per-line geometry must be recovered")
    rng_u = np.random.RandomState(seed + 81)
    n_lines, P_u, n_cells_u = 20, 300, 40
    panel_u = [f"G{i}" for i in range(P_u)]
    pidx_u = {g: i for i, g in enumerate(panel_u)}
    Lat = rng_u.randn(n_lines, 6) @ rng_u.randn(6, P_u)            # the planted line geometry
    ex_u = []
    for li in range(n_lines):
        for _ in range(n_cells_u):
            v = Lat[li] + rng_u.randn(P_u) * 0.8
            genes = " ".join(panel_u[j] for j in np.argsort(-v)[:30])
            ex_u.append({"prompt": f"Predict the response of L{li} to D at 1 uM. Mechanism: "
                                   f"unclear.\nControl cell: {genes} [END_CELL]\n\nResponse cell:",
                         "response": "x",
                         "metadata": {"cell_line_id": f"CVCL_{li}", "cell_line_name": f"L{li}"}})
    pc_u = build_cellline_biology(ex_u, pidx_u, P_u, seed, min_cells=5, max_lines=60,
                                  n_rel_splits=6)
    # the recovered roster is ordered lexicographically by cell-line id, so the planted reference
    # must be reordered to match; comparing them in the planted order instead reads ~0.0 and would
    # look exactly like a broken build path
    order_u = [int(str(c).split("_")[1]) for c in pc_u["lines"]]
    Lat_o = Lat[order_u]
    B_true_u, _ = biology_matrix(Lat_o - Lat_o.mean(axis=0, keepdims=True), P_u)
    mantel_u = spearman(offdiag(pc_u["B"]), offdiag(B_true_u)) if pc_u else None
    relb_u = reliability_B(pc_u["half_profiles"], P_u, n_boot=200, seed=3) if pc_u else None
    gate_u = rel_B_gate(relb_u)
    ok = (pc_u is not None and len(pc_u["lines"]) == n_lines and mantel_u is not None
          and mantel_u > 0.70 and relb_u is not None and relb_u["rel_B"] > 0.40
          and gate_u["headline_permitted"])
    ok_all &= _report("the cell-line B path recovers a planted geometry", ok,
                      f"{0 if pc_u is None else len(pc_u['lines'])} lines recovered (planted "
                      f"{n_lines})  Spearman(recovered B, planted B)={mantel_u:.3f} (threshold > "
                      f"0.700)  rel_B={relb_u['rel_B']:.3f} (threshold > 0.400) -> "
                      f"{gate_u['status']}")

    # -------------------------------------------------------------------------------------
    # (v) x9 SAME-PLATE CONTROL. A two-plate world where PLATE drives both matrices and nothing
    #     else does. Without x9 the partial reads a large positive; with x9 it must be annihilated.
    #     And under a single-plate roster x9 must be dropped as zero-variance rather than retained
    #     as a constant column -- that is what makes cell_line_plate scope safe.
    # -------------------------------------------------------------------------------------
    logger.info("")
    logger.info("(v) x9 SAME-PLATE — plate-driven geometry must die when the roster spans plates")
    rng_v = np.random.RandomState(seed + 91)
    names_v, toks_v = _synth_names(nd, rng_v)
    moas_v = [f"moa{rng_v.randint(8)}" for _ in range(nd)]
    plates_v = ["plate5" if i % 2 == 0 else "plate6" for i in range(nd)]
    rng_v.shuffle(plates_v)
    base_args = ([len(t) for t in toks_v], [40 + rng_v.randint(6) for _ in range(nd)])
    ctrls_v_no = build_control_matrices(names_v, toks_v, moas_v, *base_args, "annotated")
    ctrls_v_x9 = build_control_matrices(names_v, toks_v, moas_v, *base_args, "annotated",
                                        plate_of=plates_v)
    ctrls_v_one = build_control_matrices(names_v, toks_v, moas_v, *base_args, "annotated",
                                         plate_of=["plate5"] * nd)
    gen_v = 1.0 - ctrls_v_x9["x9"]                    # 0 within plate, 1 across: pure batch
    A_v = gen_v + _sym_noise(nd, rng_v, 0.15); np.fill_diagonal(A_v, 0.0)
    B_v = gen_v + _sym_noise(nd, rng_v, 0.15); np.fill_diagonal(B_v, 0.0)
    m_v_no = mrm_partial(A_v, B_v, ctrls_v_no, perms=perms)
    m_v_x9 = mrm_partial(A_v, B_v, ctrls_v_x9, perms=perms)
    m_v_one = mrm_partial(A_pos, B_pos, ctrls_v_one, perms=perms[:200])
    for mm in (m_v_no, m_v_x9, m_v_one):
        mm.pop("_perm_stats", None); mm.pop("_control_diag_design", None)
    ok = (m_v_no["r_partial"] > 0.30 and abs(m_v_x9["r_partial"]) < 0.08
          and "x9" in m_v_one["controls_dropped_zero_variance"]
          and "x9" not in m_v_x9["controls_dropped_zero_variance"])
    ok_all &= _report("x9 annihilates a plate-driven geometry, and is dropped on a single plate", ok,
                      f"WITHOUT x9: r_partial={m_v_no['r_partial']:+.3f} (threshold > +0.300)  ->  "
                      f"WITH x9: r_partial={m_v_x9['r_partial']:+.3f} (threshold |r| < 0.080)  |  "
                      f"single-plate roster: x9 dropped as zero-variance = "
                      f"{'x9' in m_v_one['controls_dropped_zero_variance']} (must be True, which is "
                      f"why cell_line_plate scope is unaffected)")

    logger.info("")
    logger.info("=" * 92)
    logger.info("HEADLINE SELFTEST LINES")
    logger.info(f"  (a) POSITIVE : r_raw={m['r_raw']:+.3f}  r_partial={m['r_partial']:+.3f}  "
                f"p={m['p']:.5f}   -> the estimator recovers planted structure")
    logger.info(f"  (b) NEGATIVE : r_raw={mn['r_raw']:+.3f}  r_partial={mn['r_partial']:+.3f}  "
                f"p={mn['p']:.3f}   -> the confound controls annihilate prompt-string geometry")
    logger.info(f"SELFTEST {'PASSED' if ok_all else 'FAILED'}")
    logger.info("=" * 92)
    return ok_all


# =============================================================================================
def main():
    ap = argparse.ArgumentParser(
        description=("Does the metric structure of the model's internal drug representation match "
                     "the metric structure of real drug effects, beyond what the prompt string "
                     "hands it for free? Partial Spearman (MRM + Freedman-Lane) of activation "
                     "distance on biological distance, controlling for name-token overlap and "
                     "mechanism annotation. Run --selftest before any sbatch."))
    ap.add_argument("--eval_dir")
    ap.add_argument("--model_paths", help="comma-separated checkpoint paths (fine-tuned first)")
    ap.add_argument("--model_names", help="comma-separated names (match paths)")
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--layers", default="all", help="'all' or a comma-separated list, 0 = embedding")
    ap.add_argument("--positions", default="name,instr,last")
    ap.add_argument("--arms", default="blind", help="blind and/or annotated, comma-separated")
    ap.add_argument("--n_drugs", type=int, default=60)
    ap.add_argument("--n_controls", type=int, default=16, help="K balanced control sentences")
    ap.add_argument("--min_cells_per_drug", type=int, default=10)
    ap.add_argument("--cell_line", default=None)
    ap.add_argument("--plate", default=None)
    ap.add_argument("--dose_str", default=None, help="override the fixed dose string")
    ap.add_argument("--headline_layers", default="12,16")
    ap.add_argument("--stratum_label", default="G1",
                    help="the label of THIS stratum (G1/G2/G3), stamped into the JSON before the "
                         "run so the primary is declared rather than chosen afterwards")
    ap.add_argument("--primary_stratum", default="G1",
                    help="which stratum label is the PRE-REGISTERED PRIMARY; every other stratum "
                         "is replication and can never supply the headline")
    ap.add_argument("--n_perm", type=int, default=4000)
    ap.add_argument("--n_perm_triplet", type=int, default=1000)
    ap.add_argument("--n_perm_direct", type=int, default=1000,
                    help="permutation draws for the direct decodability arm; taken as the PREFIX "
                         "of the shared sequence, so the arms answer to the same relabelings")
    ap.add_argument("--n_rel_splits", type=int, default=25)
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--no_rel_companions", dest="rel_companions", action="store_false",
                    help="skip the REPORT-ONLY full-panel and well-split reliability companions "
                         "(they drive no gate; they exist to make the cell-split rel_B's inflation "
                         "and the k=100 truncation visible)")
    ap.add_argument("--n_residue_rep", type=int, default=200,
                    help="replicates for the per-arm confound-residue calibration (CPU, seconds)")
    ap.add_argument("--panel_file", default=None)
    ap.add_argument("--scope", default="cell_line_plate",
                    choices=["cell_line_plate", "cell_line"],
                    help="stratum scope. cell_line_plate eliminates plate batch by stratification; "
                         "cell_line pools plates (larger n) and adjusts plate dyadically via x9. "
                         "The two are not interchangeable readings.")
    ap.add_argument("--precheck", action="store_true",
                    help="CPU ONLY: roster, biology, rel_B and the gate, then STOP before any "
                         "checkpoint is loaded. Exits 0 if the stratum is runnable, 2 if it "
                         "refuses. torch and transformers are never imported on this path.")
    ap.add_argument("--precheck_sweep", nargs="?", const="10,25,50,100", default=None,
                    help="with --precheck: also sweep min_cells_per_drug over this comma list "
                         "(default 10,25,50,100) x both scopes, one row per configuration with "
                         "n_drugs and rel_B [lo, hi].")
    ap.add_argument("--positive_control", action="store_true",
                    help="run the CELL-LINE positive control on the identical code path: drug and "
                         "dose held fixed, the cell-line name varying in the same prompt slot. "
                         "Reported as a labelled control block, never as a headline.")
    ap.add_argument("--force_through_gate", action="store_true",
                    help="DIAGNOSTICS ONLY. Continue into the model sweep even when the rel_B gate "
                         "refuses. Results produced this way are NOT interpretable in either "
                         "direction and every cell is stamped produced_under_forced_gate.")
    ap.add_argument("--out", help="the ONE json this script writes; it writes nothing else")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(seed=args.seed) else 1)

    if args.precheck:
        missing = [k for k in ("eval_dir", "out") if not getattr(args, k)]
        if missing:
            ap.error(f"--{' --'.join(missing)} required with --precheck")
        _res, runnable = precheck(args)
        sys.exit(0 if runnable else 2)

    missing = [k for k in ("eval_dir", "model_paths", "model_names", "out")
               if not getattr(args, k)]
    if missing:
        ap.error(f"--{' --'.join(missing)} required unless --selftest")
    run(args)


if __name__ == "__main__":
    main()
