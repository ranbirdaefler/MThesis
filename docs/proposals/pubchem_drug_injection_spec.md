# Spec: PubChem Drug-Knowledge Injection

**Goal.** Test whether giving the model *chemical/biological information* about each drug (not just its
name) lets it condition on the drug — especially for **unseen drugs (tier2)**, where a drug *name* is an
uninformative arbitrary token. This is the principled fix for the drug-blindness result: it turns "predict
the effect of a name you've never seen" into "predict the effect of a molecule near others you have seen."

---

## 1. Why this is the right experiment (the hypothesis)

The current model conditions on a drug *name*. For a held-out drug, the name carries **zero** learnable
information — the model never saw it paired with any effect. So drug-blindness on tier2 is partly expected
and *uninformative*: we never gave the model a way to generalize across drugs. Chemical featurization fixes
exactly this — an unseen drug becomes a point in feature space near seen drugs, enabling generalization by
similarity (the actual mechanism by which drug-response models generalize).

Two clean outcomes, both valuable:
- **Features help** → "single-cell LLMs can learn drug-specific responses, but require structured chemical
  featurization, not drug names." A *positive* contribution.
- **Features don't help** → the negative result strengthens: "even with rich featurization, single-cell
  noise prevents drug-specific learning." Rules out "we under-informed the model."

**Gate (do Option B first):** features can only help if the drug effect is predictable from features *at all*
at this noise level. Option B (task ceiling) measures that. If even structure-aware classical methods can't
discriminate drugs at pseudobulk, featurizing the LLM won't either — skip this experiment. So: **Option B
decides whether this runs.**

---

## 2. What features to pull (the decision that matters most)

Ranked by expected predictive value for *transcriptional* drug response, NOT by ease:

### Tier 1 — highest value: TARGETS + MECHANISM (biological, not structural)
The single most predictive thing for a transcriptional response is **what the drug does to the cell** —
its protein targets and mechanism. Two drugs with the same target hit the same pathway → similar
transcriptional response. This is far more predictive than raw structure.
- **Protein targets** (gene symbols, e.g. EGFR, CDK4/6, HDAC1): the key feature. PubChem exposes these via
  bioassay/target annotations, but they're **cleaner from ChEMBL** (mechanism-of-action table) keyed by the
  drug. Path: CID → (PubChem xref) → ChEMBL ID → ChEMBL `mechanism` + `target` endpoints.
- **MOA / drug class** (e.g. "tyrosine kinase inhibitor"): you likely ALREADY HAVE this in Tahoe metadata
  (you used `moa` for the scramble). Confirm coverage; if present, this is free and high-value.
- **ATC / therapeutic class** if available (PubChem "Drug and Medication Information" / pug_view heading).

### Tier 2 — medium value: interpretable physicochemical descriptors
A small set of standardized descriptors, as text. Cheap (one PUG-REST call for all CIDs), and they give the
model a coarse "kind of molecule" signal.
- `MolecularWeight`, `XLogP` (lipophilicity), `TPSA` (polar surface area),
  `HBondDonorCount`, `HBondAcceptorCount`, `RotatableBondCount`, `HeavyAtomCount`.
- One PUG-REST call:
  `/rest/pug/compound/cid/{cids}/property/MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount/CSV`

### Tier 3 — LOW value for a text LLM: raw structure (SMILES / fingerprints)
- **Canonical SMILES**: available trivially, BUT a text LLM cannot meaningfully parse SMILES into
  structure-activity intuition, and it eats context. Include ONLY as an identifier fallback, not as the main
  feature. (SMILES/fingerprints would matter for a *dedicated molecular encoder* — the deferred fusion
  architecture — not for prompt-text injection.)
- **Morgan/PubChem fingerprints**: only useful if you later build a molecular-encoder branch. Skip for the
  text-prompt version.

**Recommendation:** Tier 1 (targets + MOA) is the experiment. Add Tier 2 descriptors as cheap extra signal.
Skip Tier 3 for the prompt-text version. Rationale: transcriptional response is driven by *what protein the
drug hits*, which structure only indirectly encodes and which a text LLM can't decode from SMILES anyway.

---

## 3. How to inject (implementation options, in order of effort)

### Option 1 — features as PROMPT TEXT  (DO THIS FIRST; preprocessing-only, no architecture change)
Append a compact, structured feature line to each prompt. Example:
```
Predict the response of {cell_line} to {drug} at {dose}.
Drug info: targets AKT1, MTOR; class PI3K/mTOR inhibitor; MW 452.5; logP 3.1; TPSA 92.
Control cell: {ctrl}

Response cell:
```
- Zero architecture change. Tests the hypothesis immediately.
- Keep it SHORT and CONSISTENT in format (the model learns the template).
- Use canonical target symbols so the same target reads identically across drugs (enables similarity).
- **This is the version to build first.** If it moves the needle, escalate.

### Option 2 — drug-feature EMBEDDING token  (medium; needs a projection layer + train-loop change)
Precompute a fixed drug-feature vector (e.g. multi-hot over a target vocabulary + normalized descriptors),
project via a small MLP into the model's embedding space, insert as a virtual token at a fixed prompt slot.
Cleaner and more compact than text; lets you use higher-dim features. Justified only if Option 1 shows signal.

### Option 3 — dual-encoder fusion  (hard; defer)
A separate molecular encoder (GNN on the graph, or a fingerprint MLP) cross-attended into the LLM. This is
the "separate training data / separate model" idea in its real form — weeks of work, a research architecture.
**Defer** until Option 1/2 justify it. Do not start here.

---

## 4. Data pipeline (build once, cache to disk)

1. **Collect CIDs.** From Tahoe metadata, gather the unique (drug → PubChem CID) map. (Tahoe provides CID.)
2. **Fetch descriptors** (Tier 2), one batched PUG-REST call for all CIDs → CSV → cache
   `drug_features.json`. Rate limit: PubChem allows ~5 req/s; batch CIDs comma-separated to minimize calls.
3. **Fetch targets/MOA** (Tier 1):
   - If Tahoe `moa` covers most drugs, use it directly (free).
   - For protein targets: CID → ChEMBL ID (PubChem xref or UniChem), then ChEMBL `mechanism` endpoint for
     `action_type`, `target_pref_name`, target gene. Cache per drug.
   - Coverage will be imperfect (some drugs lack annotated targets) — record coverage %, and use a
     consistent "targets: unknown" token for missing, so the model handles it uniformly.
4. **Build the feature string per drug** (canonical, sorted targets; fixed field order). Cache
   `drug_feature_strings.json`: {drug: "targets ...; class ...; MW ...; logP ...; TPSA ..."}.
5. **Regenerate training/eval prompts** with the feature line inserted (a small change to the preprocess
   prompt template; the cell-sentence construction is UNCHANGED — this only touches the prompt prefix).
6. **Retrain** (cold, same recipe as the [END_CELL] run) with feature-augmented prompts.
7. **Grade** with the existing airtight instrument (`metric_grades_model_v2.py`) — crucially on **tier2
   (unseen drugs)**, where featurization should help if it helps anywhere. Compare to the name-only model.

---

## 5. The critical evaluation design (don't skip)

The whole point is **generalization to unseen drugs**. So the decisive comparison is:
- name-only model vs feature-augmented model, **on tier2 (held-out drugs)**, using the grading instrument.
- If feature-augmented rises above chance on tier2 while name-only stays at chance → features enable
  cross-drug generalization. That's the result.
- Also run an **ablation within features**: targets-only vs descriptors-only vs both, to see which carries
  the signal (targets should dominate if the hypothesis is right).

**Leakage caution:** ensure the feature fetch doesn't inadvertently encode the held-out drugs' *effects*
(e.g., don't pull "gene expression signatures" or LINCS/L1000 annotations from any source — that would be
target leakage). Stick to structure/target/mechanism, which are properties of the molecule, not its measured
transcriptional effect in this dataset.

---

## 6. Honest expectation

- Best case: targets/MOA give the model a real generalization handle and tier2 discrimination rises. Strong
  positive result.
- Likely case (given Parts I–V): featurization helps the model *represent* the drug (Option A would confirm
  the info now enters the representation), but the single-cell **noise floor** still caps recoverable signal,
  so tier2 accuracy rises only modestly. Even this is informative: "featurization fixes the representation
  gap; noise remains the ceiling."
- The Option-B ceiling tells you which regime you're in BEFORE you spend the retrain.

## 7. Build order
1. Option B (task ceiling) — decides if this is worth running.
2. Fetch + cache drug features (targets+MOA+descriptors); report coverage.
3. Option 1 (features-as-prompt-text) regenerate + retrain + grade on tier2 vs name-only.
4. Only if signal: Option 2 (embedding token). Defer Option 3 (fusion) unless clearly justified.
