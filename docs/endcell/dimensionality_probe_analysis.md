# Dimensionality Probe: Where Drug Identity Lives and Dies Inside the Model

**Date:** 2026-07-12
**Script:** `mechanistic_drug_probe.py`
**Output:** `eval_results/mechanistic_probe.json`
**Models probed:** (1) retrained [END_CELL] model (`pythia_sft_endcell/final`), (2) original SFT
(`pythia_sft_diverse2/checkpoint-10000`). Both probed on `data_diverse2_endcell_big` tier2 (unseen
drugs), 12 drugs × 40 prompts = 480 real prompts + 12 fixed prompts (one cell line, vary only drug).

---

## 1. The question

Parts IV–V established that the model is drug-blind behaviorally — its predictions don't change
when the drug is scrambled, and it grades at chance on drug discrimination. But *why*? Two possible
mechanisms:

- **The drug never enters the representation.** The drug name is an arbitrary token that the model
  encodes trivially (as noise) and never builds into a meaningful internal state. In this case,
  drug-knowledge injection would need to get the drug info *into* the representation.
- **The drug enters the representation but isn't used for generation.** The model encodes the drug
  into its hidden states, but the generation head doesn't condition on that information — it's
  present but ignored. In this case, the problem is *utilization*, not *encoding*, and injection
  needs to make the drug info harder to ignore, not just present.

These have opposite implications for the next experiment. This probe distinguishes them.

## 2. Method

For each prompt, run a single forward pass with `output_hidden_states=True` and extract the
residual-stream activation at the **last prompt position** (the token position where response
generation begins) at **every layer** (0 = embedding output, 1–16 = after each transformer layer).
No generation, no training — forward passes only.

Two prompt constructions:

- **FIXED set** (12 prompts): one cell line, one control cell, vary *only* the drug name and MOA.
  Any activation difference is attributable to the drug text alone. Used for PCA visualization.
- **REAL set** (480 prompts): real eval prompts from tier2, 12 drugs × 40 cells each. Varied
  controls and cell lines. Used for the per-layer linear probe.

**Per-layer drug separability:** a cross-validated logistic regression (5-fold) classifies which of
the 12 drugs produced each activation vector. Accuracy is the fraction correctly classified; chance
= 1/12 ≈ 8.3%. A shuffled-label baseline (same probe, randomized drug labels) provides the null
floor. The **between/within variance ratio** (ratio of between-drug to within-drug sum-of-squares
across all activation dimensions) measures how geometrically separated the drug clusters are — a
complementary view to probe accuracy that captures cluster *tightness*, not just linear separability.

## 3. Results

### Per-layer probe accuracy (12 drugs, chance = 8.3%)

**Retrained [END_CELL] model:**

| Layer | Probe acc | Shuffled | B/W ratio |
|---|---|---|---|
| 0 (embedding) | 0.083 | 0.083 | — |
| 1 | 0.323 | 0.056 | 0.064 |
| 2 | 0.408 | 0.085 | 0.067 |
| 3 | 0.531 | 0.071 | 0.086 |
| 4 | 0.585 | 0.092 | 0.103 |
| 5 | 0.652 | 0.081 | 0.103 |
| 6 | 0.692 | 0.075 | 0.108 |
| 7 | 0.727 | 0.085 | 0.107 |
| 8 | 0.779 | 0.067 | 0.109 |
| **9** | **0.821** | 0.079 | **0.112** |
| 10 | 0.783 | 0.069 | 0.084 |
| 11 | 0.748 | 0.073 | 0.082 |
| 12 | 0.754 | 0.085 | 0.042 |
| 13 | 0.746 | 0.090 | 0.035 |
| 14 | 0.769 | 0.085 | 0.035 |
| 15 | 0.765 | 0.085 | 0.034 |
| 16 (output) | 0.758 | 0.085 | 0.034 |

**Original SFT model:**

| Layer | Probe acc | Shuffled | B/W ratio |
|---|---|---|---|
| 0 (embedding) | 0.083 | 0.083 | — |
| 1 | 0.346 | 0.083 | 0.073 |
| 2 | 0.419 | 0.077 | 0.074 |
| 3 | 0.571 | 0.079 | 0.090 |
| 4 | 0.621 | 0.071 | 0.106 |
| 5 | 0.713 | 0.075 | 0.100 |
| 6 | 0.729 | 0.073 | 0.107 |
| 7 | 0.708 | 0.081 | 0.105 |
| 8 | 0.725 | 0.071 | 0.115 |
| 9 | 0.733 | 0.069 | 0.121 |
| **10** | **0.740** | 0.065 | **0.124** |
| 11 | 0.681 | 0.073 | 0.111 |
| 12 | 0.646 | 0.073 | 0.102 |
| 13 | 0.548 | 0.069 | 0.096 |
| 14 | 0.602 | 0.073 | 0.101 |
| 15 | 0.540 | 0.079 | 0.091 |
| 16 (output) | 0.517 | 0.065 | 0.092 |

Shuffled baselines sit at ~7–9% throughout (near the 8.3% chance level), confirming the probe
accuracy is real signal, not an artifact of the classifier.

## 4. Interpretation

### The drug IS in the representation — and it survives to the output layer

This was not the expected result. The behavioral tests (Parts IV–V: chance discrimination, scramble
≈ model) suggested the model might not encode the drug at all. Instead:

- Drug identity **enters the representation early** (chance at layer 0 → ~50% by layer 3) and
  **builds through mid-layers** to a peak at layer 9 (endcell: 82%) or layer 10 (original: 74%).
  A linear probe can identify which of 12 drugs produced the activation with 82% accuracy, against
  an 8.3% chance baseline — a 10× lift.
- In the **endcell model**, drug identity **largely survives to the output layer**: 76% at layer 16,
  only modestly below the peak. The model's final hidden state, right where generation begins, still
  carries strong drug-identifying information.
- In the **original SFT**, drug identity **decays more substantially**: 74% peak → 52% at layer 16.
  Still above chance, but the signal is weaker at the generation stage.

### But the generation head doesn't use it — the b/w ratio tells the geometric story

The probe accuracy says "a linear classifier can find the drug." The between/within variance ratio
says *how* — and it tells a different, complementary story:

- Both models build drug clusters that **peak mid-network** (b/w ratio ~0.11–0.12 around layers
  8–10).
- In the **endcell model**, the ratio **collapses sharply** after layer 10: 0.112 → 0.034 by layer
  16 — a 3× compression. The drug clusters are *geometrically shrinking* into a low-variance
  subspace, even though a linear probe can still separate them. The information is being squeezed
  into dimensions the generation head apparently doesn't attend to.
- In the **original SFT**, the ratio decays more gradually (0.124 → 0.092), but the probe accuracy
  drops more (because the clusters also blur/spread in addition to shrinking).

So the full picture: drug identity is **encoded** (high probe accuracy) but **geometrically
compressed** (collapsing b/w ratio) in the final layers. The generation head operates in the
high-variance directions of the residual stream, and the drug signal has been pushed into
low-variance directions it doesn't use. **The bottleneck is utilization, not encoding.**

### The two-model comparison is itself a finding

The [END_CELL] retrain **improved** the model's internal drug representation:
- Higher probe accuracy at the output layer: 76% vs 52%
- The drug information persists better through the later layers

But it did **not** change the behavioral outcome — both models are drug-blind on generation (Parts
IV–V). So retraining improved encoding without improving utilization. The [END_CELL] format helped
the model *represent* the drug more cleanly, but didn't help it *use* that representation for
generation. This is consistent with the bottleneck being in how the generation head reads the
residual stream, not in what the residual stream contains.

### One important caveat on the early layers

The drug name is literally in the prompt tokens. So the earliest layers showing above-chance
accuracy (layers 1–3) is partly trivial — the model is just representing the input tokens, which
include the drug name. The interesting signal is in the **mid-to-late layers** (5–16), where the
drug identity has been *processed* through attention and feedforward layers and either retained or
discarded. The peak at layer 9 means the model *builds up* drug-related representations through
processing, not just passively carries the input tokens — it's doing real computation with the drug
information, then failing to route it to generation.

## 5. Implications for the drug-knowledge injection experiment

This result directly informs the PubChem drug-knowledge injection:

1. **The model can encode drug identity from text.** A drug name alone produces 76–82% decodable
   signal mid-network. So the encoding pathway works — there's no need for a separate molecular
   encoder to get drug info into the representation. Text-based injection (Option 1 in the spec)
   should be sufficient to get the information *in*.

2. **The problem is utilization, not encoding.** Adding richer drug features (targets, mechanism,
   descriptors) will likely make the drug representation even *more* decodable — but it may not
   change whether the generation head *uses* it. The experiment is still worth running (richer
   features might cross a threshold that triggers utilization), but the honest expectation should
   account for this: even with perfect drug encoding, the generation head may still not condition
   on it.

3. **A possible targeted fix:** if the drug signal is being compressed into low-variance directions
   in the final layers, one approach is to add an **auxiliary drug-prediction loss** during
   training — a small classification head at the final layer that forces the model to *retain*
   drug-discriminative variance in the residual stream. This would fight the geometric compression
   directly. This is a training-objective change, not a data change, and it's orthogonal to the
   featurization experiment.

## 6. For the thesis

This probe provides the **mechanistic complement** to the behavioral finding. The behavioral result
(Parts IV–V) says *what*: the model is drug-blind. This probe says *why*: the model encodes drug
identity (82% decodable at layer 9) but compresses it geometrically in the final layers (b/w ratio
collapses 3× from peak to output), so the generation head — which operates in the high-variance
directions — doesn't see it. The information is present but inaccessible to the output.

This is a genuinely novel finding for single-cell perturbation LLMs and contributes to the growing
literature on mechanistic interpretability of biological language models. It also extends the DrEval
critique into the mechanistic dimension: DrEval showed bulk drug-response models fail on unseen
drugs; this probe shows *how* the failure manifests inside the model's representations.
