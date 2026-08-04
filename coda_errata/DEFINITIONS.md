# What we measured, defined from scratch

Every term used in the residual-arm evaluation, built up in dependency order. Nothing here assumes a
term defined later. Formulas are in plain text so they read the same in any viewer.

This is written to become Methods text, so it is deliberately pedantic.

---

## 1. A response is a vector

A cell's transcriptome is reduced to a **panel** of 946 genes. A drug's effect on one cell line in
one well is summarised as a **residual**: a list of 946 numbers, one per gene, where positive means
the drug pushed that gene up and negative means it pushed it down, relative to what an average drug
would have done.

Call one of these vectors `a`. It has 946 entries.

---

## 2. Cosine similarity — do two responses point the same way?

    cos(a, b)  =  (a . b) / (||a|| * ||b||)

where `a . b` is the dot product (multiply entry by entry, add them all up) and `||a||` is the
length of `a` (square root of the sum of its squared entries).

Dividing by the lengths removes magnitude, so only DIRECTION is compared:

    cos = +1    identical direction: the same genes up, the same genes down
    cos =  0    unrelated
    cos = -1    exactly opposite: everything one raises, the other lowers

A strong and a weak version of the same biological response therefore score the same. This is
deliberate — we care which genes moved, not how far.

---

## 3. NIR — Normalised Inverse Rank

The score used throughout. It asks a **discrimination** question, not an accuracy question.

Setup. We have:

- `p` — the model's predicted response for drug A in cell line C
- `t` — drug A's TRUE response in cell line C
- `o_1, ..., o_m` — the true responses of every OTHER drug measured in cell line C

Definition:

    NIR(p) = (1/m) * sum over j of:
                 1.0   if cos(p, t) >  cos(p, o_j)
                 0.5   if cos(p, t) == cos(p, o_j)
                 0.0   if cos(p, t) <  cos(p, o_j)

In words: **the fraction of rival drugs whose truth the prediction resembles LESS than it resembles
its own truth.**

    NIR = 1.00   the prediction is closest to its own truth; it beats every rival
    NIR = 0.50   chance; no closer to its own truth than to a randomly chosen rival
    NIR < 0.50   closer to OTHER drugs' truths than to its own — actively misleading

Two properties matter.

**Chance is 0.5 by construction.** No control arm is needed to know where the null sits. This is why
some of our tests need no comparator at all.

**It measures discrimination, not accuracy.** A prediction can be far from the truth in absolute
terms and still score 1.0, provided it is closer to its own truth than to any rival's. Conversely a
prediction can be quite close to the truth and score 0.5 if it is equally close to everything.

The 0.5-for-ties rule matters: without it, a degenerate predictor that is exactly equidistant from
every truth scores 0.000 instead of the 0.500 it deserves.

---

## 4. The prompt, and the scramble

The model is given a prompt of the form:

    Predict the response of MCF7 to Lapatinib at 0.5 uM. Mechanism: EGFR inhibitor.
    Control cell: <about 123 gene symbols>

    Response cell:

The **scramble** rewrites ONLY the drug name and mechanism to name a different drug. The control
cell, the cell line, the dose, the layout — every other byte — is identical.

    model arm      : prompt names the TRUE drug
    scramble arm   : prompt names a DIFFERENT drug, everything else unchanged

Because the control cell is held constant between the two arms, its effect cancels in any
difference between them. That is the point of pairing.

---

## 5. Two things vary: WHICH LIE, and WHICH CONDITIONS

### 5a. Strata — which drug do we swap TO?

The target drug is always the same. What changes is the partner we substitute. Partners are chosen
by the cosine between the target's TRUE response and the partner's TRUE response, in that cell line:

    near        the most SIMILAR available partner        cos about +0.26
    orth        the most UNRELATED available partner      cos about  0.00
    opposite    the most ANTI-CORRELATED available        cos about -0.24

These are not "drugs the model has seen more or less of". They are how big a lie we tell.

### 5b. Splits — which conditions do we score?

    train           conditions the model trained on
    unseen_combo    the WELL was held out; the drug may be known from other wells
    unseen_drug     10 drugs held out entirely; the model has never seen these names

**Important and easy to misread: `unseen_combo` means an unseen WELL, not an unseen cell line.**
A Tahoe well contains about 49 cell lines dosed together, so every cell line appears in training.
What is held out is 36 whole treated wells.

`unseen_drug` is the **control**. Those drugs were held out of fine-tuning, so the model has no
LEARNED drug-specific information about them and the measured effect should sit near zero.

This is weaker than it was originally stated ("the measured effect there MUST be zero"), and the
weakening is deliberate. Three things break the guarantee: "unseen" means absent from the fine-tuning
set, not from the pretraining corpus, and these are named drugs with a literature; the prompt
explicitly supplies a `Mechanism:` string, which is real information about the drug; and the default
scramble swaps the drug name and the mechanism TOGETHER, so the contrast is not "a name the model
knows nothing about" versus "another such name". A large value on this arm is therefore evidence
against the comparator, not proof of an instrument fault.

---

## 6. Test 1 — the gap (and why it broke)

    gap  =  NIR(model arm)  -  NIR(scramble arm)

Intended reading: if the model ignores the drug name, both arms produce the same output, so the gap
is zero. If it reads the name, the arms differ.

The flaw is that the gap has two parts:

    gap  =  (how much naming the TRUTH helps)  +  (how much naming a LIE hurts)

and only the first is wanted. The second depends entirely on WHICH lie. Tell the model a drug it
knows well and it confidently produces that drug's signature. If that signature points away from the
truth, the prediction is worse than useless.

Measured, the comparator's own score:

    scramble_near       partner cos +0.26   ->   NIR 0.572    ABOVE chance
    scramble_orth       partner cos  0.00   ->   NIR 0.497    AT chance
    scramble_opposite   partner cos -0.24   ->   NIR 0.403    BELOW chance

The comparator tracks the PARTNER, not the model's knowledge of the target. Only `orth` sits at
chance and is therefore a neutral null.

The evaluation code hard-coded `opposite` — the stratum that maximises the unwanted term.

**The control settles it:**

    unseen_drug  under `opposite`   +0.0862  [+0.0017, +0.1708]   CONTROL FAILS
    unseen_drug  under `orth`       +0.0097  [-0.0655, +0.0848]   control passes

A control that fails under one comparator and passes under another is a measurement of the fault,
not an argument about it.

---

## 7. Test 2 — the slope (comparator-free)

### What "regression" means

Given paired numbers `(x_1, y_1), ..., (x_n, y_n)`, **linear regression** fits the straight line

    y  =  a  +  b * x

choosing the intercept `a` and slope `b` to make the total squared vertical miss as small as
possible:

    minimise   sum over i of  (y_i  -  a  -  b * x_i)^2

"We regressed y on x" means exactly this: we fitted that line. **`b` is the slope** — how much `y`
moves per unit of `x`. If `y` does not depend on `x`, the best line is flat and `b = 0`.

### What we put in

Each condition gives three generations, one per named partner:

    x  =  cos(true response of TARGET, true response of NAMED partner)
    y  =  NIR of the generation produced under that name,
          always scored against the TARGET's truth

### What the slope means

    b = 0    the output does not move with the name -> the model IGNORES the drug name
    b > 0    naming a drug whose real response resembles the target's produces a generation that
             scores HIGH; naming an opposite one produces a generation that scores LOW
             -> the model EMITS THE NAMED DRUG'S SIGNATURE

So `b > 0` means the model holds a **drug -> signature map**.

**Zero slope is the natural null**, so no comparator arm exists to be contaminated. The very
behaviour that broke Test 1 — the model emitting the named drug's signature — is the signal here.

### Measured

    split           points    slope b       95% interval
    train             600     +0.390    [+0.278, +0.503]
    unseen_combo      750     +0.374    [+0.250, +0.480]
    unseen_drug       360     +0.238    [+0.081, +0.379]

Magnitude check: `b = 0.39` across the observed span of x (-0.24 to +0.26, width 0.50) predicts a
NIR swing of 0.39 * 0.50 = 0.195. Observed swing: 0.572 - 0.403 = 0.169. Consistent.

**Do not extrapolate the slope.** Naively extending it to x = 1 -- naming a drug whose true response
is identical to the target's, which is what naming the TRUE drug amounts to -- predicts NIR about
0.5 + 0.374 = 0.87, against an observed true-name NIR of 0.545 on held-out wells. That is not a
contradiction, because the three x values are not a random sample: each is the MOST similar, MOST
orthogonal and MOST anti-correlated partner available for that condition. Fitting a line through
three selected extremes licenses a test of whether the dependence exists -- the sign of b -- and not
a calibrated sensitivity that can be read outside the sampled range.

The magnitude question is better answered by the absolute cosines in section 11 than by extrapolating
this slope.

---

## 8. Test 3 — beat chance (no comparator at all)

Told the TRUE drug, is the prediction closer to its own truth than to rivals'? That is simply NIR
against 0.5.

    split           n     NIR       95% interval        verdict
    train         200    0.635   [0.5773, 0.6924]    above chance
    unseen_combo  250    0.545   [0.4951, 0.5943]    not distinguishable from chance
    unseen_drug   120    0.512   [0.4456, 0.5778]    not distinguishable from chance

---

## 9. What the intervals mean

### Clustering

Two conditions from the SAME treated well are not independent: one well holds about 49 cell lines
dosed together, so they share the drug assignment and the batch. Two conditions from the SAME cell
line are not independent either: they share the biology.

Treating conditions as independent gives intervals that are too narrow. Here, about 1.3 to 1.5 times
too narrow.

### Two-way cluster-robust variance

Because a condition belongs to a cell line AND a well at the same time, neither grouping alone
captures the dependence. The Cameron-Gelbach-Miller estimator combines them:

    V  =  V_line  +  V_well  -  V_intersection

where `V_intersection` is the ordinary independent-sampling variance (the intersection of the two
groupings is the single condition). The interval is then

    estimate  +/-  1.96 * sqrt(V)

If `V` comes out non-positive in a finite sample, we fall back to the wider of the two one-way
variances and record which branch was taken.

### Reading an interval

    excludes zero    the effect is established at this sample size
    spans zero       NOT established — which is different from "shown to be zero"

The distinction matters here. `unseen_combo` reads +0.0449 with an interval reaching +0.0972, so an
effect up to about +0.1 remains compatible with the data. That is a limit of statistical power at
n = 250, not evidence of absence.

---

## 10. The corrected results table

Against the neutral comparator (`orth`), two-way clustered:

    split           n     model   scramble     gap          95% interval        verdict
    train         200     0.635      0.489   +0.1457   [+0.0836, +0.2078]   excludes zero
    unseen_combo  250     0.545      0.500   +0.0449   [-0.0074, +0.0972]   spans zero
    unseen_drug   120     0.512      0.502   +0.0097   [-0.0655, +0.0848]   control passes

Absolute scores for context:

    ceiling         0.958     within-well SPLIT-HALF bar: one half of a condition's cells scored
                              against the other half's truth. NOT a biological replicate -- 286 of
                              287 treatments occur in a single well, so a true cross-well replicate
                              ceiling is unavailable and would be LOWER.
    model           0.569
    drug_lookup     0.941     a table of "this drug does this", fitted on TRAINING conditions only
    drug_lookup_1   0.859     the same, from ONE other cell line, no averaging
    control_copy    0.511     drug-agnostic by construction — must be 0.5
    generic         0.500     drug-agnostic by construction — must be 0.5
    random          0.521

Chance 0.5, ceiling 0.958, so the achievable range is 0.458. The model covers 15 percent of it. A
lookup table covers 96 percent.

**Memorisation premium:** train +0.1457 versus unseen_combo +0.0449. Difference +0.1008,
permutation p = 0.0054. (A **permutation test** shuffles the split labels many times and asks how
often a difference this large appears by chance. p = 0.0054 means: about 5 times in 1000.)

---

## 11. What all of this concludes

Three findings, each from a different instrument:

1. **The model holds a drug -> signature map.** Slope +0.374 on held-out wells, essentially
   unchanged from +0.390 on trained ones. Name a drug and the output resembles what that drug really
   does.

2. **The map is too coarse to identify the target among its neighbours on held-out wells.** NIR
   0.545, not distinguishable from chance. Two independent instruments agree: the neutral-comparator
   gap spans zero, and the direct NIR-versus-chance test spans zero.

3. **A memorisation premium exists.** The model does significantly better on conditions it trained
   on than on held-out ones.

These are not in conflict. The model knows roughly what a drug does; it does not know what that drug
does IN THIS WELL. NIR asks the second question, ranking against about 40 rivals measured in the same
cell line.

In the thesis's existing vocabulary: the model learned the drug **main effect** and none of the
**drug x cell-line interaction** — which Q17 measures at about 45 percent of the residual variance.
That also explains the lookup's dominance: `drug_lookup` IS the main effect, estimated directly from
data far more precisely than the model learned it.

The sentence the evidence supports:

> Re-encoding the target made the model read the drug and learn a drug-level average response. It
> learned the main effect and not the interaction. A lookup table computes that same main effect
> better, which is why retrieval beats it.

---

## 12. Glossary

| term | one line |
|---|---|
| panel | the 946 genes every response is expressed over |
| residual | a drug's effect after removing the cell's own state and what an average drug does |
| cosine similarity | do two responses point the same way; +1 identical, 0 unrelated, -1 opposite |
| NIR | fraction of rival drugs whose truth the prediction resembles less than its own; 0.5 is chance |
| ceiling | NIR achieved by one half of a condition's cells against the other half's truth. A within-well split-half bar, not a biological replicate; the true achievable bar is lower and unmeasured |
| prompt | cell line, drug, dose, mechanism, and one control cell's gene list |
| scramble | the same prompt with only the drug name and mechanism replaced |
| stratum | how big a lie the scramble tells: near, orth, opposite |
| split | which conditions are scored: train, unseen_combo, unseen_drug |
| unseen_combo | an unseen WELL — not an unseen cell line |
| control (unseen_drug) | drugs absent from FINE-TUNING; the effect should sit near zero, but this is not a proof -- pretraining exposure and the supplied mechanism string both leak information |
| gap | NIR(model) - NIR(scramble) |
| regression | fitting y = a + b*x by minimising total squared vertical miss |
| slope | b: how much y moves per unit of x; zero means no dependence |
| clustering | treating non-independent observations as a group when computing uncertainty |
| two-way cluster-robust | combining cell-line and well clustering: V = V_line + V_well - V_intersection |
| permutation test | shuffle the labels many times; how often does a difference this big appear by chance |
| drug main effect | what a drug does on average across cell lines |
| drug x cell-line interaction | how a drug's effect differs between cell lines; about 45% of the residual |
