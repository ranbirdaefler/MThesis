"""Run-C consistency audit for the thesis.

Three checks:
  1. superseded numeric values must not be quoted as current
  2. retracted claims must not be ASSERTED (stating them in order to retract them is fine,
     so the matcher is negation-aware -- a bare substring test flags every correction)
  3. canonical run-C values must actually appear
"""
import io, os, re

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")
FILES = ["Introduction", "Investigation-v4", "Limitations-and-Future-Research-Directions",
         "Conclusions", "Appendix"]
TEX = {f: io.open("thesis/Sections/%s.tex" % f, encoding="utf-8").read() for f in FILES}
ALL = "\n".join(TEX.values())

# --- 1. superseded values ------------------------------------------------------------------------
# Bare numbers only where the string is unambiguous. Values that also occur as legitimate
# quantities elsewhere (0.549 = model NIR and a CI bound; 0.569, 18.6 = ablation-table cells)
# are matched with their context instead.
BANNED = {
    "0.968": "run A ceiling (build discarded -- generic fitted over the holdout)",
    "0.963": "run A lookup",
    "0.639": "run A model",
    "0.958": "run B ceiling",
    "0.941": "run B lookup",
    "0.3559": "run B model-minus-lookup",
    "0.557": "a transfer coefficient matching NO artifact",
    "+0.1457": "run B train gap",
    "+0.0449": "run B held-out gap",
    "+0.0097": "run B unseen-drug gap",
    "0.0212": "run B exposure p",
    "0.4851": "run B drug-matched p",
    "0.0161": "run B exposure interval lo",
    # channel gate measured on the CONTAMINATED (pre-fix) partner pool
    "0.1799": "pre-fix gate: target plate-matched",
    "0.1863": "pre-fix gate: target count-matched",
    "0.0663": "pre-fix gate: moa plate-matched",
    "0.0380": "pre-fix gate: chem plate-matched",
    "0.1115": "pre-fix gate: target different-plate",
    "0.0604": "pre-fix gate: moa different-plate",
    "0.0183": "pre-fix gate: chem different-plate",
    "0.2459": "pre-fix interval bound",
    "0.1237": "pre-fix interval bound",
    "0.0617": "pre-fix interval bound",
    "0.1783": "pre-fix interval bound",
    "0.1114": "pre-fix interval bound",
    "$4114$": "pre-fix chem different-plate n",
    "$688$": "pre-fix target different-plate n",
    "$874$": "pre-fix moa different-plate n",
    "0.420": "pre-fix co-plating rate",
    # same-plate calibration before the wide, cell-line-clustered rerun
    "-0.122": "pre-wide same-plate weighted_r2",
    "-0.247": "pre-wide same-plate panel_tau",
    "-0.495": "pre-wide same-plate spearman_expr",
    # different-plate prose block, measured on a run that predates channel_gate_v3
    "0.0643": "pre-v3 different-plate target gap",
    "0.0344": "pre-v3 different-plate moa gap",
    "0.0056": "pre-v3 different-plate chem gap",
    "0.348": "pre-v4 target co-plating rate",
    "0.323": "pre-v4 target co-plating null",
    "0.344": "pre-v4 chem co-plating rate",
    "0.345": "pre-v4 chem co-plating null",
}
# superseded values needing context to disambiguate from legitimate uses of the same digits
BANNED_CTX = {
    r"absolute[^.]{0,60}below chance": "v4 different-plate arms are ABOVE their nulls",
    r"ceiling\s*-\s*\texttt\{drug\_lookup\}\s*=\s*\+0\.017": "run B, sign inverted",
    r"\\Tcoef\s*=\s*0\.549": "the 29-July pre-correction transfer coefficient",
    r"model[^.]{0,40}0\.569": "run B model NIR",
    r"coverage[^.]{0,40}18\.6": "run B coverage",
}

# --- 2. retracted claims -------------------------------------------------------------------------
NEG = re.compile(r"\b(not|never|neither|nor|without|cannot|withdrawn|withheld|rather than|"
                 r"goes with it|is not one|no longer|deliberately weaker|nothing|"
                 r"would require|must not)\b", re.I)
BANNED_PHRASES = {
    "protein target and mechanism do": "the repaired gate makes mechanism INCONCLUSIVE, not live",
    "not through chemistry": "chemistry is inconclusive, not closed",
    "nine tenths of the combined effect": "the comparator is non-neutral in opposite directions; "
                                          "model-minus-chance inverts the ordering",
    "applying to the reproducible subset": "run C does NOT filter the held-out splits",
    "five-fold below what re-encoding": "the chapter forbids this ratio (600 vs 1400 tokens)",
    "of the training gradient": "no gradient attribution was ever computed",
    "half the drug-specific variance": "1-T is a similarity scale, not a variance share",
    "ordering tracks sequencing depth": "CP10K is a per-cell constant and log is monotone; "
                                        "neither reorders genes within a cell",
    "clears it and may be written as live": "live holds on the cell-line axis only; "
        "on the drug axis the plate-matched lower bound is 0.0287 vs a 0.03 margin",
    "replicate ceiling": "renamed: within-well split-half precision reference",
    "unseen by the whole pipeline": "false -- Pythia pretraining + supplied mechanism",
    "interaction share": "1-T mixes cell line, dose, well and estimator",
    "memorisation premium": "the exposure advantage does not survive drug matching",
    "hedges rather than confabulat": "abstention is not measured",
    "did not teach the model about drugs": "the slope never scores the real held-out prompt",
    "biologically reproducible": "within-well sampling precision only",
}

# --- 3. values that must be present ---------------------------------------------------------------
REQUIRED = {
    "0.0898": "train gap vs neutral",
    "0.0332": "held-out gap vs neutral",
    "0.553": "transfer coefficient",
    "0.913": "lookup on common support",
    "1192": "common-support n",
    "0.824": "held-out ceiling (splits no longer equally hard)",
    "0.1351": "v4 gate: target plate-matched",
    "0.0805": "v4 gate: moa plate-matched",
    "0.0261": "v4 gate: chem plate-matched",
    "0.1322": "v4 gate: target different-plate",
    "0.0118": "v4 gate: chem different-plate",
    "0.604": "cluster-correct DRF lower bound",
    "0.663": "cluster-correct DRF upper bound",
    "2{,}523": "training conditions in the restricted pool",
    "0.545": "same-plate DRF, wide cell-line-clustered rerun",
    "0.516": "same-plate DRF lower bound",
    "0.569": "same-plate DRF upper bound",
    # the gate verdict depends on the clustering axis; both must stay quoted
    "0.0287": "target plate-matched, DRUG-clustered lower bound (misses the 0.03 margin)",
    "0.2416": "target plate-matched, DRUG-clustered upper bound",
    "0.0199": "target different-plate, DRUG-clustered lower bound",
}

fail = 0
print("=" * 82)
print("1. SUPERSEDED VALUES  (must appear nowhere)")
print("=" * 82)
n = 0
for v, why in sorted(BANNED.items()):
    hits = [(f, t.count(v)) for f, t in TEX.items() if v in t]
    if hits:
        fail += 1; n += 1
        print("  STILL PRESENT  %-9s %-42s %s" % (v, why, hits))
for pat, why in sorted(BANNED_CTX.items()):
    for f, t in TEX.items():
        for m in re.finditer(pat, t, re.I):
            fail += 1; n += 1
            print("  STILL PRESENT  [ctx] %-38s %s :: %r" % (why, f, m.group(0)))
if not n:
    print("  clean")

print()
print("=" * 82)
print("2. RETRACTED CLAIMS  (may be stated only in order to be retracted)")
print("=" * 82)
n = 0
for p, why in sorted(BANNED_PHRASES.items()):
    for f, t in TEX.items():
        for m in re.finditer(re.escape(p), t, re.I):
            ctx = t[max(0, m.start() - 130):m.end() + 130]
            if NEG.search(ctx):
                continue                      # negated -> this is the correction, not the claim
            fail += 1; n += 1
            print("  ASSERTED  %-32s [%s]" % (repr(p), f))
            print("            ...%s..." % " ".join(ctx.split())[:150])
if not n:
    print("  clean -- every occurrence is negated")

print()
print("=" * 82)
print("3. CANONICAL VALUES  (must appear somewhere)")
print("=" * 82)
n = 0
for v, why in sorted(REQUIRED.items()):
    if v not in ALL:
        fail += 1; n += 1
        print("  MISSING  %-9s %s" % (v, why))
if not n:
    print("  all present")

print()
print("=" * 82)
print("4. STRUCTURAL")
print("=" * 82)
b = io.open("thesis/Sections/Investigation-v4.tex", "rb").read()
ctrl = sorted({c for c in b if c < 9 or 11 <= c <= 12 or 14 <= c <= 31})
print("  control bytes:", ctrl if ctrl else "none")
stale = sum(ALL.count(t) for t in ("TODO", "in flight"))
print("  TODO / in-flight markers:", stale)
fail += bool(ctrl) + bool(stale)

# internal working notes left in the source
for f, t in TEX.items():
    for marker in ("AUTHOR NOTE", "FIXME", "XXX", "no artifact"):
        if marker in t:
            fail += 1
            print("  AUTHOR NOTE LEFT IN SOURCE [%s]: %r" % (f, marker))

# duplicated table rows: identical row text repeated inside one tabular
for f, t in TEX.items():
    for tb in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", t, re.S):
        rows = [" ".join(r.split()) for r in tb.split("\\\\") if "&" in r]
        for d in {r for r in rows if rows.count(r) > 1 and len(r) > 40}:
            fail += 1
            print("  DUPLICATE ROW [%s] %s..." % (f, d[:80]))

print()
print("RESULT:", "CLEAN" if not fail else "%d problems" % fail)
