"""Reconcile every interval macro in the BUILT thesis against the result artifacts."""
import io, re, json, glob, os

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")
FILES = ["Introduction", "Literature-review", "Investigation-v4",
         "Limitations-and-Future-Research-Directions", "Conclusions", "Appendix"]

CI_MACRO = re.compile(r"\\ci\{([+-]?[\d.]+)\}\{([+-]?[\d.]+)\}\{([+-]?[\d.]+)\}")
# also catch the hand-written form:  $+0.0183$ [$-0.0033$, $+0.0399$]
CI_PLAIN = re.compile(r"\$([+-][\d.]+)\$\s*(?:\\nolinebreak\[\d\])?\s*\$?\[\$?([+-][\d.]+)\$?,\s*\$?([+-][\d.]+)\$?\]")

cis = []
for f in FILES:
    s = io.open("thesis/Sections/%s.tex" % f, encoding="utf-8").read()
    for i, line in enumerate(s.split("\n"), 1):
        for rx, kind in ((CI_MACRO, "macro"), (CI_PLAIN, "plain")):
            for m in rx.finditer(line):
                cis.append((f, i, kind, float(m.group(1)), float(m.group(2)), float(m.group(3))))

print("interval statements in the build: %d  (%d macro, %d plain)"
      % (len(cis), sum(1 for c in cis if c[2] == "macro"), sum(1 for c in cis if c[2] == "plain")))

# harvest every 2-element numeric bound pair anywhere in any artifact
bounds = set()
def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            if (isinstance(v, list) and len(v) == 2
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
                bounds.add((round(float(v[0]), 4), round(float(v[1]), 4)))
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)

# thesis/figs/*.json matter as much as RESULTS_cluster: the figure scripts recompute some
# quantities from per-drug rows and store what they drew. Omitting them produced a false
# "unsourced interval" for the stratum contrast, which reproduces exactly from fig_difficulty.py.
SOURCES = glob.glob("RESULTS_cluster/*.json") + glob.glob("thesis/figs/*.json")
for p in SOURCES:
    try:
        walk(json.load(io.open(p, encoding="utf-8")))
    except Exception:
        pass
# CIs are also stored as sibling ci_low/ci_high keys, not only as 2-element lists
def walk_lohi(o):
    if isinstance(o, dict):
        lo, hi = o.get("ci_low", o.get("lo")), o.get("ci_high", o.get("hi"))
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            bounds.add((round(float(lo), 4), round(float(hi), 4)))
        for v in o.values():
            walk_lohi(v)
    elif isinstance(o, list):
        for v in o:
            walk_lohi(v)
for p in SOURCES:
    try:
        walk_lohi(json.load(io.open(p, encoding="utf-8")))
    except Exception:
        pass
print("distinct bound pairs across %d artifacts: %d" % (len(SOURCES), len(bounds)))

def near(lo, hi):
    """artifact match allowing for rounding to 3 or 4 dp in the text"""
    for a, b in bounds:
        if abs(a - lo) <= 5.1e-4 and abs(b - hi) <= 5.1e-4:
            return (a, b)
    return None

# Intervals legitimately absent from every stored artifact, each checked by hand.
# Keyed on (lo, hi) rounded to 4dp -> the reason it is not a defect.
ACCOUNTED = {
    (0.0407, 0.1554): "DERIVED: train model-0.5 clustered on DRUG. canon.py stores the "
                      "line-clustered variant; recomputed from re_v3.json it reproduces "
                      "exactly (+0.0981 [+0.0407,+0.1554], df=75 drugs).",
    (0.0592, 0.1010): "SUPERSEDED-AND-WITHDRAWN: name-span arm on the earlier target build, "
                      "scored against the 'opposite' comparator. Text withdraws the claim.",
    (-0.0174, 0.0383): "SUPERSEDED-AND-WITHDRAWN: mechanism-span arm, same run.",
    (0.0827, 0.1236): "SUPERSEDED-AND-WITHDRAWN: combined arm, same run.",
}

bad, ok_accounted = [], []
for f, i, kind, p, lo, hi in cis:
    if near(lo, hi) is not None:
        continue
    key = next((k for k in ACCOUNTED
                if abs(k[0] - lo) <= 5.1e-4 and abs(k[1] - hi) <= 5.1e-4), None)
    if key:
        ok_accounted.append((f, i, p, lo, hi, ACCOUNTED[key]))
    else:
        bad.append((f, i, kind, p, lo, hi))

print()
print("accounted for without a stored artifact (%d):" % len(ok_accounted))
for f, i, p, lo, hi, why in ok_accounted:
    print("   %s:%-5d %+.4f [%+.4f, %+.4f]" % (f, i, p, lo, hi))
    print("      %s" % why)

print()
if bad:
    print("!! %d interval(s) whose bounds match NO artifact:" % len(bad))
    for f, i, kind, p, lo, hi in bad:
        print("   %-34s:%-5d [%s] %+.4f [%+.4f, %+.4f]" % (f, i, kind, p, lo, hi))
else:
    print("OK: every interval in the build is present in an artifact")
