#!/usr/bin/env python3
"""
audit.py -- the integrity harness for thesis_v2.

The absolute constraint on this rewrite is "not one digit moves, no claim is
strengthened, weakened, added or dropped". Typography cannot enforce that, and
the failure is invisible in the PDF, because a draft that has quietly lost a
qualifier looks CLEANER, not worse. This script is the only guard that holds.

It compares thesis/Sections/*.tex (v1) against thesis_v2/Sections/*.tex (v2)
and fails the build on any of the checks below.

  A1  DISTINCT NUMERIC LITERALS.  set(v2) must equal set(v1), exactly.
      A literal that v1 contains and v2 does not is a DROPPED NUMBER.
      A literal that v2 contains and v1 does not is an INVENTED NUMBER.
  A2  CONFIDENCE INTERVALS.  The set of (value, lo, hi) triples must be equal.
      Both the \\ci form and v1's hand-rolled "$v$ \\nolinebreak[3]$[lo, hi]$"
      form are parsed, so the check works before and after migration.
  A3  FIXED HEDGE FORMULAS.  "not established" and "shown to be zero" must each
      occur at least as often in v2 as in v1. These are DIFFERENT STATEMENTS
      and the difference is defended in v1 against three external audits.
  A4  STRUCTURAL COUNTS.  claim blocks == 12; withdrawn/correction boxes == 5.
  A5  MARGIN PURITY.  No margin-note wrapper argument may contain a digit or a
      hedge word. The margin restates and navigates; it never argues.
  A6  OPENER PURITY.  No question-box body may contain a digit or a \\ref.
  A7  STRIP FIDELITY.  Every \\cistrip{lo}{est}{hi} must reproduce a triple that
      exists in the interval set. A strip is a redrawing of a number, never a
      new one.

  B   HEDGE INVENTORY (advisory, printed, never fatal). Per-lexeme counts in v1
      and v2 for a fixed list of qualifying words. A large drop is where a
      writer split a 60-word sentence and lost a qualifier on the way. A human
      reads this list; the script does not judge it.

Usage
  python tools/audit.py                       # full v1-vs-v2 audit
  python tools/audit.py --baseline            # v1 only: print the inventory
  exit 0 = clean, exit 1 = discrepancy, exit 2 = usage error
"""

import argparse
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2DIR = os.path.abspath(os.path.join(HERE, os.pardir))
V1DIR = os.path.abspath(os.path.join(V2DIR, os.pardir, "thesis"))

V1_FILES = [
    "Sections/Introduction.tex",
    "Sections/Literature-review.tex",
    "Sections/Investigation-v4.tex",
    "Sections/Limitations-and-Future-Research-Directions.tex",
    "Sections/Conclusions.tex",
    "Sections/Appendix.tex",
]
V2_GLOB_DIR = os.path.join(V2DIR, "Sections")

# Macros whose numeric arguments are typesetting, not science. Their arguments
# are deleted before any number is extracted.
LENGTH_MACROS = [
    "vspace", "hspace", "addvspace", "setlength", "needspace", "nolinebreak",
    "penalty", "linespread", "arraystretch", "hskip", "vskip", "rule",
    "beatstrip", "cistrip", "cistripclip", "titlerule", "cmidrule",
    "addlinespace", "raisebox", "scalebox", "hphantom", "vphantom",
]
REF_MACROS = ["label", "ref", "cref", "Cref", "autoref", "cite", "citep",
              "citet", "eqref", "pageref", "hyperref", "gref", "glentry",
              "includegraphics", "input", "graphicspath", "addbibresource"]

MARGIN_MACROS = ["gloss", "bench", "framecue", "seealso", "corrmark"]

HEDGE_LEXEMES = [
    "not established", "shown to be zero", "does not establish",
    "not identified", "inconclusive", "underpowered", "provisional",
    "descriptive", "in expectation", "cannot", "does not", "did not",
    "no more than", "at most", "only", "bound", "bounds", "withdrawn",
    "retracted", "superseded", "optimistic", "not a ceiling", "one run",
    "not a replicate", "does not license", "is not the same as",
]


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
def strip_comments(text):
    """Remove LaTeX comments. \\% is a literal percent and must survive."""
    text = text.replace(r"\%", "\x00PCT\x00")
    text = re.sub(r"(?m)%.*$", "", text)
    return text.replace("\x00PCT\x00", r"\%")


def drop_braced_arg(text, macro):
    """Delete \\macro{...} including nested braces, one occurrence at a time."""
    out, i = [], 0
    needle = "\\" + macro
    while True:
        j = text.find(needle, i)
        if j < 0:
            out.append(text[i:])
            break
        # must not be a prefix of a longer macro name
        k = j + len(needle)
        if k < len(text) and text[k].isalpha():
            out.append(text[i:k])
            i = k
            continue
        out.append(text[i:j])
        # skip optional [..] then one or more {..}
        while k < len(text) and text[k] in " \t":
            k += 1
        if k < len(text) and text[k] == "[":
            depth = 1
            k += 1
            while k < len(text) and depth:
                depth += {"[": 1, "]": -1}.get(text[k], 0)
                k += 1
        while k < len(text) and text[k] == "{":
            depth = 1
            k += 1
            while k < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[k], 0)
                k += 1
        i = k
    return "".join(out)


def drop_environment(text, env):
    return re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), " ", text,
                  flags=re.S)


def normalise_numbers(text):
    """1{,}394 and \\num{1394} and 1,394 all become 1394."""
    text = re.sub(r"\\num\{\s*([0-9.,{}\\]+?)\s*\}", r"\1", text)
    text = text.replace("{,}", "")
    text = re.sub(r"(?<=\d),(?=\d\d\d(?!\d))", "", text)
    return text


NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:\\%)?")


def extract_numbers(text):
    return collections.Counter(NUM_RE.findall(text))


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------
CI_MACRO = re.compile(r"\\ci\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}")
CI_HAND = re.compile(
    r"\$([+-]?[\d.]+)\$\s*\\nolinebreak\[3\]\s*"
    r"\$\[\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\]\$")
CI_HAND_BARE = re.compile(
    r"\\nolinebreak\[3\]\s*\$\[\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\]\$")
STRIP_RE = re.compile(r"\\cistrip\{([\d.]+)\}\{([\d.]+)\}\{([\d.]+)\}")


def extract_intervals(text):
    """Returns (triples, n_bare_unconverted)."""
    trips = collections.Counter()
    for m in CI_MACRO.finditer(text):
        trips[tuple(x.strip() for x in m.groups())] += 1
    consumed = set()
    for m in CI_HAND.finditer(text):
        trips[tuple(x.strip() for x in m.groups())] += 1
        consumed.add(m.span())
    bare = 0
    for m in CI_HAND_BARE.finditer(text):
        if not any(a <= m.start() and m.end() <= b for a, b in consumed):
            bare += 1
    return trips, bare


# --------------------------------------------------------------------------
# per-document inventory
# --------------------------------------------------------------------------
def inventory(paths, label):
    raw = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            raw.append(strip_comments(fh.read()))
    text = "\n".join(raw)

    intervals, bare = extract_intervals(text)
    strips = collections.Counter(tuple(m.groups())
                                 for m in STRIP_RE.finditer(text))

    # margin-note and opener bodies, harvested BEFORE they are stripped
    margins = []
    for mac in MARGIN_MACROS:
        for m in re.finditer(r"\\%s\{((?:[^{}]|\{[^{}]*\})*)\}"
                             r"(?:\{((?:[^{}]|\{[^{}]*\})*)\})?" % mac, text):
            margins.append((mac, " ".join(g or "" for g in m.groups())))
    openers = re.findall(r"\\begin\{question\}(?:\[[^\]]*\])?(.*?)"
                         r"\\end\{question\}", text, flags=re.S)

    counts = {
        "claim": len(re.findall(r"\\begin\{claim\}", text)),
        "withdrawn": len(re.findall(r"\\begin\{withdrawn\}", text)),
        "scopelimit": len(re.findall(r"\\begin\{scopelimit\}", text)),
        "worked": len(re.findall(r"\\begin\{worked\}", text)),
        "defn": len(re.findall(r"\\begin\{defn\}", text)),
        "question": len(openers),
    }

    hedges = collections.Counter()
    low = re.sub(r"\s+", " ", text.lower())
    low = low.replace(r"\notestablished", "not established")
    low = low.replace(r"\shownzero", "shown to be zero")
    for lex in HEDGE_LEXEMES:
        hedges[lex] = low.count(lex)

    # numbers: strip machinery, then count
    body = text
    body = drop_environment(body, "tikzpicture")
    for mac in LENGTH_MACROS + REF_MACROS:
        body = drop_braced_arg(body, mac)
    body = re.sub(r"\\begin\{tabular\}\{[^}]*\}", " ", body)
    body = re.sub(r"S\[[^\]]*\]", " ", body)
    body = normalise_numbers(body)
    numbers = extract_numbers(body)

    return dict(label=label, text=text, numbers=numbers, intervals=intervals,
                strips=strips, counts=counts, hedges=hedges, bare=bare,
                margins=margins, openers=openers)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
DIGIT = re.compile(r"\d")


def run_checks(v1, v2):
    fails, notes = [], []

    # A1 distinct numeric literals
    s1, s2 = set(v1["numbers"]), set(v2["numbers"])
    for n in sorted(s1 - s2):
        fails.append("A1 DROPPED NUMBER: %s appears %d time(s) in v1, none in v2"
                     % (n, v1["numbers"][n]))
    for n in sorted(s2 - s1):
        fails.append("A1 INVENTED NUMBER: %s appears %d time(s) in v2, none in v1"
                     % (n, v2["numbers"][n]))

    # A2 confidence intervals
    i1, i2 = set(v1["intervals"]), set(v2["intervals"])
    for t in sorted(i1 - i2):
        fails.append("A2 DROPPED INTERVAL: %s" % (t,))
    for t in sorted(i2 - i1):
        fails.append("A2 INVENTED INTERVAL: %s" % (t,))
    if v2["bare"]:
        fails.append("A2 %d hand-rolled interval(s) left unconverted in v2; "
                     "every interval goes through \\ci" % v2["bare"])

    # A3 fixed hedge formulas
    for lex in ("not established", "shown to be zero"):
        a, b = v1["hedges"][lex], v2["hedges"][lex]
        if b < a:
            fails.append('A3 HEDGE LOST: "%s" occurs %d times in v1 and %d in v2'
                         % (lex, a, b))

    # A4 structural counts
    if v2["counts"]["claim"] != 12:
        fails.append("A4 claim blocks = %d, expected 12 (v1 has %d)"
                     % (v2["counts"]["claim"], v1["counts"]["claim"]))
    if v2["counts"]["withdrawn"] != 5:
        fails.append("A4 withdrawn boxes = %d, expected 5 (the five retractions "
                     "v1 records)" % v2["counts"]["withdrawn"])
    if v2["counts"]["worked"] > 4:
        fails.append("A4 worked examples = %d, cap is 4"
                     % v2["counts"]["worked"])

    # A5 margin purity
    for mac, arg in v2["margins"]:
        if DIGIT.search(re.sub(r"\\(cref|ref|Cref)\{[^}]*\}", "", arg)):
            fails.append("A5 DIGIT IN MARGIN (\\%s): %s" % (mac, arg[:70]))
        for lex in ("not established", "shown to be zero", "cannot",
                    "does not", "may be"):
            if lex in arg.lower():
                fails.append('A5 HEDGE IN MARGIN (\\%s): "%s" in: %s'
                             % (mac, lex, arg[:70]))

    # A6 opener purity
    for body in v2["openers"]:
        if DIGIT.search(re.sub(r"\\begin\{question\}\[\d\]", "", body)):
            fails.append("A6 DIGIT IN OPENER: %s"
                         % re.sub(r"\s+", " ", body)[:80])
        if re.search(r"\\c?ref\{|\\cite", body):
            fails.append("A6 CROSS-REFERENCE IN OPENER: %s"
                         % re.sub(r"\s+", " ", body)[:80])

    # A7 strip fidelity
    for trip, k in v2["strips"].items():
        if trip not in v2["intervals"]:
            fails.append("A7 STRIP WITHOUT A MATCHING INTERVAL: %s" % (trip,))

    # B hedge inventory, advisory
    for lex in HEDGE_LEXEMES:
        a, b = v1["hedges"][lex], v2["hedges"][lex]
        if a and b < a:
            notes.append('   "%s": v1 %d -> v2 %d  (%+d)' % (lex, a, b, b - a))

    # count deltas on numbers, advisory: a restated number is legitimate,
    # a number that suddenly appears eight more times is worth a human look
    for n in sorted(s1 & s2):
        a, b = v1["numbers"][n], v2["numbers"][n]
        if abs(b - a) >= 3:
            notes.append("   literal %s: v1 %d -> v2 %d  (%+d)" % (n, a, b, b - a))

    return fails, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true",
                    help="print the v1 inventory and exit")
    args = ap.parse_args()

    v1_paths = [os.path.join(V1DIR, f) for f in V1_FILES]
    missing = [p for p in v1_paths if not os.path.exists(p)]
    if missing:
        sys.stderr.write("missing v1 source: %s\n" % missing[0])
        return 2
    v1 = inventory(v1_paths, "v1")

    if args.baseline:
        print("v1 BASELINE")
        print("  distinct numeric literals : %d" % len(v1["numbers"]))
        print("  total numeric tokens      : %d" % sum(v1["numbers"].values()))
        print("  distinct CI triples       : %d" % len(v1["intervals"]))
        print("  total CI uses             : %d" % sum(v1["intervals"].values()))
        print("  hand-rolled, unconverted  : %d" % v1["bare"])
        print("  claim blocks              : %d" % v1["counts"]["claim"])
        for lex in ("not established", "shown to be zero"):
            print('  "%s" : %d' % (lex, v1["hedges"][lex]))
        return 0

    if not os.path.isdir(V2_GLOB_DIR):
        sys.stderr.write("no thesis_v2/Sections\n")
        return 2
    v2_paths = sorted(os.path.join(V2_GLOB_DIR, f)
                      for f in os.listdir(V2_GLOB_DIR) if f.endswith(".tex"))
    v2 = inventory(v2_paths, "v2")

    fails, notes = run_checks(v1, v2)

    print("AUDIT  v1 %d files / %d distinct literals / %d CI triples"
          % (len(v1_paths), len(v1["numbers"]), len(v1["intervals"])))
    print("       v2 %d files / %d distinct literals / %d CI triples"
          % (len(v2_paths), len(v2["numbers"]), len(v2["intervals"])))
    if notes:
        print("\nADVISORY (human reads this; not a failure)")
        for n in notes:
            print(n)
    if fails:
        print("\nFAILURES (%d)" % len(fails))
        for f in fails:
            print("  " + f)
        return 1
    print("\nOK: every number, every interval and every fixed hedge formula in "
          "v1 is present in v2, and v2 introduces none of its own.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
