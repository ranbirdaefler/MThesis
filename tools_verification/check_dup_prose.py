"""Find duplicated prose -- paste errors that leave a sentence fragment repeated.

The chapter opener carried a whole block pasted twice mid-sentence. LaTeX renders that without
complaint and no numeric check can see it.

Tracks the source line of every token so a hit can be judged: a claim box legitimately restated in
a summary table is not a defect; the same sentence twice in one paragraph is.
"""
import io
import os
import re

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")

FILES = ["Introduction", "Literature-review", "Investigation-v4",
         "Limitations-and-Future-Research-Directions", "Conclusions", "Appendix"]
N = 12
NEAR = 60          # hits closer than this many lines apart are almost certainly paste errors

total, near_hits = 0, 0
for f in FILES:
    raw = io.open("thesis/Sections/%s.tex" % f, encoding="utf-8").read().split("\n")

    toks = []                                   # (word, lineno)
    in_tab = False
    for ln, line in enumerate(raw, 1):
        if line.lstrip().startswith("%"):
            continue
        if "\\begin{tabular}" in line:
            in_tab = True
        if "\\end{tabular}" in line:
            in_tab = False
            continue
        if in_tab:
            continue                            # table rows repeat legitimately
        t = re.sub(r"\\[a-zA-Z@]+\s*", " ", line)
        t = re.sub(r"[{}$&\\~^_#]", " ", t)
        for w in t.split():
            toks.append((w, ln))

    seen, hits = {}, []
    for i in range(len(toks) - N + 1):
        key = " ".join(w for w, _ in toks[i:i + N]).lower()
        if key in seen:
            hits.append((key, seen[key], toks[i][1]))
        else:
            seen[key] = toks[i][1]

    collapsed, last = [], -999
    for key, l1, l2 in hits:
        if l2 - last > 3:
            collapsed.append((key, l1, l2))
        last = l2

    if collapsed:
        print("=" * 92)
        print("%s" % f)
        for key, l1, l2 in collapsed:
            gap = abs(l2 - l1)
            flag = "  <-- SAME REGION, likely a paste error" if gap < NEAR else "  (distant; check if restatement is intended)"
            total += 1
            near_hits += gap < NEAR
            print("   lines %5d and %5d (%d apart)%s" % (l1, l2, gap, flag))
            print("       %s..." % key[:105])

print()
print("RESULT: %d duplicated passage(s), %d in the same region" % (total, near_hits))
