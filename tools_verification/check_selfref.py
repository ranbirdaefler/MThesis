"""Does any passage cross-reference the section it already sits inside?

A read-through finding: a table caption cited \\S\\ref{sec:res-scope} while sitting inside that very
section, so it rendered as a section pointing at itself. LaTeX resolves such a reference happily.

Figure captions live in thesis/figs/*.tex and are part of the built document. Omitting them from the
scan is how a second self-reference survived, in fig-scramble's caption.
"""
import glob
import io
import os
import re

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")

SECTIONS = ["Introduction", "Literature-review", "Investigation-v4",
            "Limitations-and-Future-Research-Directions", "Conclusions", "Appendix"]
PATHS = ["thesis/Sections/%s.tex" % f for f in SECTIONS] + sorted(glob.glob("thesis/figs/*.tex"))

LABEL = re.compile(re.escape("\\label") + r"\{(sec:[^}]+)\}")
REF = re.compile(re.escape("\\ref") + r"\{(sec:[^}]+)\}")
SECTIONING = re.compile(re.escape("\\subsection") + r"|" + re.escape("\\section"))

total = 0
for path in PATHS:
    lines = io.open(path, encoding="utf-8").read().split("\n")

    # A label belongs to the sectioning command it follows. Ownership resets at each new one, so a
    # \ref only counts as a self-reference when it names the label of the section enclosing it.
    owner, cur = [None] * len(lines), None
    for i, line in enumerate(lines):
        if SECTIONING.search(line):
            cur = None
        m = LABEL.search(line)
        if m:
            cur = m.group(1)
        owner[i] = cur

    for i, line in enumerate(lines):
        if line.lstrip().startswith("%"):
            continue
        for m in REF.finditer(line):
            if owner[i] == m.group(1):
                total += 1
                print("  [%s:%d] refers to its own section %s"
                      % (os.path.basename(path), i + 1, m.group(1)))
                print("       %s" % " ".join(line.split())[:100])

print()
print("scanned %d files (%d sections, %d figure files)"
      % (len(PATHS), len(SECTIONS), len(PATHS) - len(SECTIONS)))
print("RESULT:", "no self-references" if not total else "%d self-reference(s)" % total)
