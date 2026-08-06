"""Structural checks LaTeX will not fail on after a batch of mechanical edits.

88 edits authored in isolation were applied by string replacement. That can leave an environment
opened and never closed inside another environment, a claim box with no text, a sentence ending
without punctuation where a replacement was truncated, or a doubled connective at a seam. latexmk
sets most of that without complaint.
"""
import io
import os
import re
from collections import Counter

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")

FILES = ["Introduction", "Literature-review", "Investigation-v4",
         "Limitations-and-Future-Research-Directions", "Conclusions", "Appendix"]

problems = 0
for f in FILES:
    raw = io.open("thesis/Sections/%s.tex" % f, encoding="utf-8").read()
    s = re.sub(r"(?m)^%.*$", "", raw)
    lines = s.split("\n")

    # 1. environment balance
    opens = Counter(re.findall(r"\\begin\{([a-zA-Z*]+)\}", s))
    closes = Counter(re.findall(r"\\end\{([a-zA-Z*]+)\}", s))
    for env in set(opens) | set(closes):
        if opens[env] != closes[env]:
            problems += 1
            print("  [%s] UNBALANCED %-14s %d begin / %d end" % (f, env, opens[env], closes[env]))

    # 2. empty claim boxes
    for m in re.finditer(re.escape("\\begin{claim}") + r"(.*?)" + re.escape("\\end{claim}"), s, re.S):
        if len(m.group(1).strip()) < 40:
            problems += 1
            print("  [%s] near-empty claim box at line %d" % (f, s[:m.start()].count("\n") + 1))

    # 3. doubled connectives at a seam ("But But", ". And And", duplicated word)
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r"\b(\w{3,})\s+\1\b", line):
            w = m.group(1).lower()
            if w in ("that", "had", "has", "very", "the"):     # legitimate English doublings
                continue
            problems += 1
            print("  [%s:%d] doubled word %r" % (f, i, m.group(0)))

    # 4. a paragraph opening on a connective whose antecedent may have been rewritten away
    for i, line in enumerate(lines, 1):
        if i > 1 and lines[i - 2].strip() == "" and re.match(r"^(But|And|That is why|Consequently|So )", line):
            print("  [%s:%d] paragraph opens on a connective: %s..." % (f, i, line[:64]))

    # 5. sentence ending with no terminal punctuation before a blank line
    for i, line in enumerate(lines, 1):
        t = line.rstrip()
        if not t or t.startswith("\\") or t.endswith(("\\\\", "%", "}", "{", "&")):
            continue
        if i < len(lines) and lines[i].strip() == "" and not re.search(r"[.!?:;,]$|\$$", t):
            problems += 1
            print("  [%s:%d] paragraph ends without punctuation: ...%s" % (f, i, t[-58:]))

print()
print("RESULT:", "structurally clean" if not problems else "%d structural problem(s)" % problems)
