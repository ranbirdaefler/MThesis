#!/usr/bin/env python3
r"""
migrate_ci.py -- one-time, mechanical, run BEFORE any drafting begins.

v1 typesets the same object two ways: 62 uses of \ci{v}{lo}{hi} and 56 uses of
the hand-rolled "$+0.0197$ \nolinebreak[3]$[+0.0063, +0.0335]$". A reader
re-parses the notation every time it changes shape, and the margin guard in
preamble.tex can only see the macro form. This script converts the hand-rolled
form to \ci and deletes the now-redundant \nolinebreak.

It rewrites ONLY files under thesis_v2/. It refuses any path outside, so it
cannot touch thesis/ even by accident.

It converts nothing it is not certain about. 11 sites in v1 write a bare
"\nolinebreak[3]$[lo, hi]$" whose point estimate sits further back in the
sentence; those are reported with line numbers and converted BY HAND, by a
person who reads the sentence.

Usage
  python tools/migrate_ci.py Sections/04a-ruler.tex ...
  python tools/migrate_ci.py --dry-run Sections/*.tex
"""

import argparse
import os
import re
import sys

V2DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir))

FULL = re.compile(
    r"\$([+-]?[\d.]+)\$\s*\\nolinebreak\[3\]\s*"
    r"\$\[\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*\]\$")
BARE = re.compile(r"\\nolinebreak\[3\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total, leftover = 0, 0
    for path in args.files:
        full = os.path.abspath(path)
        if not full.startswith(V2DIR + os.sep):
            sys.stderr.write("REFUSED (outside thesis_v2): %s\n" % full)
            return 2
        with open(full, "r", encoding="utf-8") as fh:
            src = fh.read()
        out, n = FULL.subn(lambda m: r"\ci{%s}{%s}{%s}" % m.groups(), src)
        total += n
        rest = [i + 1 for i, ln in enumerate(out.splitlines())
                if BARE.search(ln)]
        leftover += len(rest)
        if not args.dry_run and n:
            with open(full, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(out)
        print("%-44s converted %3d" % (os.path.relpath(full, V2DIR), n)
              + ("   MANUAL: lines %s" % rest if rest else ""))
    print("\ntotal converted %d; %d site(s) need a human" % (total, leftover))
    return 1 if leftover else 0


if __name__ == "__main__":
    sys.exit(main())
