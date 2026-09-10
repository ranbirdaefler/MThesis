#!/usr/bin/env bash
# build.sh -- the only sanctioned build. Run from thesis_v4/.
#
#   bash tools/build.sh          full build + all gates
#   bash tools/build.sh --fast   build only, no audit (drafting loop)
#
# Every gate below is a FAILURE, not a warning. A \PackageWarning nobody reads
# is not enforcement, and this document is written by seven people in parallel.
set -u
cd "$(dirname "$0")/.." || exit 2

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

echo "== 0. thesis/ must be untouched =================================="
if ! git -C .. diff --quiet -- thesis/ 2>/dev/null; then
  echo "   NOTE: thesis/ has uncommitted changes."
  echo "   Compare against tools/thesis.md5 to see whether THIS work caused them."
fi
if [ -f tools/thesis.md5 ]; then
  if ! (cd .. && md5sum -c --quiet thesis_v4/tools/thesis.md5); then
    echo "   FAIL: a file under thesis/ changed since the baseline was taken."
    exit 1
  fi
  echo "   ok: thesis/ byte-identical to the recorded baseline"
else
  echo "   no baseline recorded; run:  cd .. && md5sum thesis/main_v4.tex \\"
  echo "        thesis/utilities.tex thesis/ref.bib thesis/Sections/*.tex \\"
  echo "        thesis/figs/* > thesis_v4/tools/thesis.md5"
fi

echo "== 1. latexmk ===================================================="
latexmk -pdf -interaction=nonstopmode main.tex > /dev/null 2>&1
BUILD=$?
[ -f main.log ] || { echo "   FAIL: no log produced"; exit 1; }

fail=0
gate () { # gate <label> <grep-pattern> <max-allowed>
  n=$(grep -c "$2" main.log)
  if [ "$n" -gt "$3" ]; then
    echo "   FAIL  $1: $n (cap $3)"
    grep -n "$2" main.log | head -8 | sed 's/^/         /'
    fail=1
  else
    echo "   ok    $1: $n"
  fi
}
gate "LaTeX errors"            "^! "                          0
gate "undefined control seq"   "Undefined control sequence"   0
gate "undefined references"    "LaTeX Warning: Reference"     0
gate "undefined citations"     "Citation .* undefined"        0
gate "box budget / adjacency / two openers" "Package thesisv2 Warning" 0
gate "marginpar relocated"     "Marginpar on page"            0
gate "overfull boxes"          "Overfull \\\\hbox"            0
gate "underfull boxes"         "Underfull \\\\hbox"           4

echo "== 2. margin and opener purity, box counts, numbers =============="
if [ "$FAST" = "1" ]; then
  echo "   skipped (--fast)"
else
  python tools/audit.py || fail=1
fi

echo "=================================================================="
if [ "$BUILD" != "0" ] || [ "$fail" != "0" ]; then
  echo "BUILD NOT ACCEPTABLE"
  exit 1
fi
echo "BUILD ACCEPTABLE  ($(grep -o 'Output written on main.pdf ([0-9]* pages' main.log | grep -o '[0-9]*' | head -1) pages)"
