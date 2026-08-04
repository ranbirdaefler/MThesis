#!/bin/sh
# Build the thesis. Writes thesis.pdf from main_v4.tex via -jobname.
#
# Do NOT rename main_v4.tex casually -- Sections/*.tex and the annotated PDF reference this build.
# thesis_annotated.pdf (hand markup) is never written by this script.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -jobname=thesis main_v4.tex
echo "---"
echo "errors: $(grep -cE '^! ' thesis.log)"
grep -E "Output written" thesis.log | tail -1
