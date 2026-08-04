#!/bin/sh
# Build the thesis WITHOUT touching main_v4.pdf, which carries hand annotations.
#
# -jobname keeps one source file (main_v4.tex) and writes main_v5.pdf. Do NOT run
# `latexmk main_v4.tex` bare -- it would overwrite the annotated copy.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -jobname=main_v5 main_v4.tex
echo "---"
echo "errors: $(grep -cE '^! ' main_v5.log)"
grep -E "Output written" main_v5.log | tail -1
