#!/bin/sh
# Build the revised thesis from main_v4.tex.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -jobname=thesis_revised main_v4.tex
echo "---"
echo "errors: $(grep -cE '^! ' thesis_revised.log)"
grep -E "Output written" thesis_revised.log | tail -1
