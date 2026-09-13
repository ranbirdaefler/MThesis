#!/usr/bin/env bash
# Build thesis_v7 and run its current submission checks.
# Usage: bash tools/build.sh [--fast]
# Requires pdflatex, latexmk, biber and Python 3 on PATH (see README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  "") fast=0 ;;
  --fast) fast=1 ;;
  *) echo 'Usage: bash tools/build.sh [--fast]' >&2; exit 2 ;;
esac

latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex
if [ "$fast" = 1 ]; then
  echo 'Draft compiled to main.pdf; submission audits were not run.'
else
  "${PYTHON:-python3}" tools/correct_heldout_channels.py --check
  "${PYTHON:-python3}" tools/audit.py
  cp main.pdf thesis_v7.pdf
  echo 'Build and source checks passed: thesis_v7.pdf'
  echo 'Visual inspection remains required before submission.'
fi
