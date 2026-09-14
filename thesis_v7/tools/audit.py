#!/usr/bin/env python3
"""Submission checks for v7, which intentionally corrects scientific statements.

The old v1-vs-v2 numeric-parity harness is preserved as audit_legacy.py for
historical use; its claim that no number may change is inappropriate for v7.
These checks verify source preservation, placeholders, numeric CI ordering and
compiler diagnostics. They do not replace scientific review or visual PDF QA.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

THESIS = Path(__file__).resolve().parent.parent
REPO = THESIS.parent


def uncomment(text: str) -> str:
    return re.sub(r"(?m)(?<!\\)%.*$", "", text)


def main() -> int:
    failures = []
    manifest = json.loads((THESIS / "tools/preserved_sources_sha256.json").read_text())
    for relpath, digest in manifest["sha256"].items():
        path = REPO / relpath
        if not path.is_file():
            failures.append(f"Original source changed or missing: {relpath}")
            continue
        raw = path.read_bytes()
        if path.suffix != ".pdf":
            raw = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != digest:
            failures.append(f"Original source changed: {relpath}")
    print(f"Preservation: checked {len(manifest['sha256'])} original files.")

    checked_ci = 0
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    ci_pattern = re.compile(r"\\ci\{(" + number + r")\}\{(" + number + r")\}\{(" + number + r")\}")
    for path in sorted(THESIS.rglob("*.tex")):
        if path.name.startswith("_"):
            continue
        text = uncomment(path.read_text(encoding="utf-8"))
        for match in re.finditer(r"\?\?|\[XX\]|\b(?:TODO|FIXME)\b", text):
            line = text.count("\n", 0, match.start()) + 1
            failures.append(f"Unresolved placeholder {path.relative_to(THESIS)}:{line}")
        for match in ci_pattern.finditer(text):
            estimate, lower, upper = map(float, match.groups())
            checked_ci += 1
            if not lower <= estimate <= upper:
                failures.append(f"Unordered CI in {path.relative_to(THESIS)}: {match.group()}")
    print(f"Source checks: scanned TeX files and {checked_ci} numeric confidence intervals.")

    log_path = THESIS / "main.log"
    if not log_path.is_file():
        failures.append("Missing main.log; compile first.")
    else:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        patterns = {
            "LaTeX error": r"(?m)^! ",
            "undefined control sequence": r"Undefined control sequence",
            "undefined reference": r"LaTeX Warning: Reference .* undefined",
            "undefined citation": r"Citation .* undefined",
            "remaining rerun request": r"Rerun to get cross-references right|Please \(re\)run Biber",
            "overfull box": r"Overfull \\[hv]box",
            "oversized float": r"Float too large for page",
            "unusable table columns": r"X Columns too narrow",
            "missing glyph": r"Missing character:",
            "missing PDF destination": r"has been referenced but does not exist",
            "duplicate reference label": r"Label .* multiply defined",
        }
        for label, pattern in patterns.items():
            count = len(re.findall(pattern, log))
            if count:
                failures.append(f"{label}: {count}")
        underfull = len(re.findall(r"Underfull \\[hv]box", log))
        print(f"Layout advisory: {underfull} underfull boxes; inspect rendered pages.")
        if "Output written on main.pdf" not in log:
            failures.append("No completed PDF recorded in compiler log.")

    biber_log = THESIS / "main.blg"
    if biber_log.is_file():
        for line in biber_log.read_text(errors="replace").splitlines():
            if re.search(r"\b(?:WARN|ERROR)\b", line):
                failures.append(f"Bibliography: {line}")

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("Source and compiler checks passed. Scientific and visual review remain required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
