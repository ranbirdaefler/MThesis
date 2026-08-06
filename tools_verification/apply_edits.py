"""Apply a batch of (file, old_string, new_string) edits with per-edit assertions.

Used to land workflow-proposed rewrites. Every edit must match VERBATIM and EXACTLY ONCE; anything
ambiguous or missing is reported and skipped rather than guessed at, and nothing is written until
every edit for a file has been resolved, so a file is never left half-edited.

Usage:  python apply_edits.py edits.json [--dry-run]
where edits.json is [{"file": "Investigation-v4", "old_string": ..., "new_string": ..., "why": ...}]
"""
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")

DRY = "--dry-run" in sys.argv
path = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
if not path:
    raise SystemExit("usage: apply_edits.py edits.json [--dry-run]")

edits = json.load(io.open(path, encoding="utf-8"))
by_file = {}
for e in edits:
    by_file.setdefault(e["file"], []).append(e)

applied = skipped = 0
for fname, group in by_file.items():
    p = "thesis/Sections/%s.tex" % fname
    s = io.open(p, encoding="utf-8").read()
    original = s
    ok_here = []
    print("=" * 96)
    print("%s -- %d proposed edit(s)" % (fname, len(group)))
    for i, e in enumerate(group, 1):
        old, new = e["old_string"], e["new_string"]
        n = s.count(old)
        why = (e.get("why") or e.get("addresses") or "")[:88]
        if n == 0:
            print("  SKIP %2d  no verbatim match          %s" % (i, why))
            print("           first 70 chars: %r" % old[:70])
            skipped += 1
        elif n > 1:
            print("  SKIP %2d  matches %d times, ambiguous  %s" % (i, n, why))
            skipped += 1
        elif old == new:
            print("  SKIP %2d  no-op                       %s" % (i, why))
            skipped += 1
        else:
            s = s.replace(old, new, 1)
            ok_here.append(i)
            applied += 1
            print("  ok   %2d  %s" % (i, why))
    if s != original and not DRY:
        io.open(p, "w", encoding="utf-8", newline="\n").write(s)
        print("  -> wrote %s (%d edit(s))" % (p, len(ok_here)))
    elif DRY:
        print("  (dry run, not written)")

print()
print("RESULT: %d applied, %d skipped%s" % (applied, skipped, " [DRY RUN]" if DRY else ""))
if skipped:
    print("Skipped edits need their anchors re-derived from the current file text.")
