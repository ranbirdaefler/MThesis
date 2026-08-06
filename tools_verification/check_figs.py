"""Do the numbers in each figure caption appear in that figure's own stored data?

A prior audit found a figure whose values disagreed with its corrected caption. The figure scripts
write what they drew to thesis/figs/<name>.json, so this is checkable: a caption numeral that is
absent from its own figure's data is either derived in prose or stale.
"""
import glob
import io
import json
import os
import re

os.chdir(r"C:\Users\avsd8\OneDrive\Desktop\tahoe")

tex = io.open("thesis/Sections/Investigation-v4.tex", encoding="utf-8").read()
tex = re.sub(r"(?m)^%.*$", "", tex)

figs = {os.path.basename(p)[:-5]: json.load(io.open(p, encoding="utf-8"))
        for p in glob.glob("thesis/figs/*.json")}


def nums(o, acc):
    if isinstance(o, dict):
        for v in o.values():
            nums(v, acc)
    elif isinstance(o, list):
        for v in o:
            nums(v, acc)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        f = float(o)
        acc.add(round(f, 3)); acc.add(round(f, 4))
        acc.add(round(abs(f), 3)); acc.add(round(abs(f) * 100, 3))
    return acc


INC = re.compile(re.escape("\\includegraphics") + r"\[[^\]]*\]\{([^}]+)\}(.{0,3000}?)" + re.escape("\\label"), re.S)
CAP = re.compile(re.escape("\\caption") + r"(?:\[[^\]]*\])?\{(.*)$", re.S)
NUM = re.compile(r"(?<![\d.])([+-]?\d+\.\d{2,4})(?![\d])")

for m in INC.finditer(tex):
    name, block = m.group(1), m.group(2)
    cap = CAP.search(block)
    if not cap:
        continue
    capnums = sorted({round(float(x), 3) for x in NUM.findall(cap.group(1))})
    data = figs.get(name)
    if data is None:
        print("%-18s  NO STORED DATA FILE   caption carries %d numerals" % (name, len(capnums)))
        continue
    have = nums(data, set())
    missing = [x for x in capnums
               if x not in have and not any(abs(x - h) < 6e-4 for h in have)]
    status = "clean" if not missing else "NOT IN ITS OWN DATA: %s" % missing
    print("%-18s  caption numerals %2d   %s" % (name, len(capnums), status))
