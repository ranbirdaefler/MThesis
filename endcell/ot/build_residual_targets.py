#!/usr/bin/env python
r"""
build_residual_targets.py — Arm 1b stage 1: drug-specific RESIDUAL training targets
====================================================================================
Measured rationale (target_divergence.py, 6,602 conditions / 125 groups): the current cell-sentence
target encodes almost no drug identity — only **34.6 of 200 tokens differ** between two drugs in the
same context, because ranking genes by expression puts the same housekeeping genes on top for every
drug. Under a drug-specific RESIDUAL the same data yields **120.3 of 200 tokens differing**. The
information was never missing (replicate retrieval ceiling ~0.742 under every representation) — the
TOKENIZATION discarded it. This builder changes what the tokens encode.

TARGET
  residual = (treated_pseudobulk - control_pseudobulk) - mean_over_drugs(treated - control)
    * the control subtraction removes the cell's own state;
    * the mean-over-drugs subtraction removes the GENERIC drug program (Q8: ~0.26 of "skill" that is
      drug-AGNOSTIC and matched by trivial baselines). What remains is what makes THIS drug different.
  SCOPE = cell line (measured: 62% of conditions reproducible vs 19% at plate scope — a plate has too
  few drugs to estimate the mean, so subtracting a noisy mean injects noise).

SENTENCE ENCODING (sign is biology, so we keep it)
  "<up genes, most up-regulated first> [DOWN] <down genes, most down-regulated first> [END_CELL]"
  i.e. a signed DE signature. NOTE: register [DOWN] as a special token in the trainer alongside
  [END_CELL] (both are plain strings that cannot collide with a gene symbol).

RELIABILITY FILTER (decided: filter, not weight)
  Keep a condition only if its residual reproduces across a half-split: cos(res_A, res_B) > --repro_thr.
  ~38% of conditions have an irreproducible residual; training on those is training on noise.

Also writes `reconstruction.npz` (per-cell-line generic shift + panel order) so evaluation can rebuild
full profiles:  predicted_treated = control_cell + generic_shift(cell_line) + predicted_residual.

USAGE
  python build_residual_targets.py --selftest
  python build_residual_targets.py --cache_dir /data/.../ot_cache --out_dir /data/.../residual_targets
"""
import argparse, json, os, sys, ast, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAHOE_REPO = "tahoebio/Tahoe-100M"
END = "[END_CELL]"
DOWN = "[DOWN]"


# ----------------------------------------------------------------- sentence encoding
def expr_to_sentence(vec, panel_genes):
    """control cell -> ordinary [END_CELL] sentence (expressed genes, highest expression first)."""
    idx = np.where(vec > 0)[0]
    if len(idx) == 0:
        return END
    order = sorted(idx.tolist(), key=lambda j: (-vec[j], j))
    return " ".join(panel_genes[j] for j in order) + " " + END


def residual_to_sentence(res, panel_genes, k_up, k_down):
    """signed DE signature: up-regulated block, [DOWN], down-regulated block, [END_CELL]."""
    up = [j for j in np.argsort(-res)[:k_up].tolist() if res[j] > 0]
    dn = [j for j in np.argsort(res)[:k_down].tolist() if res[j] < 0]
    parts = [" ".join(panel_genes[j] for j in up), DOWN, " ".join(panel_genes[j] for j in dn), END]
    return " ".join(p for p in parts if p)


def parse_dose(conc_str):
    try:
        parsed = ast.literal_eval(conc_str)
        if parsed and isinstance(parsed, (list, tuple)):
            _, val, unit = parsed[0]
            return f"{val} {unit}"
    except Exception:
        pass
    return "unknown"


def format_prompt(cell_line_name, drug, dose_str, moa, control_sentence):
    if not moa or moa in ("unknown", "nan", "None"):
        moa = "unclear"
    return (f"Predict the response of {cell_line_name} to {drug} at {dose_str}. Mechanism: {moa}."
            f"\nControl cell: {control_sentence}\n\nResponse cell:")


def load_meta_maps(repo=TAHOE_REPO):
    """Exact columns as the preprocessor, so prompts match the model's training/eval prompts."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    L = lambda n: pd.read_parquet(hf_hub_download(repo, f"metadata/{n}", repo_type="dataset"))
    cl, dr, sm = L("cell_line_metadata.parquet"), L("drug_metadata.parquet"), L("sample_metadata.parquet")
    cvcl = {}
    for _, r in cl.iterrows():
        cid, nm = r.get("Cell_ID_Cellosaur"), r.get("cell_name")
        if nm is None or (isinstance(nm, float) and pd.isna(nm)):
            nm = str(cid)
        if cid is not None:
            cvcl[str(cid)] = str(nm)
    moa = {str(r.get("drug")): str(r.get("moa-fine", r.get("moa_fine", "unknown")))
           for _, r in dr.iterrows() if r.get("drug")}
    conc = {str(r.get("sample")): str(r.get("drugname_drugconc", "unknown"))
            for _, r in sm.iterrows() if r.get("sample") is not None}
    logger.info(f"meta maps: {len(cvcl)} cell lines, {len(moa)} drugs, {len(conc)} samples")
    return cvcl, moa, conc


# ----------------------------------------------------------------- residual construction
def build_residuals(cache_dir, min_treated, min_control, repro_thr, seed=0):
    """-> conditions {(drug, cl, plate): {...}}, generic shift per cell line, control cell indices."""
    import pandas as pd
    from scipy import sparse
    rng = np.random.RandomState(seed)
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    X = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    plate = meta["plate"].astype(str).values
    drug = meta["drug"].astype(str).values
    dose = meta["dose"].astype(str).values

    # control pseudobulk per (cell_line, plate) + the control cell row indices (prompts come from these)
    ctrl_pb, ctrl_rows = {}, {}
    for g in set(zip(cl[is_ctrl], plate[is_ctrl])):
        idx = np.where(is_ctrl & (cl == g[0]) & (plate == g[1]))[0]
        if len(idx) >= min_control:
            ctrl_pb[g] = np.asarray(np.log1p(X[idx].todense()).mean(0)).ravel().astype(np.float32)
            ctrl_rows[g] = idx
    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], cl[i], plate[i], dose[i])].append(i)

    # per-condition treated pseudobulks (full + half split) and raw shifts
    cond = {}
    for (d, c, p, ds), idxs in by.items():
        g = (c, p)
        if len(idxs) < min_treated or g not in ctrl_pb:
            continue
        idxs = list(idxs); rng.shuffle(idxs); h = len(idxs) // 2
        L = lambda ix: np.asarray(np.log1p(X[ix].todense()).mean(0)).ravel().astype(np.float32)
        cond[(d, c, p, ds)] = {"shift_full": L(idxs) - ctrl_pb[g], "shift_A": L(idxs[:h]) - ctrl_pb[g],
                               "shift_B": L(idxs[h:]) - ctrl_pb[g], "group": g, "n_cells": len(idxs)}
    logger.info(f"conditions with enough cells: {len(cond)}")

    # GENERIC shift = mean over drugs, per CELL LINE (measured: cell-line scope >> plate scope)
    by_cl = defaultdict(list)
    for k in cond:
        by_cl[k[1]].append(k)
    generic = {}
    for c, keys in by_cl.items():
        generic[c] = {h: np.mean(np.stack([cond[k][f"shift_{h}"] for k in keys]), axis=0)
                      for h in ("full", "A", "B")}

    # residuals + reliability filter
    kept, cosines = {}, []
    for k, v in cond.items():
        c = k[1]
        rA = v["shift_A"] - generic[c]["A"]
        rB = v["shift_B"] - generic[c]["B"]
        cs = float(rA @ rB / (np.linalg.norm(rA) * np.linalg.norm(rB) + 1e-9))
        cosines.append(cs)
        if cs > repro_thr:
            kept[k] = {"residual": (v["shift_full"] - generic[c]["full"]).astype(np.float32),
                       # half-split residuals kept so evaluation can compute the achievable CEILING
                       # (a real replicate scoring against the other half's truths)
                       "residual_A": rA.astype(np.float32), "residual_B": rB.astype(np.float32),
                       "group": v["group"], "n_cells": v["n_cells"], "repro_cos": cs}
    cosines = np.array(cosines)
    logger.info(f"reliability: mean cos(A,B)={cosines.mean():+.3f}; "
                f"KEPT {len(kept)}/{len(cond)} conditions ({100*len(kept)/max(1,len(cond)):.0f}%) "
                f"at cos > {repro_thr}")
    return kept, {c: generic[c]["full"] for c in generic}, ctrl_rows, X, meta


def run(args):
    from scipy import sparse
    panel_genes = json.load(open(os.path.join(args.cache_dir, "panel_genes.json")))
    kept, generic, ctrl_rows, X, meta = build_residuals(
        args.cache_dir, args.min_treated, args.min_control, args.repro_thr, args.seed)
    if not kept:
        logger.error("no conditions survived the reliability filter"); return
    cvcl, moa_of, conc_of = load_meta_maps(args.repo)
    rng = np.random.RandomState(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "residual.jsonl")
    n_ex, n_cond = 0, 0
    with open(out_path, "w") as out:
        for (d, c, p, ds), v in kept.items():
            rows = ctrl_rows.get(v["group"])
            if rows is None or len(rows) == 0:
                continue
            take = rows if len(rows) <= args.max_ctrl else rows[rng.choice(len(rows), args.max_ctrl,
                                                                          replace=False)]
            resp = residual_to_sentence(v["residual"], panel_genes, args.k_up, args.k_down)
            cname = cvcl.get(c, c)
            dstr = parse_dose(conc_of.get(ds, "unknown"))
            moa = moa_of.get(d, "unclear")
            for ri in take:
                ctrl_vec = np.asarray(X[ri].todense()).ravel()
                prompt = format_prompt(cname, d, dstr, moa, expr_to_sentence(ctrl_vec, panel_genes))
                out.write(json.dumps({
                    "prompt": prompt, "response": resp,
                    "metadata": {"drug": d, "cell_line_id": c, "plate": p, "dose_float": ds,
                                 "target": "residual", "repro_cos": v["repro_cos"]}}) + "\n")
                n_ex += 1
            n_cond += 1
            if n_cond % 200 == 0:
                logger.info(f"  {n_cond}/{len(kept)} conditions, {n_ex} examples")

    # reconstruction assets: predicted_treated = control + generic_shift(cell_line) + predicted_residual
    np.savez_compressed(os.path.join(args.out_dir, "reconstruction.npz"),
                        cell_lines=np.array(list(generic.keys()), dtype=object),
                        generic_shift=np.stack([generic[c] for c in generic]),
                        panel_genes=np.array(panel_genes, dtype=object))
    json.dump({"n_conditions_kept": len(kept), "n_examples": n_ex, "k_up": args.k_up,
               "k_down": args.k_down, "repro_thr": args.repro_thr, "scope": "cell_line",
               "down_token": DOWN, "end_token": END},
              open(os.path.join(args.out_dir, "report.json"), "w"), indent=2)
    logger.info(f"wrote {n_ex} examples from {n_cond} conditions -> {out_path}")
    logger.info(f"reconstruction assets -> {args.out_dir}/reconstruction.npz")
    logger.info(f"NOTE: register '{DOWN}' as a special token in the trainer alongside '{END}'.")


def selftest():
    """Synthetic: known residual -> the sentence must list the planted up genes first, then [DOWN],
    then the planted down genes; and the reliability filter must drop a noise-only condition."""
    panel = [f"G{i}" for i in range(20)]
    res = np.zeros(20, dtype=np.float32)
    res[3], res[7] = 5.0, 3.0          # up
    res[11], res[15] = -4.0, -2.0      # down
    s = residual_to_sentence(res, panel, 5, 5)
    toks = s.split()
    up_block = toks[:toks.index(DOWN)]
    dn_block = toks[toks.index(DOWN) + 1:toks.index(END)]
    ok = up_block == ["G3", "G7"] and dn_block == ["G11", "G15"] and s.endswith(END)
    logger.info(f"  sentence: {s}")
    logger.info(f"  up={up_block} down={dn_block}  ({'ok' if ok else 'WRONG'})")
    # reliability: correlated halves pass, independent halves fail
    rng = np.random.RandomState(0)
    sig = rng.randn(50)
    good = float((sig + rng.randn(50) * 0.3) @ (sig + rng.randn(50) * 0.3) /
                 (np.linalg.norm(sig + rng.randn(50) * 0.3) ** 2 + 1e-9))
    a, b = rng.randn(50), rng.randn(50)
    bad = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    ok2 = good > 0.2 > bad
    logger.info(f"  reliability: signal cos~{good:+.2f} (keep) vs noise cos~{bad:+.2f} (drop) "
                f"({'ok' if ok2 else 'WRONG'})")
    logger.info(f"SELFTEST {'PASSED' if (ok and ok2) else 'FAILED'}")
    if not (ok and ok2):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2, help="cos(res_A,res_B) filter")
    ap.add_argument("--k_up", type=int, default=100)
    ap.add_argument("--k_down", type=int, default=100)
    ap.add_argument("--max_ctrl", type=int, default=60, help="control cells (=examples) per condition")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="RESULTS/residual_targets")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")
    run(args)


if __name__ == "__main__":
    main()
