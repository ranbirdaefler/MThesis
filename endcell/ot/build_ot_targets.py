#!/usr/bin/env python
r"""
build_ot_targets.py — OT/meta-cell training targets (Arm 1c, step 6, decoder-free)
==================================================================================
From the validated cache (panel CP10K expression + shipped scVI latent + metadata), build the target
ladder and emit training JSONLs drop-in compatible with the endcell trainer/eval.

For each condition (cell_line, plate, drug, sample) with enough treated cells, and using the plate's DMSO
CONTROL cells as input prompts:
  * coupling: entropic-Sinkhorn OT pi between control and treated cells IN THE EMBEDDING (scVI latent or
    PCA), cost = squared Euclidean. (The coupling matches control<->treated by cell STATE.)
  * targets in EXPRESSION space (decoder-free, always real-cell convex combos):
      T0 consensus         : target_i = mean(Y_expr)                    (eps->inf; = Arm 1a)
      T1 mean-displacement : target_i = X_expr_i + (mean(Y)-mean(X))    (control identity + mean shift)
      T2 OT-barycentric    : target_i = (pi @ Y_expr)_i / rowsum        (control-specific, state-matched)
  * each target -> rank panel genes desc -> top-K (K = median expressed panel genes among treated) ->
    "[END_CELL]" sentence; prompt = the SAME control cell + drug/dose/MoA text.

Also computes the STEP-0 gates and writes a report:
  0a signal survival (MW2 location shift treated-vs-control in the embedding, permutation p),
  0c target sanity (target pseudobulk ~ real treated; per-control target variance > 0),
  0d target-contrast (top-K genes differing from the control prompt: T2 vs T0).

USAGE
  python build_ot_targets.py --selftest
  python build_ot_targets.py --cache_dir /data/.../ot_cache --emb scvi --epsilon 0.5 \
      --min_treated 60 --min_control 40 --out_dir /data/.../ot_targets
"""
import argparse, json, os, sys, logging, ast
from collections import defaultdict
import numpy as np

# shared/ locally; flat alongside the script on the cluster
for _p in (os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "shared"),
           os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tahoe_design as td

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TAHOE_REPO = "tahoebio/Tahoe-100M"
END = "[END_CELL]"


# ----------------------------------------------------------------- OT + sentence helpers
def sinkhorn(C, eps, n_iter=250):
    """Entropic OT plan with uniform marginals. C: (n_c, n_t) cost. Returns pi (n_c, n_t)."""
    nc, nt = C.shape
    K = np.exp(-C / max(eps, 1e-6))
    u = np.ones(nc) / nc
    v = np.ones(nt) / nt
    a = np.ones(nc) / nc
    b = np.ones(nt) / nt
    for _ in range(n_iter):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    return u[:, None] * K * v[None, :]


def cost_matrix(X, Y):
    """squared Euclidean (n_c, n_t)."""
    return (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T


def expr_to_sentence(vec, panel_genes, K=None):
    """panel expression vector -> [END_CELL] sentence: expressed genes ranked desc, tie by panel index,
    optionally truncated to top-K (for dense averaged targets)."""
    idx = np.where(vec > 0)[0]
    if len(idx) == 0:
        return END
    if K is not None and len(idx) > K:
        idx = idx[np.argsort(-vec[idx])[:K]]
    order = sorted(idx.tolist(), key=lambda j: (-vec[j], j))
    return " ".join(panel_genes[j] for j in order) + " " + END


def parse_dose(conc_str):
    """Display string for the prompt only.

    The numeric concentration comes from `tahoe_design.parse_treatment`, which also returns
    every component of a combination instead of silently taking the first."""
    st = td.parse_treatment(conc_str)
    if st.primary is not None:
        return st.primary.dose.display()
    return "unknown"


# ----------------------------------------------------------------- Tahoe metadata maps
def load_meta_maps(repo):
    """Exact columns as the preprocessor: Cell_ID_Cellosaur->cell_name, drug->moa-fine,
    sample->drugname_drugconc (so OT prompts match the model's training/eval prompts verbatim)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    def L(name):
        return pd.read_parquet(hf_hub_download(repo, f"metadata/{name}", repo_type="dataset"))
    cl, dr, sm = L("cell_line_metadata.parquet"), L("drug_metadata.parquet"), L("sample_metadata.parquet")
    cvcl_to_name = {}
    for _, r in cl.iterrows():
        cid, nm = r.get("Cell_ID_Cellosaur"), r.get("cell_name")
        if nm is None or (isinstance(nm, float) and pd.isna(nm)):
            nm = str(cid)
        if cid is not None:
            cvcl_to_name[str(cid)] = str(nm)
    drug_to_moa = {}
    for _, r in dr.iterrows():
        d = r.get("drug")
        if d:
            drug_to_moa[str(d)] = str(r.get("moa-fine", r.get("moa_fine", "unknown")))
    sample_to_conc = {}
    for _, r in sm.iterrows():
        s = r.get("sample")
        if s is not None:
            sample_to_conc[str(s)] = str(r.get("drugname_drugconc", "unknown"))
    logger.info(f"meta maps: {len(cvcl_to_name)} cell lines, {len(drug_to_moa)} drugs, {len(sample_to_conc)} samples")
    return cvcl_to_name, drug_to_moa, sample_to_conc


def format_prompt(cell_line_name, drug_name, dose_str, moa, control_sentence):
    if not moa or moa in ("unknown", "nan", "None"):
        moa = "unclear"
    p = f"Predict the response of {cell_line_name} to {drug_name} at {dose_str}. Mechanism: {moa}."
    p += f"\nControl cell: {control_sentence}"
    p += "\n\nResponse cell:"
    return p


# ----------------------------------------------------------------- core
def build(cache_dir, emb, epsilon, min_treated, min_control, targets, max_ctrl_per_cond,
          out_dir, repo, gate_conditions=200):
    import pandas as pd
    from scipy import sparse
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    Xp = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    panel_genes = json.load(open(os.path.join(cache_dir, "panel_genes.json")))
    if emb == "scvi":
        z = np.load(os.path.join(cache_dir, "scvi_latent.npz"), allow_pickle=True)
        E = z["latent"].astype(np.float32); found = z["found"]
    else:
        z = np.load(os.path.join(cache_dir, "pca.npz"))
        E = z["coords"].astype(np.float32); found = np.ones(len(E), bool)
    logger.info(f"cache: {len(meta)} cells, embedding='{emb}' dim={E.shape[1]}, {int(found.sum())} with embedding")

    cvcl_to_name, drug_to_moa, sample_to_conc = load_meta_maps(repo)

    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    plate = meta["plate"].astype(str).values
    drug = meta["drug"].astype(str).values
    scol, how = td.sample_column(meta)
    if scol is None:
        raise SystemExit("no treatment/well identifier in the cache -- rebuild it with a "
                         "`sample_id` column (see build_embeddings.py)")
    logger.info(f"treatment identifier: column '{scol}' ({how})")
    sample_id_source = how          # recorded in step0_gates.json so the branch is auditable
    sample = meta[scol].astype(str).values
    ok = found & np.array([np.isfinite(E[i]).all() for i in range(len(E))])

    # index controls by (cl,plate); treated by (cl,plate,drug,sample). The fourth element is the
    # treated WELL -- the physical assignment unit -- and was never a concentration.
    ctrl_by_plate = defaultdict(list)
    treat_by_cond = defaultdict(list)
    for i in range(len(meta)):
        if not ok[i]:
            continue
        if is_ctrl[i]:
            ctrl_by_plate[(cl[i], plate[i])].append(i)
        else:
            treat_by_cond[(cl[i], plate[i], drug[i], sample[i])].append(i)

    conds = [k for k, v in treat_by_cond.items()
             if len(v) >= min_treated and len(ctrl_by_plate.get((k[0], k[1]), [])) >= min_control]
    logger.info(f"{len(conds)} usable conditions (>= {min_treated} treated & >= {min_control} control)")
    if not conds:
        logger.error("no usable conditions — lower --min_treated/--min_control or build a bigger cache")
        return

    os.makedirs(out_dir, exist_ok=True)
    # each arm is written to a .partial path and renamed only once the dose check below passes,
    # so a refusal leaves nothing behind that could be mistaken for a finished build
    writer_paths = {t: os.path.join(out_dir, f"{t}.jsonl") for t in targets}
    writers = {t: open(p + ".partial", "w") for t, p in writer_paths.items()}
    gates = {"n_conditions": len(conds), "sample_id_source": sample_id_source,
             "signal_shift": [], "target_sanity": [], "target_contrast": []}
    n_emit = 0
    emitted_doses = []
    rng = np.random.RandomState(0)

    for ci, (c, pl, dg, ds) in enumerate(conds):
        t_idx = np.array(treat_by_cond[(c, pl, dg, ds)])
        ctrl_idx = np.array(ctrl_by_plate[(c, pl)])
        if len(ctrl_idx) > max_ctrl_per_cond:
            ctrl_idx = ctrl_idx[rng.choice(len(ctrl_idx), max_ctrl_per_cond, replace=False)]
        Y = E[t_idx]; X = E[ctrl_idx]
        Yexpr = np.asarray(Xp[t_idx].todense(), dtype=np.float32)          # (n_t, 946) CP10K
        Xexpr = np.asarray(Xp[ctrl_idx].todense(), dtype=np.float32)
        muY, muX = Yexpr.mean(0), Xexpr.mean(0)
        K = int(np.median((Yexpr > 0).sum(1)))                            # target sentence length

        # OT coupling in embedding space
        pi = sinkhorn(cost_matrix(X, Y), epsilon)
        pi_row = pi / (pi.sum(1, keepdims=True) + 1e-300)
        T2expr = pi_row @ Yexpr                                            # (n_c, 946) barycentric

        # ---- gates (sampled conditions) ----
        if ci < gate_conditions:
            shift = float(np.linalg.norm(Y.mean(0) - X.mean(0)))
            null = []
            allE = np.vstack([X, Y]); nX = len(X)
            for _ in range(50):
                p = rng.permutation(len(allE))
                null.append(np.linalg.norm(allE[p[:nX]].mean(0) - allE[p[nX:]].mean(0)))
            gates["signal_shift"].append({"cond": f"{dg}|{c}", "shift": shift,
                                          "p": float((np.sum(np.array(null) >= shift) + 1) / (len(null) + 1))})
            # 0c: target pseudobulk vs real treated pseudobulk; per-control variance
            t2_pb = T2expr.mean(0)
            gates["target_sanity"].append({
                "cond": f"{dg}|{c}",
                "mw2_target_vs_truth": float(np.linalg.norm(t2_pb - muY)),
                "mw2_consensus_vs_truth": float(np.linalg.norm(muY - muY)),  # 0 by construction (ref)
                "per_control_var_T2": float(np.mean(T2expr.var(0))),
                "per_control_var_T0": 0.0})
            # 0d: top-K genes differing from the control prompt (T2 vs T0)
            def topk_set(v):
                idx = np.where(v > 0)[0]
                return set(idx[np.argsort(-v[idx])[:K]].tolist()) if len(idx) > K else set(idx.tolist())
            t0set = topk_set(muY)
            diffs_T0, diffs_T2 = [], []
            for j in range(min(len(ctrl_idx), 20)):
                cset = topk_set(Xexpr[j])
                diffs_T0.append(len(t0set ^ cset) / 2)
                diffs_T2.append(len(topk_set(T2expr[j]) ^ cset) / 2)
            gates["target_contrast"].append({"cond": f"{dg}|{c}", "K": K,
                                              "T0_genes_diff": float(np.mean(diffs_T0)),
                                              "T2_genes_diff": float(np.mean(diffs_T2))})

        # ---- emit training examples ----
        cname = cvcl_to_name.get(c, c)
        moa = drug_to_moa.get(dg, "unclear")
        _st = td.parse_treatment(sample_to_conc.get(ds, "unknown"), ds)
        _dose = _st.primary.dose if _st.primary is not None else None
        dstr = _dose.display() if _dose is not None else "unknown"
        emitted_doses.append(_dose.molar if (_dose is not None and _dose.molar is not None) else ds)
        for j, gi in enumerate(ctrl_idx):
            ctrl_sent = expr_to_sentence(Xexpr[j], panel_genes, K=None)
            prompt = format_prompt(cname, dg, dstr, moa, ctrl_sent)
            tvecs = {}
            if "T0" in targets:
                tvecs["T0"] = muY
            if "T1" in targets:
                tvecs["T1"] = np.maximum(Xexpr[j] + (muY - muX), 0)
            if "T2" in targets:
                tvecs["T2"] = T2expr[j]
            for t, vec in tvecs.items():
                resp = expr_to_sentence(vec, panel_genes, K=K)
                writers[t].write(json.dumps({
                    "prompt": prompt, "response": resp,
                    "metadata": {"drug": dg, "cell_line_id": c, "plate": pl,
                                 "sample_id": ds,
                                 "dose_molar": (_dose.molar if _dose is not None else None),
                                 "dose_raw": (_dose.raw if _dose is not None else "unknown"),
                                 "target": t}}) + "\n")
            n_emit += 1
        if ci % 50 == 0:
            logger.info(f"  condition {ci}/{len(conds)}  emitted {n_emit} prompts")

    for w in writers.values():
        w.close()
    # the defect tahoe_design exists to stop: never ship a dose field that is actually holding
    # sample identifiers
    if td.looks_like_sample_id(emitted_doses):
        for p in writer_paths.values():
            if os.path.exists(p + ".partial"):
                os.remove(p + ".partial")
        raise SystemExit("REFUSING TO WRITE: the emitted dose field holds sample identifiers.")
    for p in writer_paths.values():
        os.replace(p + ".partial", p)
    json.dump(gates, open(os.path.join(out_dir, "step0_gates.json"), "w"), indent=2, default=float)
    _report_gates(gates)
    logger.info(f"emitted {n_emit} prompts x {len(targets)} target arms -> {out_dir}")


def _report_gates(gates):
    ss = gates["signal_shift"]; tc = gates["target_contrast"]; tsan = gates["target_sanity"]
    if ss:
        sig = np.mean([g["p"] < 0.05 for g in ss])
        logger.info(f"GATE 0a signal survival: {100*sig:.0f}% of {len(ss)} conditions have a significant "
                    f"latent treated-vs-control shift (p<0.05)")
    if tsan:
        var2 = np.mean([g["per_control_var_T2"] for g in tsan])
        mw = np.mean([g["mw2_target_vs_truth"] for g in tsan])
        logger.info(f"GATE 0c target sanity: mean per-control T2 variance={var2:.4f} (want >0, i.e. targets "
                    f"vary across controls); mean ||T2 pseudobulk - real treated||={mw:.3f} (want small)")
    if tc:
        d0 = np.mean([g["T0_genes_diff"] for g in tc]); d2 = np.mean([g["T2_genes_diff"] for g in tc])
        logger.info(f"GATE 0d target-contrast (top-K genes differing from control prompt): "
                    f"T0(consensus)={d0:.1f}  T2(OT)={d2:.1f}  (higher => more for the model to learn; "
                    f"if both tiny, Q13 predicts objective still dominates)")


def selftest():
    """Synthetic: 2 plates x 2 drugs; treated = control shifted by a drug-specific gene program.
    Verify OT runs, T2 varies across controls, and T2 recovers the treated pseudobulk."""
    rng = np.random.RandomState(0)
    P, d = 30, 8
    panel = [f"G{i}" for i in range(P)]
    # controls: random baseline; treated: baseline + drug program on a few genes
    def cells(n, base, prog):
        out = np.maximum(0, base + prog + rng.randn(n, P) * 0.3)
        return out
    base = np.abs(rng.rand(P))
    prog = np.zeros(P); prog[[2, 5, 9]] = 3.0
    Xexpr = cells(40, base, 0)
    Yexpr = cells(40, base, prog)
    # embeddings ~ expression PCA-ish (use expression directly for the test)
    X, Y = Xexpr[:, :d], Yexpr[:, :d]
    pi = sinkhorn(cost_matrix(X, Y), 0.5)
    pi_row = pi / (pi.sum(1, keepdims=True) + 1e-300)
    T2 = pi_row @ Yexpr
    var_t2 = float(np.mean(T2.var(0)))
    mw_t2 = float(np.linalg.norm(T2.mean(0) - Yexpr.mean(0)))
    mw_t0 = float(np.linalg.norm(Yexpr.mean(0) - Yexpr.mean(0)))
    s = expr_to_sentence(T2[0], panel, K=10)
    logger.info(f"  T2 per-control variance={var_t2:.4f} (want >0)  ||T2pb-truth||={mw_t2:.3f} (want small)")
    logger.info(f"  sample sentence: {s[:60]}...")
    ok = var_t2 > 1e-4 and mw_t2 < 0.5 and s.endswith(END)
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--emb", choices=["scvi", "pca"], default="scvi")
    ap.add_argument("--epsilon", type=float, default=0.5, help="Sinkhorn regularization (T2 sharpness)")
    ap.add_argument("--min_treated", type=int, default=60)
    ap.add_argument("--min_control", type=int, default=40)
    ap.add_argument("--targets", default="T0,T1,T2")
    ap.add_argument("--max_ctrl_per_cond", type=int, default=60)
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--out_dir", default="RESULTS/ot_targets")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")
    build(args.cache_dir, args.emb, args.epsilon, args.min_treated, args.min_control,
          [t.strip() for t in args.targets.split(",") if t.strip()], args.max_ctrl_per_cond,
          args.out_dir, args.repo)


if __name__ == "__main__":
    main()
