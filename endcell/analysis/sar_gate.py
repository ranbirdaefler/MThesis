#!/usr/bin/env python
r"""
sar_gate.py — Arm 2 Step-0: does chemical structure predict drug RESPONSE in Tahoe? (leak-safe)
==============================================================================================
Before building the chemistry-injection / SAR-CLIP architecture, verify its load-bearing assumption:
that structurally similar drugs have similar transcriptional responses. If not, no chemistry-based
method can help. Reuses the response-distance machinery from the scramble sweep and the SMILES loader
from drug_biology_atlas.

Two measurements, on TRAIN drugs only (tier2 unseen drugs never enter — mirrored by leave-drug-out CV):
  (A) UNSUPERVISED SAR: within cell line, Spearman( chem_dist(A,B) , response_dist(A,B) ) over drug
      pairs, averaged over cell lines. "Do raw chemical distances track response distances?"
  (B) SUPERVISED SAR = the CLIP CEILING (the decisive test): leave-ONE-DRUG-out ridge from the drug's
      chemical embedding -> its response shift Delta. For each held-out drug, report cosine(pred, true)
      and RETRIEVAL — is the predicted Delta closest to the true drug's Delta among all drugs (a
      chemistry->response NIR)? This directly measures whether a FROZEN encoder + learned map
      generalizes to UNSEEN drugs. High here => CLIP works without fine-tuning; low here => the
      frozen ceiling is the reason to consider fine-tuning MolFormer.

Featurizers (run all available): Morgan fingerprints (rdkit) and MolFormer embeddings (transformers).

USAGE
  python sar_gate.py --cache_dir /data/.../ot_cache --out RESULTS/sar_gate.json
  python sar_gate.py --selftest
"""
import argparse, json, os, sys, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
TAHOE_REPO = "tahoebio/Tahoe-100M"


# ----------------------------------------------------------------- chemistry featurizers
def load_drug_smiles(repo=TAHOE_REPO):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    df = pd.read_parquet(hf_hub_download(repo, "metadata/drug_metadata.parquet", repo_type="dataset"))
    col = next((c for c in ("canonical_smiles", "smiles", "SMILES") if c in df.columns), None)
    if col is None:
        logger.warning(f"no SMILES column in drug_metadata ({list(df.columns)})")
        return {}
    return {r["drug"]: r[col] for _, r in df.iterrows() if r.get("drug") and r.get(col)}


def featurize_morgan(smiles_of, n_bits=2048):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception:
        logger.warning("rdkit not available -> skipping Morgan (pip install rdkit)")
        return {}
    out = {}
    for d, smi in smiles_of.items():
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.float32)
                from rdkit.DataStructs import ConvertToNumpyArray
                ConvertToNumpyArray(fp, arr)
                out[d] = arr
        except Exception:
            continue
    logger.info(f"Morgan fingerprints: {len(out)} drugs")
    return out


def featurize_molformer(smiles_of, model_name="ibm/MoLFormer-XL-both-10pct", batch=64):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, deterministic_eval=True).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    except Exception as e:
        logger.warning(f"MolFormer unavailable -> skipping ({type(e).__name__}: {e})")
        return {}
    drugs = [d for d in smiles_of]
    out = {}
    with torch.no_grad():
        for i in range(0, len(drugs), batch):
            chunk = drugs[i:i + batch]
            enc = tok([smiles_of[d] for d in chunk], padding=True, truncation=True,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)      # mean-pool over tokens
            for j, d in enumerate(chunk):
                out[d] = emb[j].float().cpu().numpy()
    logger.info(f"MolFormer embeddings: {len(out)} drugs (dim {next(iter(out.values())).shape[0]})")
    return out


# ----------------------------------------------------------------- response shifts from the cache
def load_response_shifts(cache_dir, min_treated=40, min_control=20):
    """Delta[(drug, cell_line)] = mean(log1p treated) - mean(log1p control) over the panel (CP10K)."""
    import pandas as pd
    from scipy import sparse
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    X = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    drug = meta["drug"].astype(str).values
    # control mean per cell line (pool plates)
    ctrl_mean = {}
    for c in np.unique(cl[is_ctrl]):
        idx = np.where(is_ctrl & (cl == c))[0]
        if len(idx) >= min_control:
            ctrl_mean[c] = np.asarray(np.log1p(X[idx].todense()).mean(0)).ravel()
    delta = {}
    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], cl[i])].append(i)
    for (d, c), idxs in by.items():
        if len(idxs) >= min_treated and c in ctrl_mean:
            tmean = np.asarray(np.log1p(X[idxs].todense()).mean(0)).ravel()
            delta[(d, c)] = (tmean - ctrl_mean[c]).astype(np.float32)
    logger.info(f"response shifts: {len(delta)} (drug,cell_line) with >= {min_treated} treated / "
                f"{min_control} control; {len(set(k[0] for k in delta))} distinct drugs")
    return delta


# ----------------------------------------------------------------- SAR measurements
def unsupervised_sar(feats, delta):
    """within cell line: Spearman(chem_dist, response_dist) over drug pairs, averaged over cell lines."""
    from scipy.stats import spearmanr
    by_cl = defaultdict(list)
    for (d, c), v in delta.items():
        if d in feats:
            by_cl[c].append(d)
    rhos = []
    for c, drugs in by_cl.items():
        drugs = [d for d in drugs if d in feats]
        if len(drugs) < 5:
            continue
        cd, rd = [], []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                a, b = drugs[i], drugs[j]
                cd.append(float(np.linalg.norm(feats[a] - feats[b])))
                rd.append(float(np.linalg.norm(delta[(a, c)] - delta[(b, c)])))
        if len(cd) >= 10:
            rho = spearmanr(cd, rd).statistic
            if rho == rho:
                rhos.append(rho)
    return (float(np.mean(rhos)), len(rhos)) if rhos else (float("nan"), 0)


def supervised_sar(feats, delta, ridge=10.0, n_retr_drugs=None, seed=0):
    """Leave-ONE-DRUG-out: fit ridge chem_embedding -> response Delta on all OTHER drugs (pooled over
    cell lines with a cell-line one-hot), predict the held-out drug, and report cosine(pred,true) +
    retrieval NIR (is pred closest to the true drug's Delta among all drugs?). This is the CLIP ceiling
    and mirrors tier2 (the held-out drug is unseen by the map)."""
    drugs = sorted(set(d for (d, c) in delta if d in feats))
    cls = sorted(set(c for (d, c) in delta if d in feats))
    cl_ix = {c: i for i, c in enumerate(cls)}
    # design rows: (drug, cell_line) with chem feat + cell-line one-hot -> Delta
    rows = [(d, c) for (d, c) in delta if d in feats]
    Fdim = len(next(iter(feats.values())))
    def x_of(d, c):
        oh = np.zeros(len(cls), np.float32); oh[cl_ix[c]] = 1.0
        return np.concatenate([feats[d], oh])
    X = np.stack([x_of(d, c) for (d, c) in rows])
    Y = np.stack([delta[(d, c)] for (d, c) in rows])
    row_drug = np.array([d for (d, c) in rows])
    cos_list, retr_list = [], []
    for held in drugs:
        tr = row_drug != held
        te = row_drug == held
        if tr.sum() < 10 or te.sum() == 0:
            continue
        Xt, Yt = X[tr], Y[tr]
        mu = Xt.mean(0); Xtc = Xt - mu
        A = Xtc.T @ Xtc + ridge * np.eye(Xt.shape[1])
        W = np.linalg.solve(A, Xtc.T @ (Yt - Yt.mean(0)))
        b = Yt.mean(0)
        for k in np.where(te)[0]:
            pred = (X[k] - mu) @ W + b
            true = Y[k]
            cos = float(pred @ true / (np.linalg.norm(pred) * np.linalg.norm(true) + 1e-9))
            cos_list.append(cos)
            # retrieval: among all (drug) Deltas in this cell line, is the true drug the closest to pred?
            c = rows[k][1]
            cands = [(d2, delta[(d2, c)]) for d2 in drugs if (d2, c) in delta]
            if len(cands) >= 3:
                dists = np.array([np.linalg.norm(pred - v) for _, v in cands])
                names = [d2 for d2, _ in cands]
                worse = np.sum(dists > dists[names.index(held)])
                retr_list.append(worse / (len(cands) - 1))
    return {"n_drugs_cv": len(set(row_drug)),
            "cosine_mean": float(np.mean(cos_list)) if cos_list else float("nan"),
            "retrieval_nir": float(np.mean(retr_list)) if retr_list else float("nan"),
            "n_pred": len(cos_list)}


def report(name, feats, delta):
    if not feats:
        return None
    rho, n_cl = unsupervised_sar(feats, delta)
    sup = supervised_sar(feats, delta)
    logger.info(f"[{name}] UNSUPERVISED SAR: mean within-cell-line Spearman(chem,resp) = {rho:+.3f} "
                f"({n_cl} cell lines)")
    logger.info(f"[{name}] SUPERVISED (CLIP ceiling, leave-drug-out): cosine(pred,true)={sup['cosine_mean']:+.3f}  "
                f"retrieval-NIR={sup['retrieval_nir']:.3f} (chance 0.50; >>0.5 => chemistry predicts "
                f"UNSEEN-drug response)  n={sup['n_pred']}")
    return {"unsupervised_spearman": rho, "n_cell_lines": n_cl, "supervised": sup}


def selftest():
    """Synthetic: response Delta is a fixed linear function of a random chem feature + cell-line offset.
    SAR must be strongly positive (both unsupervised and supervised)."""
    rng = np.random.RandomState(0)
    D, F, P, C = 30, 16, 40, 4
    Wtrue = rng.randn(F, P)
    feats = {f"d{i}": rng.randn(F).astype(np.float32) for i in range(D)}
    cl_off = {f"c{j}": rng.randn(P) * 0.3 for j in range(C)}
    delta = {}
    for i in range(D):
        for j in range(C):
            delta[(f"d{i}", f"c{j}")] = (feats[f"d{i}"] @ Wtrue + cl_off[f"c{j}"]
                                         + rng.randn(P) * 0.2).astype(np.float32)
    r = report("SELFTEST", feats, delta)
    ok = r["unsupervised_spearman"] > 0.3 and r["supervised"]["retrieval_nir"] > 0.8
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--out", default="RESULTS/sar_gate.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")

    smiles = load_drug_smiles(args.repo)
    logger.info(f"drug->SMILES: {len(smiles)} drugs")
    delta = load_response_shifts(args.cache_dir, args.min_treated, args.min_control)
    drugs_with_resp = set(k[0] for k in delta)
    smiles = {d: s for d, s in smiles.items() if d in drugs_with_resp}
    logger.info(f"drugs with BOTH SMILES and response: {len(smiles)}")

    out = {"n_drugs": len(smiles), "n_conditions": len(delta), "featurizers": {}}
    out["featurizers"]["morgan"] = report("Morgan", featurize_morgan(smiles), delta)
    out["featurizers"]["molformer"] = report("MolFormer", featurize_molformer(smiles), delta)

    logger.info("=" * 90)
    logger.info("GATE READING: SUPERVISED retrieval-NIR is decisive. >>0.5 (say >0.65) => a FROZEN encoder")
    logger.info("  + learned map predicts unseen-drug response => CLIP will work, no fine-tuning needed.")
    logger.info("  ~0.5 for BOTH featurizers => chemistry is not predictive at this resolution => the")
    logger.info("  frozen ceiling is low (consider fine-tuning MolFormer, or reconsider the arm).")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
