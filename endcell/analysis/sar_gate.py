#!/usr/bin/env python
r"""
sar_gate.py — Arm 2 Step-0: does chemical structure predict drug RESPONSE in Tahoe? (leak-safe)
==============================================================================================
Before building the chemistry-injection / SAR-CLIP architecture, verify its load-bearing assumption:
structurally similar drugs have similar transcriptional responses. Crucially, this is only a fair
test if the response itself is REPRODUCIBLE — Q1/Q11 say the drug-specific residual is noise-limited
and ~27% of drugs are inert, so chemistry failing could mean "our response is noise", not "chemistry
is uninformative". So we always report a NOISE CEILING and compare chemistry to it.

Everything is on the DRUG-SPECIFIC RESIDUAL: within a cell line we subtract the mean-over-drugs shift
(the generic stress program shared by all drugs) so the metric isolates WHICH drug, not the common
response. Leave-DRUG-out CV mirrors tier2 (the held-out drug is unseen), so no leakage.

Measurements (per featurizer: Morgan fingerprints + MolFormer embeddings):
  * NOISE CEILING (retrieval): use a drug's half-B residual to retrieve its half-A residual among all
    drugs in the cell line. This is the MAX achievable retrieval given response noise. ~0.5 => the
    drug-specific response is not even self-reproducible -> the SAR test is underpowered (not chemistry).
  * REPRODUCIBLE subset: (drug,cell_line) whose half-A/half-B residuals agree (cosine > thr).
  * UNSUPERVISED SAR: within cell line, Spearman(chem_dist, residual_dist).
  * SUPERVISED SAR (the CLIP ceiling): leave-drug-out ridge chem -> residual; RETRIEVAL-NIR of the
    held-out drug. Compare to the NOISE CEILING: chemistry near the ceiling => chemistry works;
    chemistry ~0.5 while ceiling is high => chemistry genuinely uninformative (redesign / fine-tune).

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
        logger.warning(f"no SMILES column in drug_metadata ({list(df.columns)})"); return {}
    return {r["drug"]: r[col] for _, r in df.iterrows() if r.get("drug") and r.get(col)}


def featurize_morgan(smiles_of, n_bits=2048):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.DataStructs import ConvertToNumpyArray
    except Exception:
        logger.warning("rdkit not available -> skipping Morgan (pip install rdkit)"); return {}
    out = {}
    for d, smi in smiles_of.items():
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            arr = np.zeros(n_bits, dtype=np.float32)
            ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits), arr)
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
        logger.warning(f"MolFormer unavailable -> skipping ({type(e).__name__}: {e})"); return {}
    drugs = list(smiles_of); out = {}
    with torch.no_grad():
        for i in range(0, len(drugs), batch):
            chunk = drugs[i:i + batch]
            enc = tok([smiles_of[d] for d in chunk], padding=True, truncation=True, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            for j, d in enumerate(chunk):
                out[d] = emb[j].float().cpu().numpy()
    logger.info(f"MolFormer embeddings: {len(out)} drugs (dim {next(iter(out.values())).shape[0]})")
    return out


# ----------------------------------------------------------------- response shifts + residuals
def load_response_shifts(cache_dir, min_treated=40, min_control=20, seed=0):
    """Per (drug, cell_line): full / halfA / halfB response shift = mean(log1p treated) - mean(log1p ctrl)."""
    import pandas as pd
    from scipy import sparse
    rng = np.random.RandomState(seed)
    meta = pd.read_parquet(os.path.join(cache_dir, "meta.parquet"))
    X = sparse.load_npz(os.path.join(cache_dir, "panel_expr.npz")).tocsr()
    is_ctrl = meta["is_control"].values.astype(bool)
    cl = meta["cell_line_id"].astype(str).values
    drug = meta["drug"].astype(str).values
    ctrl_mean = {}
    for c in np.unique(cl[is_ctrl]):
        idx = np.where(is_ctrl & (cl == c))[0]
        if len(idx) >= min_control:
            ctrl_mean[c] = np.asarray(np.log1p(X[idx].todense()).mean(0)).ravel()
    by = defaultdict(list)
    for i in range(len(meta)):
        if not is_ctrl[i]:
            by[(drug[i], cl[i])].append(i)
    shifts = {}
    for (d, c), idxs in by.items():
        if len(idxs) < min_treated or c not in ctrl_mean:
            continue
        idxs = list(idxs); rng.shuffle(idxs); h = len(idxs) // 2
        L = lambda ix: np.asarray(np.log1p(X[ix].todense()).mean(0)).ravel() - ctrl_mean[c]
        shifts[(d, c)] = {"full": L(idxs).astype(np.float32),
                          "A": L(idxs[:h]).astype(np.float32), "B": L(idxs[h:]).astype(np.float32)}
    logger.info(f"response shifts: {len(shifts)} (drug,cell_line); "
                f"{len(set(k[0] for k in shifts))} distinct drugs")
    return shifts


def residualize(shifts, key):
    """Subtract the per-cell-line mean-over-drugs shift -> the DRUG-SPECIFIC residual for `key`."""
    by_cl = defaultdict(list)
    for (d, c) in shifts:
        by_cl[c].append(d)
    res = {}
    for c, drugs in by_cl.items():
        M = np.mean(np.stack([shifts[(d, c)][key] for d in drugs]), axis=0)
        for d in drugs:
            res[(d, c)] = shifts[(d, c)][key] - M
    return res


def retrieval_nir(query, gallery):
    """query/gallery: {(drug,cl): vec}. For each (d,c), is query closest to gallery[(d,c)] among all
    drugs in the SAME cell line? Returns mean retrieval-NIR (0.5 = chance)."""
    by_cl = defaultdict(list)
    for (d, c) in gallery:
        by_cl[c].append(d)
    vals = []
    for (d, c), q in query.items():
        cand = [dd for dd in by_cl[c] if (dd, c) in gallery]
        if len(cand) < 3 or d not in cand:
            continue
        dists = np.array([np.linalg.norm(q - gallery[(dd, c)]) for dd in cand])
        worse = np.sum(dists > dists[cand.index(d)])
        vals.append(worse / (len(cand) - 1))
    return float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------- SAR
def unsupervised_sar(feats, res):
    from scipy.stats import spearmanr
    by_cl = defaultdict(list)
    for (d, c) in res:
        if d in feats:
            by_cl[c].append(d)
    rhos = []
    for c, drugs in by_cl.items():
        if len(drugs) < 5:
            continue
        cd, rd = [], []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                a, b = drugs[i], drugs[j]
                cd.append(float(np.linalg.norm(feats[a] - feats[b])))
                rd.append(float(np.linalg.norm(res[(a, c)] - res[(b, c)])))
        if len(cd) >= 10:
            rho = spearmanr(cd, rd).statistic
            if rho == rho:
                rhos.append(rho)
    return (float(np.mean(rhos)), len(rhos)) if rhos else (float("nan"), 0)


def supervised_sar(feats, res, ridge=10.0):
    """leave-DRUG-out ridge chem -> residual; predicted residual retrieval-NIR (the CLIP ceiling)."""
    rows = [(d, c) for (d, c) in res if d in feats]
    cls = sorted(set(c for (_, c) in rows)); cl_ix = {c: i for i, c in enumerate(cls)}
    drugs = sorted(set(d for (d, _) in rows))
    def x_of(d, c):
        oh = np.zeros(len(cls), np.float32); oh[cl_ix[c]] = 1.0
        return np.concatenate([feats[d], oh])
    X = np.stack([x_of(d, c) for (d, c) in rows]); Y = np.stack([res[(d, c)] for (d, c) in rows])
    row_drug = np.array([d for (d, _) in rows])
    pred = {}
    for held in drugs:
        tr, te = row_drug != held, row_drug == held
        if tr.sum() < 10 or te.sum() == 0:
            continue
        Xt, Yt = X[tr], Y[tr]; mu = Xt.mean(0); Xtc = Xt - mu
        W = np.linalg.solve(Xtc.T @ Xtc + ridge * np.eye(Xt.shape[1]), Xtc.T @ (Yt - Yt.mean(0)))
        for k in np.where(te)[0]:
            pred[rows[k]] = (X[k] - mu) @ W + Yt.mean(0)
    return {"n_drugs": len(drugs), "retrieval_nir": retrieval_nir(pred, res), "n_pred": len(pred)}


def report(name, feats, res_full, res_A, res_B, repro_keys):
    if not feats:
        return None
    rho_all, ncl = unsupervised_sar(feats, res_full)
    sup_all = supervised_sar(feats, res_full)
    res_full_r = {k: v for k, v in res_full.items() if k in repro_keys}
    rho_r, ncl_r = unsupervised_sar(feats, res_full_r)
    sup_r = supervised_sar(feats, res_full_r)
    logger.info(f"[{name}] ALL drugs      : unsup Spearman={rho_all:+.3f}  supervised retrieval-NIR={sup_all['retrieval_nir']:.3f}  (n={sup_all['n_pred']})")
    logger.info(f"[{name}] REPRODUCIBLE   : unsup Spearman={rho_r:+.3f}  supervised retrieval-NIR={sup_r['retrieval_nir']:.3f}  (n={sup_r['n_pred']})")
    return {"unsup_all": rho_all, "sup_all": sup_all, "unsup_repro": rho_r, "sup_repro": sup_r}


def selftest():
    rng = np.random.RandomState(0)
    D, F, P, C = 30, 16, 40, 4
    Wt = rng.randn(F, P); feats = {f"d{i}": rng.randn(F).astype(np.float32) for i in range(D)}
    shifts = {}
    for i in range(D):
        for j in range(C):
            base = feats[f"d{i}"] @ Wt + rng.randn(P) * 0.3 * 5     # cell-line offset via noise scale
            shifts[(f"d{i}", f"c{j}")] = {"full": (base + rng.randn(P) * 0.1).astype(np.float32),
                                          "A": (base + rng.randn(P) * 0.15).astype(np.float32),
                                          "B": (base + rng.randn(P) * 0.15).astype(np.float32)}
    rf, rA, rB = residualize(shifts, "full"), residualize(shifts, "A"), residualize(shifts, "B")
    ceil = retrieval_nir(rB, rA)
    repro = {k for k in shifts if float(rA[k] @ rB[k] / (np.linalg.norm(rA[k]) * np.linalg.norm(rB[k]) + 1e-9)) > 0.2}
    logger.info(f"SELFTEST noise-ceiling retrieval={ceil:.3f}  reproducible {len(repro)}/{len(shifts)}")
    r = report("SELFTEST", feats, rf, rA, rB, repro)
    ok = r["sup_all"]["retrieval_nir"] > 0.8 and ceil > 0.8
    logger.info(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--repo", default=TAHOE_REPO)
    ap.add_argument("--min_treated", type=int, default=40)
    ap.add_argument("--min_control", type=int, default=20)
    ap.add_argument("--repro_thr", type=float, default=0.2, help="cosine(res_A,res_B) to call a "
                    "(drug,cell_line) response reproducible above noise")
    ap.add_argument("--out", default="RESULTS/sar_gate.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.cache_dir:
        ap.error("--cache_dir required (unless --selftest)")

    smiles = load_drug_smiles(args.repo)
    shifts = load_response_shifts(args.cache_dir, args.min_treated, args.min_control)
    drugs_with_resp = set(k[0] for k in shifts)
    smiles = {d: s for d, s in smiles.items() if d in drugs_with_resp}
    logger.info(f"drugs with BOTH SMILES and response: {len(smiles)}")

    res_full = residualize(shifts, "full")
    res_A, res_B = residualize(shifts, "A"), residualize(shifts, "B")
    # NOISE CEILING: retrieve half-A residual from half-B residual (max achievable given noise)
    ceil = retrieval_nir(res_B, res_A)
    repro_keys = set()
    for k in shifts:
        a, b = res_A[k], res_B[k]
        if float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) > args.repro_thr:
            repro_keys.add(k)
    logger.info("=" * 92)
    logger.info(f"NOISE CEILING (drug retrieval from its OWN response replicate): {ceil:.3f}  "
                f"(0.5=chance). This is the ceiling any predictor can reach.")
    logger.info(f"REPRODUCIBLE (drug,cell_line): {len(repro_keys)}/{len(shifts)} have res_A~res_B > {args.repro_thr}")
    logger.info("=" * 92)

    out = {"n_drugs": len(smiles), "n_conditions": len(shifts), "noise_ceiling_retrieval": ceil,
           "n_reproducible": len(repro_keys), "featurizers": {}}
    out["featurizers"]["morgan"] = report("Morgan", featurize_morgan(smiles), res_full, res_A, res_B, repro_keys)
    out["featurizers"]["molformer"] = report("MolFormer", featurize_molformer(smiles), res_full, res_A, res_B, repro_keys)

    logger.info("=" * 92)
    logger.info("READ: compare each featurizer's SUPERVISED retrieval-NIR to the NOISE CEILING above.")
    logger.info("  chemistry near ceiling  => chemistry predicts unseen-drug response (Arm 2 grounded).")
    logger.info("  chemistry ~0.5 while ceiling high => chemistry genuinely uninformative (redesign/fine-tune).")
    logger.info("  ceiling ~0.5 => response residual not self-reproducible => test underpowered, not a verdict.")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
