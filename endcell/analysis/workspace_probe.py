#!/usr/bin/env python
r"""
workspace_probe.py — is drug identity encoded in a CAUSALLY INERT subspace?
===========================================================================
Bridges the two standing observations — (i) drug identity is DECODABLE from activations (~82% at
layer 9) and (ii) the model IGNORES the drug (scramble / output-invariance null) — into a single
mechanistic claim: the direction that ENCODES drug identity is not the direction that DRIVES
generation.

WHY A NAIVE ABLATION IS UNFALSIFIABLE (and how we fix it):
  We ALREADY know the model is drug-blind. So ablating a direction it doesn't use is GUARANTEED to
  change nothing — a null there is indistinguishable from a broken hook. Guards:
    * POSITIVE CONTROL subspace: the CELL-LINE subspace, which the model demonstrably DOES use.
      Ablating it MUST produce a large causal effect — that is the proof the instrument works, and
      it now GATES the headline instead of merely being printed.
    * NULL subspace: random orthonormal subspaces of matched dimension, AVERAGED OVER SEVERAL DRAWS.
    * ABLATION-VERIFICATION GATE, two-sided, on the purified slab AND in the state the intervention
      actually instantiates.

THE IDENTITY TEST, AND THE TWO CONFOUNDS IT HAD TO SURVIVE
  v1 compared drug-B injection against a random vector drawn in the FULL 2048-dim space while the
  swap vector lived in a <=10-dim slab the hook ablated. ~99.5% of the random energy landed in
  directions that were never ablated. Retracted.

  v2 drew the random vector INSIDE the slab, which looked like a fix and was not. The hook did
  ablate-then-add, so the delivered change was (added - removed):

      swap arm   comp_B - s        random arm   rv - s      (s = the cell's own slab component)
      E||rv - s||^2 - ||comp_B - s||^2 = 2<comp_B, s>  > 0   ALWAYS

  because drug_mean was the RAW, UNCENTERED class mean, so comp_B and s shared the global-mean term.
  The swap arm put back roughly what it removed; the random arm put back something orthogonal. With
  no model in the loop the random arm delivers 5.3x the perturbation in 100% of draws. Retracted.

  An intermediate design scored delta-logP(r_B) against a norm-matched random direction. Confounded
  the OTHER way: logP(t) = z_t - logsumexp(z), and an off-manifold random direction inflates the
  normalizer (+2.75 nats measured) while an on-manifold class-mean displacement does not (-0.22), so
  every target is depressed under the control whether or not it carries B's identity. Forcing the
  scored target to carry ZERO identity information still left 36% of the effect standing.
  Norm-matched is not disruption-matched.

  The current design removes the ablation from the identity test and makes the arms structurally
  identical, so that only IDENTITY differs:

      score     logP(r_B | prompt_A) - logP(r_A | prompt_A)     <- normalizer cancels algebraically
      swap      delta = P(mu_B - mu_A)                          <- global mean cancels exactly
      permuted  delta = P(mu_C - mu_A)   HEADLINE CONTROL       <- same anchor, same construction,
                                                                   same norm, WRONG drug
      isotropic delta = random in slab                          <- off-manifold DAMAGE diagnostic
                                                                   only, never the headline

  r_B comes from a THIRD disjoint split, never from the rows that define mu_B.

WHAT A NULL HERE DOES AND DOES NOT MEAN
  Subspace recovery is SNR-limited. On synthetic data with the cells per drug this design can afford,
  the estimated drug axis overlaps the planted one by only ~0.65, reaching 0.90 near 500 cells per
  class. That bounds POWER, not validity -- both arms are projected into the SAME estimated slab and
  differ only in which drug they point at. So a null must be read as "not resolved at this n", and
  the verdict strings enforce that: EQUIVALENT is earned only when the whole CI lies inside the
  margin AND contains zero; otherwise the answer is INCONCLUSIVE / underpowered.

EVERYTHING IS FIT OUT OF SAMPLE
  Subspaces AND class means are estimated on the est half; every KL, gate and swap is evaluated on
  the disjoint eval half. v2 held out only the means, which left the subspace — the object the
  causal claim is about — fit on the very activations whose KL it then measured.

SELFTEST (no model/data) exercises the SWAP PATH ITSELF under THREE planted truths --
drug inert, drug read, and drug inert behind a NORM-SENSITIVE readout (the world that
catches manifold asymmetry between arms):
  python workspace_probe.py --selftest
"""
# --- repo path bootstrap: works in BOTH the reorganized repo AND the flat cluster layout ---
import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
_cands = [_HERE, os.path.join(_HERE, "src")]
if os.path.isdir(os.path.join(_ROOT, "shared")):
    _cands += [os.path.join(_ROOT, "shared")] + sorted(glob.glob(os.path.join(_PIPE, "*")))
for _p in _cands:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse, json, logging
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- linear-algebra core (numpy)
def build_subspace(X, labels, max_dims, idx=None, spec_tol=0.05):
    """Class-mean (between-class) subspace, optionally fit on a SUBSET of rows (idx).

    The rank cutoff is RELATIVE to the leading singular value, not the old absolute S > 1e-8. With C
    classes the old rule kept min(max_dims, C-1) directions whatever the data looked like, so when
    the label truly varies along fewer directions than that, the surplus dimensions are pure
    estimation noise -- and noise directions are not orthogonal to the context subspace, so they
    carry context leakage straight into the 'purified' slab. The selftest catches this: it plants a
    5-dim drug subspace among 8 classes and the old rule returned 7 dims, two of them noise, which
    leaked enough cell-line signal to fail the absolute inertness bound.
    """
    X = np.asarray(X, dtype=np.float64)
    if idx is not None:
        X, labels = X[idx], [labels[i] for i in idx]
    uniq = sorted(set(labels))
    gmean = X.mean(0)
    M = np.stack([X[[i for i, l in enumerate(labels) if l == u]].mean(0) - gmean for u in uniq])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(max_dims, len(uniq) - 1, int(np.sum(S > max(spec_tol * S[0], 1e-8))))
    return Vt[:max(k, 1)].T.astype(np.float64)


def build_union_subspace(X, label_sets, max_dims, idx=None, min_class=5, spec_tol=0.05):
    """Subspace spanned by the class means of SEVERAL label sets at once (the 'context / batch'
    subspace). Classes with fewer than min_class members are SKIPPED: a singleton class mean is that
    one cell's activation, which would let a near-continuous field (dose) degenerate the context
    subspace into a generic top-variance basis and orthogonalize away the drug axis wholesale."""
    X = np.asarray(X, dtype=np.float64)
    if idx is not None:
        X = X[idx]
        label_sets = [[labels[i] for i in idx] for labels in label_sets]
    gmean = X.mean(0)
    rows, dropped = [], 0
    for labels in label_sets:
        for u in sorted(set(labels)):
            sel = [i for i, l in enumerate(labels) if l == u]
            if len(sel) >= min_class:
                rows.append(X[sel].mean(0) - gmean)
            else:
                dropped += 1
    if not rows:
        return np.zeros((X.shape[1], 0)), dropped
    M = np.stack(rows)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # same RELATIVE rule as build_subspace. An absolute 1e-8 here would pad the context subspace with
    # noise directions, and every one of those is subtracted from the drug axis by orthogonalize() --
    # so an over-parameterised context subspace silently eats real drug signal.
    k = min(max_dims, int(np.sum(S > max(spec_tol * S[0], 1e-8))))
    return Vt[:max(k, 1)].T.astype(np.float64), dropped


def random_subspace(H, k, rng):
    k = max(int(k), 1)
    Q, _ = np.linalg.qr(rng.randn(H, k))
    return Q[:, :k]


def orthogonalize(V, W, rel_tol=0.2):
    """Component of V's span orthogonal to W's span.

    The cutoff is RELATIVE, not the old absolute 1e-6. V's columns are unit norm, so |R_ii| is
    literally the fraction of that column surviving the projection; an absolute 1e-6 retained
    directions whose surviving component was numerically negligible and then RENORMALISED them to
    unit length, amplifying context and estimation noise by up to six orders of magnitude and
    inflating d_pure with directions that are pure noise.
    """
    Vp = V - W @ (W.T @ V) if W.shape[1] else V.copy()
    Q, R = np.linalg.qr(Vp)
    resid = np.abs(np.diag(R))
    keep = resid > rel_tol
    return Q[:, keep], resid


def shared_component(V, W):
    """The part of V's span that lies INSIDE W — i.e. exactly what orthogonalize() throws away.

    Without this, 'the pure drug subspace is causally inert' is indistinguishable from 'purification
    deleted the drug's causally live component'. If the shared part carries the effect, the correct
    conclusion is 'the drug is read only via context-shared directions', which is a different and
    much weaker claim than the one the chapter was making.
    """
    if W.shape[1] == 0:
        return np.zeros((V.shape[0], 0))
    Vs = W @ (W.T @ V)
    Q, R = np.linalg.qr(Vs)
    return Q[:, np.abs(np.diag(R)) > 0.2]


def subspace_overlap(V, W):
    """Mean squared cosine of principal angles — how much V's span lies inside W's span (0..1)."""
    if V.shape[1] == 0 or W.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(V.T @ W, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def project_out(X, V):
    return X - (X @ V) @ V.T if V.shape[1] else X


def variance_share(X, V):
    if V.shape[1] == 0:
        return 0.0
    Xc = X - X.mean(0)
    return float(np.sum((Xc @ V) ** 2) / (float(np.sum(Xc ** 2)) + 1e-12))


def probe_acc(X, labels, rng, n_splits=5):
    """Cross-validated multinomial logistic-regression accuracy + shuffled-label chance floor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    classes = sorted(set(labels))
    if len(classes) < 2:
        return None, None          # a collapsed label set is missing metadata, not a result
    y = np.array([classes.index(l) for l in labels])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    nsp = min(n_splits, int(np.min(np.bincount(y))))
    if nsp < 2:
        return None, None
    acc = float(np.mean(cross_val_score(clf, X, y, cv=nsp)))
    ys = y.copy(); rng.shuffle(ys)
    shuf = float(np.mean(cross_val_score(clf, X, ys, cv=nsp)))
    return acc, shuf


def kl_rows(logp_clean, logp_alt):
    p = np.exp(logp_clean)
    return np.sum(p * (logp_clean - logp_alt), axis=-1)


# ----------------------------------------------------------------- statistics
def cluster_bootstrap(diffs, clusters, n_boot, rng):
    """Bootstrap the mean of `diffs` by resampling whole CLUSTERS (drugs), not rows.

    v2 resampled prompts. Rows sharing a drug share drug B's estimated mean and the same injected
    displacement, so they are not independent; resampling them treats one drug's idiosyncrasy as
    many observations and gives an interval that is too narrow. The evaluation slice also spans only
    a handful of drug identities, which makes the distinction large rather than academic.
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    groups = defaultdict(list)
    for v, c in zip(diffs, clusters):
        groups[c].append(v)
    keys = sorted(groups)
    arrs = [np.array(groups[k]) for k in keys]
    if len(keys) < 2:
        return float(diffs.mean()), float("nan"), float("nan"), 1.0, len(keys)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(arrs), len(arrs))
        boots[b] = np.mean(np.concatenate([arrs[i] for i in pick]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    p = 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
    return float(diffs.mean()), lo, hi, min(max(p, 1.0 / n_boot), 1.0), len(keys)


def benjamini_hochberg(pvals):
    """BH-adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        prev = min(prev, p[order[rank]] * m / (rank + 1))
        out[order[rank]] = min(prev, 1.0)
    return out


def stratified_take(rows, n, key, rng):
    """Take ~n rows spread evenly across the values of `key`, round-robin.

    load_prompts emits rows drug-major, so `rows[:n]` covers only the first two or three drugs. Every
    KL, every ablation ratio and every swap in v2 was measured on that slice — seven layers of
    results resting on a handful of drug identities, with a sem computed as if they were independent.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for k in groups:
        rng.shuffle(groups[k])
    keys = sorted(groups)
    out, i = [], 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        i += 1
    return out


def bin_dose(vals, n_bins=4):
    """Quantile-bin a continuous field so its class means are estimated from real groups."""
    v = np.array([np.nan if x is None else float(x) for x in vals], dtype=float)
    good = ~np.isnan(v)
    if good.sum() < n_bins:
        return ["NA"] * len(v)
    edges = np.unique(np.quantile(v[good], np.linspace(0, 1, n_bins + 1)[1:-1]))
    return ["NA" if np.isnan(x) else f"q{int(np.searchsorted(edges, x))}" for x in v]


# ----------------------------------------------------------------- model plumbing (torch)
def get_layers(model):
    for attr in ("gpt_neox", "model", "transformer"):
        base = getattr(model, attr, None)
        if base is not None:
            if hasattr(base, "layers"):
                return base.layers
            if hasattr(base, "h"):
                return base.h
    raise RuntimeError("could not locate transformer layer list on this model")


def extract_activations(model, tok, prompts, device, layers, bf16):
    """activations[L] = (n_prompts, H) at the LAST PROMPT position, read FROM THE SAME POINT THE
    HOOK WRITES TO.

    v3.0 read `out.hidden_states[L]` while the intervention hooked `layers_mod[L-1]`. Those agree for
    interior layers but NOT for the last one: in GPT-NeoX `hidden_states[-1]` has `final_layer_norm`
    applied, whereas the block's own output does not. Pythia-1b has 16 layers and 16 was in the
    default layer list, so that layer estimated its subspace and class means in one coordinate frame
    and injected into a different one. Capturing from the block output makes the two frames identical
    by construction, at every depth, with no special-casing.
    """
    import torch
    acts = {L: [] for L in layers}
    grab = {}

    def capture(L):
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            grab[L] = hs[0, -1, :].detach().float().cpu().numpy()
        return hook

    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        hs_handles = [get_layers(model)[L - 1].register_forward_hook(capture(L)) for L in layers]
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in hs_handles:
                h.remove()
        for L in layers:
            acts[L].append(grab[L])
    return {L: np.vstack(v) for L, v in acts.items()}


def make_hook(V_t=None, delta=None):
    """Forward hook on layer (L-1). Two modes, MUTUALLY EXCLUSIVE and enforced:
         V_t   -> ABLATE the subspace at every position           (the ablation arm)
         delta -> ADD a fixed displacement at every position      (the identity arm)
    The v2 hook did ablate-then-add, which is what made the swap's control unmatchable: the delivered
    change was (added - removed) and only `added` was norm-matched. Silently ignoring one argument
    when both are passed is exactly how that seam failed before, so it raises instead."""
    if (V_t is None) == (delta is None):
        raise ValueError("make_hook: pass exactly one of V_t (ablate) or delta (displace)")

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        if V_t is not None:
            Vd = V_t.to(hs.dtype)
            hs2 = hs - (hs @ Vd) @ Vd.transpose(0, 1)
        else:
            hs2 = hs + delta.to(hs.dtype)
        return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2
    return hook


def teacher_forced_logits(model, tok, prompt, response, device):
    import torch
    p_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=3500).input_ids
    r_ids = tok(" " + response, add_special_tokens=False, truncation=True, max_length=1200).input_ids
    if len(r_ids) < 2:
        return None, None
    ids = torch.tensor([p_ids + r_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    pos = [q for q in range(len(p_ids) - 1, len(p_ids) - 1 + len(r_ids)) if q < logits.shape[0]]
    return torch.log_softmax(logits[pos], dim=-1).cpu().numpy(), pos


def kl_under_hook(model, tok, layers_mod, target_layer, prompt, response, device,
                  V_np=None, delta_np=None):
    """Mean KL over response positions between clean logits and logits under the intervention."""
    import torch
    lp_clean, _ = teacher_forced_logits(model, tok, prompt, response, device)
    if lp_clean is None:
        return None
    V_t = torch.tensor(V_np, dtype=torch.float32, device=device) if V_np is not None else None
    dl = torch.tensor(delta_np, dtype=torch.float32, device=device) if delta_np is not None else None
    h = layers_mod[target_layer - 1].register_forward_hook(make_hook(V_t, dl))
    try:
        p_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=3500).input_ids
        r_ids = tok(" " + response, add_special_tokens=False, truncation=True, max_length=1200).input_ids
        ids = torch.tensor([p_ids + r_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        qpos = [q for q in range(len(p_ids) - 1, len(p_ids) - 1 + len(r_ids)) if q < logits.shape[0]]
        lp_alt = torch.log_softmax(logits[qpos], dim=-1).cpu().numpy()
    finally:
        h.remove()
    n = min(len(lp_clean), len(lp_alt))
    return float(np.mean(kl_rows(lp_clean[:n], lp_alt[:n])))


def resp_logprob(model, tok, layers_mod, target_layer, prompt, response, device, delta_np=None):
    """Mean log-probability the model assigns to `response` after `prompt`, optionally with the
    residual stream displaced by delta_np at every position of layer `target_layer`.

    This is the quantity the identity test needs: not how far the output moved, but whether it moved
    toward a particular drug's actual response."""
    import torch
    p_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=3500).input_ids
    r_ids = tok(" " + response, add_special_tokens=False, truncation=True, max_length=1200).input_ids
    if len(r_ids) < 2:
        return None
    ids = torch.tensor([p_ids + r_ids], device=device)
    h = None
    if delta_np is not None:
        dl = torch.tensor(delta_np, dtype=torch.float32, device=device)
        h = layers_mod[target_layer - 1].register_forward_hook(make_hook(None, dl))
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
    finally:
        if h is not None:
            h.remove()
    lp = torch.log_softmax(logits, dim=-1)
    tot, cnt = 0.0, 0
    for j, t in enumerate(r_ids):
        q = len(p_ids) - 1 + j
        if q < lp.shape[0]:
            tot += float(lp[q, t]); cnt += 1
    return tot / cnt if cnt else None


# ----------------------------------------------------------------- THE IDENTITY TEST
def run_swap(score_fn, eval_rows, drug_mean, V_pure, rng, resp_for, n_boot=4000, equiv_frac=0.1):
    """Directional drug-identity test. Shared verbatim by the selftest and by main(), so the selftest
    exercises THE ACTUAL HEADLINE CODE PATH — in v2 the entire swap block lived inside main() and
    `SELFTEST PASSED` said nothing whatever about it.

    WHY NOT KL-AGAINST-CLEAN. The obvious test — inject the A->B displacement, measure how far the
    output distribution moved, compare against a random displacement of the same norm — cannot work,
    and the selftest proves it: in a planted world where the readout DOES read the drug subspace, a
    random direction inside that subspace moves the output just as far as the correct one, because
    the norms are matched. KL-against-clean measures DISTANCE MOVED, and distance moved is exactly
    what norm-matching equalises. The test had no power in the one world it needed power in.

    WHY NOT delta-logP(r_B) EITHER. The next design scored logP(r_B) under the injection against a
    norm-matched RANDOM direction. Also confounded, in the opposite direction. logP(t) = z_t -
    logsumexp(z), and a random direction inside the slab is OFF the class-mean manifold: it inflates
    the normalizer (measured +2.75 nats) while the on-manifold displacement P(mu_B - mu_A) does not
    (-0.22). Every target is then depressed under the control arm whether or not it carries B's
    identity. Measured directly: forcing the scored target to carry ZERO identity information still
    left 36% of the effect standing. Norm-matched is not disruption-matched.

    WHAT THIS MEASURES. Two changes make the arms structurally identical so that only IDENTITY
    differs between them.

    (1) A CONTRAST score, so the normalizer cancels algebraically rather than approximately:

            score = logP(r_B | prompt_A) - logP(r_A | prompt_A)

        Any shift common to both responses -- including the whole logsumexp inflation -- subtracts
        out. What survives is the model's PREFERENCE between drug B's response and drug A's own.

    (2) Every arm anchored at drug A, so every arm is an on-manifold class-mean displacement with the
        same cross term against h:

            swap      delta = P(mu_B - mu_A)      points at the drug whose response we score
            permuted  delta = P(mu_C - mu_A)      same construction, same anchor, WRONG identity
            isotropic delta = random in slab      kept only as an off-manifold DAMAGE diagnostic,
                                                  never as the headline

    The headline is swap vs permuted. Both are real drug displacements from A of identical norm; they
    differ in nothing but which drug they point at. If the readout reads identity, pointing at B must
    raise B's response relative to A's more than pointing at C does.

    r_B comes from a THIRD disjoint split, never from the rows that define mu_B -- otherwise the
    injected vector and the scored response share cells and the test is partly circular.
    """
    if V_pure.shape[1] == 0:
        return None
    P = lambda v: V_pure @ (V_pure.T @ v)
    drugs = sorted(drug_mean)
    sw, iso, perm, base_gap, clusters, norms = [], [], [], [], [], []
    for r in eval_rows:
        a = r["drug"]
        others = [d for d in drugs if d != a]
        if a not in drug_mean or len(others) < 2:
            continue
        b = others[rng.randint(len(others))]
        pool = [x for x in others if x != b]          # the permuted control must not BE drug B
        if not pool:
            continue
        c = pool[rng.randint(len(pool))]
        r_b = resp_for(b, r)
        if r_b is None or not r.get("response"):
            continue
        d_swap = P(drug_mean[b] - drug_mean[a])       # global mean cancels exactly
        nrm = np.linalg.norm(d_swap)
        if nrm < 1e-9:
            continue
        d_perm = P(drug_mean[c] - drug_mean[a])       # SAME anchor, same construction, wrong drug
        pn = np.linalg.norm(d_perm)
        if pn < 1e-9:
            continue
        d_perm = d_perm * (nrm / pn)
        u = V_pure @ rng.randn(V_pure.shape[1])
        d_iso = u * (nrm / (np.linalg.norm(u) + 1e-12))

        def contrast(delta):
            hb = score_fn(r, delta, r_b)
            ha = score_fn(r, delta, r["response"])
            return None if (hb is None or ha is None) else hb - ha

        base = contrast(None)
        if base is None:
            continue
        vals = [contrast(dv.astype(np.float32)) for dv in (d_swap, d_perm, d_iso)]
        if any(v is None for v in vals):
            continue
        sw.append(vals[0] - base); perm.append(vals[1] - base); iso.append(vals[2] - base)
        base_gap.append(base)
        # cluster on the ORDERED PAIR: rows sharing (A,B) share both the injected vector and the
        # scored response, which is the dependence the bootstrap has to respect
        clusters.append((a, b))
        # norms of what is actually DELIVERED, after the float32 cast, for all three arms -- unlike
        # v3.0's ratio, which re-measured the vector three lines after constructing it and was
        # therefore the same species of tautology as v2's frac_in
        norms.append([float(np.linalg.norm(dv.astype(np.float32))) for dv in (d_swap, d_perm, d_iso)])
    if not sw:
        return None
    sw, perm, iso = np.array(sw), np.array(perm), np.array(iso)
    nz = np.array(norms)
    # scale: how big is the model's OWN preference gap between A's and B's responses, with no
    # intervention at all? Identity-bearing, measured clean, and interpretable -- the margin then
    # reads "the injection closes less than equiv_frac of the A-vs-B gap".
    scale = float(np.mean(np.abs(base_gap)))
    margin = equiv_frac * scale if scale > 0 else None
    out = {"n": len(sw), "n_pair_clusters": len(set(clusters)),
           "n_drug_clusters": len(set(a for a, _ in clusters)),
           "effect_swap": float(sw.mean()), "effect_perm": float(perm.mean()),
           "effect_iso": float(iso.mean()),
           "clean_ab_gap": scale, "equiv_margin": margin,
           "delivered_norm_ratio_perm": float(np.mean(nz[:, 1] / nz[:, 0])),
           "delivered_norm_ratio_iso": float(np.mean(nz[:, 2] / nz[:, 0]))}
    for name, ctrl in (("perm", perm), ("iso", iso)):
        m, lo, hi, p, nc = cluster_bootstrap(sw - ctrl, clusters, n_boot, rng)
        out[f"diff_{name}"] = m
        out[f"ci_{name}"] = [lo, hi]
        out[f"p_{name}"] = p
        out[f"verdict_{name}"] = _verdict(lo, hi, margin)
    return out


def _verdict(lo, hi, margin):
    """TOST-style. v2 read any interval containing zero as an affirmative 'the readout is NOT
    drug-specific', with no equivalence margin and no power statement — a non-significant result
    converted into a positive claim. Equivalence must be EARNED: the whole interval has to sit
    inside the margin, and if it does not, the honest answer is that we are underpowered."""
    if np.isnan(lo) or np.isnan(hi):
        return "insufficient clusters"
    if margin is None or not np.isfinite(margin) or margin <= 0:
        return ("drug displacement moves output TOWARD drug B (identity IS read)" if lo > 0 else
                "drug displacement moves output AWAY from drug B" if hi < 0 else
                "no margin available — inconclusive")
    half = (hi - lo) / 2
    if half > 0 and margin < 0.5 * half:
        # the margin is finer than the experiment can resolve; any verdict would be an artifact of
        # precision rather than a finding
        return f"NO VERDICT — margin {margin:.4f} below resolution (CI half-width {half:.4f})"
    if lo > margin:
        return "drug displacement moves output TOWARD drug B (identity IS read)"
    if hi < -margin:
        return "drug displacement moves output AWAY from drug B (anomalous)"
    if lo > -margin and hi < margin:
        # equivalence requires the interval to contain zero as well as sit inside the margin. Without
        # the zero check a significantly NONZERO effect that happens to be small gets reported as
        # 'identity is NOT read', which is a directional finding dressed up as a null.
        if lo <= 0 <= hi:
            return "EQUIVALENT to control within margin (identity is NOT read)"
        return "DIRECTIONAL EFFECT present but smaller than the margin"
    return f"INCONCLUSIVE / underpowered (CI half-width {half:.4f} vs margin {margin:.4f})"


# ----------------------------------------------------------------- data
def load_prompts(eval_dir, tier, n_drugs, n_per_drug, rng):
    path = os.path.join(eval_dir, f"eval_{tier}.jsonl")
    ex = [json.loads(l) for l in open(path)]
    by_drug = defaultdict(list)
    for e in ex:
        m = e.get("metadata", {})
        if m.get("drug"):
            by_drug[m["drug"]].append(e)
    drugs = [d for d, v in by_drug.items() if len(v) >= n_per_drug]
    rng.shuffle(drugs); drugs = drugs[:n_drugs]
    rows = []
    for d in drugs:
        cells = by_drug[d]
        for i in rng.choice(len(cells), n_per_drug, replace=False):
            e = cells[i]; m = e["metadata"]
            rows.append({"prompt": e["prompt"], "response": e["response"], "drug": d,
                         "cell_line": m.get("cell_line_id"), "plate": m.get("plate"),
                         "dose": m.get("dose_float")})
    logger.info(f"  {len(drugs)} drugs x {n_per_drug} = {len(rows)} prompts")
    return rows


# ----------------------------------------------------------------- selftest
def _logsm(Z):
    Z = Z - Z.max(-1, keepdims=True)
    return Z - np.log(np.sum(np.exp(Z), -1, keepdims=True))


def _synthetic_world(reads_drug, rng, norm_sensitive=False, H=1024, C=12, n=60, kdim=5):
    """A linear readout that reads the cell-line subspace and, if reads_drug, the drug subspace too.

    H is deliberately close to the model's 2048. A k-dim random subspace captures ~k/H of the
    variance, so every ratio-against-random threshold in this file is dimension-dependent: at H=64
    (the v2 selftest) a matched-dim random ablation has a large effect and the inertness threshold
    had to be loose enough to tolerate it, which is precisely why v2's `kl_pure < 3 * kl_randp`
    passed while the purified slab was leaking.

    norm_sensitive puts an RMS normalisation in front of the readout. That world is the one that
    catches MANIFOLD asymmetry between arms: with a normaliser in the path an injection can move the
    output while carrying no identity at all, purely by changing ||h||. The delta-logP design this
    replaced had exactly that defect -- an off-manifold control inflated the logsumexp and depressed
    every target -- and no planted world that could see it.
    """
    Q, _ = np.linalg.qr(rng.randn(H, 2 * kdim))
    drug_sub, cl_sub = Q[:, :kdim], Q[:, kdim:2 * kdim]
    drug_code = {d: (rng.randn(kdim) * 3) @ drug_sub.T for d in range(C)}
    cl_code = {c: (rng.randn(kdim) * 3) @ cl_sub.T for c in range(C)}
    W = rng.randn(200, kdim) @ cl_sub.T
    if reads_drug:
        W = W + rng.randn(200, kdim) @ drug_sub.T
    X, dl, cll = [], [], []
    # a LARGE shared offset, so the synthetic activations have the uncentered structure that broke v2
    offset = rng.randn(H) * 4.0
    for _ in range(n * C):
        d, c = rng.randint(C), rng.randint(C)
        X.append(offset + rng.randn(H) * 0.4 + drug_code[d] + cl_code[c]); dl.append(d); cll.append(c)

    def readout(h):
        if norm_sensitive:
            h = h / (np.linalg.norm(h) + 1e-9) * np.sqrt(h.shape[-1])
        return h @ W.T

    return np.array(X), dl, cll, readout, drug_sub, kdim


def selftest(args):
    """Three planted worlds, and the estimator must separate them. v2's selftest never touched the
    swap code and had no world in which the drug was causally live, so it could only ever confirm
    the conclusion we were hoping for."""
    ok = True
    res = {}
    WORLDS = [("drug-is-inert", False, False),
              ("drug-IS-read", True, False),
              ("inert+norm-sensitive", False, True)]
    for tag, reads_drug, norm_sensitive in WORLDS:
        rng = np.random.RandomState(0)
        X, dl, cll, readout, drug_sub, kdim = _synthetic_world(reads_drug, rng, norm_sensitive)
        C = len(set(dl))
        chance = 1.0 / C
        idx = np.arange(len(X)); rng.shuffle(idx)
        half = len(idx) // 2
        # a small response slice, matching main(): responses are cheap, class means are not
        n_resp = max(C * 5, 30)
        mean_i, resp_i, ev = idx[:half - n_resp], idx[half - n_resp:half], idx[half:]

        V_drug = build_subspace(X, dl, args.n_dims, idx=mean_i)
        V_conf, _ = build_union_subspace(X, [cll], args.n_dims * args.ctx_mult, idx=mean_i)
        V_cl = build_subspace(X, cll, args.n_dims, idx=mean_i)
        V_pure, _ = orthogonalize(V_drug, V_conf)
        d_pure = V_pure.shape[1]
        recov = subspace_overlap(V_pure, drug_sub) if d_pure else 0.0

        def readout_kl(Vsub):
            if Vsub is None or Vsub.shape[1] == 0:
                return 0.0
            lp0 = _logsm(np.stack([readout(h) for h in X[ev]]))
            lp1 = _logsm(np.stack([readout(h) for h in project_out(X[ev], Vsub)]))
            return float(np.mean(kl_rows(lp0, lp1)))

        a_drug, _ = probe_acc(X, dl, np.random.RandomState(1))
        a_abl, _ = probe_acc(project_out(X, V_drug), dl, np.random.RandomState(1))
        frac = (a_drug - a_abl) / (a_drug - chance)
        kl_pure, kl_cl = readout_kl(V_pure), readout_kl(V_cl)
        kl_rand = float(np.mean([readout_kl(random_subspace(X.shape[1], max(d_pure, 1), rng))
                                 for _ in range(5)]))

        # --- the swap path, through THE SAME run_swap() that main() uses ---
        drug_mean = {d: X[[i for i in mean_i if dl[i] == d]].mean(0) for d in set(dl)}
        # each drug's "response" is the token it most favours; r_B comes from the DISJOINT response
        # slice, never from the rows that define mu_B
        tok_for = {d: int(np.argmax(readout(X[[i for i in resp_i if dl[i] == d]].mean(0))))
                   for d in set(dl)}
        rows_ev = [{"drug": dl[i], "_i": i, "response": tok_for[dl[i]]} for i in ev]

        def score_fn(r, delta, resp):
            h = X[r["_i"]] if delta is None else X[r["_i"]] + delta
            return float(_logsm(readout(h))[resp])

        sw = run_swap(score_fn, rows_ev, drug_mean, V_pure, np.random.RandomState(7),
                      resp_for=lambda d, r: tok_for[d], n_boot=2000, equiv_frac=args.equiv_frac)

        logger.info(f"  [{tag}] drug decodable {a_drug:.2f} -> ablated {a_abl:.2f} "
                    f"({100*frac:.0f}% removed) | pure slab = {d_pure} dims, overlap with planted "
                    f"drug subspace {recov:.3f}")
        logger.info(f"  [{tag}] ablation KL: pure={kl_pure:.4f} rand@matched={kl_rand:.4f} "
                    f"cell_line={kl_cl:.4f}")
        if sw:
            logger.info(f"  [{tag}] contrast effects: A->B {sw['effect_swap']:+.4f} | A->C "
                        f"{sw['effect_perm']:+.4f} | isotropic {sw['effect_iso']:+.4f}")
            logger.info(f"  [{tag}] HEADLINE vs A->C {sw['diff_perm']:+.4f} "
                        f"[{sw['ci_perm'][0]:+.4f},{sw['ci_perm'][1]:+.4f}] -> {sw['verdict_perm']}")
            logger.info(f"  [{tag}] norm ratios perm {sw['delivered_norm_ratio_perm']:.6f} "
                        f"iso {sw['delivered_norm_ratio_iso']:.6f}")
        res[tag] = {"kl_pure": kl_pure, "kl_cl": kl_cl, "kl_rand": kl_rand, "swap": sw}

        if not (a_drug > 0.8):
            logger.error(f"  FAIL[{tag}]: drug not decodable in synthetic data"); ok = False
        if not (0.8 < frac <= 1.05):
            logger.error(f"  FAIL[{tag}]: removal gate {frac:.3f} outside (0.8, 1.05]"); ok = False
        if d_pure != kdim:
            logger.info(f"  note[{tag}]: recovered {d_pure} pure dims against {kdim} planted "
                        f"(surplus dims are estimation noise; the leak bound is what polices them)")
        if d_pure < kdim:
            logger.error(f"  FAIL[{tag}]: recovered {d_pure} < {kdim} planted dims -- the rank "
                         f"cutoff is discarding real signal"); ok = False
        # Recovery is SNR-limited, not a correctness property: a swept check shows it reaches 0.90
        # only at ~500 cells per class, against the 25-30 the real data can supply. It bounds POWER,
        # not validity -- both arms are projected into the SAME estimated slab and differ only in
        # which drug they point at, so a misaligned slab weakens the test without biasing it. The
        # floor here only catches a slab that is essentially noise.
        if recov < 0.5:
            logger.error(f"  FAIL[{tag}]: purified slab only {recov:.3f} inside the planted drug "
                         f"subspace -- that is noise, not a drug axis"); ok = False
        if subspace_overlap(V_pure, V_conf) > 0.01:
            logger.error(f"  FAIL[{tag}]: purified slab overlaps context "
                         f"{subspace_overlap(V_pure, V_conf):.4f}"); ok = False
        if sw is None:
            logger.error(f"  FAIL[{tag}]: swap did not run"); ok = False
            continue
        for k in ("delivered_norm_ratio_perm", "delivered_norm_ratio_iso"):
            if abs(sw[k] - 1.0) > 1e-5:
                logger.error(f"  FAIL[{tag}]: {k} = {sw[k]} -- arms not norm-matched"); ok = False

        if reads_drug:
            # the world v2 never had: the estimator MUST detect a readout that reads the drug, and
            # with a real effect rather than a positive sign on a tight interval
            if not (sw["ci_perm"][0] > 0):
                logger.error(f"  FAIL[{tag}]: failed to DETECT a drug-reading readout "
                             f"(CI {sw['ci_perm']}) -- no power"); ok = False
            if not sw["verdict_perm"].startswith("drug displacement moves output TOWARD"):
                logger.error(f"  FAIL[{tag}]: verdict was '{sw['verdict_perm']}' where the planted "
                             f"truth is that identity IS read"); ok = False
            if not (kl_pure > 5 * kl_rand):
                logger.error(f"  FAIL[{tag}]: ablation failed to detect a live drug slab"); ok = False
        else:
            # ABSOLUTE bound. v2 asserted kl_pure < 3*kl_randp, calibrating the inertness threshold
            # against a leakage measurement, so leakage in both arms cancelled and passed silently.
            if not (kl_pure < 0.05 * kl_cl):
                logger.error(f"  FAIL[{tag}]: pure-drug KL not ~0 in absolute terms "
                             f"({kl_pure:.5f} vs 5% of cell-line {0.05*kl_cl:.5f}) -- leaking")
                ok = False
            if not sw["verdict_perm"].startswith("EQUIVALENT"):
                logger.error(f"  FAIL[{tag}]: verdict was '{sw['verdict_perm']}' where the planted "
                             f"truth is that identity is NOT read"); ok = False

    inert, read = res.get("drug-is-inert", {}), res.get("drug-IS-read", {})
    norm = res.get("inert+norm-sensitive", {})
    if inert.get("swap") and read.get("swap"):
        if not (read["swap"]["diff_perm"] > 10 * abs(inert["swap"]["diff_perm"])):
            logger.error("  FAIL: estimator does not separate the planted worlds by a clear margin")
            ok = False
        else:
            nv = norm.get("swap", {}).get("diff_perm", float("nan"))
            logger.info(f"  SEPARATION: inert {inert['swap']['diff_perm']:+.4f} | drug-read "
                        f"{read['swap']['diff_perm']:+.4f} | inert+norm {nv:+.4f}")
    logger.info(f"  SELFTEST {'PASSED' if ok else 'FAILED'}  (three planted worlds incl. a "
                f"norm-sensitive readout; swap path exercised; arms norm-matched after cast; "
                f"subspace recovery and absolute leak bounds checked)")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir")
    ap.add_argument("--model_path")
    ap.add_argument("--tier", default="tier2_unseen_drugs")
    ap.add_argument("--layers", default="2,4,6,8,9,12,16")
    # 24, not v2's 12: the bootstrap resamples DRUGS, so the cluster count -- not the prompt count --
    # sets the interval width, and the selftest lands on "underpowered" rather than "equivalent" at
    # 12. load_prompts degrades gracefully if fewer drugs have enough cells.
    ap.add_argument("--n_drugs", type=int, default=24)
    ap.add_argument("--n_per_drug", type=int, default=60)
    ap.add_argument("--n_dims", type=int, default=10, help="max subspace dimension")
    ap.add_argument("--ctx_mult", type=int, default=3, help="context subspace dim budget = n_dims*this")
    ap.add_argument("--n_kl_prompts", type=int, default=60, help="prompts for the KL causal test")
    ap.add_argument("--n_rand_draws", type=int, default=3, help="random null subspaces to average")
    ap.add_argument("--headroom", type=float, default=5.0,
                    help="require cell-line KL >= this x random KL before quoting a layer")
    ap.add_argument("--equiv_frac", type=float, default=0.1,
                    help="equivalence margin as a fraction of the model's own clean A-vs-B "
                         "response-preference gap")
    ap.add_argument("--do_swap", action="store_true")
    ap.add_argument("--out", default="RESULTS/workspace_probe.json")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return
    if not (args.eval_dir and args.model_path):
        ap.error("--eval_dir and --model_path required (unless --selftest)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.RandomState(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device).eval()
    layers_mod = get_layers(model)
    logger.info(f"Model {args.model_path}: {len(layers_mod)} layers; testing hidden_states {layers}")

    rows = load_prompts(args.eval_dir, args.tier, args.n_drugs, args.n_per_drug, rng)
    drug_lab = [r["drug"] for r in rows]
    cl_lab = [r["cell_line"] for r in rows]
    plate_lab = [str(r["plate"]) for r in rows]
    dose_lab = bin_dose([r["dose"] for r in rows])
    n_drug_classes = len(set(drug_lab))
    chance = 1.0 / n_drug_classes

    # The class-mean subspace of C classes can span C-1 directions. Capping it below that ablates
    # only part of the drug axis and the rest survives -- which is what --n_dims 10 did at 24 drugs.
    need = n_drug_classes - 1
    if args.n_dims < need:
        logger.info(f"  raising --n_dims {args.n_dims} -> {need}: with {n_drug_classes} drugs the "
                    f"class-mean subspace spans up to {need} dims, and a lower cap ablates only "
                    f"part of it")
        args.n_dims = need

    logger.info("Extracting activations (last prompt position) ...")
    acts = extract_activations(model, tok, [r["prompt"] for r in rows], device, layers, args.bf16)

    # ---- est / eval split. EVERYTHING is fit on est and measured on eval: subspaces, class means,
    # gates, KL, swap. v2 held out only the class means.
    RESP_PER_DRUG = 5
    by_drug_idx = defaultdict(list)
    for i, r in enumerate(rows):
        by_drug_idx[r["drug"]].append(i)
    est_mask = np.zeros(len(rows), dtype=bool)
    for d, idxs in by_drug_idx.items():
        order = list(idxs); rng.shuffle(order)
        for i in order[:len(order) // 2]:
            est_mask[i] = True
    est_idx = np.where(est_mask)[0]
    ev_idx = np.where(~est_mask)[0]
    # THIRD slice, carved out of est, that supplies r_B and NOTHING else. If r_B came from the rows
    # that define mu_B, the injected vector and the scored response would share cells and a positive
    # result would be partly circular.
    resp_only = np.zeros(len(rows), dtype=bool)
    for d, idxs in by_drug_idx.items():
        sel = [i for i in idxs if est_mask[i]]
        rng.shuffle(sel)
        # a HANDFUL is enough -- run_swap draws one response per row -- and every cell spent here is
        # a cell taken away from the class-mean estimate that defines the injected displacement
        for i in sel[:min(RESP_PER_DRUG, max(1, len(sel) // 4))]:
            resp_only[i] = True
    mean_idx = np.array([i for i in est_idx if not resp_only[i]])   # defines mu and the subspaces
    resp_idx = np.array([i for i in est_idx if resp_only[i]])       # supplies r_B only
    ev_rows = [rows[i] for i in ev_idx]
    # stratified across drugs, not the head of a drug-major list
    kl_prompts = stratified_take(ev_rows, args.n_kl_prompts, lambda r: r["drug"], rng)
    swap_prompts = stratified_take(ev_rows, args.n_kl_prompts, lambda r: r["drug"], rng)
    logger.info(f"  split: {len(mean_idx)} mean-est / {len(resp_idx)} response-pool / "
                f"{len(ev_idx)} eval | KL prompts span "
                f"{len(set(r['drug'] for r in kl_prompts))} drugs")

    result = {"layers": {}, "config": {k: v for k, v in vars(args).items()},
              "n_drug_classes": n_drug_classes, "chance": chance,
              "layer_set_preregistered": args.layers,
              "design": "v3 displacement swap; out-of-sample subspaces; cluster bootstrap over drugs"}

    for L in layers:
        X = acts[L]
        H = X.shape[1]
        Xe, Xv = X[est_idx], X[ev_idx]
        ev_drug = [drug_lab[i] for i in ev_idx]
        ev_cl = [cl_lab[i] for i in ev_idx]
        ev_plate = [plate_lab[i] for i in ev_idx]
        ev_dose = [dose_lab[i] for i in ev_idx]
        logger.info("")
        logger.info(f"===== hidden_states[{L}] =====")

        # (1) subspaces — fit on est ONLY
        V_drug = build_subspace(X, drug_lab, args.n_dims, idx=mean_idx)
        V_cl = build_subspace(X, cl_lab, args.n_dims, idx=mean_idx)
        V_conf, n_dropped = build_union_subspace(X, [cl_lab, plate_lab, dose_lab],
                                                 args.n_dims * args.ctx_mult, idx=mean_idx)
        V_pure, resid = orthogonalize(V_drug, V_conf)
        V_shared = shared_component(V_drug, V_conf)
        d_pure = V_pure.shape[1]
        # There is no ground truth here, so report how well-DETERMINED the slab is instead: the
        # residual fractions from orthogonalisation. A slab whose columns barely survive projection
        # is estimation noise wearing a drug label, and on synthetic data with this many cells per
        # class the estimated axis overlaps the true one by only ~0.65 -- which bounds power, and is
        # why a null here must be read as "not resolved", never as "not there".
        det = float(np.mean(resid[resid > 0.2])) if np.any(resid > 0.2) else 0.0

        # (1b) LEAKAGE on the subspace the causal claims actually use. v2 ran every confound check on
        # the RAW slab, which is the one no causal statement is made about.
        Xp_ev = Xv @ V_pure if d_pure else None
        leak, leak_chance = {}, {}
        if d_pure:
            for nm, lab in (("cell_line", ev_cl), ("plate", ev_plate), ("dose", ev_dose)):
                a, _ = probe_acc(Xp_ev, lab, np.random.RandomState(1))
                leak[nm] = a
                # each label has its OWN chance level, and for an imbalanced label the majority-class
                # rate is the floor, not 1/n_classes. Judging dose against the DRUG chance (1/24)
                # made a 90% majority-class baseline look like catastrophic leakage.
                cnt = np.bincount(np.array([sorted(set(lab)).index(x) for x in lab]))
                leak_chance[nm] = float(cnt.max() / cnt.sum())
        cross = {}
        for nm, lab in (("cell_line", ev_cl), ("plate", ev_plate), ("dose", ev_dose)):
            a, _ = probe_acc(Xv @ V_drug, lab, np.random.RandomState(1))
            cross[nm] = a
        overlaps = {"drug_vs_cell_line": subspace_overlap(V_drug, V_cl),
                    "drug_vs_context": subspace_overlap(V_drug, V_conf),
                    "pure_vs_context": subspace_overlap(V_pure, V_conf)}

        # (2) gates, TWO-SIDED (over-removal is a failure mode too: it means we destroyed the
        # representation rather than cleaning it, and v2's one-sided test passed that silently)
        a_drug, _ = probe_acc(Xv, ev_drug, np.random.RandomState(1))
        a_abl, _ = probe_acc(project_out(Xv, V_drug), ev_drug, np.random.RandomState(1))
        Xc = project_out(Xv, V_conf)
        a_pureonly, _ = probe_acc(Xc, ev_drug, np.random.RandomState(1))
        a_pure_abl, _ = probe_acc(project_out(Xc, V_pure), ev_drug, np.random.RandomState(1)) \
            if d_pure else (None, None)
        # the gate in the state the INTERVENTION instantiates: the hook ablates V_pure from the FULL
        # residual stream, not from context-removed activations
        a_full_pure_abl, _ = probe_acc(project_out(Xv, V_pure), ev_drug, np.random.RandomState(1)) \
            if d_pure else (None, None)

        def _frac(a0, a1):
            return ((a0 - a1) / (a0 - chance)) if (a0 and a1 is not None and a0 > chance) else None
        frac_removed = _frac(a_drug, a_abl)
        frac_removed_pure = _frac(a_pureonly, a_pure_abl)
        two_sided = lambda f: (f is not None and 0.8 < f <= 1.05)
        gate_ok = two_sided(frac_removed) and two_sided(frac_removed_pure)
        # preconditions for the IDENTITY test, computed here so the results entry can record them
        leak_ok = all(v is None or v < leak_chance.get(k, 1.0) + 0.15
                      for k, v in leak.items()) if leak else False
        slab_informative = (a_pureonly is not None and a_pureonly > chance + 0.10)

        logger.info(f"  GATE (pure slab, the one the causal claims use): {_f(a_pureonly)} -> "
                    f"{_f(a_pure_abl)} | removed {100*(frac_removed_pure or 0):.0f}% "
                    f"[{'PASS' if two_sided(frac_removed_pure) else 'FAIL'}]")
        logger.info(f"  GATE (raw): drug decodable {_f(a_drug)} (chance {chance:.3f}) -> "
                    f"{_f(a_abl)} | removed {100*(frac_removed or 0):.0f}% "
                    f"[{'PASS' if two_sided(frac_removed) else 'FAIL'}]")
        logger.info(f"  GATE (intervention state): ablating pure slab from the FULL stream leaves "
                    f"drug at {_f(a_full_pure_abl)}")
        logger.info("  LEAKAGE in the purified slab (accuracy vs majority-class floor): "
                    + "  ".join(f"{k}={_f(leak.get(k))}/{leak_chance.get(k, float('nan')):.3f}"
                                for k in ("cell_line", "plate", "dose"))
                    + f" | overlap(pure, context)={overlaps['pure_vs_context']:.4f}")
        logger.info(f"  CONFOUND (raw slab): cell_line={_f(cross['cell_line'])} "
                    f"plate={_f(cross['plate'])} dose={_f(cross['dose'])}")
        logger.info(f"    overlap(drug, context)={overlaps['drug_vs_context']:.3f} | pure slab = "
                    f"{d_pure}/{V_drug.shape[1]} dims (shared part = {V_shared.shape[1]} dims, "
                    f"mean surviving fraction {det:.3f}) | "
                    f"{n_dropped} context classes dropped for size")

        def mean_kl(Vsub):
            if Vsub is None or Vsub.shape[1] == 0:
                return (None, None, 0)
            vals = [kl_under_hook(model, tok, layers_mod, L, r["prompt"], r["response"], device,
                                  V_np=Vsub) for r in kl_prompts]
            vals = [v for v in vals if v is not None]
            return ((float(np.mean(vals)), float(np.std(vals, ddof=1) / max(1, len(vals)) ** 0.5),
                     len(vals)) if vals else (None, None, 0))

        # (3) causal KL. The random null is averaged over several draws, so the ratio is not hostage
        # to one lucky subspace.
        kl_pure = mean_kl(V_pure) if d_pure else (None, None, 0)
        rand_draws = [mean_kl(random_subspace(H, max(d_pure, 1), rng))[0]
                      for _ in range(args.n_rand_draws)] if d_pure else []
        rand_draws = [v for v in rand_draws if v is not None]
        kl_rand_pure = (float(np.mean(rand_draws)) if rand_draws else None,
                        float(np.std(rand_draws, ddof=1)) if len(rand_draws) > 1 else None,
                        len(rand_draws))
        kl_shared = mean_kl(V_shared)
        kl_cl = mean_kl(V_cl)
        kl_conf = mean_kl(V_conf)
        kl_drug = mean_kl(V_drug)

        logger.info(f"  CAUSAL KL (mean+-sem over {len(kl_prompts)} stratified prompts):")
        logger.info(f"     PURE drug (perp context) {_kf(kl_pure)}")
        logger.info(f"     random @ matched dim     {_kf(kl_rand_pure)}  <- mean of "
                    f"{args.n_rand_draws} draws")
        logger.info(f"     DELETED drug component   {_kf(kl_shared)}  <- what purification removed")
        logger.info(f"     cell_line                {_kf(kl_cl)}  <- POSITIVE CONTROL")
        logger.info(f"     context (all confounds)  {_kf(kl_conf)}")
        logger.info(f"     raw drug                 {_kf(kl_drug)}  (confounded - do not quote)")

        # (3b) HEADROOM. v2 printed the positive control and gated nothing on it, so layer 12 —
        # where the instrument's cell-line/random ratio collapsed to 1.02x — carried a "<<< HEADLINE"
        # label with no measurable dynamic range behind it.
        headroom = ((kl_cl[0] / kl_rand_pure[0]) if (kl_cl[0] and kl_rand_pure[0]) else None)
        live = headroom is not None and headroom >= args.headroom
        logger.info(f"  HEADROOM: cell-line / random = "
                    f"{'NA' if headroom is None else f'{headroom:.2f}x'} "
                    f"(need >={args.headroom:.0f}x) [{'INSTRUMENT LIVE' if live else 'SATURATED - LAYER EXCLUDED'}]")

        vs = {"drug": variance_share(Xv, V_drug), "drug_pure": variance_share(Xv, V_pure),
              "shared": variance_share(Xv, V_shared), "cell_line": variance_share(Xv, V_cl),
              "context": variance_share(Xv, V_conf)}
        logger.info(f"  VARIANCE SHARE: pure-drug={vs['drug_pure']:.3f} shared={vs['shared']:.3f} "
                    f"cell_line={vs['cell_line']:.3f} context={vs['context']:.3f}")

        entry = {"gate_pass": bool(gate_ok), "instrument_live": bool(live),
                 "slab_informative": bool(slab_informative),
                 "ablation_gate_blocks_ablation_claims_only": True, "headroom": headroom,
                 "frac_signal_removed": frac_removed, "frac_signal_removed_pure": frac_removed_pure,
                 "drug_decode": a_drug, "drug_decode_ablated": a_abl,
                 "drug_decode_context_removed": a_pureonly, "drug_decode_pure_ablated": a_pure_abl,
                 "drug_decode_full_pure_ablated": a_full_pure_abl, "chance": chance,
                 "leakage_pure_slab": leak,
                 "leakage_majority_floor": leak_chance, "confound_crossdecode_raw": cross, "overlaps": overlaps,
                 "pure_drug_dims": int(d_pure), "raw_drug_dims": int(V_drug.shape[1]),
                 "shared_dims": int(V_shared.shape[1]),
                 "orthogonalize_residuals": [float(x) for x in resid],
                 "slab_determinacy": det,
                 "context_classes_dropped": int(n_dropped),
                 "kl": {"pure_drug": kl_pure, "random_matched": kl_rand_pure,
                        "deleted_shared": kl_shared, "cell_line": kl_cl, "context": kl_conf,
                        "raw_drug": kl_drug},
                 "variance_share": vs}

        # (4) THE IDENTITY TEST — only where the instrument is live and the gates passed.
        # THE IDENTITY TEST IS NOT GATED ON THE ABLATION GATE, and that is deliberate.
        #
        # The removal gate exists to stop an unfalsifiable ABLATION null: if projecting the slab out
        # did not remove the drug, a null KL says nothing. That logic does not transfer to an
        # INJECTION. We add a displacement; the hook demonstrably moves the output; the question is
        # only whether the movement is identity-specific, and swap-vs-permuted answers that with an
        # internal control. Whether ablation also erases decodability is a separate property.
        #
        # It also cannot be required. A swept synthetic check shows that even when the drug is
        # planted as EXACTLY a 23-dim subspace and all 23 dims are ablated, out-of-sample removal
        # reaches only 22%: a subspace fit on held-out rows is slightly misaligned, and with classes
        # separated at 3 sigma the surviving fraction stays linearly decodable. The 0.8 threshold is
        # unreachable out of sample. v2 passed it only because it fit the subspace IN-SAMPLE on the
        # very rows it then probed, which makes removal near-perfect by construction.
        #
        # What the injection DOES require is that the slab carry drug identity at all -- otherwise we
        # are injecting noise. That is the precondition gated here.
        if args.do_swap and d_pure and live and leak_ok and slab_informative:
            drug_mean = {}
            for d, idxs in by_drug_idx.items():
                sel = [i for i in idxs if i in set(mean_idx.tolist())]
                if sel:
                    drug_mean[d] = X[sel].mean(0)
            # a held-out response really emitted under drug b, matched on cell line where possible
            resp_pool = defaultdict(list)
            for i in resp_idx:
                resp_pool[(rows[i]["drug"], rows[i]["cell_line"])].append(rows[i]["response"])
                resp_pool[(rows[i]["drug"], None)].append(rows[i]["response"])

            def resp_for(b, r):
                pool = resp_pool.get((b, r["cell_line"])) or resp_pool.get((b, None))
                return pool[rng.randint(len(pool))] if pool else None

            score_fn = lambda r, delta, resp: resp_logprob(
                model, tok, layers_mod, L, r["prompt"], resp, device, delta_np=delta)
            sw = run_swap(score_fn, swap_prompts, drug_mean, V_pure, rng, resp_for=resp_for,
                          equiv_frac=args.equiv_frac)
            if sw:
                logger.info(f"  SWAP (displacement from drug A -> preference for B's response over "
                            f"A's own, n={sw['n']} over {sw['n_pair_clusters']} (A,B) pairs / "
                            f"{sw['n_drug_clusters']} drugs):")
                logger.info(f"     delta[logP(r_B) - logP(r_A)]:  A->B {sw['effect_swap']:+.4f} | "
                            f"A->C (wrong drug) {sw['effect_perm']:+.4f} | isotropic "
                            f"{sw['effect_iso']:+.4f}")
                logger.info(f"     vs A->C {sw['diff_perm']:+.4f} "
                            f"[{sw['ci_perm'][0]:+.4f},{sw['ci_perm'][1]:+.4f}] p={sw['p_perm']:.4f} "
                            f"-> {sw['verdict_perm']}   <<< HEADLINE")
                logger.info(f"     vs isotropic {sw['diff_iso']:+.4f} "
                            f"[{sw['ci_iso'][0]:+.4f},{sw['ci_iso'][1]:+.4f}] p={sw['p_iso']:.4f} "
                            f"-> {sw['verdict_iso']}   (off-manifold: DAMAGE diagnostic, not identity)")
                logger.info(f"     delivered norm ratios perm/swap {sw['delivered_norm_ratio_perm']:.6f} "
                            f"iso/swap {sw['delivered_norm_ratio_iso']:.6f} (both must be 1.000000)")
                logger.info(f"     margin {sw['equiv_margin']:.4f} = {args.equiv_frac:g} x the clean "
                            f"A-vs-B preference gap {sw['clean_ab_gap']:.4f}")
                entry["swap"] = sw
        if args.do_swap and "swap" not in entry:
            why = ("no pure slab" if not d_pure
                   else "instrument saturated" if not live
                   else "purified slab leaks context" if not leak_ok
                   else "slab carries no drug identity" if not slab_informative
                   else "run_swap returned nothing (no usable rows)")
            logger.info(f"  SWAP SKIPPED at this layer: {why}")
            entry["swap_skipped"] = why

        result["layers"][str(L)] = entry

    # multiplicity across depths — seven layers, seven intervals, one claim
    ps = [result["layers"][str(L)].get("swap", {}).get("p_perm", float("nan")) for L in layers]
    for L, q in zip(layers, benjamini_hochberg(ps)):
        if "swap" in result["layers"][str(L)]:
            result["layers"][str(L)]["swap"]["q_perm_bh"] = None if np.isnan(q) else float(q)

    measured = [L for L in layers if "swap" in result["layers"][str(L)]]
    result["swap_measured_at_layers"] = measured
    result["swap_skipped_reasons"] = {str(L): result["layers"][str(L)].get("swap_skipped")
                                      for L in layers if "swap_skipped" in result["layers"][str(L)]}
    result["headline_status"] = "measured" if measured else "NOT_MEASURED"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=float)
    live_layers = [L for L in layers if result["layers"][str(L)].get("instrument_live")]
    if args.do_swap and not measured:
        logger.error("")
        logger.error("=" * 78)
        logger.error("  HEADLINE NOT MEASURED AT ANY LAYER — the identity test never ran.")
        logger.error("  Reasons per layer: %s" % result["swap_skipped_reasons"])
        logger.error("  This is NOT a null result. Nothing below may be read as one.")
        logger.error("=" * 78)
    logger.info("")
    logger.info("READ (narrative order):")
    logger.info("  (1) SWAP is the HEADLINE and it is now a DISPLACEMENT test: inject the A->B drug")
    logger.info("      displacement vs a random direction of IDENTICAL norm in the SAME slab. Norm")
    logger.info("      match is exact by construction, so a difference means DIRECTION matters.")
    logger.info("  (2) Read the verdict, not the sign: 'EQUIVALENT' is earned only when the whole CI")
    logger.info("      sits inside the margin; otherwise the honest answer is underpowered.")
    logger.info("  (3) ABLATION is SUPPORTING ONLY — functional variance, never drug identity.")
    logger.info("  (4) DELETED drug component: if it carries the causal effect, the claim is 'drug is")
    logger.info("      read only via context-shared directions', NOT 'drug is inert'.")
    logger.info(f"  instrument live at layers {live_layers} of {layers}; q-values are BH across depths")
    logger.info(f"-> {args.out}")


def _f(x):
    return "NA" if x is None else f"{x:.3f}"


def _kf(t):
    if t is None or t[0] is None:
        return "NA"
    return f"{t[0]:.4f} +- {'NA' if t[1] is None else f'{t[1]:.4f}'} (n={t[2]})"


if __name__ == "__main__":
    main()
