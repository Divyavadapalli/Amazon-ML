"""V-dense: the single validation protocol for every Exp-3 comparison (baseline, adaptive
top-K, AR-7, combined). Identical to the protocol that measured the submitted system at
0.9628 (public LB 0.9621), plus the eval-set extension needed for top-K > 3:

  * entities E: India fold-1 S1 in hash bucket < 100 that appear in the cached eval
    candidates (the same entity set as the 0.9628 measurement); half-A (bucket(e,2)==0)
    tunes the threshold, half-B reports.
  * queries: every query owned by E or with an E member in its retrieval top-10
    (cache + experiments/build_eval_ext.py), candidates from the full training S1 index.
  * test density: each distractor query (owns no S1) is cloned under a fresh id with
    probability 0.94 (numpy seed 0, cache order first -> the same clones as the 0.9628 run),
    bringing distractors per S1 from ~1.21 (train) to ~2.35 (test).
  * metric: src.evaluation.sweep (exact macro F0.5, asserted against src.metrics).
"""
import time
import zlib

import numpy as np
import pandas as pd

from src.config import WORK_DIR
from src.evaluation import GRID, check_against_reference, predictions, sweep
from src.io_utils import read_ground_truth
from src.prep import s1_fold

CACHE = WORK_DIR / "m2cache_v2_blend_ad15_India_50"
EXP = WORK_DIR / "exp3"


def bucket(x, mod=1000):
    return (zlib.crc32(x.encode()) >> 1) % mod


def load_india(with_ext=True):
    """-> ev pairs (q_id, s1_id, score, rank, y, own), X (49 features), truth, A, B, owner"""
    ev = pd.read_parquet(CACHE / "ev_pairs.parquet")
    X = pd.read_parquet(CACHE / "Xev.parquet")
    ev["part"] = "cache"
    if with_ext:
        e2 = pd.read_parquet(EXP / "ev_ext_pairs.parquet")
        x2 = pd.read_parquet(EXP / "Xev_ext.parquet")[X.columns]
        e2["part"] = "ext"
        ev = pd.concat([ev, e2], ignore_index=True)
        X = pd.concat([X, x2], ignore_index=True)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    cache_s1 = set(pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["s1_id"]).s1_id)
    E = sorted({s for s in gt if bucket(s) < 100 and s1_fold(s) == 1} & cache_s1)
    A = [e for e in E if bucket(e, 2) == 0]
    B = [e for e in E if bucket(e, 2) == 1]
    ev["own"] = ev.q_id.map(owner)
    ev["y"] = (ev["own"] == ev["s1_id"]).to_numpy(np.int8)
    return ev, X, {e: gt[e] for e in E}, A, B, owner


def load_us():
    """US spot-check: the 5 % full-index blocking sample (entities bucket < 50, their queries
    plus distractors sampled at the same rate). Features come from the final-model training
    cache. Caveat: records owned by OTHER US entities are not in the sample, so other-entity
    false merges (13 % of India false merges) are not counted -> optimistic in absolute terms,
    still a paired comparison between systems."""
    pairs = pd.read_parquet(WORK_DIR / "final_train_blend_ad15_100" / "pairs.parquet")
    X = pd.read_parquet(WORK_DIR / "final_train_blend_ad15_100" / "X.parquet")
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
    us = set(s1c.loc[s1c.country == "US", "entity_id"])
    m = pairs["s1_id"].isin(us).to_numpy()
    ev, X = pairs[m].reset_index(drop=True), X[m].reset_index(drop=True)
    gt = read_ground_truth()
    owner = {k: s for s, ms in gt.items() for k in ms}
    E = sorted({s for s in us if bucket(s) < 50})
    A = [e for e in E if bucket(e, 2) == 0]
    B = [e for e in E if bucket(e, 2) == 1]
    ev["part"] = "us"
    ev["own"] = ev.q_id.map(owner)
    ev["y"] = (ev["own"] == ev["s1_id"]).to_numpy(np.int8)
    return ev, X, {e: gt[e] for e in E}, A, B, owner


SLIM = ["nc_tsort", "ad_tset", "q_addr_empty"]


def slim(ev, X):
    """Pair frame for evaluation: ids, rank, owner + the 3 feature columns the breakdowns use
    (keeps memory low; model probabilities are added as columns before densify)."""
    out = ev[["q_id", "s1_id", "rank", "own", "y", "part"]].copy()
    for c in SLIM:
        out[c] = X[c].to_numpy()
    return out


def densify(ev, seed=0, rate=0.94):
    """Clone distractor queries (with all their candidate rows and every column, including
    model probabilities) to test density."""
    dq = ev.q_id[ev["own"].isna()].unique()
    clone = set(dq[np.random.default_rng(seed).random(len(dq)) < rate])
    m = ev.q_id.isin(clone).to_numpy()
    e2 = ev[m].copy()
    e2["q_id"] = "CLONE_" + e2["q_id"]
    return pd.concat([ev, e2], ignore_index=True), len(clone)


def policy_mask(ev, prob, policy):
    """k3: ranks 0-2.  ('adapt', h, kmax): ranks 0-2, plus ranks 3..kmax-1 for queries whose
    best top-3 probability is below h.  ('k', n): ranks 0..n-1."""
    r = ev["rank"].to_numpy()
    if isinstance(policy, tuple) and policy[0] == "mask":      # precomputed (staged K50 policies)
        return np.asarray(policy[1], dtype=bool)
    if policy == "k3":
        return r < 3
    if policy[0] == "k":
        return r < policy[1]
    _, h, kmax = policy
    top3 = pd.Series(np.where(r < 3, prob, -1.0)).groupby(ev["q_id"].to_numpy()).transform("max").to_numpy()
    return (r < 3) | ((r < kmax) & (top3 < h))


def per_entity_f(pred, truth, ents):
    """Per-entity F0.5 vector (same formula as src.evaluation.sweep) for paired bootstrap."""
    got = pred.groupby("s1_id")["q_id"].agg(set).to_dict()
    f = np.empty(len(ents))
    for i, e in enumerate(ents):
        g, t = got.get(e, set()), truth[e]
        if not t:
            f[i] = float(not g)
            continue
        tp = len(g & t)
        if tp == 0:
            f[i] = 0.0
            continue
        p, r = tp / len(g), tp / len(t)
        f[i] = 1.25 * p * r / (0.25 * p + r)
    return f


def evaluate(ev, prob, truth, A, B, owner, policy="k3", label=""):
    """Tune the threshold on A, report on B, with breakdowns. Returns (summary dict,
    per-entity F on B, accepted predictions on B)."""
    t0 = time.time()
    m = policy_mask(ev, prob, policy)
    sub = ev[m]
    pr = predictions(sub, prob[m], True)
    sa = sweep(pr, truth, A, GRID)
    thr = float(sa.loc[sa.f05_macro.idxmax(), "threshold"])
    sb = sweep(pr, truth, B, [thr]).iloc[0]
    check_against_reference(pr, truth, B, thr, sb.f05_macro)
    acc = pr[pr["prob"] >= thr]
    Bs = set(B)
    accB = acc[acc["s1_id"].isin(Bs)].copy()
    accB["ok"] = accB["q_id"].map(owner) == accB["s1_id"]
    fvec = per_entity_f(accB, truth, B)
    assert abs(fvec.mean() - sb.f05_macro) < 1e-9
    # attributes of accepted pairs: predictions() keeps each row's position within `sub`
    # as its index label, so map back to the row of X
    ix = np.flatnonzero(m)[accB.index.to_numpy()]
    assert (ev["q_id"].to_numpy()[ix] == accB["q_id"].to_numpy()).all()
    accB["near_copy"] = ((ev["nc_tsort"].to_numpy()[ix] >= 90) & (ev["ad_tset"].to_numpy()[ix] >= 90))
    accB["addr_empty"] = ev["q_addr_empty"].to_numpy()[ix] == 1
    accB["src"] = np.where(accB.q_id.str.replace("CLONE_", "").str.startswith("S3-"), "S3", "S2")
    fp = accB[~accB.ok]
    # true pairs of B for recall by source / empty address
    tru = pd.DataFrame([(q, e) for e in B for q in truth[e]], columns=["q_id", "s1_id"])
    tru["src"] = np.where(tru.q_id.str.startswith("S3-"), "S3", "S2")
    tp_set = set(zip(accB.q_id[accB.ok], accB.s1_id[accB.ok]))
    tru["found"] = [(q, s) in tp_set for q, s in zip(tru.q_id, tru.s1_id)]
    qempty = pd.Series(ev["q_addr_empty"].to_numpy(), index=ev["q_id"].to_numpy())
    qempty = qempty[~qempty.index.duplicated()]
    tru["addr_empty"] = tru.q_id.map(qempty).fillna(0).to_numpy() == 1
    out = {
        "label": label, "policy": "mask" if isinstance(policy, tuple) and policy[0] == "mask" else str(policy), "threshold": thr,
        "f05": float(sb.f05_macro), "precision": float(sb.pair_precision), "recall": float(sb.pair_recall),
        "tp": int(sb.tp), "fp": int(sb.fp), "fn": int(sb.fn),
        "singleton_acc": float(sb.singleton_acc), "multi_f05": float(sb.multi_f05),
        "fp_distractor": int(fp.q_id.map(owner).isna().sum()),
        "fp_near_copy": int(fp.near_copy.sum()),
        "candidate_pairs": int(m.sum()), "candidate_pairs_orig_queries": int(m[~ev.q_id.str.startswith("CLONE_").to_numpy()].sum()),
        "pairs_per_query": float(m.sum() / ev.q_id.nunique()),
        "secs": round(time.time() - t0, 1),
    }
    for s in ("S2", "S3"):
        out[f"recall_{s}"] = float(tru.found[tru.src == s].mean())
        a = accB[accB.src == s]
        out[f"precision_{s}"] = float(a.ok.mean()) if len(a) else None
    out["recall_addr_empty"] = float(tru.found[tru.addr_empty].mean()) if tru.addr_empty.any() else None
    a = accB[accB.addr_empty]
    out["precision_addr_empty"] = float(a.ok.mean()) if len(a) else None
    out["n_accepted_addr_empty"] = int(len(a))
    return out, fvec, accB


def paired_ci(f_new, f_base, reps=1000, seed=1):
    d = f_new - f_base
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
