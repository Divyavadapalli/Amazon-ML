"""K50 Phase B: retrieval ceilings, oracle F0.5 and E' sizing (plan section 4, Phase B).

Streams the compact India K50 training cache chunk by chunk (never all 210M pairs at once).
For every candidate E' (India fold-1, S1 hash bucket < b, b in BUCKETS; b=100 is the V-dense E):
  * recall@K (K = 1,3,5,10,20,30,50) of the true (query, E') pairs, rank-bin distribution,
    retrieval misses (true owner not in the query's top 50)
  * oracle macro-F0.5@K (perfect matcher on the candidates, exact challenge definition,
    cross-checked against src.metrics.macro_f05 -- test T5)
  * segment recall@10 / @50: source S2/S3, empty query address, non-Latin name, name_core equal
  * volumes: queries touching E' in top-10 / top-50, eval pairs, clean training queries left
Also the mixed India+US 5 % sample (blk2_blend_ad15_k50.parquet): recall@K and oracle by country.
Outputs: K50_ROOT/reports/k50_retrieval_stats.json, k50_oracle.json

Usage: python -m experiments.k50_ceiling
"""
import time

import numpy as np
import pandas as pd

from experiments.k50_common import K50_ROOT, RANK_BINS, atomic_json, bucket
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records
from src.metrics import macro_f05
from src.prep import s1_fold

CAND = K50_ROOT / "cand_train"
BUCKETS = [5, 10, 20, 100]
KS = [1, 3, 5, 10, 20, 30, 50]


def oracle(truth, found, ents, K):
    """found: {entity: {q: rank}}; perfect matcher predicts exactly the found true members."""
    f = np.empty(len(ents))
    pred = {}
    for i, e in enumerate(ents):
        n = len(truth[e])
        got = {q for q, r in found.get(e, {}).items() if r < K}
        pred[e] = got
        if n == 0:
            f[i] = 1.0
        elif not got:
            f[i] = 0.0
        else:
            r = len(got) / n
            f[i] = 1.25 * r / (0.25 + r)
    ref = macro_f05(truth, pred, ents)["f05_macro"]          # T5 cross-check
    assert abs(ref - f.mean()) < 1e-9, (ref, f.mean())
    return float(f.mean())


if __name__ == "__main__":
    t0 = time.time()
    assert (CAND / "DONE").exists() or (CAND / "manifest.json").exists(), "run Phase A first"
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    s1_ids = pd.read_parquet(CAND / "index_India_s1.parquet")["entity_id"].to_numpy()
    q_ids = pd.read_parquet(CAND / "index_India_q.parquet")["entity_id"].to_numpy()
    s1b = np.array([bucket(s) if s1_fold(s) == 1 else 1000 for s in s1_ids], dtype=np.int16)   # 1000 = never in E'
    s1pos = pd.Series(np.arange(len(s1_ids), dtype=np.int32), index=s1_ids)
    own_pos = pd.Series(q_ids).map(owner).map(s1pos).fillna(-1).to_numpy(np.int32)             # owner's s1_pos or -1
    own_b = np.where(own_pos >= 0, s1b[np.maximum(own_pos, 0)], 1000)
    qbkt = np.array([bucket(x) for x in q_ids], dtype=np.int16)
    n_q = len(q_ids)
    touch10 = {b: np.zeros(n_q, bool) for b in BUCKETS}
    touch50 = {b: np.zeros(n_q, bool) for b in BUCKETS}
    ncand = np.zeros(n_q, np.int8)
    true_rank = np.full(n_q, -1, np.int16)            # rank of the owner among the query's candidates, -1 = miss
    files = sorted(p for p in CAND.glob("India_*.parquet") if not p.name.startswith("index"))
    for p in files:
        c = pd.read_parquet(p)
        qp, sp, rk = c["q_pos"].to_numpy(), c["s1_pos"].to_numpy(), c["rank"].to_numpy()
        np.add.at(ncand, qp, 1)
        b_row = s1b[sp]
        for b in BUCKETS:
            hit = b_row < b
            touch50[b][qp[hit]] = True
            touch10[b][qp[hit & (rk < 10)]] = True
        t = sp == own_pos[qp]
        true_rank[qp[t]] = rk[t]
        print(f"  {p.name}: {len(c)} pairs", flush=True)
    # ---------------- per E' statistics
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id")["country"]
    stats, orac = {}, {}
    for b in BUCKETS:
        ents = sorted(s for s in gt if s1c.get(s) == "India" and bucket(s) < b and s1_fold(s) == 1)
        truth = {e: gt[e] for e in ents}
        owned = own_b < b
        tr = true_rank[owned]
        n_true = int(owned.sum())
        st = {"entities": len(ents), "singletons": int(sum(not truth[e] for e in ents)), "true_pairs": n_true,
              "retrieval_misses_k50": int((tr < 0).sum()),
              "recall_at": {k: float(((tr >= 0) & (tr < k)).sum() / max(n_true, 1)) for k in KS},
              "rank_bins": {lab: int(((tr >= lo) & (tr < hi)).sum()) for lo, hi, lab in RANK_BINS}}
        st["rank_bins"][">50 (not retrieved)"] = int((tr < 0).sum())
        # volumes
        tq50, tq10 = touch50[b] | owned, touch10[b] | owned
        st["eval_queries_top10"] = int(tq10.sum())
        st["eval_queries_top50"] = int(tq50.sum())
        st["eval_pairs_top50"] = int(ncand[tq50].sum())
        st["new_eval_pairs_vs_k10_cache"] = int(ncand[tq50].sum() - 10 * int(tq10.sum()))
        clean = (qbkt < 50) & ~tq50
        st["clean_train_queries_bucket50"] = int(clean.sum())
        st["clean_train_pairs_bucket50"] = int(ncand[clean].sum())
        st["train_queries_bucket50_before_exclusion"] = int((qbkt < 50).sum())
        stats[f"E_bucket_lt_{b}"] = st
        found = {}
        qi = np.flatnonzero(owned & (true_rank >= 0))
        for i in qi:
            found.setdefault(owner[q_ids[i]], {})[q_ids[i]] = int(true_rank[i])
        orac[f"E_bucket_lt_{b}"] = {f"oracle_f05@{k}": oracle(truth, found, ents, k) for k in (3, 10, 20, 50)}
        print(b, st["recall_at"][10], st["recall_at"][50], orac[f"E_bucket_lt_{b}"], flush=True)
    # ---------------- segments (E = bucket < 100)
    owned = own_b < 100
    idx = np.flatnonzero(owned)
    qs = q_ids[idx]
    rec = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(qs), ["entity_id", "business_name", "addr_norm", "name_core"])
                     for k in (2, 3)]).set_index("entity_id").loc[qs]
    s1n = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "name_core"]).set_index("entity_id")["name_core"]
    seg = pd.DataFrame({"rank": true_rank[idx], "source": [x[:2] for x in qs],
                        "addr_empty": (rec["addr_norm"] == "").to_numpy(),
                        "script": (~rec["business_name"].map(str.isascii)).to_numpy(),
                        "core_exact": rec["name_core"].to_numpy() == s1n.loc[[owner[x] for x in qs]].to_numpy()})
    segs = {}
    for col in ("source", "addr_empty", "script", "core_exact"):
        for v, g in seg.groupby(col):
            r = g["rank"].to_numpy()
            segs[f"{col}={v}"] = {"n": len(g), **{f"recall@{k}": float(((r >= 0) & (r < k)).mean()) for k in (10, 20, 50)}}
    stats["segments_E_bucket_lt_100"] = segs
    # ---------------- mixed India+US 5 % sample (existing K50 retrieval)
    smp = pd.read_parquet(WORK_DIR / "blk2_blend_ad15_k50.parquet", columns=["q_id", "s1_id", "rank"])
    ents = [s for s in gt if bucket(s) < 50]
    tp = smp[smp["q_id"].map(owner) == smp["s1_id"]]
    rk = dict(zip(tp.q_id, tp["rank"]))
    for ctry in ("India", "US", "ALL"):
        es = [e for e in ents if ctry == "ALL" or s1c.get(e) == ctry]
        truth = {e: gt[e] for e in es}
        pairs = [(q, e) for e in es for q in gt[e]]
        r = np.array([rk.get(q, -1) for q, _ in pairs])
        found = {}
        for (q, e), x in zip(pairs, r):
            if x >= 0:
                found.setdefault(e, {})[q] = int(x)
        stats[f"sample5pct_{ctry}"] = {"entities": len(es), "true_pairs": len(pairs),
                                       "recall_at": {k: float(((r >= 0) & (r < k)).mean()) for k in KS}}
        orac[f"sample5pct_{ctry}"] = {f"oracle_f05@{k}": oracle(truth, found, es, k) for k in (10, 50)}
    stats["secs"] = round(time.time() - t0)
    atomic_json(stats, K50_ROOT / "reports" / "k50_retrieval_stats.json")
    atomic_json(orac, K50_ROOT / "reports" / "k50_oracle.json")
    print(pd.DataFrame({k: v["recall_at"] for k, v in stats.items() if isinstance(v, dict) and "recall_at" in v}).round(4).to_string())
    print(pd.DataFrame(orac).round(4).to_string())
    print(pd.DataFrame({k: {x: v[x] for x in ("entities", "eval_queries_top50", "eval_pairs_top50", "new_eval_pairs_vs_k10_cache",
                                              "clean_train_queries_bucket50")}
                        for k, v in stats.items() if k.startswith("E_bucket")}).to_string())
