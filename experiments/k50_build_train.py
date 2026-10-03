"""K50 Phase C: leakage-free K50 training pairs with features (plan section 4, Phase C).

Queries: India, hash bucket < 50 (same sampling as the V-dense cache), NOT owned by E' and
with NO E' member anywhere in their top 50 (plan D4). All of their retrieved candidates
(ranks 0..49) become rows; y = 1 iff the candidate is the query's owner (plan D3: an owner
outside the top 50 creates no row, it is counted as a retrieval miss).
Features: experiments.k50_common.k50_features (the one path shared with validation and test).
Checkpoints: K50_ROOT/train/b<bucket>/block_<i>.parquet (atomic, skipped on resume).
Stats: K50_ROOT/reports/k50_train_stats_b<bucket>.json

Usage: python -m experiments.k50_build_train <bucket>
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiments.ar7_features import _init, s1_counts
from experiments.exp_match2 import REC
from experiments.k50_common import (K50_ROOT, atomic_json, atomic_parquet, band_counts, bucket, fingerprint,
                                    k50_features, labels, mem_ok)
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records
from src.prep import s1_fold

CAND = K50_ROOT / "cand_train"
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"
BLOCK = 1_000_000

if __name__ == "__main__":
    b = int(sys.argv[1])
    t0 = time.time()
    out = K50_ROOT / "train" / f"b{b}"
    out.mkdir(parents=True, exist_ok=True)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    s1_ids = pd.read_parquet(CAND / "index_India_s1.parquet")["entity_id"].to_numpy()
    in_e = np.array([bucket(s) < b and s1_fold(s) == 1 for s in s1_ids])       # E' membership by s1_pos
    e_set = set(s1_ids[in_e])
    # ---- select clean queries chunk by chunk (a query's 50 rows live in one chunk)
    sel = []
    for p in sorted(p for p in CAND.glob("India_*.parquet")):
        c = load_compact_chunk(CAND, p.stem)
        c = c[c["q_id"].map(bucket).to_numpy() < 50]
        touch = set(c.loc[in_e[c["s1_pos"].to_numpy()], "q_id"])
        own_e = c["q_id"].map(owner).isin(e_set).to_numpy()
        keep = ~c["q_id"].isin(touch).to_numpy() & ~own_e
        sel.append(c.loc[keep, ["q_id", "s1_id", "score", "rank"]])
    pairs = pd.concat(sel, ignore_index=True).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
    del sel
    y = labels(pairs, owner)
    # T3 / T6 asserts
    assert not pairs["s1_id"].isin(e_set).any(), "leakage: an E' entity is a training candidate"
    assert not pairs["q_id"].map(owner).isin(e_set).any(), "leakage: a training query is owned by E'"
    owned_q = set(pairs.loc[pairs["q_id"].map(owner).notna(), "q_id"])
    found_q = set(pairs.loc[y == 1, "q_id"])
    assert int(y.sum()) == len(found_q), "a query has two positive rows"
    stats = {"E_bucket": b, "queries": int(pairs.q_id.nunique()), "rows": len(pairs), "positives": int(y.sum()),
             "negatives": int((y == 0).sum()), "positive_rate": float(y.mean()),
             "owned_queries": len(owned_q), "retrieval_misses_owner_beyond_50": len(owned_q - found_q),
             "distractor_queries": int(pairs.q_id.nunique() - len(owned_q)),
             "by_band": band_counts(pairs["rank"].to_numpy(), y),
             "by_source": {s: int((pairs.q_id.str[:2] == s).sum()) for s in ("S2", "S3")},
             "country": "India", "fingerprint": fingerprint(pairs.q_id.to_numpy(), pairs.s1_id.to_numpy())}
    print(stats, flush=True)
    atomic_json(stats, K50_ROOT / "reports" / f"k50_train_stats_b{b}.json")
    # ---- features in blocks of whole queries
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    s1c = s1_counts(s1[["country", "name_core", "addr_norm"]])
    freq = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"]).value_counts().to_dict()
    qv = pairs["q_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    bounds, i = [], 0
    while i < len(pairs):
        j = starts[np.searchsorted(starts, i + BLOCK)] if i + BLOCK < starts[-1] else len(pairs)
        bounds.append((i, j))
        i = j
    with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
        for n, (i, j) in enumerate(bounds):
            f = out / f"block_{n:03d}.parquet"
            if f.exists():
                continue
            t = time.time()
            rows = pairs.iloc[i:j].reset_index(drop=True)
            q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(rows.q_id), REC) for k in (2, 3)]).set_index("entity_id")
            X = k50_features(rows, rows[rows["rank"] < 10], s1, q, freq, s1c, pool)
            blk = pd.concat([rows[["q_id", "s1_id", "score", "rank"]], pd.Series(y[i:j], name="y"), X], axis=1)
            atomic_parquet(blk, f)
            print(f"  block {n + 1}/{len(bounds)}: {len(rows)} pairs {time.time() - t:.0f}s {mem_ok(tag='build_train')}", flush=True)
    atomic_json({**stats, "blocks": len(bounds), "secs": round(time.time() - t0)}, K50_ROOT / "reports" / f"k50_train_stats_b{b}.json")
    (out / "DONE").write_text("ok")
    print(f"done {time.time() - t0:.0f}s", flush=True)
