"""Extend the cached India validation set so that top-K > 3 policies are measured exactly.

The m2cache eval set holds every query owned by an evaluation entity E plus every query with
an E member in its retrieval top-3. A policy that scores ranks 3..9 can also assign OTHER
queries (distractors, records of other entities) to an E entity sitting at rank 3..9; those
queries are missing from the cache, so their false merges would go uncounted. This script
adds them: all queries not already in the cache whose top-10 contains an E entity, with all
of their retrieved pairs (per-query retrieval features need the whole list), featurised with
the exact production feature code.

Output (new directory, the original cache is never modified):
  WORK_DIR/exp3/ev_ext_pairs.parquet, WORK_DIR/exp3/Xev_ext.parquet

Usage: python -m experiments.build_eval_ext
"""
import time

import numpy as np
import pandas as pd
import psutil

from experiments.exp_match2 import REC, bucket, featurize, s1_name_freq
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records
from src.prep import s1_fold

CACHE = WORK_DIR / "m2cache_v2_blend_ad15_India_50"
OUT = WORK_DIR / "exp3"

if __name__ == "__main__":
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    gt = read_ground_truth()
    E = {s for s in gt if bucket(s) < 100 and s1_fold(s) == 1}
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
    E &= set(s1c.loc[s1c.country == "India", "entity_id"])
    have = set(pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["q_id"]).q_id)
    parts, n_touch = [], 0
    for p in sorted((WORK_DIR / "cand_train_blend_ad15").glob("India_*.parquet")):
        c = pd.read_parquet(p)
        touch = set(c.loc[c["s1_id"].isin(E).to_numpy(), "q_id"])
        n_touch += len(touch)
        new = touch - have
        parts.append(c[c["q_id"].isin(new).to_numpy()])
    ext = pd.concat(parts, ignore_index=True)
    print(f"E={len(E)} queries touching E in top-10: {n_touch}; already cached: {len(have)}; "
          f"new: {ext.q_id.nunique()} queries / {len(ext)} pairs  {time.time() - t0:.0f}s", flush=True)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(ext.q_id), REC) for k in (2, 3)]).set_index("entity_id")
    t = time.time()
    ext, X = featurize(ext, s1, q, s1_name_freq())
    cols = pd.read_parquet(CACHE / "Xtr.parquet").columns
    assert set(cols) == set(X.columns), (set(cols) ^ set(X.columns))
    X = X[cols]
    print(f"features {X.shape} in {time.time() - t:.0f}s  rss={psutil.Process().memory_info().rss / 2**30:.1f}GB", flush=True)
    ext.to_parquet(OUT / "ev_ext_pairs.tmp.parquet", index=False)
    X.to_parquet(OUT / "Xev_ext.tmp.parquet", index=False)
    (OUT / "ev_ext_pairs.tmp.parquet").replace(OUT / "ev_ext_pairs.parquet")
    (OUT / "Xev_ext.tmp.parquet").replace(OUT / "Xev_ext.parquet")
    print(f"done {time.time() - t0:.0f}s", flush=True)
