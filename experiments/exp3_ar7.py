"""Exp-3 stage 1: AR-7 features for every pair set of the comparison, one part at a time.

Parts (each written atomically to WORK_DIR/exp3/ar7_<part>.parquet, skipped if present):
  tr        training pairs of the V-dense models (m2 cache, 5 % India queries)
  ev_cache  cached India eval pairs;  ev_ext  the eval extension (top-10 touching E)
  us        US spot-check pairs (5 % full-index sample)
Rows are aligned to the row order of the corresponding pair file.

Usage: python -m experiments.exp3_ar7
"""
import pickle
import time
from multiprocessing import Pool

import pandas as pd
import psutil

from experiments.ar7_features import _init, ar7_features, build_stats, s1_counts
from src.config import WORK_DIR
from src.io_utils import read_records

EXP = WORK_DIR / "exp3"
CACHE = WORK_DIR / "m2cache_v2_blend_ad15_India_50"


def part_pairs(name):
    if name == "tr":
        return pd.read_parquet(CACHE / "tr_pairs.parquet", columns=["q_id", "s1_id"])
    if name == "ev_cache":
        return pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["q_id", "s1_id"])
    if name == "ev_ext":
        return pd.read_parquet(EXP / "ev_ext_pairs.parquet", columns=["q_id", "s1_id"])
    p = pd.read_parquet(WORK_DIR / "final_train_blend_ad15_100" / "pairs.parquet", columns=["q_id", "s1_id"])
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
    us = set(s1c.loc[s1c.country == "US", "entity_id"])
    return p[p["s1_id"].isin(us).to_numpy()].reset_index(drop=True)


if __name__ == "__main__":
    t0 = time.time()
    EXP.mkdir(exist_ok=True)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country", "name_core", "addr_norm"])
    stats_path = EXP / "ar7_stats_train_v2.pkl"
    if not stats_path.exists():
        with open(EXP / "ar7_stats_train_v2.tmp.pkl", "wb") as f:
            pickle.dump(build_stats(s1), f)
        (EXP / "ar7_stats_train_v2.tmp.pkl").replace(stats_path)
    s1 = s1_counts(s1.set_index("entity_id"))
    print(f"stats {stats_path.stat().st_size / 2**20:.0f} MB, s1 counts ready {time.time() - t0:.0f}s", flush=True)
    with Pool(8, initializer=_init, initargs=(str(stats_path),)) as pool:
        for name in ("tr", "ev_cache", "ev_ext", "us"):
            out = EXP / f"ar7_{name}.parquet"
            if out.exists():
                print(f"{name}: exists", flush=True)
                continue
            t = time.time()
            pairs = part_pairs(name)
            q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(pairs.q_id), ["entity_id", "name_core", "addr_norm"])
                           for k in (2, 3)]).set_index("entity_id")
            F = ar7_features(pairs, s1, q, pool)
            F.to_parquet(EXP / f"ar7_{name}.tmp.parquet", index=False)
            (EXP / f"ar7_{name}.tmp.parquet").replace(out)
            secs = time.time() - t
            print(f"{name}: {len(pairs)} pairs in {secs:.0f}s ({len(pairs) / secs:.0f} pairs/s) "
                  f"rss={psutil.Process().memory_info().rss / 2**30:.1f}GB", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)
