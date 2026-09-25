"""Smoke test: features on a small slice of saved sample candidates (bugs + throughput).
Usage: python -m experiments.smoke_features [n_queries]"""
import sys
import time

import pandas as pd

from src.config import WORK_DIR
from src.io_utils import read_records
from experiments.exp_match2 import REC, featurize, s1_name_freq

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    c = pd.read_parquet(WORK_DIR / "blk2_blend_ad15.parquet")
    qs = set(c.q_id.drop_duplicates().sample(n, random_state=0))
    c = c[c.q_id.isin(qs)]
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", qs, REC) for k in (2, 3)])
    q = q.set_index("entity_id")
    t = time.time()
    c2, X = featurize(c, s1, q, s1_name_freq(), chunk=20000)
    dt = time.time() - t
    print(f"pairs {len(c2)} feats {X.shape} in {dt:.1f}s -> {len(c2) / dt:.0f} pairs/s")
    na = X.isna().sum()
    print("NaN columns:", na[na > 0].to_dict())
    print(X.describe().T[["mean", "min", "max"]].round(3).to_string())
