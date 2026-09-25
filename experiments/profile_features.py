"""Profile pair_features on a small slice (second call, so pool start-up is excluded)."""
import cProfile
import pstats
import time

import pandas as pd

from src.config import WORK_DIR
from src.io_utils import read_records
from src.features import pair_features
from experiments.exp_match2 import REC, s1_name_freq

if __name__ == "__main__":
    c = pd.read_parquet(WORK_DIR / "blk2_blend_ad15.parquet")
    qs = set(c.q_id.drop_duplicates().sample(4000, random_state=0))
    c = c[c.q_id.isin(qs)].sort_values(["q_id", "rank"]).reset_index(drop=True)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", qs, REC) for k in (2, 3)]).set_index("entity_id")
    freq = s1_name_freq()
    t = time.time(); pair_features(c.iloc[:2000], s1, q, freq); print(f"warm-up (pool spawn) {time.time()-t:.1f}s")
    pr = cProfile.Profile(); t = time.time(); pr.enable()
    pair_features(c, s1, q, freq)
    pr.disable(); dt = time.time() - t
    print(f"{len(c)} pairs in {dt:.1f}s -> {len(c)/dt:.0f} pairs/s")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
