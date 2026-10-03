"""final_v2 = final_v1's training pairs and 49 features + the 21 AR-7 features.

Same pairs (India 10 % of queries + US 5 % full-index sample, full-index retrieval), same
LightGBM parameters and early-stopping split as final_v1 (src.matching.train_model); only the
AR-7 columns are added. AR-7 statistics come from the TRAINING S1 table. The threshold and
candidate policy frozen into the meta are the ones tuned on V-dense half-A for system D.

Usage: python -m experiments.train_final_v2 <threshold> <h> <kmax>
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiments.ar7_features import NAMES as AR7, _init, ar7_features, s1_counts
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records
from src.matching import LGB_PARAMS, train_model

TAG = "final_v2"
CACHE = WORK_DIR / "final_train_blend_ad15_100"
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"

if __name__ == "__main__":
    thr, h, kmax = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    t0 = time.time()
    pairs = pd.read_parquet(CACHE / "pairs.parquet")
    X = pd.read_parquet(CACHE / "X.parquet")
    f7 = CACHE / "ar7.parquet"
    if not f7.exists():
        s1 = s1_counts(pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country", "name_core", "addr_norm"]).set_index("entity_id"))
        q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(pairs.q_id), ["entity_id", "name_core", "addr_norm"])
                       for k in (2, 3)]).set_index("entity_id")
        with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
            A = ar7_features(pairs[["q_id", "s1_id"]], s1, q, pool)
        A.to_parquet(CACHE / "ar7.tmp.parquet", index=False)
        (CACHE / "ar7.tmp.parquet").replace(f7)
        print(f"AR-7 features {A.shape} in {time.time() - t0:.0f}s", flush=True)
    A = pd.read_parquet(f7)
    assert len(A) == len(X)
    X = pd.concat([X, A[AR7]], axis=1)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    y = (pairs["q_id"].map(owner) == pairs["s1_id"]).to_numpy(np.int8)
    assert not X.isna().any().any(), "NaN features"
    t = time.time()
    model = train_model(X, y, pairs["q_id"].to_numpy())
    model.save_model(str(WORK_DIR / f"m2_model_{TAG}.txt"))
    meta = {"model_tag": TAG, "threshold": thr, "one_to_one": True, "cand_tag": "blend_ad15",
            "policy": ["adapt", h, kmax], "features": list(X.columns), "ar7_features": AR7,
            "n_train_pairs": len(X), "n_pos": int(y.sum()), "best_iteration": model.best_iteration,
            "lgb_params": LGB_PARAMS, "train_secs": round(time.time() - t), "secs": round(time.time() - t0)}
    (WORK_DIR / f"m2_model_{TAG}.meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", json.dumps({k: v for k, v in meta.items() if k not in ("features", "ar7_features")}), flush=True)
