"""final_v2_s7 = final_v2 retrained with seed 7 (for the V3 'max' two-seed ensemble, E1).

Identical training pairs, features (49 + 21 AR-7, cached in final_train_blend_ad15_100) and
LightGBM parameters as final_v2; only the seed changes, through experiments.exp3.train (the
same routine that produced the validated seed-7 model M7c_s7: seed 7 for both the 10 %
query-group early-stopping split and LightGBM). final_v2 itself is never modified.
Memory-lean ground truth: only the training queries' owners are read.

Usage: python -m experiments.train_final_v2_s7
"""
import json
import time

import numpy as np
import pandas as pd

from experiments.ar7_features import NAMES as AR7
from experiments.exp3 import train
from experiments.v3_common import GT_PATH, Log
from src.config import WORK_DIR
from src.matching import LGB_PARAMS

TAG = "final_v2_s7"
CACHE = WORK_DIR / "final_train_blend_ad15_100"

if __name__ == "__main__":
    t0 = time.time()
    log = Log("train_final_v2_s7")
    out_model, out_meta = WORK_DIR / f"m2_model_{TAG}.txt", WORK_DIR / f"m2_model_{TAG}.meta.json"
    assert not out_model.exists(), "refusing to overwrite an existing model"
    ref = json.loads((WORK_DIR / "m2_model_final_v2.meta.json").read_text())
    pairs = pd.read_parquet(CACHE / "pairs.parquet", columns=["q_id", "s1_id"])
    X = pd.concat([pd.read_parquet(CACHE / "X.parquet"), pd.read_parquet(CACHE / "ar7.parquet")[AR7]], axis=1)
    assert list(X.columns) == ref["features"], "feature list differs from final_v2"
    need = set(pairs.q_id)
    owner = {}
    with open(GT_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            for m in rest.split(","):
                if m in need:
                    owner[m] = s1
    y = (pairs["q_id"].map(owner) == pairs["s1_id"]).to_numpy(np.int8)
    assert len(X) == ref["n_train_pairs"] and int(y.sum()) == ref["n_pos"], "training data differs from final_v2"
    assert not X.isna().any().any(), "NaN features"
    log(f"training data identical to final_v2: {len(X)} pairs, {int(y.sum())} positives")
    log.set_stage("train seed 7")
    t = time.time()
    model = train(X, y, pairs["q_id"].to_numpy(), 7)
    model.save_model(str(out_model))
    meta = {**ref, "model_tag": TAG, "seed": 7, "best_iteration": model.best_iteration,
            "lgb_params": {**LGB_PARAMS, "seed": 7}, "train_secs": round(time.time() - t), "secs": round(time.time() - t0),
            "note": "final_v2 with seed 7 (exp3.train); used only as the second member of the max ensemble"}
    out_meta.write_text(json.dumps(meta, indent=1))
    log("saved " + json.dumps({k: v for k, v in meta.items() if k not in ("features", "ar7_features")}))
