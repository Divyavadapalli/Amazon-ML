"""Train the final matcher on training data and freeze it for test inference.

Training pairs (all produced by the locked blocking config against the FULL S1 index, so
per-query retrieval features match test time exactly):
  * India: uniform `india_permille` share of queries from full training candidates
  * US:    every query of the 5 % full-index sample (experiments/exp_blocking2, blend_ad15),
           whose owned/distractor mix (74/26) matches the full data
Labels come from train_ground_truth only. The threshold is passed in (tuned on held-out
validation entities) and frozen into model_meta.json together with the feature list.

Usage: python -m src.train_final <model_tag> <threshold> [india_permille=200]
"""
import json
import sys
import time
import zlib

import numpy as np
import pandas as pd

from .config import WORK_DIR
from .features import pair_features
from .io_utils import read_ground_truth, read_records
from .matching import LGB_PARAMS, train_model

REC = ["entity_id", "country", "name_norm", "name_core", "name_alt", "addr_norm", "addr_digits",
       "pobox", "business_name", "business_address"]
CAND_TAG = "blend_ad15"


def bucket(x, mod=1000):
    return (zlib.crc32(x.encode()) >> 1) % mod


def featurize(pairs, s1i, qi, freq, chunk=1_000_000):
    """Features in chunks of whole queries (per-query features need all of a query's pairs)."""
    pairs = pairs.sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
    qv = pairs["q_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    out, i = [], 0
    while i < len(pairs):
        j = starts[np.searchsorted(starts, i + chunk)] if i + chunk < starts[-1] else len(pairs)
        out.append(pair_features(pairs.iloc[i:j].reset_index(drop=True), s1i, qi, freq))
        i = j
    return pairs, pd.concat(out, ignore_index=True)


def build_training_pairs(india_permille):
    parts = []
    for p in sorted((WORK_DIR / f"cand_train_{CAND_TAG}").glob("India_*.parquet")):
        c = pd.read_parquet(p)
        parts.append(c[c["q_id"].map(bucket).to_numpy() < india_permille])
    us = pd.read_parquet(WORK_DIR / f"blk2_{CAND_TAG}.parquet")
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
    us_s1 = set(s1c.loc[s1c.country == "US", "entity_id"])
    parts.append(us[us["s1_id"].isin(us_s1)])
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    tag, thr = sys.argv[1], float(sys.argv[2])
    india_permille = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    t0 = time.time()
    cache = WORK_DIR / f"final_train_{CAND_TAG}_{india_permille}"
    if (cache / "X.parquet").exists():
        pairs = pd.read_parquet(cache / "pairs.parquet")
        X = pd.read_parquet(cache / "X.parquet")
    else:
        pairs = build_training_pairs(india_permille)
        print(f"training pairs {len(pairs)} ({pairs.q_id.nunique()} queries)", flush=True)
        s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
        need = set(pairs["q_id"])
        q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", need, REC) for k in (2, 3)]).set_index("entity_id")
        freq = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"]).value_counts().to_dict()
        pairs, X = featurize(pairs, s1, q, freq)
        cache.mkdir(exist_ok=True)
        pairs.to_parquet(cache / "pairs.parquet", index=False)
        X.to_parquet(cache / "X.parquet", index=False)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    y = (pairs["q_id"].map(owner) == pairs["s1_id"]).to_numpy(np.int8)
    assert not X.isna().any().any(), "NaN features"
    print(f"features {X.shape}, positives {y.sum()} ({y.mean():.3f})  {time.time() - t0:.0f}s", flush=True)
    model = train_model(X, y, pairs["q_id"].to_numpy())
    model.save_model(str(WORK_DIR / f"m2_model_{tag}.txt"))
    meta = {"model_tag": tag, "threshold": thr, "one_to_one": True, "cand_tag": CAND_TAG,
            "features": list(X.columns), "n_train_pairs": len(X), "n_pos": int(y.sum()),
            "india_permille": india_permille, "best_iteration": model.best_iteration,
            "lgb_params": LGB_PARAMS, "secs": round(time.time() - t0)}
    (WORK_DIR / f"m2_model_{tag}.meta.json").write_text(json.dumps(meta, indent=1))
    print("saved", json.dumps({k: v for k, v in meta.items() if k != "features"}), flush=True)
