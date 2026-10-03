"""K50 production model (run only after gate G1 passes).

Training queries: the final_v1/final_v2 set (India queries with hash bucket < 100 + the US
queries of the 5 % full-index sample), now with their top-50 lists:
  * ranks 0-9: reuse the final_train_blend_ad15_100 features (X + ar7) -- valid under plan D1
    (top-10 context), rows matched on (q_id, s1_id)
  * ranks 10-49: negatives kept at the G1-selected budget r (deterministic hash, weight 1/r),
    positives always; features from k50_common.k50_features (checkpointed blocks)
Production model => trained on data that includes V-dense entities (leak L2): it is never
evaluated on V-dense. Saves WORK_DIR/m2_model_final_k50.txt + .meta.json (new tag).

Usage: python -m experiments.k50_train_final <threshold> <r> '<policy json>'
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiments.ar7_features import NAMES as AR7, _init, s1_counts
from experiments.exp_match2 import REC
from experiments.k50_common import (K50_ROOT, atomic_json, atomic_parquet, band_counts, bucket, fingerprint,
                                    k50_features, labels, mem_ok)
from experiments.k50_train import keep_rate, train_w
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records
from src.matching import LGB_PARAMS

TAG = "final_k50"
FT = WORK_DIR / "final_train_blend_ad15_100"
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"

if __name__ == "__main__":
    thr, r, policy = float(sys.argv[1]), float(sys.argv[2]), json.loads(sys.argv[3])
    t0 = time.time()
    out = K50_ROOT / "train" / "final"
    out.mkdir(parents=True, exist_ok=True)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    # ---- deep rows (rank >= 10) of the production training queries
    deep_f = out / "deep_pairs.parquet"
    if not deep_f.exists():
        parts = []
        for p in sorted((K50_ROOT / "cand_train").glob("India_*.parquet")):
            c = load_compact_chunk(K50_ROOT / "cand_train", p.stem)
            parts.append(c[(c["q_id"].map(bucket).to_numpy() < 100)][["q_id", "s1_id", "score", "rank"]])
        smp = pd.read_parquet(WORK_DIR / "blk2_blend_ad15_k50.parquet", columns=["q_id", "s1_id", "score", "rank"])
        s1c_ = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
        us = set(s1c_.loc[s1c_.country == "US", "entity_id"])
        parts.append(smp[smp["s1_id"].isin(us)])
        allp = pd.concat(parts, ignore_index=True)
        atomic_parquet(allp[allp["rank"] < 10][["q_id", "s1_id", "score", "rank"]], out / "top10_pairs.parquet")
        d = allp[allp["rank"] >= 10].reset_index(drop=True)
        y = labels(d, owner)
        keep = (y == 1) | (keep_rate(d.q_id.to_numpy(), d.s1_id.to_numpy()) < r)
        atomic_parquet(d[keep].reset_index(drop=True), deep_f)
    deep = pd.read_parquet(deep_f).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
    top10 = pd.read_parquet(out / "top10_pairs.parquet")
    print(f"deep rows kept {len(deep)} {mem_ok(tag='train_final')}", flush=True)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    s1c = s1_counts(s1[["country", "name_core", "addr_norm"]])
    freq = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"]).value_counts().to_dict()
    qv = deep["q_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    bounds, i = [], 0
    while i < len(deep):
        j = starts[np.searchsorted(starts, i + 1_000_000)] if i + 1_000_000 < starts[-1] else len(deep)
        bounds.append((i, j)); i = j
    with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
        for n, (i, j) in enumerate(bounds):
            f = out / f"deep_block_{n:03d}.parquet"
            if f.exists():
                continue
            t = time.time()
            rows = deep.iloc[i:j].reset_index(drop=True)
            ctx = top10[top10["q_id"].isin(set(rows.q_id))]
            q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(rows.q_id), REC) for k in (2, 3)]).set_index("entity_id")
            X = k50_features(rows, ctx, s1, q, freq, s1c, pool)
            atomic_parquet(pd.concat([rows, X], axis=1), f)
            print(f"  deep block {n + 1}/{len(bounds)} {len(rows)} rows {time.time() - t:.0f}s {mem_ok(tag='train_final')}", flush=True)
    # ---- assemble: final_train (ranks 0-9) + deep blocks
    P0 = pd.read_parquet(FT / "pairs.parquet")
    X0 = pd.concat([pd.read_parquet(FT / "X.parquet"), pd.read_parquet(FT / "ar7.parquet")[AR7]], axis=1)
    Dd = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("deep_block_*.parquet"))], ignore_index=True)
    feats = list(X0.columns)
    X = pd.concat([X0, Dd[feats]], ignore_index=True)
    ids = pd.concat([P0[["q_id", "s1_id", "rank"]], Dd[["q_id", "s1_id", "rank"]]], ignore_index=True)
    y = labels(ids, owner)
    rank = ids["rank"].to_numpy()
    w = np.where((rank >= 10) & (y == 0), 1.0 / r, 1.0).astype(np.float32)
    assert not X.isna().any().any(), "NaN features"
    t = time.time()
    model = train_w(X, y, w, ids["q_id"].to_numpy(), 42)
    model.save_model(str(WORK_DIR / f"m2_model_{TAG}.txt"))
    meta = {"model_tag": TAG, "threshold": thr, "one_to_one": True, "cand_tag": "blend_ad15_k50", "policy": policy,
            "budget_rank_ge10": r, "features": feats, "ar7_features": AR7, "context": "top-10 (plan D1)",
            "n_train_pairs": len(X), "n_pos": int(y.sum()), "by_band": band_counts(rank, y),
            "by_source": {s: int((ids.q_id.str[:2] == s).sum()) for s in ("S2", "S3")},
            "fingerprint": fingerprint(ids.q_id.to_numpy(), ids.s1_id.to_numpy()),
            "best_iteration": model.best_iteration, "lgb_params": {**LGB_PARAMS, "seed": 42},
            "train_secs": round(time.time() - t), "secs": round(time.time() - t0)}
    (WORK_DIR / f"m2_model_{TAG}.meta.json").write_text(json.dumps(meta, indent=1))
    atomic_json({k: v for k, v in meta.items() if k not in ("features", "ar7_features")}, K50_ROOT / "reports" / "k50_final_model_meta.json")
    print("saved", json.dumps({k: v for k, v in meta.items() if k not in ("features", "ar7_features")}), flush=True)
