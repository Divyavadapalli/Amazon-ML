"""Blocking recall against the FULL training S1 index (test-like competition density).

Queries = every S2/S3 record owned by a hash-sampled set E of S1 entities, plus unmatched
(distractor) records sampled at the same rate. Recall is measured over the true pairs of E.
Each config appends one JSON line to experiments/blocking2_results.jsonl and saves its
candidates + missed pairs to WORK_DIR for error analysis.

Usage: python -m experiments.exp_blocking2 <config_name> [sample_frac=0.05]
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src.blocking import DEFAULT, VIEW_DF, retrieve_all
from src.config import WORK_DIR
from src.io_utils import read_ground_truth

B = {"nw": 1.0, "nc": 1.0, "ad": 0.8}
CONFIGS = {
    "blend10": {"passes": [{"name": "blend", "w": B, "k": 10}]},
    "blend20": {"passes": [{"name": "blend", "w": B, "k": 20}]},
    "name10": {"passes": [{"name": "name", "w": {"nw": 1.0, "nc": 1.0}, "k": 10}]},
    "addr10": {"passes": [{"name": "addr", "w": {"ad": 1.0}, "k": 10}]},
    "char10": {"passes": [{"name": "char", "w": {"nc": 1.0}, "k": 10}]},
    "word10": {"passes": [{"name": "word", "w": {"nw": 1.0}, "k": 10}]},
    "multi": {"passes": [{"name": "blend", "w": B, "k": 10},
                         {"name": "name", "w": {"nw": 1.0, "nc": 1.0}, "k": 5},
                         {"name": "addr", "w": {"ad": 1.0, "nc": 0.5}, "k": 5}],
              "exact": True},
    "blend_ab10": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 0.6, "ab": 0.6}, "k": 10}]},
    "multi_ab": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 0.6, "ab": 0.6}, "k": 10},
                            {"name": "addr", "w": {"ad": 0.6, "ab": 1.0}, "k": 5}],
                 "exact": True},
    "blend10_df": {"passes": [{"name": "blend", "w": B, "k": 10}],
                   "max_df": {"nw": 0.05, "nc": 0.03, "ad": 0.05}},
}


def bucket(x, mod=1000):
    return (zlib.crc32(x.encode()) >> 1) % mod


def build_queries(frac):
    gt = read_ground_truth()
    E = {s for s in gt if bucket(s) < frac * 1000}
    owner = {m: s for s, ms in gt.items() for m in ms}
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet")
    qs = []
    for k in (2, 3):
        q = pd.read_parquet(WORK_DIR / f"train_s{k}.parquet")
        own = q["entity_id"].map(owner)
        keep = own.isin(E) | (own.isna() & (q["entity_id"].map(bucket) < frac * 1000))
        qs.append(q[keep.to_numpy()])
    q = pd.concat(qs, ignore_index=True)
    q["owner"] = q["entity_id"].map(owner)
    truth = {s: gt[s] for s in E}
    return s1, q, truth


def evaluate(c, q, truth, s1):
    tp = q[q["owner"].notna()][["entity_id", "owner", "country", "business_name", "addr_norm",
                                "name_core"]].rename(columns={"entity_id": "q_id"})
    got = set(zip(c["q_id"].to_numpy(), c["s1_id"].to_numpy()))
    tp["hit"] = [(a, b) in got for a, b in zip(tp["q_id"], tp["owner"])]
    s1c = s1.set_index("entity_id")["name_core"]
    tp["source"] = tp["q_id"].str[:2]
    tp["addr_empty"] = tp["addr_norm"] == ""
    tp["script"] = ~tp["business_name"].map(str.isascii)
    tp["core_exact"] = tp["name_core"].to_numpy() == s1c.loc[tp["owner"]].to_numpy()
    out = {"pair_recall": tp["hit"].mean(), "n_true": len(tp), "n_cand": len(c),
           "cand_per_query": len(c) / max(len(q), 1)}
    for col in ("country", "source", "addr_empty", "script", "core_exact"):
        out[f"recall_by_{col}"] = tp.groupby(col)["hit"].agg(["mean", "size"]).round(4).to_dict("index")
    rk = c.merge(tp[["q_id", "owner"]], left_on=["q_id", "s1_id"], right_on=["q_id", "owner"])["rank"]
    out["recall@k"] = {k: float((rk < k).sum() / len(tp)) for k in (1, 2, 3, 5, 10, 20)}
    # oracle macro-F0.5 (perfect matcher on these candidates) over sampled entities
    hit_by_s1 = tp[tp["hit"]].groupby("owner").size()
    f = []
    for s, ms in truth.items():
        if not ms:
            f.append(1.0)
            continue
        h = hit_by_s1.get(s, 0)
        if h == 0:
            f.append(0.0)
            continue
        r = h / len(ms)
        f.append(1.25 * r / (0.25 + r))
    out["oracle_f05"] = float(np.mean(f))
    return out, tp[~tp["hit"]]


if __name__ == "__main__":
    name = sys.argv[1]
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    cfg = {**DEFAULT, **CONFIGS[name]}
    cfg["max_df"] = {**VIEW_DF, **cfg.get("max_df", {})}
    t = time.time()
    s1, q, truth = build_queries(frac)
    print(f"[{name}] full S1={len(s1)} queries={len(q)} sampled entities={len(truth)} "
          f"load={time.time() - t:.0f}s", flush=True)
    t = time.time()
    c = retrieve_all(s1, q, cfg)
    secs = time.time() - t
    res, miss = evaluate(c, q, truth, s1)
    res.update({"config": name, "frac": frac, "cfg": cfg, "retrieval_secs": round(secs),
                "peak_rss_gb": round(psutil.Process().memory_info().peak_wset / 2**30, 1)
                if hasattr(psutil.Process().memory_info(), "peak_wset") else None})
    print(json.dumps(res, default=str, indent=1), flush=True)
    with open(Path(__file__).with_name("blocking2_results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(res, default=str) + "\n")
    c.to_parquet(WORK_DIR / f"blk2_{name}.parquet", index=False)
    miss.to_parquet(WORK_DIR / f"blk2_{name}_miss.parquet", index=False)
