"""K50 Phase F data: the K50 V-dense evaluation set, scored by every model (plan section 4, Phase F).

Entities E' = India fold-1 S1 with hash bucket < <bucket> (nested inside the V-dense E).
Queries   = every India query owned by E' or with an E' member anywhere in its top 50 (so any
            policy up to rank 50 is evaluated exactly: no uncounted false merges).
Staged, per K50 cache chunk (checkpointed, resumable):
  1. ranks 0-9 of every eval query: features (k50_common.k50_features) -> every model's prob
  2. ranks 10-49 only for queries whose best top-3 prob is < 0.95 under ANY model. Every tuned
     gate is capped at h <= 0.95 (k50_eval), so a query without deep rows can never open them.
Stored per row: ids, rank, score, owner, y, the 3 slim columns vdense.evaluate uses, and one
probability column per model -- never the feature matrix (disk budget).
US spot-check (--us): the same on the US part of the 5 % full-index K50 sample
(blk2_blend_ad15_k50.parquet); V-dense K50 models are India-only, so it is held out.

Usage: python -m experiments.k50_build_eval <bucket> [--us]
"""
import sys
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.ar7_features import _init, s1_counts
from experiments.exp_match2 import REC
from experiments.k50_common import K50_ROOT, atomic_json, atomic_parquet, bucket, k50_features, mem_ok
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records
from src.prep import s1_fold

CAND = K50_ROOT / "cand_train"
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"
GATE_CAP = 0.95
SLIM = ["nc_tsort", "ad_tset", "q_addr_empty"]


def load_models():
    ms = {p.stem: lgb.Booster(model_file=str(p)) for p in sorted((K50_ROOT / "model").glob("*.txt"))}
    for tag, path in (("ref_exp3c_M7c_s42", WORK_DIR / "exp3" / "model_M7c_s42.txt"),
                      ("ref_exp3_M7_s42_leaky", WORK_DIR / "exp3" / "model_M7_s42.txt")):
        if path.exists():
            ms[tag] = lgb.Booster(model_file=str(path))
    assert any(k.startswith("K50") for k in ms) and any(k.startswith("M7_K10reg") for k in ms), "train models first"
    return ms


def score(rows, X, models, owner):
    out = rows[["q_id", "s1_id", "rank", "score"]].reset_index(drop=True).copy()
    out["own"] = out["q_id"].map(owner)
    out["y"] = (out["own"] == out["s1_id"]).to_numpy(np.int8)
    for c in SLIM:
        out[c] = X[c].to_numpy(np.float32)
    for k, m in models.items():
        out[k] = m.predict(X[m.feature_name()], num_threads=12).astype(np.float32)
    return out


def staged(rows, models, owner, s1, q, freq, s1c, pool):
    """rows: all K50 rows of a set of whole queries. Returns scored rows (top-10 + gated deep)."""
    top = rows[rows["rank"] < 10].reset_index(drop=True)
    X = k50_features(top, top, s1, q, freq, s1c, pool)
    a = score(top, X, models, owner)
    r3 = a[a["rank"] < 3]
    p3 = r3.groupby("q_id")[[k for k in models]].max()
    deep_q = set(p3.index[(p3.to_numpy() < GATE_CAP).any(axis=1)])
    deep = rows[(rows["rank"] >= 10) & rows["q_id"].isin(deep_q)].reset_index(drop=True)
    parts = [a]
    if len(deep):
        Xd = k50_features(deep, top[top["q_id"].isin(deep_q)], s1, q, freq, s1c, pool)
        parts.append(score(deep, Xd, models, owner))
    return pd.concat(parts, ignore_index=True), len(deep_q)


if __name__ == "__main__":
    b = int(sys.argv[1])
    us = "--us" in sys.argv
    t0 = time.time()
    out_dir = K50_ROOT / "eval" / (f"us" if us else f"b{b}")
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    models = load_models()
    print("models:", list(models), flush=True)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    s1c = s1_counts(s1[["country", "name_core", "addr_norm"]])
    freq = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"]).value_counts().to_dict()
    stats = {"E_bucket": b, "us": us, "gate_cap": GATE_CAP, "models": list(models), "chunks": {}}
    with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
        if us:
            smp = pd.read_parquet(WORK_DIR / "blk2_blend_ad15_k50.parquet", columns=["q_id", "s1_id", "score", "rank"])
            us_s1 = set(s1.index[s1["country"].to_numpy() == "US"])
            smp = smp[smp["s1_id"].isin(us_s1)].sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            qv = smp["q_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
            blocks, i = [], 0
            while i < len(smp):
                j = starts[np.searchsorted(starts, i + 2_000_000)] if i + 2_000_000 < starts[-1] else len(smp)
                blocks.append((f"us_{len(blocks):03d}", smp.iloc[i:j]))
                i = j
            src_iter = iter(blocks)
        else:
            s1_ids = pd.read_parquet(CAND / "index_India_s1.parquet")["entity_id"].to_numpy()
            in_e = np.array([bucket(s) < b and s1_fold(s) == 1 for s in s1_ids])
            e_set = set(s1_ids[in_e])

            def gen():
                for p in sorted(CAND.glob("India_*.parquet")):
                    yield p.stem, None
            src_iter = gen()
        for stem, rows in src_iter:
            f = out_dir / f"{stem}.parquet"
            if f.exists():
                continue
            t = time.time()
            if rows is None:
                c = load_compact_chunk(CAND, stem)
                touch = set(c.loc[in_e[c["s1_pos"].to_numpy()], "q_id"]) | set(c.loc[c["q_id"].map(owner).isin(e_set), "q_id"])
                rows = c[c["q_id"].isin(touch)][["q_id", "s1_id", "score", "rank"]]
            rows = rows.sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(rows.q_id), REC) for k in (2, 3)]).set_index("entity_id")
            res, n_deep = staged(rows, models, owner, s1, q, freq, s1c, pool)
            atomic_parquet(res, f)
            stats["chunks"][stem] = {"queries": int(rows.q_id.nunique()), "rows_all": len(rows), "rows_scored": len(res),
                                     "deep_queries": n_deep}
            atomic_json(stats, out_dir / "manifest.json")
            print(f"  {stem}: {rows.q_id.nunique()} queries, scored {len(res)} of {len(rows)} rows, deep {n_deep} "
                  f"{time.time() - t:.0f}s {mem_ok(tag='build_eval')}", flush=True)
    stats["secs"] = round(time.time() - t0)
    atomic_json(stats, out_dir / "manifest.json")
    (out_dir / "DONE").write_text("ok")
    print(f"done {time.time() - t0:.0f}s", flush=True)
