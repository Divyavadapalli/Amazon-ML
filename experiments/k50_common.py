"""Shared pieces of the K50 experiment (plan: C:\\Users\\Divya\\.claude\\plans\\nested-sniffing-perlis.md).

ONE feature path for training, validation and test inference: `k50_features(rows, ctx, ...)`.
  rows : the (q_id, s1_id, score, rank) pairs to featurise -- any ranks 0..49
  ctx  : the query's top-10 retrieval list (rank < 10) -- the K-invariant context (plan D1)
Base features come from src.features.pair_features on `rows`, then every per-query retrieval
feature is overwritten from the top-10 context, so a rank-37 pair and a rank-2 pair of the
same query see the same ret_top1 / ret_ncand, and the rank 0-9 features are identical to the
validated K10 caches (parity test: experiments/k50_parity.py). AR-7 contrast features use the
top-10 context (ar7_features ctx_mask).
"""
import hashlib
import json
import os
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from experiments.ar7_features import NAMES as AR7, ar7_features
from src.features import pair_features

K50_ROOT = Path(os.environ.get("AML_K50_ROOT", r"C:\aml2026\k50_exp"))
CTX_K = 10
BANDS = [(0, 3), (3, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
RANK_BINS = [(0, 1, "1"), (1, 3, "2-3"), (3, 5, "4-5"), (5, 10, "6-10"), (10, 15, "11-15"), (15, 20, "16-20"),
             (20, 30, "21-30"), (30, 40, "31-40"), (40, 50, "41-50")]
PROTECTED = [Path(p).resolve() for p in (r"C:\aml2026\output", r"C:\aml2026\submission2", r"C:\aml2026\submission3",
                                           r"C:\aml2026\submissions", r"C:\aml2026\work\cand_train_blend_ad15",
                                           r"C:\aml2026\work\cand_test_blend_ad15")]


def bucket(x, mod=1000):
    return (zlib.crc32(x.encode()) >> 1) % mod


def guard(path):
    """Refuse to write into a protected location (plan S7)."""
    p = Path(path).resolve()
    for g in PROTECTED:
        assert p != g and g not in p.parents, f"refusing to write into protected path {g}"
    return p


def atomic_parquet(df, path):
    path = guard(path)
    tmp = path.with_name(path.stem + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def atomic_json(obj, path):
    path = guard(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def fingerprint(q_ids, s1_ids):
    order = np.lexsort((np.asarray(s1_ids, dtype=str), np.asarray(q_ids, dtype=str)))
    h = hashlib.sha256()
    for a, b in zip(np.asarray(q_ids)[order], np.asarray(s1_ids)[order]):
        h.update(f"{a}\t{b}\n".encode())
    return h.hexdigest()


def mem_ok(min_avail_gb=1.5, tag=""):
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss / 2**30
    msg = f"rss={rss:.1f}GB avail={vm.available / 2**30:.1f}GB"
    if vm.available / 2**30 < min_avail_gb:
        raise MemoryError(f"[{tag}] available RAM below {min_avail_gb} GB ({msg}); stopping safely (checkpoints kept)")
    return msg


def disk_ok(need_gb, root=r"C:\\", tag=""):
    free = shutil.disk_usage(root).free / 2**30
    assert free >= need_gb + 5, f"[{tag}] only {free:.1f} GB free, need {need_gb} + 5 GB margin"
    return free


def e_prime(gt, s1_country, bucket_lt):
    """E' = India fold-1 S1 in hash bucket < bucket_lt (nested inside the V-dense E, bucket < 100)."""
    from src.prep import s1_fold
    india = set(s1_country.index[s1_country.to_numpy() == "India"])
    return sorted(s for s in gt if s in india and bucket(s) < bucket_lt and s1_fold(s) == 1)


def k50_features(rows, ctx, s1i, q, freq, s1c, pool):
    """rows / ctx: DataFrames with q_id, s1_id, score, rank. ctx must hold, for every query in
    rows, its complete rank < CTX_K list. Returns X (pair_features columns + AR-7), aligned to rows."""
    rows = rows.reset_index(drop=True)
    assert (ctx["rank"] < CTX_K).all()
    base = rows[["q_id", "s1_id", "score", "rank"]].copy()
    base["sc_blend"] = base["score"].astype(np.float32)
    base["rk_blend"] = base["rank"].astype(np.int16)
    X = pair_features(base, s1i, q, freq)
    g = ctx.groupby("q_id")
    top1 = g["score"].max()
    ncand = g.size()
    r1 = ctx[ctx["rank"] == 1].set_index("q_id")["score"]
    qv = rows["q_id"]
    t1 = qv.map(top1)
    assert t1.notna().all(), "a row's query has no top-10 context"
    X["ret_top1"] = t1.to_numpy(np.float32)
    X["ret_gap_top1"] = (X["ret_top1"] - X["ret_score"]).astype(np.float32)
    sec = qv.map(r1).fillna(0.0).to_numpy(np.float32)
    X["ret_margin"] = np.where(rows["rank"].to_numpy() == 0, X["ret_score"].to_numpy() - sec, 0.0).astype(np.float32)
    X["ret_ncand"] = qv.map(ncand).to_numpy(np.float32)
    # AR-7 on rows plus their context (context rows not already in rows), contrast vs context only
    key_r = rows["q_id"] + "|" + rows["s1_id"]
    extra = ctx[ctx["q_id"].isin(set(qv)) & ~(ctx["q_id"] + "|" + ctx["s1_id"]).isin(set(key_r))]
    both = pd.concat([rows[["q_id", "s1_id", "rank"]], extra[["q_id", "s1_id", "rank"]]], ignore_index=True)
    A = ar7_features(both[["q_id", "s1_id"]], s1c, q, pool, ctx_mask=(both["rank"].to_numpy() < CTX_K))
    X = pd.concat([X.reset_index(drop=True), A.iloc[:len(rows)].reset_index(drop=True)], axis=1)
    return X.astype(np.float32)


def labels(pairs, owner):
    return (pairs["q_id"].map(owner) == pairs["s1_id"]).to_numpy(np.int8)


def band_counts(rank, y):
    out = {}
    for lo, hi in BANDS:
        m = (rank >= lo) & (rank < hi)
        out[f"{lo}-{hi - 1}"] = {"neg": int((m & (y == 0)).sum()), "pos": int((m & (y == 1)).sum())}
    return out
