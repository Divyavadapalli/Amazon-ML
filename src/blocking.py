"""Stage 2: candidate generation (blocking).

Every Source-2/3 record queries the Source-1 records of the *same country* (country is an
open set of labels; the training labels show 100 % same-country matches). Retrieval runs
S2/S3 -> S1 because each S2/S3 record belongs to at most one S1 entity in the training
labels, so a small per-query K suffices while an S1 entity can still collect any number
of matches.

Views (TF-IDF, vocabulary + IDF fitted on the S1 side of each country, no labels used):
    nw  name core words        nc  name character 3-grams        ad  address tokens
    ab  address token bigrams (e.g. '27_1', 'krupa_complex'): rare and discriminative even
        when every single address token is common
A *pass* is a weighted blend of views with its own top-K. Several passes can be run and
their candidates unioned (multi-pass blocking); `exact` adds an exact name_core join.
Features with document frequency above max_df are dropped from the index: they carry
little ranking signal but dominate sparse-matmul cost. Their recall impact is measured in
experiments/exp_blocking.py rather than assumed.
"""
import time

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import N_JOBS

VIEW_DF = {"nw": 0.02, "nc": 0.01, "ad": 0.02, "ab": 0.02}
DEFAULT = {
    "passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 0.8}, "k": 10}],
    "exact": False, "exact_cap": 20, "max_df": VIEW_DF, "min_score": 0.05,
}


def _mem():
    return (f"rss={psutil.Process().memory_info().rss / 2**30:.1f}GB "
            f"avail={psutil.virtual_memory().available / 2**30:.1f}GB")


def _text(df, view):
    if view in ("ad", "ab"):
        return df["addr_norm"].fillna("")
    name = df["name_core"].fillna("")
    alt = df["name_alt"].fillna("").str.replace("|", " ", regex=False)
    return (name + " " + alt).str.strip()


def build_views(s1: pd.DataFrame, q: pd.DataFrame, views, max_df: dict):
    """{view: (S1 matrix, query matrix)}, each row L2-normalised TF-IDF."""
    out = {}
    for v in views:
        kw = dict(dtype=np.float32, sublinear_tf=True, max_df=max_df[v], min_df=1, lowercase=False)
        if v == "nc":
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
        elif v == "ab":
            vec = TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(2, 2), **kw)
        else:
            vec = TfidfVectorizer(token_pattern=r"[^ ]+", **kw)
        m1 = vec.fit_transform(_text(s1, v))
        out[v] = (m1, vec.transform(_text(q, v)))
    return out


def _blend(mats, weights, side):
    ws = {v: w for v, w in weights.items() if w > 0}
    norm = np.sqrt(sum(w * w for w in ws.values()))
    return sp.hstack([mats[v][side] * (w / norm) for v, w in ws.items()], format="csr")


def _topk(A, BT, k, min_score):
    C = sp_matmul_topn(A, BT, top_n=k, threshold=min_score, sort=True, n_threads=N_JOBS).tocoo()
    out = pd.DataFrame({"q_idx": C.row.astype(np.int32), "s1_idx": C.col.astype(np.int32),
                        "score": C.data.astype(np.float32)})
    out = out.sort_values(["q_idx", "score"], ascending=[True, False], kind="stable")
    out["rank"] = out.groupby("q_idx").cumcount().astype(np.int16)
    return out


def _exact(s1, q, cap):
    """Queries whose name_core equals an S1 name_core (skip names shared by > cap S1)."""
    a = pd.DataFrame({"key": s1["name_core"].values, "s1_idx": np.arange(len(s1), dtype=np.int32)})
    a = a[a["key"] != ""]
    sizes = a.groupby("key")["s1_idx"].transform("size")
    a = a[sizes <= cap]
    b = pd.DataFrame({"key": q["name_core"].values, "q_idx": np.arange(len(q), dtype=np.int32)})
    return b.merge(a, on="key")[["q_idx", "s1_idx"]]


def retrieve(s1: pd.DataFrame, q: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """Candidates for one country: q_idx, s1_idx, plus score/rank per pass
    (NaN/-1 when the pair was not produced by that pass) and `score`=max pass score."""
    cfg = {**DEFAULT, **(cfg or {})}
    t = time.time()
    views = sorted({v for p in cfg["passes"] for v, w in p["w"].items() if w > 0})
    mats = build_views(s1, q, views, cfg["max_df"])
    t_vec = time.time() - t
    merged = None
    for p in cfg["passes"]:
        tp = time.time()
        A = _blend(mats, p["w"], 1)
        BT = _blend(mats, p["w"], 0).T.tocsr()
        r = _topk(A, BT, p["k"], cfg["min_score"])
        r = r.rename(columns={"score": f"sc_{p['name']}", "rank": f"rk_{p['name']}"})
        print(f"      pass {p['name']}: {len(r)} pairs {time.time() - tp:.0f}s nnzA={A.nnz} "
              f"nnzB={BT.nnz} {_mem()}", flush=True)
        merged = r if merged is None else merged.merge(r, on=["q_idx", "s1_idx"], how="outer")
        del A, BT
    if cfg["exact"]:
        e = _exact(s1, q, cfg["exact_cap"]).assign(ex=np.int8(1))
        merged = merged.merge(e, on=["q_idx", "s1_idx"], how="outer")
        merged["ex"] = merged["ex"].fillna(0).astype(np.int8)
    sc_cols = [c for c in merged.columns if c.startswith("sc_")]
    for c in merged.columns:
        if c.startswith("rk_"):
            merged[c] = merged[c].fillna(-1).astype(np.int16)
    merged["score"] = merged[sc_cols].max(axis=1).fillna(0).astype(np.float32)
    # exact-only pairs lack pass scores; give them the true blended cosine of pass 0
    missing = merged[sc_cols[0]].isna().to_numpy()
    if missing.any():
        p0 = cfg["passes"][0]["w"]
        A, B = _blend(mats, p0, 1), _blend(mats, p0, 0)
        qi, si = merged["q_idx"].to_numpy()[missing], merged["s1_idx"].to_numpy()[missing]
        merged.loc[missing, "score"] = np.asarray(A[qi].multiply(B[si]).sum(axis=1)).ravel()
    merged = merged.sort_values(["q_idx", "score"], ascending=[True, False], kind="stable")
    merged["rank"] = merged.groupby("q_idx").cumcount().astype(np.int16)
    print(f"    retrieve: {len(q)} q x {len(s1)} S1 vec={t_vec:.0f}s total={time.time() - t:.0f}s "
          f"pairs={len(merged)} {_mem()}", flush=True)
    return merged.reset_index(drop=True)


def retrieve_all(s1_all: pd.DataFrame, q_all: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """Per-country retrieval; returns q_id, s1_id and all score/rank columns."""
    parts = []
    for c in sorted(set(q_all["country"]) | set(s1_all["country"])):
        s1 = s1_all[s1_all["country"] == c].reset_index(drop=True)
        q = q_all[q_all["country"] == c].reset_index(drop=True)
        if len(s1) == 0 or len(q) == 0:
            continue
        print(f"  country={c}", flush=True)
        r = retrieve(s1, q, cfg)
        r.insert(0, "q_id", q["entity_id"].values[r.pop("q_idx").values])
        r.insert(1, "s1_id", s1["entity_id"].values[r.pop("s1_idx").values])
        parts.append(r)
    return pd.concat(parts, ignore_index=True)
