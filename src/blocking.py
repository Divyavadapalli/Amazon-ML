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


class CountryIndex:
    """TF-IDF views and per-pass S1 matrices for one country, built once and then
    queried in chunks (vectorisers are fitted on S1 only)."""

    def __init__(self, s1: pd.DataFrame, cfg: dict):
        self.cfg = {**DEFAULT, **(cfg or {})}
        self.s1 = s1
        t = time.time()
        views = sorted({v for p in self.cfg["passes"] for v, w in p["w"].items() if w > 0})
        self.vecs, self.m1 = {}, {}
        for v in views:
            kw = dict(dtype=np.float32, sublinear_tf=True, max_df=self.cfg["max_df"][v],
                      min_df=1, lowercase=False)
            if v == "nc":
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw)
            elif v == "ab":
                vec = TfidfVectorizer(token_pattern=r"[^ ]+", ngram_range=(2, 2), **kw)
            else:
                vec = TfidfVectorizer(token_pattern=r"[^ ]+", **kw)
            self.m1[v] = vec.fit_transform(_text(s1, v))
            self.vecs[v] = vec
        self.BT = {p["name"]: _blend(self.m1, p["w"]).T.tocsr() for p in self.cfg["passes"]}
        self.p0 = self.cfg["passes"][0]
        self.B0 = _blend(self.m1, self.p0["w"])
        if self.cfg["exact"]:
            a = pd.DataFrame({"key": s1["name_core"].values,
                              "s1_idx": np.arange(len(s1), dtype=np.int32)})
            a = a[a["key"] != ""]
            self.exact_tab = a[a.groupby("key")["s1_idx"].transform("size") <= self.cfg["exact_cap"]]
        print(f"    index: {len(s1)} S1, views={views} built in {time.time() - t:.0f}s {_mem()}",
              flush=True)

    def query(self, q: pd.DataFrame) -> pd.DataFrame:
        """Candidates for a chunk of queries: q_idx (positional in q), s1_idx, per-pass
        sc_*/rk_* columns, optional exact flag `ex`, blended `score` and final `rank`."""
        t = time.time()
        mq = {v: vec.transform(_text(q, v)) for v, vec in self.vecs.items()}
        merged = None
        for p in self.cfg["passes"]:
            A = _blend(mq, p["w"])
            r = _topk(A, self.BT[p["name"]], p["k"], self.cfg["min_score"])
            r = r.rename(columns={"score": f"sc_{p['name']}", "rank": f"rk_{p['name']}"})
            merged = r if merged is None else merged.merge(r, on=["q_idx", "s1_idx"], how="outer")
        if self.cfg["exact"]:
            b = pd.DataFrame({"key": q["name_core"].values, "q_idx": np.arange(len(q), dtype=np.int32)})
            e = b.merge(self.exact_tab, on="key")[["q_idx", "s1_idx"]].assign(ex=np.int8(1))
            merged = merged.merge(e, on=["q_idx", "s1_idx"], how="outer")
            merged["ex"] = merged["ex"].fillna(0).astype(np.int8)
        for c in merged.columns:
            if c.startswith("rk_"):
                merged[c] = merged[c].fillna(-1).astype(np.int16)
        # the blended cosine of the first pass for EVERY pair (pairs found only by other
        # passes or by the exact join get it computed directly)
        sc0 = f"sc_{self.p0['name']}"
        miss = merged[sc0].isna().to_numpy()
        if miss.any():
            A0 = _blend(mq, self.p0["w"])
            qi, si = merged["q_idx"].to_numpy()[miss], merged["s1_idx"].to_numpy()[miss]
            merged.loc[miss, sc0] = np.asarray(A0[qi].multiply(self.B0[si]).sum(axis=1)).ravel().astype(np.float32)
        for c in merged.columns:
            if c.startswith("sc_"):
                merged[c] = merged[c].fillna(0).astype(np.float32)
        merged["score"] = merged[sc0]
        merged = merged.sort_values(["q_idx", "score"], ascending=[True, False], kind="stable")
        merged["rank"] = merged.groupby("q_idx").cumcount().astype(np.int16)
        print(f"    query: {len(q)} q -> {len(merged)} pairs in {time.time() - t:.0f}s {_mem()}",
              flush=True)
        return merged.reset_index(drop=True)


def _blend(mats, weights):
    """Weighted hstack of L2-normalised views; cosine of blends = weighted mean of view cosines."""
    ws = {v: w for v, w in weights.items() if w > 0}
    norm = np.sqrt(sum(w * w for w in ws.values()))
    return sp.hstack([mats[v] * (w / norm) for v, w in ws.items()], format="csr")


def _topk(A, BT, k, min_score):
    C = sp_matmul_topn(A, BT, top_n=k, threshold=min_score, sort=True, n_threads=N_JOBS).tocoo()
    out = pd.DataFrame({"q_idx": C.row.astype(np.int32), "s1_idx": C.col.astype(np.int32),
                        "score": C.data.astype(np.float32)})
    out = out.sort_values(["q_idx", "score"], ascending=[True, False], kind="stable")
    out["rank"] = out.groupby("q_idx").cumcount().astype(np.int16)
    return out


def retrieve(s1: pd.DataFrame, q: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """One-shot convenience wrapper: build the index for `s1` and query all of `q`."""
    return CountryIndex(s1, cfg).query(q)


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
