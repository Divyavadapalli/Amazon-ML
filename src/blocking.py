"""Stage 2: candidate generation (blocking).

Each Source-2/3 record queries the Source-1 records of the *same country* and keeps its
top-K by a blended TF-IDF cosine over three views:
    name core words  |  name character 3-grams  |  address tokens (incl. house numbers)
Retrieving S2/S3 -> S1 (not the other way round) matches the data: in training every
S2/S3 record belongs to at most one S1 entity, so each query needs only a few S1 options.

Very frequent features (e.g. 'pvt', 'rd', char-grams like 'ent') are dropped from the
index (max_df) because they add little ranking signal but dominate sparse-matmul cost.
Vocabulary and IDF are fitted on the Source-1 side of each country only, so the
procedure is identical for train and test and needs no labels.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

from .config import N_JOBS

DEFAULT = {
    "top_k": 10,
    "w_name_word": 1.0, "w_name_char": 1.0, "w_addr": 0.8,
    "max_df_word": 0.02, "max_df_char": 0.01, "max_df_addr": 0.02,
    "min_score": 0.05,
}


def _views(df):
    name = df["name_core"].fillna("")
    alt = df["name_alt"].fillna("").str.replace("|", " ", regex=False)
    name = (name + " " + alt).str.strip()
    addr = df["addr_norm"].fillna("")
    return name, addr


def _vec(kind, max_df):
    common = dict(dtype=np.float32, sublinear_tf=True, max_df=max_df, min_df=1, lowercase=False)
    if kind == "word":
        return TfidfVectorizer(token_pattern=r"[^ ]+", **common)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **common)


def build_matrices(s1: pd.DataFrame, q: pd.DataFrame, cfg: dict):
    """Fit vectorisers on S1, transform S1 and queries; return (S1 matrix, query matrix)."""
    n1, a1 = _views(s1)
    nq, aq = _views(q)
    blocks1, blocksq = [], []
    for kind, txt1, txtq, w, mdf in (
        ("word", n1, nq, cfg["w_name_word"], cfg["max_df_word"]),
        ("char", n1, nq, cfg["w_name_char"], cfg["max_df_char"]),
        ("word", a1, aq, cfg["w_addr"], cfg["max_df_addr"]),
    ):
        if w <= 0:
            continue
        v = _vec(kind, mdf)
        m1 = v.fit_transform(txt1)
        mq = v.transform(txtq)
        blocks1.append(m1 * w)
        blocksq.append(mq * w)
    # divide by total weight^2 so a perfect match on every view scores ~1
    norm = np.sqrt(sum(w * w for w in (cfg["w_name_word"], cfg["w_name_char"], cfg["w_addr"]) if w > 0))
    A = sp.hstack(blocksq, format="csr") / norm
    B = sp.hstack(blocks1, format="csr") / norm
    return B, A


def retrieve(s1: pd.DataFrame, q: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """Top-K S1 candidates for every query row (same country assumed).

    Returns a long DataFrame: q_idx, s1_idx (positional), score, rank.
    """
    cfg = {**DEFAULT, **(cfg or {})}
    t = time.time()
    B, A = build_matrices(s1, q, cfg)
    t_vec = time.time() - t
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=cfg["top_k"], threshold=cfg["min_score"],
                       sort=True, n_threads=N_JOBS)
    C = C.tocoo()
    out = pd.DataFrame({"q_idx": C.row.astype(np.int32), "s1_idx": C.col.astype(np.int32),
                        "score": C.data.astype(np.float32)})
    out = out.sort_values(["q_idx", "score"], ascending=[True, False], kind="stable")
    out["rank"] = out.groupby("q_idx").cumcount().astype(np.int8)
    print(f"    retrieve: {len(q)} queries x {len(s1)} S1  vec={t_vec:.0f}s "
          f"total={time.time() - t:.0f}s  pairs={len(out)}", flush=True)
    return out.reset_index(drop=True)


def retrieve_all(s1_all: pd.DataFrame, q_all: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """Run `retrieve` per country (country is an open set of labels) and return
    candidate pairs with entity ids: q_id, s1_id, score, rank."""
    parts = []
    for c in sorted(set(q_all["country"]) | set(s1_all["country"])):
        s1 = s1_all[s1_all["country"] == c].reset_index(drop=True)
        q = q_all[q_all["country"] == c].reset_index(drop=True)
        if len(s1) == 0 or len(q) == 0:
            continue
        print(f"  country={c}", flush=True)
        r = retrieve(s1, q, cfg)
        parts.append(pd.DataFrame({
            "q_id": q["entity_id"].values[r["q_idx"].values],
            "s1_id": s1["entity_id"].values[r["s1_idx"].values],
            "score": r["score"].values, "rank": r["rank"].values,
        }))
    return pd.concat(parts, ignore_index=True)
