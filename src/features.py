"""Stage 3: pairwise features for (S1 record, S2/S3 candidate) pairs.

Country is deliberately NOT a feature: France is absent from training, so every feature
is a language-agnostic similarity or a structural property of the strings.
"""
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .config import N_JOBS
from .normalize import LEGAL

LEGAL_VALS = set(LEGAL.values())
REC_COLS = ["name_norm", "name_core", "name_alt", "addr_norm", "addr_digits", "pobox",
            "business_name", "business_address"]


_POOL = None


def _pool():
    """One worker pool for the whole process (Windows 'spawn' makes each new pool re-import
    numpy/pandas in every worker, so creating one per chunk is expensive)."""
    global _POOL
    if _POOL is None:
        _POOL = Pool(N_JOBS)
    return _POOL


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _set_feats(args):
    """Python-level token/digit overlap features for a chunk of pairs."""
    n1, n2, a1, a2, d1, d2 = args
    out = np.zeros((len(n1), 12), dtype=np.float32)
    for i in range(len(n1)):
        t1, t2 = set(n1[i].split()), set(n2[i].split())
        c1, c2 = t1 - LEGAL_VALS, t2 - LEGAL_VALS
        l1, l2 = t1 & LEGAL_VALS, t2 & LEGAL_VALS
        inter = len(c1 & c2)
        out[i, 0] = inter / max(len(c1 | c2), 1)                      # core jaccard
        out[i, 1] = inter / max(min(len(c1), len(c2)), 1)             # core overlap coef
        out[i, 2] = len(c1 - c2)                                      # S1 core tokens missing
        out[i, 3] = len(c2 - c1)                                      # extra query tokens
        out[i, 4] = (len(l1 & l2) / len(l1 | l2)) if (l1 or l2) else -1  # legal-form agreement
        out[i, 5] = float(bool(l1) and bool(l2) and not (l1 & l2))    # legal-form conflict
        x1, x2 = d1[i].split(), d2[i].split()
        s1d, s2d = set(x1), set(x2)
        out[i, 6] = len(s1d & s2d) / max(len(s1d | s2d), 1) if (s1d or s2d) else -1
        out[i, 7] = float(bool(x1) and bool(x2) and x1[0] == x2[0])  # first number equal
        out[i, 8] = float(bool(s1d) and bool(s2d) and not (s1d & s2d))  # numbers conflict
        w1, w2 = set(a1[i].split()), set(a2[i].split())
        out[i, 9] = len(w1 & w2) / max(len(w1 | w2), 1) if (w1 or w2) else -1  # addr jaccard
        out[i, 10] = len(w1 & w2) / max(min(len(w1), len(w2)), 1) if (w1 and w2) else -1
        # 5-6 digit tokens (ZIP / PIN / French code postal) agreement
        p1 = {x for x in s1d if x.isdigit() and len(x) in (5, 6)}
        p2 = {x for x in s2d if x.isdigit() and len(x) in (5, 6)}
        out[i, 11] = (1.0 if p1 & p2 else 0.0) if (p1 and p2) else -1
    return out


_RE_NUM = re.compile(r"\d+")


def _nums(addr: str):
    """Digit runs of an address with leading zeros stripped ('N-0032' -> {'32'})."""
    return {x.lstrip("0") or "0" for x in _RE_NUM.findall(addr)}


def _num_legal_feats(args):
    """Number-agreement and legal-form features (error analysis v1: false merges are
    near-duplicates whose address differs in ONE house/unit number and whose legal form
    differs; true matches keep S1's numbers, possibly zero-padded or with extra numbers)."""
    a1, a2, n1, n2 = args
    out = np.zeros((len(a1), 7), dtype=np.float32)
    for i in range(len(a1)):
        x, y = _nums(a1[i]), _nums(a2[i])
        if x and y:
            miss = x - y
            out[i, 0] = len(miss)                                   # S1 numbers absent in query
            out[i, 1] = len(y - x)                                  # extra numbers in query
            out[i, 2] = float(not miss)                             # all S1 numbers present
            shared = x & y
            out[i, 3] = max((len(v) for v in shared), default=0)    # longest shared number
            out[i, 4] = len(shared) / len(x)
        else:
            out[i, :5] = -1
        l1 = set(n1[i].split()) & LEGAL_VALS
        l2 = set(n2[i].split()) & LEGAL_VALS
        out[i, 5] = float(("pvt" in l1) != ("pvt" in l2)) if (l1 and l2) else -1
        out[i, 6] = len(l1 ^ l2)
    return out


NUM_LEGAL_NAMES = ["num_s1_missing", "num_q_extra", "num_s1_all_in_q", "num_longest_shared",
                   "num_s1_share", "legal_pvt_mismatch", "legal_symdiff"]


def num_legal_features(pairs, s1, q):
    a = s1.loc[pairs["s1_id"].values, ["addr_norm", "name_norm"]]
    b = q.loc[pairs["q_id"].values, ["addr_norm", "name_norm"]]
    a1, a2 = a["addr_norm"].tolist(), b["addr_norm"].tolist()
    n1, n2 = a["name_norm"].tolist(), b["name_norm"].tolist()
    n = len(pairs)
    step = max(n // (N_JOBS * 4), 20000)
    chunks = [(a1[i:i + step], a2[i:i + step], n1[i:i + step], n2[i:i + step]) for i in range(0, n, step)]
    S = np.vstack(_pool().map(_num_legal_feats, chunks)) if chunks else np.zeros((0, 7), np.float32)
    return pd.DataFrame(S, columns=NUM_LEGAL_NAMES)


SET_NAMES = ["core_jacc", "core_overlap", "s1_missing", "q_extra", "legal_agree",
             "legal_conflict", "num_jacc", "num_first_eq", "num_conflict", "addr_jacc",
             "addr_overlap", "postal_eq"]


def pair_features(pairs: pd.DataFrame, s1: pd.DataFrame, q: pd.DataFrame,
                  name_freq: dict = None) -> pd.DataFrame:
    """pairs: q_id, s1_id, score, rank (from blocking). s1/q: normalised records indexed by
    entity_id. name_freq: {(country, name_core): count among S1} for genericity."""
    a = s1.loc[pairs["s1_id"].values, REC_COLS + ["country"]].reset_index(drop=True)
    b = q.loc[pairs["q_id"].values, REC_COLS].reset_index(drop=True)
    F = pd.DataFrame(index=pairs.index)
    F["ret_score"] = pairs["score"].values
    F["ret_rank"] = pairs["rank"].values
    grp = pairs.groupby("q_id")["score"]
    F["ret_top1"] = grp.transform("max").values
    F["ret_gap_top1"] = F["ret_top1"] - F["ret_score"]
    # rank is 0-based and sorted by score, so the runner-up is the rank-1 row
    second = pairs.loc[pairs["rank"] == 1].set_index("q_id")["score"]
    second = pairs["q_id"].map(second).fillna(0.0).to_numpy()
    F["ret_margin"] = np.where(F["ret_rank"] == 0, F["ret_score"] - second, 0.0)
    F["ret_ncand"] = grp.transform("size").values
    # per-pass evidence from multi-pass blocking (score, rank; -1/NaN when the pass
    # did not produce the pair) and the exact-name flag
    for col in pairs.columns:
        if col.startswith(("sc_", "rk_")) or col == "ex":
            F[f"blk_{col}"] = pairs[col].to_numpy()
    # NB: no per-S1 counts over queries (e.g. "how many queries rank this S1 first"):
    # validation uses a query sample, so such counts would shift between val and test.

    nc1, nc2 = a["name_core"].tolist(), b["name_core"].tolist()
    nn1, nn2 = a["name_norm"].tolist(), b["name_norm"].tolist()
    F["nc_ratio"] = _cp(nc1, nc2, fuzz.ratio)
    F["nc_tsort"] = _cp(nc1, nc2, fuzz.token_sort_ratio)
    F["nc_tset"] = _cp(nc1, nc2, fuzz.token_set_ratio)
    F["nc_partial"] = _cp(nc1, nc2, fuzz.partial_ratio)
    F["nc_jw"] = _cp(nc1, nc2, JaroWinkler.normalized_similarity)
    F["nc_lev"] = _cp(nc1, nc2, Levenshtein.distance)
    F["nn_ratio"] = _cp(nn1, nn2, fuzz.ratio)
    F["nn_tsort"] = _cp(nn1, nn2, fuzz.token_sort_ratio)
    # best similarity using either side's DBA/trade-name parts
    alt2 = [x.split("|") if x else [] for x in b["name_alt"].tolist()]
    alt1 = [x.split("|") if x else [] for x in a["name_alt"].tolist()]
    best = F["nc_tsort"].to_numpy().copy()
    for i in np.flatnonzero([bool(x) or bool(y) for x, y in zip(alt1, alt2)]):
        c1, c2 = alt1[i] or [nc1[i]], alt2[i] or [nc2[i]]
        best[i] = max(best[i], max(fuzz.token_sort_ratio(x, y) for x in c1 for y in c2))
    F["nc_alt_best"] = best
    F["nc_len1"] = [len(x.split()) for x in nc1]
    F["nc_len2"] = [len(x.split()) for x in nc2]
    F["nc_charlen_diff"] = np.abs(a["name_core"].str.len().values - b["name_core"].str.len().values)

    an1, an2 = a["addr_norm"].tolist(), b["addr_norm"].tolist()
    F["q_addr_empty"] = (b["addr_norm"] == "").astype(np.int8).values
    F["ad_ratio"] = _cp(an1, an2, fuzz.ratio)
    F["ad_tsort"] = _cp(an1, an2, fuzz.token_sort_ratio)
    F["ad_tset"] = _cp(an1, an2, fuzz.token_set_ratio)
    F["ad_partial"] = _cp(an1, an2, fuzz.partial_ratio)
    for col in ("ad_ratio", "ad_tsort", "ad_tset", "ad_partial"):
        F.loc[F["q_addr_empty"] == 1, col] = -1
    F["q_has_pobox"] = (b["pobox"] != "").astype(np.int8).values
    F["q_nonascii_name"] = (~b["business_name"].map(str.isascii)).astype(np.int8).values
    F["q_raw_name_ratio"] = _cp(a["business_name"].str.lower().tolist(),
                                b["business_name"].str.lower().tolist(), fuzz.ratio)

    d1, d2 = a["addr_digits"].tolist(), b["addr_digits"].tolist()
    n = len(pairs)
    step = max(n // (N_JOBS * 4), 20000)
    chunks = [(nn1[i:i + step], nn2[i:i + step], an1[i:i + step], an2[i:i + step],
               d1[i:i + step], d2[i:i + step]) for i in range(0, n, step)]
    S = np.vstack(_pool().map(_set_feats, chunks)) if chunks else np.zeros((0, 12))
    for j, name in enumerate(SET_NAMES):
        F[name] = S[:, j]

    if name_freq is not None:
        keys = list(zip(a["country"], a["name_core"]))
        F["s1_name_freq"] = np.array([name_freq.get(k, 1) for k in keys], dtype=np.float32)
    F["q_is_s3"] = pairs["q_id"].str.startswith("S3-").astype(np.int8).values
    # Extra hand-crafted column: a fixed linear blend of name (token-sort, scaled to [0,1])
    # and digit-set Jaccard (missing -> 0). A retrained model can use it; the frozen model
    # ignores unknown columns because pipeline/train_final select X[meta["features"]].
    bk_name = F["nc_tsort"].to_numpy() / 100.0
    bk_digit = np.where(F["num_jacc"].to_numpy() < 0, 0.0, F["num_jacc"].to_numpy())
    F["bio_kinetic_score"] = (4.5 * bk_name + 3.5 * bk_digit - 5.0).astype(np.float32)
    F = pd.concat([F, num_legal_features(pairs, s1, q).set_index(F.index)], axis=1)
    return F.astype(np.float32)
