"""AR-7: difference-aware features (experiment only; not used by src/ until adopted).

HYPOTHESIS (H7). Among candidate pairs whose name and address are already near-identical,
a legitimate variant of the same business and a near-copy that belongs to no S1 (or to a
different S1) differ in the TYPE and RARITY of their residual differences:
  * variants: formatting edits -- abbreviations, typos, joined/split words, extra or
    dropped common words, numbers that are extended or re-segmented;
  * near-copies: substitution edits -- one house/unit number with one digit changed or
    transposed, or a rare (high-IDF) distinctive token replaced by another.
The current 49 features measure how MUCH the strings differ, and number SET overlap, but
not WHICH KIND of difference it is, nor how rare the differing tokens are, nor whether the
query carries the tokens that distinguish this S1 from the other retrieved S1 look-alikes.
Prediction: adding these features lowers near-copy false merges on V-dense half-B at equal
or better recall, raising macro F0.5 by >= +0.0010 (acceptance gate). Falsified if the
half-B gain is < +0.0010, its paired 95 % CI includes 0, or near-copy false merges do not
fall.

Everything is derived from the supplied data only: token document frequencies and address
sharing counts come from the S1 table of the split being scored; the model learns how much
each difference type matters from the training labels.

Feature groups (prefix e7_):
  nm_*  name-core residual tokens classified typo / abbreviation / join / novel, IDF mass
  ad_*  same for alphabetic address tokens
  num_* address numbers: S1 numbers missing from the query, typed as one-digit
        substitution, transposition, extension/segmentation, or other
  s1_*  how many S1 share this S1's exact address / exact name+address (multi-tenant sites)
  ctr_* namesake contrast inside the query's retrieved list: share of this S1's
        distinguishing tokens (absent from every other candidate) present in the query, and
        the most query tokens explained by a rival candidate but not by this S1
"""
import math
import pickle
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.normalize import LEGAL

LEGAL_VALS = set(LEGAL.values())
_RE_NUM = re.compile(r"\d+")
NAMES = ["e7_nm_idf_shared_frac", "e7_nm_idf_novel_s1", "e7_nm_idf_novel_q", "e7_nm_max_idf_novel",
         "e7_nm_n_novel_s1", "e7_nm_n_novel_q", "e7_nm_n_soft",
         "e7_ad_idf_shared_frac", "e7_ad_idf_novel_s1", "e7_ad_idf_novel_q", "e7_ad_n_soft",
         "e7_num_sub1", "e7_num_transp", "e7_num_ext", "e7_num_other", "e7_num_q_novel",
         "e7_s1_addr_share", "e7_s1_dup",
         "e7_ctr_n", "e7_ctr_hit", "e7_ctr_rival"]


def build_stats(s1: pd.DataFrame):
    """Per-country token document frequencies from an S1 table (country, name_core,
    addr_norm). Tokens seen in only one S1 record are dropped (a missing token is looked up
    as df=0, which changes its IDF by log 2 at most) to keep worker memory small."""
    st = {"n": s1["country"].value_counts().to_dict(), "nm": {}, "ad": {}}
    for c, g in s1.groupby("country"):
        nm, ad = Counter(), Counter()
        for x in g["name_core"]:
            nm.update(set(x.split()))
        for x in g["addr_norm"]:
            ad.update({t for t in x.split() if not any(ch.isdigit() for ch in t)})
        st["nm"][c] = {t: v for t, v in nm.items() if v >= 2}
        st["ad"][c] = {t: v for t, v in ad.items() if v >= 2}
    return st


def s1_counts(s1: pd.DataFrame) -> pd.DataFrame:
    """S1-level columns: how many S1 (same country) share this exact address, and this
    exact name_core + address."""
    out = s1[["country", "name_core", "addr_norm"]].copy()
    out["addr_share"] = out.groupby(["country", "addr_norm"])["name_core"].transform("size").astype(np.float32)
    out["dup"] = out.groupby(["country", "name_core", "addr_norm"])["addr_norm"].transform("size").astype(np.float32)
    return out


_ST = None


def _init(path):
    """Worker initialiser. The pickle is written by this experiment (build_stats output) in
    the same run, never an external file, so loading it is safe."""
    global _ST
    with open(path, "rb") as f:
        _ST = pickle.load(f)


def _idf(t, table, n):
    return math.log((n + 1) / (table.get(t, 0) + 1))


def _subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def _soft(t, others, joins):
    """True if residual token t is explained by some token of the other side."""
    for u in others:
        if min(len(t), len(u)) >= 4 and Levenshtein.distance(t, u) <= (1 if max(len(t), len(u)) < 8 else 2):
            return True
        if t[0] == u[0] and min(len(t), len(u)) >= 2 and (_subseq(t, u) or _subseq(u, t)):
            return True
    return t in joins


def _residual(S, Q, sl, ql, table, n):
    """S, Q: token sets; sl, ql: token lists (for joins). Returns
    (shared_frac, idf_novel_s1, idf_novel_q, max_idf_novel, n_novel_s1, n_novel_q, n_soft)."""
    if not S and not Q:
        return (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    shared = S & Q
    rs, rq = S - Q, Q - S
    jq = {ql[i] + ql[i + 1] for i in range(len(ql) - 1)}
    js = {sl[i] + sl[i + 1] for i in range(len(sl) - 1)}
    tot = sum(_idf(t, table, n) for t in S) or 1.0
    ns, nq, soft, isn, iqn, mx = 0, 0, 0, 0.0, 0.0, 0.0
    for t in rs:
        if _soft(t, rq, jq):
            soft += 1
        else:
            w = _idf(t, table, n); ns += 1; isn += w; mx = max(mx, w)
    for t in rq:
        if _soft(t, rs, js):
            soft += 1
        else:
            w = _idf(t, table, n); nq += 1; iqn += w; mx = max(mx, w)
    return (sum(_idf(t, table, n) for t in shared) / tot, isn, iqn, mx, ns, nq, soft)


def _nums(s):
    return {x.lstrip("0") or "0" for x in _RE_NUM.findall(s)}


def _num_types(X, Y):
    if not X or not Y:
        return (-1.0, -1.0, -1.0, -1.0, -1.0)
    # sorted: set iteration order depends on the per-process hash seed, and which query number
    # gets "used" (-> e7_num_q_novel) depended on it (K50 parity T2/T4, 2026-09-27)
    mx, my = sorted(X - Y), sorted(Y - X)
    xy = sorted(Y & X)
    sub1 = tr = ext = oth = 0
    used = set()
    for x in mx:
        kind = "other"
        for y in my:
            if len(x) == len(y) and sum(a != b for a, b in zip(x, y)) == 1:
                kind = "sub1"; used.add(y); break
        if kind == "other":
            for y in my:
                if len(x) == len(y) and sorted(x) == sorted(y):
                    kind = "tr"; used.add(y); break
        if kind == "other":
            for y in my + xy:
                if x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x)):
                    kind = "ext"; used.add(y); break
        sub1 += kind == "sub1"; tr += kind == "tr"; ext += kind == "ext"; oth += kind == "other"
    qnov = sum(1 for y in my if y not in used)
    return (sub1, tr, ext, oth, qnov)


def _tok(name_core, addr):
    nm = [t for t in name_core.split() if t not in LEGAL_VALS]
    ad = [t for t in addr.split() if not any(ch.isdigit() for ch in t)]
    return nm, ad


def _chunk(args):
    """args: lists over pairs of ONE block of whole queries (sorted by q_id)."""
    qid, cty, s_nc, s_ad, q_nc, q_ad, s_share, s_dup = args[:8]
    ctx = args[8] if len(args) > 8 else None      # None: every row of the query is context
    n = len(qid)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)
    i = 0
    while i < n:
        j = i
        while j < n and qid[j] == qid[i]:
            j += 1
        c = cty[i]
        N, tn, ta = _ST["n"].get(c, 1), _ST["nm"].get(c, {}), _ST["ad"].get(c, {})
        qnl, qal = _tok(q_nc[i], q_ad[i])
        qn, qa, qnum = set(qnl), set(qal), _nums(q_ad[i])
        cand = []
        for k in range(i, j):
            snl, sal = _tok(s_nc[k], s_ad[k])
            cand.append((snl, sal, set(snl) | set(sal) | _nums(s_ad[k])))
        Tq = qn | qa | qnum
        for k in range(i, j):
            snl, sal, T = cand[k - i]
            r = out[k]
            r[0:7] = _residual(set(snl), qn, snl, qnl, tn, N)
            if q_ad[k].strip():
                a = _residual(set(sal), qa, sal, qal, ta, N)
                r[7], r[8], r[9], r[10] = a[0], a[1], a[2], a[6]
            else:
                r[7:11] = -1.0
            r[11:16] = _num_types(_nums(s_ad[k]), qnum)
            r[16] = s_share[k]
            r[17] = s_dup[k]
            others = set()
            rival = 0
            for m in range(i, j):
                if m != k and (ctx is None or ctx[m]):
                    To = cand[m - i][2]
                    others |= To
                    rival = max(rival, len(Tq & (To - T)))
            D = T - others
            r[18] = len(D)
            r[19] = len(D & Tq) / len(D) if D else -1.0
            r[20] = rival
        i = j
    return out


def ar7_features(pairs: pd.DataFrame, s1: pd.DataFrame, q: pd.DataFrame, pool, block=2_000_000, ctx_mask=None):
    """pairs: q_id, s1_id (any order; every query's full candidate list must be present).
    ctx_mask: optional bool per row; the contrast features (ctr_*) of every row are computed
    against the OTHER rows of its query with ctx_mask=True (K50 plan D1: context = the
    query's top-10 for rows at every rank). None = every row is context (original behaviour).
    s1: indexed by entity_id with country, name_core, addr_norm, addr_share, dup (s1_counts);
    q: indexed by entity_id with name_core, addr_norm. pool: multiprocessing Pool initialised
    with _init(stats_path). Processed in blocks of whole queries to bound main-process
    memory. Returns a DataFrame aligned to `pairs` rows."""
    order = np.lexsort((pairs["s1_id"].to_numpy(), pairs["q_id"].to_numpy()))
    qs = pairs["q_id"].to_numpy()[order]
    ss = pairs["s1_id"].to_numpy()[order]
    cm = None if ctx_mask is None else np.asarray(ctx_mask, dtype=bool)[order]
    n = len(order)
    S = np.zeros((n, len(NAMES)), dtype=np.float32)
    i = 0
    while i < n:
        j = min(i + block, n)
        while j < n and qs[j] == qs[j - 1]:
            j += 1
        a = s1.loc[ss[i:j], ["country", "name_core", "addr_norm", "addr_share", "dup"]]
        b = q.loc[qs[i:j], ["name_core", "addr_norm"]]
        qid = qs[i:j].tolist()
        cols = (a["country"].tolist(), a["name_core"].tolist(), a["addr_norm"].tolist(),
                b["name_core"].tolist(), b["addr_norm"].tolist(),
                a["addr_share"].to_numpy(), a["dup"].to_numpy())
        if cm is not None:
            cols = cols + (cm[i:j],)
        m = j - i
        step = max(m // 64, 5000)
        chunks, u = [], 0
        while u < m:
            v = min(u + step, m)
            while v < m and qid[v] == qid[v - 1]:
                v += 1
            chunks.append((qid[u:v],) + tuple(x[u:v] for x in cols))
            u = v
        S[i:j] = np.vstack(pool.map(_chunk, chunks))
        del a, b, qid, cols, chunks
        i = j
    out = np.empty_like(S)
    out[order] = S
    return pd.DataFrame(out, columns=NAMES)
