"""V3 / M4: retrieval depth and specialist-view diagnostic on the V-dense population E.

For every true (query, owner) pair of E (India fold-1 bucket < 100) the owner's rank is found
in the top-1000 of: the production blend (nw 1 / nc 1 / ad 1.5, = blend_ad15) and each single
view (nw name words, nc name char-3grams, ad address tokens), same TF-IDF fit on India S1 and the
same min_score 0.05 as src.blocking.  Reported per K in (3, 10, 20, 50, 100, 200, 500, 1000):
recall@K, oracle macro F0.5@K (P = 1, R = found/|truth|, singletons 1), by segment (S2/S3, empty
address, script name, core_exact, high/low S1 name frequency); per view: unique true pairs it
recovers within its top-10 / top-50 that the blend misses in its top-10 / top-50.
Runtime and RSS per config. Queries only (no candidate files are written).
Output: C:\\aml2026\\v3_exp\\reports\\kdepth.json
Usage: python -m experiments.v3_kdepth
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from experiments.k50_common import atomic_json, bucket, mem_ok
from src.blocking import CountryIndex, _blend, _text, _topk
from src.io_utils import read_ground_truth
from src.config import WORK_DIR
from src.prep import s1_fold

OUT = Path(r"C:\aml2026\v3_exp\reports")
KS = (3, 10, 20, 50, 100, 200, 500, 1000)
KMAX = 1000
CFGS = {"blend": {"nw": 1.0, "nc": 1.0, "ad": 1.5}, "nw": {"nw": 1.0}, "nc": {"nc": 1.0}, "ad": {"ad": 1.0}}
COLS = ["entity_id", "country", "business_name", "name_core", "name_alt", "addr_norm"]
_NONASCII = re.compile(r"[^\x00-\x7f]")


def oracle(rank, pair_ent, n_ent, nt, K):
    found = np.bincount(pair_ent[(rank >= 0) & (rank < K)], minlength=n_ent).astype(float)
    r = np.divide(found, nt, out=np.zeros_like(found), where=nt > 0)
    f = np.where(found > 0, 1.25 * r / (0.25 + r), 0.0)
    return float(np.where(nt == 0, 1.0, f).mean())


if __name__ == "__main__":
    t0 = time.time()
    gt = read_ground_truth()
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=COLS)
    s1 = s1[s1.country == "India"].reset_index(drop=True)
    cache_s1 = set(pd.read_parquet(WORK_DIR / "m2cache_v2_blend_ad15_India_50" / "ev_pairs.parquet", columns=["s1_id"]).s1_id)
    E = sorted({s for s in gt if bucket(s) < 100 and s1_fold(s) == 1} & cache_s1)
    eix = pd.Series(np.arange(len(E)), index=E)
    nt = np.array([len(gt[e]) for e in E], float)
    tp = pd.DataFrame([(q, e) for e in E for q in gt[e]], columns=["q_id", "s1_id"])
    q = pd.concat([pd.read_parquet(WORK_DIR / f"train_s{k}.parquet", columns=COLS) for k in (2, 3)], ignore_index=True)
    q = q[q.entity_id.isin(set(tp.q_id))].drop_duplicates("entity_id").set_index("entity_id").loc[tp.q_id].reset_index()
    s1pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    owner_pos = s1pos.loc[tp.s1_id].to_numpy()
    pair_ent = eix.loc[tp.s1_id].to_numpy()
    freq = s1.name_core.map(s1.name_core.value_counts()).to_numpy()
    seg = {"S2": tp.q_id.str.startswith("S2").to_numpy(), "S3": tp.q_id.str.startswith("S3").to_numpy(),
           "addr_empty": (q.addr_norm.fillna("").str.strip() == "").to_numpy(),
           "script": q.business_name.fillna("").map(lambda x: bool(_NONASCII.search(x))).to_numpy(),
           "core_exact": (q.name_core.fillna("").to_numpy() == s1.name_core.to_numpy()[owner_pos]),
           "s1_name_freq>=5": freq[owner_pos] >= 5}
    seg["core_nonexact"], seg["s1_name_freq<5"] = ~seg["core_exact"], ~seg["s1_name_freq>=5"]
    print(f"E {len(E)} entities, {len(tp)} true pairs; India S1 {len(s1)}", mem_ok(tag="M4"), flush=True)
    idx = CountryIndex(s1, {"passes": [{"name": "blend", "w": CFGS["blend"], "k": KMAX}], "min_score": 0.05})
    mq = {v: vec.transform(_text(q, v)) for v, vec in idx.vecs.items()}
    ranks, res = {}, {"true_pairs": len(tp), "entities": len(E), "configs": {}}
    for name, w in CFGS.items():
        t = time.time()
        A, BT = _blend(mq, w), _blend(idx.m1, w).T.tocsr()
        rk = np.full(len(tp), -1, np.int32)
        for i in range(0, len(tp), 5000):
            r = _topk(A[i:i + 5000], BT, KMAX, 0.05)
            hit = r[r["s1_idx"].to_numpy() == owner_pos[i + r["q_idx"].to_numpy()]]
            rk[i + hit["q_idx"].to_numpy()] = hit["rank"].to_numpy()
        ranks[name] = rk
        c = {"secs": round(time.time() - t), "rss_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}
        c["recall"] = {K: float(((rk >= 0) & (rk < K)).mean()) for K in KS}
        c["oracle_f05"] = {K: oracle(rk, pair_ent, len(E), nt, K) for K in KS}
        c["recall_by_segment"] = {s: {K: float(((rk[m] >= 0) & (rk[m] < K)).mean()) for K in (10, 50, 200, 1000)}
                                  for s, m in seg.items()}
        res["configs"][name] = c
        print(name, json.dumps({k: c[k] for k in ("secs", "recall", "oracle_f05")}), mem_ok(tag="M4"), flush=True)
    b = ranks["blend"]
    for name in ("nw", "nc", "ad"):
        v = ranks[name]
        res["configs"][name]["unique_recovered"] = {
            f"view@{kv}_vs_blend@{kb}": int(((v >= 0) & (v < kv) & ~((b >= 0) & (b < kb))).sum())
            for kv, kb in ((10, 10), (10, 50), (50, 50))}
    res["blend_misses"] = {f"@{K}": int((~((b >= 0) & (b < K))).sum()) for K in (10, 50)}
    res["secs"] = round(time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(res, OUT / "kdepth.json")
    print(json.dumps({n: (c["oracle_f05"], c.get("unique_recovered")) for n, c in res["configs"].items()}, indent=1))
