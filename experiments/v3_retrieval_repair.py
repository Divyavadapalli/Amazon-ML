"""V3 / R1: focused retrieval repair for the measured top-10 miss segments (offline, read-only).

Base = D's candidate set: blend (nw 1 / nc 1 / ad 1.5) top-10 (== cand_*_blend_ad15, parity T1).
Specialist additions (union with the base, no change to the base ranking):
  A   address-heavy blend (nw 1 / nc 1 / ad 4) top-k   -> common-name collisions (address decides)
  N   name char-3gram only (nc) top-k                  -> non-exact / script / typo names
  X   blend top-50 for EMPTY-ADDRESS queries only       -> empty-address records (depth is the only lever)
Baselines: blend top-20, top-50 (plain deeper K, as in the K50 experiment).

Evaluation population: V-dense E (44,360 India fold-1 entities; == vdense.load_india), every
true (query, owner) pair; recall = owner inside the candidate set; oracle macro F0.5 (P = 1,
R = found/|truth|, singletons 1) on half-B, E; the D top-10 half-B oracle must reproduce
0.986494 (M2). Candidate growth = mean NEW candidates per query beyond the base top-10,
measured on a hash sample of ALL India train queries (owned + distractors, bucket < 25 permille),
because growth must be measured on the test-like mix, not only on owned queries.
Same TF-IDF fit on India S1 and min_score 0.05 as src.blocking. No candidate files are written.

Logs: reports/R1_retrieval_repair.log (heartbeat 60 s). Output: reports/retrieval_repair.json
Usage: python -m experiments.v3_retrieval_repair
"""
import json
import re
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.dataset as ds

from experiments.k50_common import atomic_json, bucket, mem_ok
from experiments.v3_common import REPORTS, Log, eval_truth
from src.blocking import CountryIndex, _blend, _text, _topk
from src.config import WORK_DIR

COLS = ["entity_id", "country", "business_name", "name_core", "name_alt", "addr_norm"]
PASSES = {"blend": ({"nw": 1.0, "nc": 1.0, "ad": 1.5}, 50),
          "A": ({"nw": 1.0, "nc": 1.0, "ad": 4.0}, 10),
          "N": ({"nc": 1.0}, 10)}
_NONASCII = re.compile(r"[^\x00-\x7f]")
CHUNK = 5000
M2_ORACLE10_B = 0.9864944990404427


def oracle(found_mask, pair_ent, n_ent, nt):
    found = np.bincount(pair_ent[found_mask], minlength=n_ent).astype(float)
    r = np.divide(found, nt, out=np.zeros_like(found), where=nt > 0)
    f = np.where(found > 0, 1.25 * r / (0.25 + r), 0.0)
    return np.where(nt == 0, 1.0, f)


if __name__ == "__main__":
    t0 = time.time()
    log = Log("R1_retrieval_repair")
    log.set_stage("ground truth + records")
    g = eval_truth(us=False)
    E, truth = g["E"], g["truth"]
    eix = pd.Series(np.arange(len(E)), index=E)
    nt = np.array([len(truth[e]) for e in E], float)
    isB = np.array([bucket(e, 2) == 1 for e in E])
    tp = pd.DataFrame([(q, e) for e in E for q in truth[e]], columns=["q_id", "s1_id"])
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=COLS)
    s1 = s1[s1.country == "India"].reset_index(drop=True)
    files = [str(WORK_DIR / f"train_s{k}.parquet") for k in (2, 3)]
    ids = pd.concat([ds.dataset(f).to_table(columns=["entity_id"], filter=ds.field("country") == "India").to_pandas()
                     for f in files], ignore_index=True).entity_id
    samp_ids = ids[(ids.map(bucket) < 25).to_numpy()].to_numpy()            # growth sample, test-like mix
    del ids
    want = pd.unique(np.concatenate([tp.q_id.to_numpy(dtype=object), samp_ids]))
    qall = pd.concat([ds.dataset(f).to_table(columns=COLS, filter=(ds.field("country") == "India") & ds.field("entity_id").isin(want)).to_pandas()
                      for f in files], ignore_index=True).drop_duplicates("entity_id").set_index("entity_id")
    missing = set(tp.q_id) - set(qall.index)
    tp = tp[~tp.q_id.isin(missing)].reset_index(drop=True)          # true pairs whose query is not India (none expected)
    q_own = qall.loc[tp.q_id].reset_index()
    samp = qall.loc[samp_ids].reset_index()
    del qall
    s1pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    owner_pos = s1pos.loc[tp.s1_id].to_numpy()
    pair_ent = eix.loc[tp.s1_id].to_numpy()
    empty_own = (q_own.addr_norm.fillna("").str.strip() == "").to_numpy()
    empty_samp = (samp.addr_norm.fillna("").str.strip() == "").to_numpy()
    vc = s1.name_core.value_counts()
    seg = {"common_name(s1 freq>=5)": s1.name_core.map(vc).to_numpy()[owner_pos] >= 5,
           "script": q_own.business_name.fillna("").map(lambda x: bool(_NONASCII.search(x))).to_numpy(),
           "empty_address": empty_own,
           "non_exact_name": q_own.name_core.fillna("").to_numpy() != s1.name_core.to_numpy()[owner_pos]}
    log(f"E {len(E)} entities, {len(tp)} true pairs (dropped {len(missing)} non-India), growth sample {len(samp)} queries "
        f"({int(empty_samp.sum())} empty-address) {mem_ok(tag='R1')}")

    log.set_stage("build India S1 index")
    idx = CountryIndex(s1, {"passes": [{"name": "blend", "w": PASSES["blend"][0], "k": 50}], "min_score": 0.05})
    log(f"index built {mem_ok(tag='R1')}")
    mq_own = {v: vec.transform(_text(q_own, v)) for v, vec in idx.vecs.items()}
    mq_samp = {v: vec.transform(_text(samp, v)) for v, vec in idx.vecs.items()}
    rank, cand, stats = {}, {}, {}
    for name, (w, k) in PASSES.items():
        log.set_stage(f"pass {name} top-{k}")
        t = time.time()
        BT = _blend(idx.m1, w).T.tocsr()
        A = _blend(mq_own, w)
        rk = np.full(len(tp), -1, np.int16)
        for i in range(0, len(tp), CHUNK):
            r = _topk(A[i:i + CHUNK], BT, k, 0.05)
            hit = r[r["s1_idx"].to_numpy() == owner_pos[i + r["q_idx"].to_numpy()]]
            rk[i + hit["q_idx"].to_numpy()] = hit["rank"].to_numpy()
        rank[name] = rk
        t_own = time.time() - t
        As = _blend(mq_samp, w)
        parts = []
        for i in range(0, As.shape[0], CHUNK):
            r = _topk(As[i:i + CHUNK], BT, k, 0.05)
            r["q_idx"] += i
            parts.append(r[["q_idx", "s1_idx", "rank"]])
        cand[name] = pd.concat(parts, ignore_index=True)
        stats[name] = {"k": k, "weights": w, "secs_owned_queries": round(t_own), "secs_total": round(time.time() - t),
                       "queries_per_sec": round((len(tp) + As.shape[0]) / (time.time() - t)),
                       "rss_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}
        log(f"  pass {name}: {stats[name]} {mem_ok(tag='R1')}")
        del BT, A, As
    # ---- candidate sets as membership tests
    b = rank["blend"]
    inb = lambda K: (b >= 0) & (b < K)
    inA = lambda K: (rank["A"] >= 0) & (rank["A"] < K)
    inN = lambda K: (rank["N"] >= 0) & (rank["N"] < K)
    variants = {
        "D_top10": inb(10), "blend_top20": inb(20), "blend_top50": inb(50),
        "top10+A5": inb(10) | inA(5), "top10+A10": inb(10) | inA(10),
        "top10+N5": inb(10) | inN(5), "top10+N10": inb(10) | inN(10),
        "top10+X(empty->50)": inb(10) | (empty_own & inb(50)),
        "top10+A5+N5+X": inb(10) | inA(5) | inN(5) | (empty_own & inb(50)),
        "top10+A10+N10+X": inb(10) | inA(10) | inN(10) | (empty_own & inb(50)),
    }
    # growth: new candidates per query beyond the base top-10, on the test-like sample
    cb = cand["blend"]
    base_key = set(zip(cb.q_idx[cb["rank"] < 10], cb.s1_idx[cb["rank"] < 10]))
    def growth(parts):
        u = pd.concat(parts, ignore_index=True)[["q_idx", "s1_idx"]].drop_duplicates()
        k = pd.Series(list(zip(u.q_idx, u.s1_idx)))
        return float((~k.isin(base_key)).sum() / len(samp))
    sel = lambda name, K, mask=None: (lambda c: c[(c["rank"] < K).to_numpy() & (np.ones(len(c), bool) if mask is None else mask[c.q_idx.to_numpy()])])(cand[name])
    gro = {"D_top10": 0.0, "blend_top20": growth([sel("blend", 20)]), "blend_top50": growth([sel("blend", 50)]),
           "top10+A5": growth([sel("A", 5)]), "top10+A10": growth([sel("A", 10)]),
           "top10+N5": growth([sel("N", 5)]), "top10+N10": growth([sel("N", 10)]),
           "top10+X(empty->50)": growth([sel("blend", 50, empty_samp)]),
           "top10+A5+N5+X": growth([sel("A", 5), sel("N", 5), sel("blend", 50, empty_samp)]),
           "top10+A10+N10+X": growth([sel("A", 10), sel("N", 10), sel("blend", 50, empty_samp)])}
    base_n = float(((cb["rank"] < 10).sum()) / len(samp))
    res = {}
    fB0 = oracle(variants["D_top10"], pair_ent, len(E), nt)[isB]
    for v, m in variants.items():
        fE = oracle(m, pair_ent, len(E), nt)
        r = {"recall_E": float(m.mean()), "recall_halfB": float(m[isB[pair_ent]].mean()),
             "found_halfB": int(m[isB[pair_ent]].sum()),
             "oracle_f05_E": float(fE.mean()), "oracle_f05_halfB": float(fE[isB].mean()),
             "oracle_gain_halfB_vs_D10": float(fE[isB].mean() - fB0.mean()),
             "new_candidates_per_query": gro[v], "total_candidates_per_query": base_n + gro[v],
             "recovered_D10_misses_halfB": int((m & ~variants["D_top10"])[isB[pair_ent]].sum()),
             "recall_by_segment": {s: float(m[mk].mean()) for s, mk in seg.items()}}
        res[v] = r
        log(f"  {v:22s} recall_B={r['recall_halfB']:.5f} oracle_B={r['oracle_f05_halfB']:.6f} "
            f"gain={r['oracle_gain_halfB_vs_D10']:+.6f} new/q={gro[v]:.2f} recovered={r['recovered_D10_misses_halfB']}")
    assert abs(res["D_top10"]["oracle_f05_halfB"] - M2_ORACLE10_B) < 2e-4, \
        f"STOP: D top-10 oracle {res['D_top10']['oracle_f05_halfB']} does not reproduce M2 {M2_ORACLE10_B}"
    out = {"population": "V-dense E true pairs (India)", "true_pairs": len(tp), "entities": len(E),
           "growth_sample_queries": len(samp), "base_candidates_per_query": base_n,
           "recall_blend_at": {K: float(inb(K).mean()) for K in (10, 20, 50)},
           "passes": stats, "variants": res, "d_top10_oracle_reproduced": res["D_top10"]["oracle_f05_halfB"],
           "m2_reference": M2_ORACLE10_B, "secs": round(time.time() - t0),
           "peak_rss_gb": round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**30, 2)}
    atomic_json(out, REPORTS / "retrieval_repair.json")
    log("RESULT " + json.dumps({v: (round(r["oracle_f05_halfB"], 6), round(r["oracle_gain_halfB_vs_D10"], 6),
                                     round(r["new_candidates_per_query"], 2)) for v, r in res.items()}))
    log(f"R1 done in {time.time() - t0:.0f}s")
