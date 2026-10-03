"""K50 test inference: final_k50 model, staged scoring over the compact K50 test cache.

Per chunk (checkpointed atomically, resumable):
  stage 1: ranks 0-2 of every query                       -> best-so-far p
  later stages from the model meta "policy" (the gate chosen on V-dense half-A), e.g.
     {"family": "B", "h": 0.5}                 ranks 3-9  if p_top3 < h
     {"family": "C"|"D", "h": ..}              ranks 3-19 / 3-49 if p_top3 < h
     {"family": "E", "h1":.., "lo":.., "h2":..} 3-9 if p3<h1; 10-19 if p10 in [lo,h2); 20-49 if p20 in [lo,h2)
Features: k50_common.k50_features with the query's top-10 as context (plan D1); AR-7 test
statistics from the TEST S1 table. candidate_pairs.tsv = exactly the scored pairs.
Writes into <out_dir> (never OUTPUT_DIR / earlier submissions: k50_common.guard).

Usage: python -m experiments.predict_k50 <out_dir>
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from experiments.ar7_features import _init, s1_counts
from experiments.k50_common import K50_ROOT, guard, k50_features, mem_ok
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.pipeline import REC, _write

TAG = "final_k50"
STATS = WORK_DIR / "exp3" / "ar7_stats_test_v2.pkl"
CAND = K50_ROOT / "cand_test"


def stages(policy):
    """[(rank_lo, rank_hi, gate(pbest) -> bool array)] in order; pbest = best prob so far."""
    f = policy["family"]
    if f == "A":
        return []
    if f in ("B", "C", "D"):
        hi = {"B": 10, "C": 20, "D": 50}[f]
        return [(3, hi, lambda p, h=policy["h"]: p < h)]
    h1, lo, h2 = policy["h1"], policy["lo"], policy["h2"]
    return [(3, 10, lambda p: p < h1), (10, 20, lambda p: (p >= lo) & (p < h2)), (20, 50, lambda p: (p >= lo) & (p < h2))]


if __name__ == "__main__":
    out_dir = guard(Path(sys.argv[1]))
    t0 = time.time()
    meta = json.loads((WORK_DIR / f"m2_model_{TAG}.meta.json").read_text())
    model = lgb.Booster(model_file=str(WORK_DIR / f"m2_model_{TAG}.txt"))
    feats, thr, policy = meta["features"], meta["threshold"], meta["policy"]
    assert model.feature_name() == feats
    assert (CAND / "DONE").exists(), "test K50 retrieval incomplete"
    ck = K50_ROOT / "test_ckpt"
    ck.mkdir(exist_ok=True)
    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    s1_pos = pd.Index(s1_ids)
    s1i = s1.set_index("entity_id")
    s1c = s1_counts(s1i[["country", "name_core", "addr_norm"]])
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
                            for k in (2, 3)])
    q_pos = pd.Index(q_ids)
    freq = s1[["country", "name_core"]].value_counts().to_dict()
    cs, cq, cp, reach = [], [], [], {}
    qcache, country_loaded = None, None
    with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
        for p in sorted(p for p in CAND.glob("*_*.parquet") if not p.name.startswith("index")):
            t = time.time()
            cf = ck / f"{p.stem}.npz"
            if cf.exists():
                z = np.load(cf)
                si, qi, pr = z["si"], z["qi"], z["prob"]
                reach[p.stem] = json.loads(str(z["reach"]))
            else:
                c = load_compact_chunk(CAND, p.stem).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
                country = p.stem.rsplit("_", 1)[0]
                if country != country_loaded:
                    qcache = pd.concat([ds.dataset(str(WORK_DIR / f"test_s{k}.parquet")).to_table(
                        columns=REC, filter=ds.field("country") == country).to_pandas() for k in (2, 3)]).set_index("entity_id")
                    country_loaded = country
                ctx = c[c["rank"] < 10]
                rows = c[c["rank"] < 3].reset_index(drop=True)
                X = k50_features(rows, ctx, s1i, qcache, freq, s1c, pool)
                probs = [model.predict(X[feats], num_threads=12).astype(np.float32)]
                scored = [rows[["q_id", "s1_id"]]]
                best = pd.Series(probs[0]).groupby(rows["q_id"].to_numpy()).max()
                rc = {"queries": int(c.q_id.nunique())}
                prev_open = None                     # nested stages, exactly as k50_eval.masks
                for lo_r, hi_r, gate in stages(policy):
                    open_q = set(best.index[gate(best.to_numpy())])
                    if prev_open is not None:
                        open_q &= prev_open
                    prev_open = open_q
                    st = c[c["q_id"].isin(open_q).to_numpy() & (c["rank"].to_numpy() >= lo_r) & (c["rank"].to_numpy() < hi_r)].reset_index(drop=True)
                    rc[f"stage_{lo_r}_{hi_r}_queries"] = len(open_q)
                    if not len(st):
                        continue
                    Xs = k50_features(st, ctx[ctx["q_id"].isin(open_q)], s1i, qcache, freq, s1c, pool)
                    ps = model.predict(Xs[feats], num_threads=12).astype(np.float32)
                    probs.append(ps); scored.append(st[["q_id", "s1_id"]])
                    best = pd.concat([best, pd.Series(ps).groupby(st["q_id"].to_numpy()).max()]).groupby(level=0).max()
                allp = pd.concat(scored, ignore_index=True)
                pr = np.concatenate(probs)
                si = s1_pos.get_indexer(allp["s1_id"]).astype(np.int32)
                qi = q_pos.get_indexer(allp["q_id"]).astype(np.int32)
                assert (si >= 0).all() and (qi >= 0).all(), f"unknown ids in {p.name}"
                np.savez(ck / f"{p.stem}.tmp.npz", si=si, qi=qi, prob=pr, reach=json.dumps(rc))
                (ck / f"{p.stem}.tmp.npz").replace(cf)
                reach[p.stem] = rc
            cs.append(si); cq.append(qi); cp.append(pr)
            print(f"  {p.name}: {len(si)} scored pairs {reach[p.stem]} {time.time() - t:.0f}s {mem_ok(tag='predict_k50')}", flush=True)
    cs, cq, cp = (np.concatenate(x) for x in (cs, cq, cp))
    sel = pd.DataFrame({"qi": cq, "si": cs, "prob": cp}).sort_values("prob", ascending=False, kind="stable").drop_duplicates("qi")
    sel = sel[sel["prob"] >= thr]
    ms, mq = sel["si"].to_numpy(np.int32), sel["qi"].to_numpy(np.int32)
    ckey = cs.astype(np.int64) * len(q_ids) + cq
    mkey = ms.astype(np.int64) * len(q_ids) + mq
    assert len(np.unique(ckey)) == len(ckey), "duplicate candidate pairs"
    assert np.isin(mkey, ckey).all(), "a match is not in the candidate set"
    assert len(np.unique(mq)) == len(mq), "a query was assigned to two S1 entities"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    _write(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cs, cq, q_ids)
    stats = {"model_tag": TAG, "policy": policy, "threshold": thr, "n_s1": len(s1_ids), "n_scored_pairs": int(len(cs)),
             "scored_pairs_per_query": len(cs) / len(q_ids), "n_matches": int(len(ms)), "s1_with_match": int(len(np.unique(ms))),
             "candidates_per_s1": len(cs) / len(s1_ids), "stage_reach": reach, "secs": round(time.time() - t0)}
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "stage_reach"}, indent=1))
