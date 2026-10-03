"""Test inference V3 = system D: final_v2 (49 + AR-7 features) with gated top-K expansion.

Per test candidate chunk (all 10 retrieved candidates of each query, sorted by q_id, rank):
  1. AR-7 features on the full candidate list (contrast features need every candidate),
     statistics from the TEST S1 table (as validation used the training S1 table)
  2. ranks 0-2: 49 base features exactly as src.pipeline (ret_ncand from the full list)
  3. gate: queries whose best top-3 probability < h also get ranks 3..kmax-1 scored; their
     per-query retrieval features come from the full list (as in training)
  4. checkpoint si / qi / prob (atomic) -> resumable
Then one-to-one assignment at the frozen threshold, hard checks, both TSVs to a NEW
directory. The submitted OUTPUT_DIR is never read or written.

Usage: python -m experiments.predict_v3 <out_dir>
"""
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from experiments.ar7_features import NAMES as AR7, _init, ar7_features, build_stats, s1_counts
from experiments.predict_v2 import extra_features
from src.config import OUTPUT_DIR, WORK_DIR
from src.features import pair_features
from src.pipeline import REC, _write

TAG = "final_v2"
STATS = WORK_DIR / "exp3" / "ar7_stats_test_v2.pkl"

if __name__ == "__main__":
    out_dir = Path(sys.argv[1]).resolve()
    assert out_dir != OUTPUT_DIR.resolve(), "never overwrite the submitted output directory"
    t0 = time.time()
    meta = json.loads((WORK_DIR / f"m2_model_{TAG}.meta.json").read_text())
    model = lgb.Booster(model_file=str(WORK_DIR / f"m2_model_{TAG}.txt"))
    feats, thr = meta["features"], meta["threshold"]
    _, h, kmax = meta["policy"]
    assert model.feature_name() == feats
    base = [f for f in feats if f not in AR7]
    d = WORK_DIR / f"cand_test_{meta['cand_tag']}"
    assert (d / "DONE").exists()
    ck = WORK_DIR / f"pred_{TAG}_adapt{h}_k{kmax}"
    ck.mkdir(exist_ok=True)

    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    assert len(set(s1_ids)) == len(s1_ids)
    s1_pos = pd.Index(s1_ids)
    s1i = s1.set_index("entity_id")
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])
                            ["entity_id"].to_numpy(dtype=object) for k in (2, 3)])
    assert len(set(q_ids)) == len(q_ids)
    q_pos = pd.Index(q_ids)
    freq = s1[["country", "name_core"]].value_counts().to_dict()
    if not STATS.exists():
        with open(STATS.with_suffix(".tmp"), "wb") as f:
            pickle.dump(build_stats(s1[["country", "name_core", "addr_norm"]]), f)
        STATS.with_suffix(".tmp").replace(STATS)
    s1c = s1_counts(s1i[["country", "name_core", "addr_norm"]])

    cs, cq, cp, n_extra = [], [], [], 0
    qcache, country_loaded = None, None
    pool = Pool(8, initializer=_init, initargs=(str(STATS),))
    for p in sorted(d.glob("*.parquet")):
        t = time.time()
        cf = ck / f"{p.stem}.npz"
        if cf.exists():
            z = np.load(cf)
            si, qi, pr, ne = z["si"], z["qi"], z["prob"], int(z["n_extra"])
            note = "checkpoint"
        else:
            c = pd.read_parquet(p).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            country = p.stem.rsplit("_", 1)[0]
            if country != country_loaded:
                qcache = pd.concat([ds.dataset(str(WORK_DIR / f"test_s{k}.parquet")).to_table(
                    columns=REC, filter=ds.field("country") == country).to_pandas() for k in (2, 3)]).set_index("entity_id")
                country_loaded = country
            A = ar7_features(c[["q_id", "s1_id"]], s1c, qcache, pool)          # aligned to c rows
            n_full = c.groupby("q_id")["q_id"].transform("size").to_numpy()
            top_m = c["rank"].to_numpy() < 3
            top = c[top_m].reset_index(drop=True)
            X = pair_features(top, s1i, qcache, freq)
            X["ret_ncand"] = n_full[top_m].astype(np.float32)
            X = pd.concat([X[base], A[top_m].reset_index(drop=True)], axis=1)[feats]
            p3 = model.predict(X, num_threads=12).astype(np.float32)
            best3 = pd.Series(p3).groupby(top["q_id"].to_numpy()).max()
            gated = set(best3.index[best3.to_numpy() < h])
            gm = c["q_id"].isin(gated).to_numpy()
            full = c[gm].reset_index(drop=True)
            ex_m = gm & (c["rank"].to_numpy() >= 3) & (c["rank"].to_numpy() < kmax)
            ex = c[ex_m].reset_index(drop=True)
            if len(ex):
                _, Xe = extra_features(full, s1i, qcache, freq)
                keep = full.loc[full["rank"] >= 3, "rank"].to_numpy() < kmax
                Xe = Xe[keep].reset_index(drop=True)
                Xe = pd.concat([Xe[base], A[ex_m].reset_index(drop=True)], axis=1)[feats]
                pe = model.predict(Xe, num_threads=12).astype(np.float32)
            else:
                pe = np.zeros(0, np.float32)
            both = pd.concat([top[["q_id", "s1_id"]], ex[["q_id", "s1_id"]]], ignore_index=True)
            si = s1_pos.get_indexer(both["s1_id"]).astype(np.int32)
            qi = q_pos.get_indexer(both["q_id"]).astype(np.int32)
            assert (si >= 0).all() and (qi >= 0).all(), f"unknown ids in {p.name}"
            pr, ne = np.concatenate([p3, pe]), len(ex)
            np.savez(ck / f"{p.stem}.tmp.npz", si=si, qi=qi, prob=pr, n_extra=ne)
            (ck / f"{p.stem}.tmp.npz").replace(cf)
            note = f"{len(gated)} gated queries"
        n_extra += ne
        cs.append(si); cq.append(qi); cp.append(pr)
        print(f"  {p.name}: {len(si)} pairs ({ne} extra, {note}) {time.time() - t:.0f}s", flush=True)
    pool.close(); pool.join()

    cs, cq, cp = (np.concatenate(x) for x in (cs, cq, cp))
    sel = pd.DataFrame({"qi": cq, "si": cs, "prob": cp})
    sel = sel.sort_values("prob", ascending=False, kind="stable").drop_duplicates("qi")
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
    stats = {"model_tag": TAG, "policy": meta["policy"], "threshold": thr, "n_s1": len(s1_ids),
             "n_candidate_pairs": int(len(cs)), "n_extra_pairs": int(n_extra), "n_matches": int(len(ms)),
             "s1_with_match": int(len(np.unique(ms))), "candidates_per_s1": len(cs) / len(s1_ids),
             "matches_by_country": pd.Series(s1["country"].to_numpy()[ms]).value_counts().to_dict(),
             "secs": round(time.time() - t0)}
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
