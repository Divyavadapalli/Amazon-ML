"""Test inference V4 = two-seed 'max' ensemble of System D (V3 experiment E1).

Structure identical to experiments/predict_v3.py (System D, Submission 3) with two changes:
  * every scored pair is scored by final_v2 (seed 42) AND final_v2_s7 (seed 7); prob = max(p42, p7)
  * the gate for ranks 3..9 uses the ensemble probability: best top-3 max-prob < h (0.5)
Threshold 0.84 = Submission 3's 0.80 + the +0.04 shift measured on V-dense half-A
(D 0.84 -> max 0.88); this transfer is an assumption, stated in the report.
Per-chunk checkpoints (si, qi, p42, p7, prob) in WORK_DIR/pred_final_v2ens_max_adapt0.5_k10
(atomic, resumable). Writes into a NEW directory; refuses OUTPUT_DIR and any existing output.

Usage: python -m experiments.predict_v4_ens <out_dir>
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

from experiments.ar7_features import NAMES as AR7, _init, ar7_features, s1_counts
from experiments.k50_common import guard, mem_ok
from experiments.predict_v2 import extra_features
from experiments.v3_common import Log
from src.config import WORK_DIR
from src.features import pair_features
from src.pipeline import REC, _write

TAGS = ("final_v2", "final_v2_s7")
STATS = WORK_DIR / "exp3" / "ar7_stats_test_v2.pkl"
THR = 0.84

if __name__ == "__main__":
    out_dir = guard(Path(sys.argv[1]))
    assert not (out_dir / "matching_results.tsv").exists(), "refusing to overwrite an existing output"
    t0 = time.time()
    log = Log("V4_ens_test")
    metas = [json.loads((WORK_DIR / f"m2_model_{t}.meta.json").read_text()) for t in TAGS]
    models = [lgb.Booster(model_file=str(WORK_DIR / f"m2_model_{t}.txt")) for t in TAGS]
    feats = metas[0]["features"]
    assert all(m.feature_name() == feats for m in models) and metas[1]["features"] == feats
    _, h, kmax = metas[0]["policy"]
    base = [f for f in feats if f not in AR7]
    d = WORK_DIR / f"cand_test_{metas[0]['cand_tag']}"
    assert (d / "DONE").exists()
    ck = WORK_DIR / f"pred_final_v2ens_max_adapt{h}_k{kmax}"
    ck.mkdir(exist_ok=True)

    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    s1_pos = pd.Index(s1_ids)
    s1i = s1.set_index("entity_id")
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])
                            ["entity_id"].to_numpy(dtype=object) for k in (2, 3)])
    q_pos = pd.Index(q_ids)
    freq = s1[["country", "name_core"]].value_counts().to_dict()
    assert STATS.exists()
    s1c = s1_counts(s1i[["country", "name_core", "addr_norm"]])

    def score(X):
        p = [m.predict(X, num_threads=12).astype(np.float32) for m in models]
        return p[0], p[1], np.maximum(p[0], p[1])

    cs, cq, cp, c42, c7, n_extra = [], [], [], [], [], 0
    qcache, country_loaded = None, None
    pool = Pool(6, initializer=_init, initargs=(str(STATS),))
    files = sorted(d.glob("*.parquet"))
    for n, p in enumerate(files):
        t = time.time()
        cf = ck / f"{p.stem}.npz"
        if cf.exists():
            z = np.load(cf)
            si, qi, pr, p42, p7, ne = z["si"], z["qi"], z["prob"], z["p42"], z["p7"], int(z["n_extra"])
            note = "checkpoint"
        else:
            c = pd.read_parquet(p).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            country = p.stem.rsplit("_", 1)[0]
            if country != country_loaded:
                qcache = pd.concat([ds.dataset(str(WORK_DIR / f"test_s{k}.parquet")).to_table(
                    columns=REC, filter=ds.field("country") == country).to_pandas() for k in (2, 3)]).set_index("entity_id")
                country_loaded = country
            A = ar7_features(c[["q_id", "s1_id"]], s1c, qcache, pool)
            n_full = c.groupby("q_id")["q_id"].transform("size").to_numpy()
            top_m = c["rank"].to_numpy() < 3
            top = c[top_m].reset_index(drop=True)
            X = pair_features(top, s1i, qcache, freq)
            X["ret_ncand"] = n_full[top_m].astype(np.float32)
            X = pd.concat([X[base], A[top_m].reset_index(drop=True)], axis=1)[feats]
            a42, a7, p3 = score(X)
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
                e42, e7, pe = score(Xe)
            else:
                e42 = e7 = pe = np.zeros(0, np.float32)
            both = pd.concat([top[["q_id", "s1_id"]], ex[["q_id", "s1_id"]]], ignore_index=True)
            si = s1_pos.get_indexer(both["s1_id"]).astype(np.int32)
            qi = q_pos.get_indexer(both["q_id"]).astype(np.int32)
            assert (si >= 0).all() and (qi >= 0).all(), f"unknown ids in {p.name}"
            pr, p42, p7, ne = np.concatenate([p3, pe]), np.concatenate([a42, e42]), np.concatenate([a7, e7]), len(ex)
            np.savez(ck / f"{p.stem}.tmp.npz", si=si, qi=qi, prob=pr, p42=p42, p7=p7, n_extra=ne)
            (ck / f"{p.stem}.tmp.npz").replace(cf)
            note = f"{len(gated)} gated queries"
        n_extra += ne
        cs.append(si); cq.append(qi); cp.append(pr); c42.append(p42); c7.append(p7)
        log(f"  [{n + 1}/{len(files)}] {p.name}: {len(si)} pairs ({ne} extra, {note}) {time.time() - t:.0f}s {mem_ok(tag='V4')}")
    pool.close(); pool.join()

    log.set_stage("assemble + write")
    cs, cq, cp, c42, c7 = (np.concatenate(x) for x in (cs, cq, cp, c42, c7))
    sel = pd.DataFrame({"qi": cq, "si": cs, "prob": cp}).sort_values("prob", ascending=False, kind="stable").drop_duplicates("qi")
    sel = sel[sel["prob"] >= THR]
    ms, mq = sel["si"].to_numpy(np.int32), sel["qi"].to_numpy(np.int32)
    ckey = cs.astype(np.int64) * len(q_ids) + cq
    mkey = ms.astype(np.int64) * len(q_ids) + mq
    assert len(np.unique(ckey)) == len(ckey), "duplicate candidate pairs"
    assert np.isin(mkey, ckey).all(), "a match is not in the candidate set"
    assert len(np.unique(mq)) == len(mq), "a query was assigned to two S1 entities"
    # comparison with Submission 3 (its stored checkpoints; read-only)
    sub3 = [np.load(f) for f in sorted((WORK_DIR / "pred_final_v2_adapt0.5_k10").glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    s3s, s3q, s3p = (np.concatenate([z[k] for z in sub3]) for k in ("si", "qi", "prob"))
    s3sel = pd.DataFrame({"qi": s3q, "si": s3s, "prob": s3p}).sort_values("prob", ascending=False, kind="stable").drop_duplicates("qi")
    s3sel = s3sel[s3sel["prob"] >= 0.80]
    s3key = set(s3sel["si"].astype(np.int64) * len(q_ids) + s3sel["qi"])
    newkey = set(mkey)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    _write(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cs, cq, q_ids)
    stats = {"system": "V4 = max(final_v2, final_v2_s7), gate on ensemble top-3 < 0.5 for ranks 3-9, threshold 0.84",
             "models": list(TAGS), "threshold": THR, "policy": metas[0]["policy"], "n_s1": len(s1_ids),
             "n_candidate_pairs": int(len(cs)), "n_extra_pairs": int(n_extra), "n_matches": int(len(ms)),
             "s1_with_match": int(len(np.unique(ms))), "candidates_per_s1": len(cs) / len(s1_ids),
             "matches_by_country": pd.Series(s1["country"].to_numpy()[ms]).value_counts().to_dict(),
             "vs_submission3": {"sub3_matches": len(s3key), "common": len(s3key & newkey), "only_v4": len(newkey - s3key),
                                "only_sub3": len(s3key - newkey)},
             "p42_p7_corr": float(np.corrcoef(c42, c7)[0, 1]), "secs": round(time.time() - t0)}
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    log("RESULT " + json.dumps(stats))
