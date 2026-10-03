"""Test inference for system X = Submission 3 (System D) + empty-address depth-50.

Only for use after the R2 validation gate passes. Submission 3 is never read-modified: its
stored test probabilities (WORK_DIR/pred_final_v2_adapt0.5_k10/*.npz) are reused unchanged and
the new rows are ADDED:
  1. empty-address test queries (addr_norm empty) per country
  2. blend_ad15 retrieval with k = 50 for those queries only (same TF-IDF config as the test K10
     cache; parity check: their ranks 0-9 must equal cand_test_blend_ad15 for >= 99.9 % of pairs)
  3. ranks 10-49 featurised with k50_common.k50_features (top-10 context, test S1 statistics,
     exp3/ar7_stats_test_v2.pkl) -- the same path used in R2 validation
  4. scored by the production model final_v2 (Submission 3's model), checkpointed per country block
  5. union with Submission 3's scored pairs; one-to-one; threshold 0.80 (Submission 3's rule)
  6. matching_results.tsv + candidate_pairs.tsv (= exactly the scored pairs) in a NEW directory
Reproduction test first: Submission 3 is rebuilt byte-for-byte from its checkpoints.

Logs: C:\\aml2026\\v3_exp\\reports\\X_test.log
Usage: python -m experiments.predict_x_test <new_out_dir>
"""
import hashlib
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
from experiments.k50_common import guard, k50_features, mem_ok
from experiments.v3_common import Log
from src.blocking import CountryIndex
from src.candidates import config
from src.config import WORK_DIR
from src.pipeline import REC, _write

CK = WORK_DIR / "pred_final_v2_adapt0.5_k10"
XCK = Path(r"C:\aml2026\v3_exp\x_test_ckpt")
STATS = WORK_DIR / "exp3" / "ar7_stats_test_v2.pkl"
THR = 0.80
BLOCK = 400_000
SUB3 = {"matching_results.tsv": "af65628dc037c5cec4b233282557fed68a07ac8fcba88547f9ca1cc18b965862",
        "candidate_pairs.tsv": "970c6d89effd7f78cfed0271f9536f8a22a49ccb2ec84026f84582d8b32165ca"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assemble(cs, cq, cp):
    df = pd.DataFrame({"qi": cq, "si": cs, "prob": cp}).sort_values("prob", ascending=False, kind="stable").drop_duplicates("qi")
    df = df[df["prob"] >= THR]
    return df["si"].to_numpy(np.int32), df["qi"].to_numpy(np.int32)


if __name__ == "__main__":
    out_dir = guard(Path(sys.argv[1]))
    assert not (out_dir / "matching_results.tsv").exists(), "refusing to overwrite an existing output"
    t0 = time.time()
    log = Log("X_test")
    XCK.mkdir(parents=True, exist_ok=True)
    meta = json.loads((WORK_DIR / "m2_model_final_v2.meta.json").read_text())
    model = lgb.Booster(model_file=str(WORK_DIR / "m2_model_final_v2.txt"))
    feats = meta["features"]
    assert model.feature_name() == feats and meta["threshold"] == THR
    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    s1_pos = pd.Index(s1_ids)
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
                            for k in (2, 3)])
    q_pos = pd.Index(q_ids)

    log.set_stage("load Submission 3 checkpoints + reproduction test")
    files = sorted(p for p in CK.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    assert len(files) == 41
    zs = [np.load(f) for f in files]
    cs, cq, cp = (np.concatenate([z[k] for z in zs]) for k in ("si", "qi", "prob"))
    rep = out_dir / "_repro_sub3"
    rep.mkdir(parents=True, exist_ok=True)
    ms, mq = assemble(cs, cq, cp)
    _write(rep / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    _write(rep / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cs, cq, q_ids)
    for n, want in SUB3.items():
        assert sha(rep / n) == want, f"REPRODUCTION FAILED: {n}"
    for n in SUB3:
        (rep / n).unlink()
    rep.rmdir()
    log(f"reproduction test PASSED ({len(cs)} Submission-3 scored pairs) {mem_ok(tag='X')}")

    log.set_stage("X retrieval + features + scoring")
    s1i = s1.set_index("entity_id")
    s1c = s1_counts(s1i[["country", "name_core", "addr_norm"]])
    freq = s1[["country", "name_core"]].value_counts().to_dict()
    cfg = config("blend_ad15")
    cfg = {**cfg, "passes": [{**cfg["passes"][0], "k": 50}]}
    parity = {}
    new_s, new_q, new_p = [], [], []
    with Pool(6, initializer=_init, initargs=(str(STATS),)) as pool:
        for country in sorted(s1["country"].unique()):
            qd = pd.concat([ds.dataset(str(WORK_DIR / f"test_s{k}.parquet")).to_table(columns=REC, filter=ds.field("country") == country).to_pandas()
                            for k in (2, 3)], ignore_index=True)
            qe = qd[qd["addr_norm"].fillna("").str.strip() == ""].reset_index(drop=True)
            f_country = XCK / f"{country}.npz"
            if f_country.exists():
                z = np.load(f_country)
                new_s.append(z["si"]); new_q.append(z["qi"]); new_p.append(z["prob"])
                parity[country] = json.loads(str(z["parity"]))
                log(f"  {country}: checkpoint ({len(z['si'])} new pairs)")
                continue
            s1c_ = s1[s1["country"] == country].reset_index(drop=True)
            t = time.time()
            r = CountryIndex(s1c_, cfg).query(qe)
            r.insert(0, "q_id", qe["entity_id"].to_numpy()[r.pop("q_idx").to_numpy()])
            r.insert(1, "s1_id", s1c_["entity_id"].to_numpy()[r.pop("s1_idx").to_numpy()])
            r = r[["q_id", "s1_id", "score", "rank"]].sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            # parity: ranks 0-9 == Submission 3's K10 candidates for these queries
            top = r[r["rank"] < 10]
            k10 = pd.concat([pd.read_parquet(p, columns=["q_id", "s1_id"], filters=[("q_id", "in", list(set(qe.entity_id)))])
                             for p in sorted((WORK_DIR / "cand_test_blend_ad15").glob(f"{country}_*.parquet"))], ignore_index=True)
            same = len(set(zip(top.q_id, top.s1_id)) & set(zip(k10.q_id, k10.s1_id)))
            parity[country] = {"empty_address_queries": len(qe), "top10_pairs": len(top), "k10_cache_pairs": len(k10),
                               "overlap": same, "overlap_share": same / max(len(k10), 1)}
            assert parity[country]["overlap_share"] >= 0.999, f"parity failed {parity[country]}"
            rows = r[r["rank"] >= 10].reset_index(drop=True)
            log(f"  {country}: {len(qe)} empty-address queries, {len(rows)} new rows, parity {parity[country]['overlap_share']:.5f} "
                f"({time.time() - t:.0f}s) {mem_ok(tag='X')}")
            qcache = qe.set_index("entity_id")
            si_c, qi_c, p_c = [], [], []
            qv = rows["q_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]]) if len(rows) else np.array([0])
            i = 0
            while i < len(rows):
                j = starts[np.searchsorted(starts, i + BLOCK)] if i + BLOCK < starts[-1] else len(rows)
                blk = rows.iloc[i:j].reset_index(drop=True)
                X = k50_features(blk, top[top.q_id.isin(set(blk.q_id))], s1i, qcache, freq, s1c, pool)
                p_c.append(model.predict(X[feats], num_threads=12).astype(np.float32))
                si_c.append(s1_pos.get_indexer(blk["s1_id"]).astype(np.int32))
                qi_c.append(q_pos.get_indexer(blk["q_id"]).astype(np.int32))
                log(f"    {country} rows {j}/{len(rows)} {mem_ok(tag='X')}")
                i = j
            si_c = np.concatenate(si_c) if si_c else np.zeros(0, np.int32)
            qi_c = np.concatenate(qi_c) if qi_c else np.zeros(0, np.int32)
            p_c = np.concatenate(p_c) if p_c else np.zeros(0, np.float32)
            assert (si_c >= 0).all() and (qi_c >= 0).all()
            np.savez(XCK / f"{country}.tmp.npz", si=si_c, qi=qi_c, prob=p_c, parity=json.dumps(parity[country]))
            (XCK / f"{country}.tmp.npz").replace(f_country)
            new_s.append(si_c); new_q.append(qi_c); new_p.append(p_c)

    log.set_stage("assemble + write")
    ns, nq, npr = (np.concatenate(x) for x in (new_s, new_q, new_p))
    ckey = cs.astype(np.int64) * len(q_ids) + cq
    nkey = ns.astype(np.int64) * len(q_ids) + nq
    assert not np.isin(nkey, ckey).any(), "a new pair duplicates a Submission-3 pair"
    CS, CQ, CP = np.concatenate([cs, ns]), np.concatenate([cq, nq]), np.concatenate([cp, npr])
    allk = CS.astype(np.int64) * len(q_ids) + CQ
    assert len(np.unique(allk)) == len(allk)
    ms, mq = assemble(CS, CQ, CP)
    assert np.isin(ms.astype(np.int64) * len(q_ids) + mq, allk).all() and len(np.unique(mq)) == len(mq)
    _write(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    _write(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, CS, CQ, q_ids)
    base_ms, base_mq = assemble(cs, cq, cp)
    stats = {"system": "X = Submission 3 (final_v2, adapt 0.5/10, thr 0.80) + empty-address blend top-50 (ranks 10-49 scored)",
             "threshold": THR, "n_scored_pairs": int(len(CS)), "n_new_pairs": int(len(ns)), "n_matches": int(len(ms)),
             "n_matches_sub3": int(len(base_ms)), "matches_changed_vs_sub3": int(len(set(zip(ms, mq)) ^ set(zip(base_ms, base_mq)))),
             "new_pair_matches": int(np.isin(ms.astype(np.int64) * len(q_ids) + mq, nkey).sum()),
             "parity": parity, "sha256": {n: sha(out_dir / n) for n in SUB3}, "secs": round(time.time() - t0)}
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    log("RESULT " + json.dumps({k: v for k, v in stats.items() if k != "parity"}))
