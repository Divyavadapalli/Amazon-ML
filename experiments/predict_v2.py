"""Test inference V2: gated top-K expansion with the frozen final_v1 model, to a NEW directory.

Policy ('adapt', h, kmax), identical to experiments/vdense.policy_mask: every query keeps its
retrieval ranks 0-2; a query whose best top-3 probability is below h also gets ranks
3..kmax-1 scored. Rank 0-2 scores are reused from the submitted run's checkpoints
(pred_final_v1_k3, bit-identical features); only the extra rows are featurised.

Per-query retrieval features of the extra rows are taken from the FULL retrieved list
(ret_top1, ret_gap_top1, ret_ncand), exactly as in training and in the submitted run; this
is asserted against a full-list computation on the first chunk.

The submitted files in OUTPUT_DIR are never read or written. Output: <out_dir>/
matching_results.tsv, candidate_pairs.tsv, prediction_stats.json.

Usage: python -m experiments.predict_v2 <h> <kmax> <threshold> <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.config import OUTPUT_DIR, WORK_DIR
from src.features import pair_features
from src.pipeline import REC, _write

MODEL = "final_v1"


def extra_features(c_all, s1i, qcache, freq):
    """Features for rows rank >= 3 of `c_all` (all rows of the gated queries, sorted by q_id,
    rank), with per-query retrieval features from the full list."""
    ex = c_all[c_all["rank"] >= 3].reset_index(drop=True)
    X = pair_features(ex, s1i, qcache, freq)
    g = c_all.groupby("q_id")["score"]
    top1 = ex["q_id"].map(g.max()).to_numpy(np.float32)
    X["ret_top1"] = top1
    X["ret_gap_top1"] = top1 - X["ret_score"].to_numpy(np.float32)
    X["ret_margin"] = 0.0                                   # rank != 0
    X["ret_ncand"] = ex["q_id"].map(g.size()).to_numpy(np.float32)
    return ex, X


if __name__ == "__main__":
    h, kmax, thr, out_dir = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    out_dir = out_dir.resolve()
    assert out_dir != OUTPUT_DIR.resolve(), "never overwrite the submitted output directory"
    t0 = time.time()
    meta = json.loads((WORK_DIR / f"m2_model_{MODEL}.meta.json").read_text())
    model = lgb.Booster(model_file=str(WORK_DIR / f"m2_model_{MODEL}.txt"))
    feats = meta["features"]
    assert model.feature_name() == feats
    d = WORK_DIR / f"cand_test_{meta['cand_tag']}"
    assert (d / "DONE").exists()
    k3 = WORK_DIR / f"pred_{MODEL}_k3"
    ck = WORK_DIR / f"pred_{MODEL}_adapt{h}_k{kmax}"
    ck.mkdir(exist_ok=True)

    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    s1_pos = pd.Index(s1_ids)
    s1i = s1.set_index("entity_id")
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])
                            ["entity_id"].to_numpy(dtype=object) for k in (2, 3)])
    q_pos = pd.Index(q_ids)
    freq = s1[["country", "name_core"]].value_counts().to_dict()

    cs, cq, cp, n_extra, checked = [], [], [], 0, False
    qcache, country_loaded = None, None
    for p in sorted(d.glob("*.parquet")):
        t = time.time()
        if (k3 / f"{p.stem}.npz").exists():
            z = np.load(k3 / f"{p.stem}.npz")
            si3, qi3, pr3 = z["si"], z["qi"], z["prob"]
        else:   # chunks the submitted run took from the full top-10 checkpoint (rank mask), as src.pipeline
            cr = pd.read_parquet(p, columns=["q_id", "rank"]).sort_values(["q_id", "rank"], kind="stable")
            z = np.load(WORK_DIR / f"pred_{MODEL}" / f"{p.stem}.npz")
            assert len(z["si"]) == len(cr), f"top-10 checkpoint misaligned in {p.name}"
            keep3 = cr["rank"].to_numpy() < 3
            si3, qi3, pr3 = z["si"][keep3], z["qi"][keep3], z["prob"][keep3]
        cf = ck / f"{p.stem}.npz"
        if cf.exists():
            e = np.load(cf)
            sie, qie, pre = e["si"], e["qi"], e["prob"]
            note = "checkpoint"
        else:
            c = pd.read_parquet(p).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            top = c[c["rank"] < 3]
            assert len(top) == len(si3), f"k3 checkpoint misaligned in {p.name}"
            assert (q_pos.get_indexer(top["q_id"]) == qi3).all() and (s1_pos.get_indexer(top["s1_id"]) == si3).all()
            best3 = pd.Series(pr3).groupby(top["q_id"].to_numpy()).max()
            gated = set(best3.index[best3.to_numpy() < h])
            c_all = c[c["q_id"].isin(gated).to_numpy() & (c["rank"].to_numpy() < kmax)].reset_index(drop=True)
            country = p.stem.rsplit("_", 1)[0]
            if country != country_loaded:
                qcache = pd.concat([ds.dataset(str(WORK_DIR / f"test_s{k}.parquet")).to_table(
                    columns=REC, filter=ds.field("country") == country).to_pandas() for k in (2, 3)]).set_index("entity_id")
                country_loaded = country
            # full per-query lists (all 10 ranks) for the per-query retrieval features
            full = c[c["q_id"].isin(gated).to_numpy()].reset_index(drop=True)
            ex = c_all[c_all["rank"] >= 3].reset_index(drop=True)
            if len(ex):
                _, X = extra_features(full, s1i, qcache, freq)
                keep = (full.loc[full["rank"] >= 3, "rank"].to_numpy() < kmax)
                X = X[keep].reset_index(drop=True)
                if not checked:          # equivalence with a full-list pair_features call
                    Xf = pair_features(full, s1i, qcache, freq)
                    Xf = Xf[(full["rank"].to_numpy() >= 3)][keep].reset_index(drop=True)
                    diff = (X[feats] - Xf[feats]).abs().to_numpy().max()
                    assert diff < 1e-4, f"extra-row features differ from full computation: {diff}"
                    checked = True
                pre = model.predict(X[feats], num_threads=12).astype(np.float32)
            else:
                pre = np.zeros(0, np.float32)
            sie = s1_pos.get_indexer(ex["s1_id"]).astype(np.int32)
            qie = q_pos.get_indexer(ex["q_id"]).astype(np.int32)
            assert (sie >= 0).all() and (qie >= 0).all()
            np.savez(ck / f"{p.stem}.tmp.npz", si=sie, qi=qie, prob=pre)
            (ck / f"{p.stem}.tmp.npz").replace(cf)
            note = f"{len(gated)} gated queries"
        n_extra += len(sie)
        cs += [si3, sie]; cq += [qi3, qie]; cp += [pr3, pre]
        print(f"  {p.name}: top3 {len(si3)} + extra {len(sie)} ({note}) {time.time() - t:.0f}s", flush=True)

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
    stats = {"model_tag": MODEL, "policy": ["adapt", h, kmax], "threshold": thr, "n_s1": len(s1_ids),
             "n_candidate_pairs": int(len(cs)), "n_extra_pairs": int(n_extra), "n_matches": int(len(ms)),
             "s1_with_match": int(len(np.unique(ms))), "candidates_per_s1": len(cs) / len(s1_ids),
             "secs": round(time.time() - t0)}
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
