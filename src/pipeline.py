"""Test-set inference: cached candidates -> features -> frozen LightGBM -> assignment -> TSVs.

    python -m src.prep                          # once: normalised parquet
    python -m src.candidates test <cand_tag>    # resumable full candidate generation
    python -m src.pipeline predict <model_tag>  # threshold/one-to-one read from the model meta

candidate_pairs.tsv is written from exactly the pairs the model scores (the final candidate
set) and matching_results.tsv only from those pairs, so every match is a candidate.

Memory: ~100M candidate pairs are held as int32 row indices into the test tables (not as
Python strings, which would need several GB) and IDs are only materialised when writing,
copied verbatim from the input files.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import OUTPUT_DIR, WORK_DIR
from .features import pair_features

REC = ["entity_id", "country", "name_norm", "name_core", "name_alt", "addr_norm", "addr_digits",
       "pobox", "business_name", "business_address"]


def _write(path, header_col, s1_ids, s_idx, q_idx, q_ids):
    """One row per S1 (in test_source1 order); comma-joined ids; empty when none."""
    order = np.lexsort((q_idx, s_idx))
    s_sorted, q_sorted = s_idx[order], q_idx[order]
    bounds = np.searchsorted(s_sorted, np.arange(len(s1_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for i, s in enumerate(s1_ids):
            a, b = bounds[i], bounds[i + 1]
            f.write(s + "\t" + ",".join(q_ids[q_sorted[a:b]]) + "\n")


def predict(model_tag: str, out_dir=OUTPUT_DIR, only_countries=None):
    """only_countries: smoke-test mode on a subset of complete country chunks, written to a
    scratch directory (never to OUTPUT_DIR)."""
    meta = json.loads((WORK_DIR / f"m2_model_{model_tag}.meta.json").read_text())
    model = lgb.Booster(model_file=str(WORK_DIR / f"m2_model_{model_tag}.txt"))
    feats, thr, one_to_one = meta["features"], meta["threshold"], meta["one_to_one"]
    assert model.feature_name() == feats, "model/meta feature mismatch"
    d = WORK_DIR / f"cand_test_{meta['cand_tag']}"
    if only_countries:
        assert out_dir != OUTPUT_DIR, "smoke-test mode must not write to OUTPUT_DIR"
    else:
        assert (d / "DONE").exists(), "test candidate generation incomplete"

    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    assert len(set(s1_ids)) == len(s1_ids), "duplicate S1 ids in test_source1"
    s1_pos = pd.Index(s1_ids)
    s1i = s1.set_index("entity_id")
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])
                            ["entity_id"].to_numpy(dtype=object) for k in (2, 3)])
    assert len(set(q_ids)) == len(q_ids), "duplicate S2/S3 ids in test sources"
    q_pos = pd.Index(q_ids)
    freq = s1[["country", "name_core"]].value_counts().to_dict()

    # Per-chunk scores are checkpointed (atomic write) so a crash resumes at the next chunk;
    # the frozen threshold / assignment is applied once, below, to the stored probabilities.
    ckpt = WORK_DIR / f"pred_{model_tag}"   # chunk scores depend only on (model, chunk): shared by smoke and final runs
    ckpt.mkdir(exist_ok=True)
    cs, cq, cp = [], [], []
    qcache, country_loaded, t0 = None, None, time.time()
    files = sorted(d.glob("*.parquet"))
    if only_countries:
        files = [f for f in files if f.stem.rsplit("_", 1)[0] in only_countries or f.stem in only_countries]
    for p in files:
        t = time.time()
        cf = ckpt / f"{p.stem}.npz"
        if cf.exists():
            z = np.load(cf)
            si, qi, prob = z["si"], z["qi"], z["prob"]
            print(f"  {p.name}: {len(si)} pairs (checkpoint)", flush=True)
        else:
            c = pd.read_parquet(p).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
            country = p.stem.rsplit("_", 1)[0]
            if country != country_loaded:          # load only this country's query records
                qcache = pd.concat([ds.dataset(str(WORK_DIR / f"test_s{k}.parquet")).to_table(
                    columns=REC, filter=ds.field("country") == country).to_pandas() for k in (2, 3)]
                ).set_index("entity_id")
                country_loaded = country
            X = pair_features(c, s1i, qcache, freq)[feats]
            prob = model.predict(X, num_threads=12).astype(np.float32)
            si = s1_pos.get_indexer(c["s1_id"]).astype(np.int32)
            qi = q_pos.get_indexer(c["q_id"]).astype(np.int32)
            assert (si >= 0).all() and (qi >= 0).all(), f"unknown ids in {p.name}"
            tmp = ckpt / f"{p.stem}.tmp.npz"
            np.savez(tmp, si=si, qi=qi, prob=prob)
            tmp.replace(cf)
            print(f"  {p.name}: {len(si)} pairs scored in {time.time() - t:.0f}s", flush=True)
        cs.append(si); cq.append(qi); cp.append(prob)

    cs, cq, cp = (np.concatenate(x) for x in (cs, cq, cp))
    sel = pd.DataFrame({"qi": cq, "si": cs, "prob": cp})
    if one_to_one:
        sel = sel.sort_values("prob", ascending=False, kind="stable").drop_duplicates("qi")
    sel = sel[sel["prob"] >= thr]
    ms, mq = sel["si"].to_numpy(np.int32), sel["qi"].to_numpy(np.int32)
    # ---- hard checks before writing anything
    ckey = cs.astype(np.int64) * len(q_ids) + cq
    mkey = ms.astype(np.int64) * len(q_ids) + mq
    assert len(np.unique(ckey)) == len(ckey), "duplicate candidate pairs across chunks"
    assert np.isin(mkey, ckey).all(), "a match is not in the candidate set"
    if one_to_one:
        assert len(np.unique(mq)) == len(mq), "a query was assigned to two S1 entities"
    assert all(str(x).startswith(("S2-", "S3-")) for x in q_ids[np.unique(mq)]), "non S2/S3 match id"

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    _write(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cs, cq, q_ids)
    stats = {"model_tag": model_tag, "threshold": thr, "one_to_one": one_to_one,
             "n_s1": len(s1_ids), "n_candidate_pairs": int(len(cs)), "n_matches": int(len(ms)),
             "s1_with_match": int(len(np.unique(ms))), "share_s1_with_match": len(np.unique(ms)) / len(s1_ids),
             "matches_per_matched_s1": len(ms) / max(len(np.unique(ms)), 1),
             "secs": round(time.time() - t0)}
    by_c = pd.Series(s1["country"].to_numpy()[ms]).value_counts().to_dict()
    stats["matches_by_country"] = by_c
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2])
    elif sys.argv[1] == "smoke":            # python -m src.pipeline smoke <model_tag> <country,...>
        predict(sys.argv[2], WORK_DIR / "smoke_output", sys.argv[3].split(","))
