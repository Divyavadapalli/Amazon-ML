"""End-to-end: train the final matcher on training data, then produce the two
submission files for the test set.

    python -m src.prep                      # once: normalised parquet
    python -m src.pipeline train  [frac]    # model + threshold -> WORK_DIR/model.txt, model_meta.json
    python -m src.pipeline predict          # -> OUTPUT_DIR/matching_results.tsv, candidate_pairs.tsv

Test inference runs per country and in chunks of queries, so the ~50M candidate pairs never
sit in memory at once. Every S2/S3 record contributes its top-K S1 candidates to
candidate_pairs.tsv (exactly the pairs the model scores) and at most one match to
matching_results.tsv.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import DEFAULT, retrieve
from .config import OUTPUT_DIR, WORK_DIR
from .devset import load_universe
from .features import pair_features
from .io_utils import write_id_lists
from .matching import train_model

CHUNK = 1_000_000


def s1_name_freq(split: str) -> dict:
    s1 = pd.read_parquet(WORK_DIR / f"{split}_s1.parquet", columns=["country", "name_core"])
    return s1.value_counts().to_dict()


def _pairs_for_country(s1c, qc, cfg):
    r = retrieve(s1c, qc, cfg)
    return pd.DataFrame({"q_id": qc["entity_id"].values[r["q_idx"].values],
                         "s1_id": s1c["entity_id"].values[r["s1_idx"].values],
                         "score": r["score"].values, "rank": r["rank"].values})


def train(frac: float, threshold: float, top_k: int):
    s1, q, truth = load_universe(frac)
    cfg = {**DEFAULT, "top_k": top_k}
    owner = {m: s for s, ms in truth.items() for m in ms}
    freq = s1_name_freq("train")
    s1i, qi = s1.set_index("entity_id"), q.set_index("entity_id")
    Xs, ys, gs = [], [], []
    for c in sorted(set(s1.country)):
        p = _pairs_for_country(s1[s1.country == c].reset_index(drop=True),
                               q[q.country == c].reset_index(drop=True), cfg)
        Xs.append(pair_features(p, s1i, qi, freq))
        ys.append((p["q_id"].map(owner) == p["s1_id"]).to_numpy(np.int8))
        gs.append(p["q_id"].to_numpy())
    X = pd.concat(Xs, ignore_index=True)
    model = train_model(X, np.concatenate(ys), np.concatenate(gs))
    model.save_model(str(WORK_DIR / "model.txt"))
    meta = {"threshold": threshold, "top_k": top_k, "blocking": cfg, "features": list(X.columns),
            "train_frac": frac, "best_iteration": model.best_iteration}
    (WORK_DIR / "model_meta.json").write_text(json.dumps(meta, indent=1))
    print("saved model", meta)


def predict():
    meta = json.loads((WORK_DIR / "model_meta.json").read_text())
    model = lgb.Booster(model_file=str(WORK_DIR / "model.txt"))
    cfg, thr = meta["blocking"], meta["threshold"]
    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet")
    q = pd.concat([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet") for k in (2, 3)],
                  ignore_index=True)
    freq = s1_name_freq("test")
    s1i, qi = s1.set_index("entity_id"), q.set_index("entity_id")
    cands, matches = {}, {}
    for c in sorted(set(s1.country) | set(q.country)):
        s1c = s1[s1.country == c].reset_index(drop=True)
        qc_all = q[q.country == c].reset_index(drop=True)
        if not len(s1c) or not len(qc_all):
            continue
        for start in range(0, len(qc_all), CHUNK):
            t = time.time()
            qc = qc_all.iloc[start:start + CHUNK].reset_index(drop=True)
            p = _pairs_for_country(s1c, qc, cfg)
            X = pair_features(p, s1i, qi, freq)[meta["features"]]
            p["prob"] = model.predict(X, num_threads=12)
            for s, qq in zip(p["s1_id"].values, p["q_id"].values):
                cands.setdefault(s, []).append(qq)
            best = p.sort_values("prob", ascending=False, kind="stable").drop_duplicates("q_id")
            best = best[best["prob"] >= thr]
            for s, qq in zip(best["s1_id"].values, best["q_id"].values):
                matches.setdefault(s, []).append(qq)
            print(f"  {c} queries {start}-{start + len(qc)}: {len(p)} pairs, "
                  f"{len(best)} matches, {time.time() - t:.0f}s", flush=True)
    ids = s1["entity_id"].tolist()
    write_id_lists(OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids", ids, matches)
    write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", ids, cands)
    n_m = sum(map(len, matches.values()))
    print(f"wrote {len(ids)} rows; {n_m} matches; {len(matches)} S1 with >=1 match; "
          f"{sum(map(len, cands.values()))} candidate pairs")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(float(sys.argv[2]) if len(sys.argv) > 2 else 0.3,
              float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
              int(sys.argv[4]) if len(sys.argv) > 4 else 5)
    else:
        predict()
