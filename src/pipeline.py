"""Test-set inference: cached candidates -> features -> LightGBM -> assignment -> TSVs.

    python -m src.prep                          # once: normalised parquet
    python -m src.candidates test <cand_tag>    # resumable full candidate generation
    python -m src.pipeline predict <model_tag> <cand_tag> <threshold>

candidate_pairs.tsv is written from exactly the pairs the model scores (the final candidate
set), and matching_results.tsv only from those pairs, so every match is a candidate.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, WORK_DIR
from .features import pair_features
from .io_utils import write_id_lists

REC = ["entity_id", "country", "name_norm", "name_core", "name_alt", "addr_norm", "addr_digits",
       "pobox", "business_name", "business_address"]


def s1_name_freq(split: str) -> dict:
    s1 = pd.read_parquet(WORK_DIR / f"{split}_s1.parquet", columns=["country", "name_core"])
    return s1.value_counts().to_dict()


def predict(model_tag: str, cand_tag: str, threshold: float, one_to_one: bool = True):
    model = lgb.Booster(model_file=str(WORK_DIR / f"m2_model_{model_tag}.txt"))
    feats = model.feature_name()
    s1 = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=REC)
    s1_ids = s1["entity_id"].tolist()
    s1i = s1.set_index("entity_id")
    freq = s1_name_freq("test")
    d = WORK_DIR / f"cand_test_{cand_tag}"
    assert (d / "DONE").exists(), "run src.candidates test first"
    qcache = {}
    cands, matches, n_pairs = {}, {}, 0
    for p in sorted(d.glob("*.parquet")):
        t = time.time()
        c = pd.read_parquet(p).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
        country = p.stem.rsplit("_", 1)[0]
        if country not in qcache:          # load only this country's query records
            qcache.clear()
            q = pd.concat([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=REC) for k in (2, 3)])
            qcache[country] = q[q["country"] == country].set_index("entity_id")
        X = pair_features(c, s1i, qcache[country], freq)[feats]
        prob = model.predict(X, num_threads=12)
        for s, qq in zip(c["s1_id"].to_numpy(), c["q_id"].to_numpy()):
            cands.setdefault(s, []).append(qq)
        sel = pd.DataFrame({"q_id": c["q_id"].to_numpy(), "s1_id": c["s1_id"].to_numpy(), "prob": prob})
        if one_to_one:
            sel = sel.sort_values("prob", ascending=False, kind="stable").drop_duplicates("q_id")
        sel = sel[sel["prob"] >= threshold]
        for s, qq in zip(sel["s1_id"].to_numpy(), sel["q_id"].to_numpy()):
            matches.setdefault(s, []).append(qq)
        n_pairs += len(c)
        print(f"  {p.name}: {len(c)} pairs, {len(sel)} matches, {time.time() - t:.0f}s", flush=True)

    # ---- sanity checks before writing
    s1_set = set(s1_ids)
    assert len(s1_set) == len(s1_ids), "duplicate S1 ids in test_source1"
    assert set(matches) <= s1_set and set(cands) <= s1_set
    all_q = [x for v in matches.values() for x in v]
    assert all(x.startswith(("S2-", "S3-")) for x in all_q), "non S2/S3 id in matches"
    if one_to_one:
        assert len(all_q) == len(set(all_q)), "a query was assigned to two S1 entities"
    for s, v in matches.items():
        assert set(v) <= set(cands[s]), f"match outside candidate set for {s}"
    write_id_lists(OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids", s1_ids, matches)
    write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cands)
    print(f"wrote {len(s1_ids)} rows; {len(all_q)} matches for {len(matches)} S1 "
          f"({len(matches) / len(s1_ids):.3f} non-empty); {n_pairs} candidate pairs")


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        predict(sys.argv[2], sys.argv[3], float(sys.argv[4]))
