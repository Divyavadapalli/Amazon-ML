"""Apply a frozen V3 decision rule to Submission 3's stored test probabilities (no new inference).

Reads the System-D test checkpoints WORK_DIR/pred_final_v2_adapt0.5_k10/*.npz (si, qi, prob of
every scored pair, same chunk order as predict_v3), then:
  1. REPRODUCTION TEST: rebuilds Submission 3 with its own rule (one-to-one, prob >= 0.80) and
     asserts both TSV sha256 values equal the submitted files (af65628d... / 970c6d89...).
     Nothing else runs if this fails.
  2. applies the rule given as JSON and writes matching_results.tsv + candidate_pairs.tsv
     (candidate set unchanged = exactly the scored pairs) into a NEW directory.
Rules (after one-to-one: each query -> its best S1):
  {"rule": "R0", "t": ..}                  accept p >= t
  {"rule": "R1", "t1": .., "t2": ..}       S1's best query needs t1, its other queries t2 (best must be accepted)
  {"rule": "R3", "t": .., "m": ..}         p >= t and p - (query's 2nd-best prob) >= m

Usage: python -m experiments.apply_decision_v3 <out_dir> '<rule json>'
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.k50_common import guard
from src.config import WORK_DIR
from src.pipeline import _write

CK = WORK_DIR / "pred_final_v2_adapt0.5_k10"
SUB3 = {"matching_results.tsv": "af65628dc037c5cec4b233282557fed68a07ac8fcba88547f9ca1cc18b965862",
        "candidate_pairs.tsv": "970c6d89effd7f78cfed0271f9536f8a22a49ccb2ec84026f84582d8b32165ca"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def accept(sel, p2, rule):
    """sel: one row per query (qi, si, prob) sorted by prob desc. Returns boolean mask."""
    p = sel["prob"].to_numpy()
    r = rule["rule"]
    if r == "R0":
        return p >= rule["t"]
    if r == "R3":
        return (p >= rule["t"]) & (p - p2 >= rule["m"])
    if r == "R1":
        si = sel["si"].to_numpy()
        first = ~pd.Series(si).duplicated().to_numpy()              # sel is prob-desc: first row per S1 = its best
        top_ok = pd.Series(p[first] >= rule["t1"], index=si[first])
        return np.where(first, p >= rule["t1"], (p >= rule["t2"]) & top_ok.reindex(si).to_numpy())
    raise ValueError(rule)


def build(cs, cq, cp, rule, q_n):
    df = pd.DataFrame({"qi": cq, "si": cs, "prob": cp}).sort_values("prob", ascending=False, kind="stable")
    k = df.groupby("qi").cumcount().to_numpy()
    p2 = pd.Series(df["prob"].to_numpy()[k == 1], index=df["qi"].to_numpy()[k == 1])
    sel = df[k == 0]
    m = accept(sel, sel["qi"].map(p2).fillna(0.0).to_numpy(), rule)
    sel = sel[m]
    return sel["si"].to_numpy(np.int32), sel["qi"].to_numpy(np.int32)


if __name__ == "__main__":
    out_dir = guard(Path(sys.argv[1]))
    rule = json.loads(sys.argv[2])
    t0 = time.time()
    s1_ids = pd.read_parquet(WORK_DIR / "test_s1.parquet", columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
    q_ids = np.concatenate([pd.read_parquet(WORK_DIR / f"test_s{k}.parquet", columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
                            for k in (2, 3)])
    files = sorted(p for p in CK.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    assert len(files) == 41, len(files)
    cs, cq, cp = [], [], []
    for f in files:
        z = np.load(f)
        cs.append(z["si"]); cq.append(z["qi"]); cp.append(z["prob"])
    cs, cq, cp = (np.concatenate(x) for x in (cs, cq, cp))
    ckey = cs.astype(np.int64) * len(q_ids) + cq
    assert len(np.unique(ckey)) == len(ckey), "duplicate candidate pairs"
    print(f"{len(cs)} scored pairs from {len(files)} checkpoints", flush=True)

    # 1. reproduction test of Submission 3
    rep = out_dir / "_repro_sub3"
    rep.mkdir(parents=True, exist_ok=True)
    ms, mq = build(cs, cq, cp, {"rule": "R0", "t": 0.80}, len(q_ids))
    _write(rep / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    _write(rep / "candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cs, cq, q_ids)
    for name, want in SUB3.items():
        got = sha(rep / name)
        assert got == want, f"REPRODUCTION FAILED for {name}: {got} != {want}"
    print("reproduction test PASSED: both Submission 3 files rebuilt byte-for-byte", flush=True)
    base_matches = len(ms)

    # 2. new rule
    ms, mq = build(cs, cq, cp, rule, len(q_ids))
    mkey = ms.astype(np.int64) * len(q_ids) + mq
    assert np.isin(mkey, ckey).all(), "a match is not in the candidate set"
    assert len(np.unique(mq)) == len(mq), "a query was assigned to two S1 entities"
    _write(out_dir / "matching_results.tsv", "matched_entity_ids", s1_ids, ms, mq, q_ids)
    (rep / "candidate_pairs.tsv").replace(out_dir / "candidate_pairs.tsv")     # identical candidate set
    (rep / "matching_results.tsv").unlink()
    rep.rmdir()
    assert sha(out_dir / "candidate_pairs.tsv") == SUB3["candidate_pairs.tsv"]
    stats = {"base": "Submission 3 / System D test probabilities (final_v2, adapt 0.5/10)", "rule": rule,
             "n_scored_pairs": int(len(cs)), "n_matches": int(len(ms)), "n_matches_sub3": int(base_matches),
             "s1_with_match": int(len(np.unique(ms))), "sha256": {n: sha(out_dir / n) for n in SUB3},
             "secs": round(time.time() - t0)}
    (out_dir / "prediction_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
