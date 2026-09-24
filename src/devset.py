"""Validation universes built from the training data.

A universe is a closed world: a set of S1 entities, all of their true S2/S3 matches, and
the same *fraction* of unmatched (distractor) S2/S3 records, so distractor density matches
the full data. Folds come from prep.s1_fold (0 = model training, 1 = held-out validation).

    sample_frac=1.0  -> the full training universe (most realistic, slowest)
    sample_frac=0.2  -> a 20 % universe for fast iteration
"""
import zlib

import pandas as pd

from .config import WORK_DIR
from .io_utils import read_ground_truth


def _bucket(x: str, mod: int) -> int:
    return (zlib.crc32(x.encode()) >> 1) % mod


def load_universe(sample_frac: float = 1.0):
    """Return (s1_df, q_df, truth) restricted to a deterministic sample of S1 entities."""
    gt = read_ground_truth()
    keep = lambda x: sample_frac >= 1.0 or _bucket(x, 1000) < int(sample_frac * 1000)
    truth = {s: ms for s, ms in gt.items() if keep(s)}
    matched_all = {m for ms in gt.values() for m in ms}
    matched_keep = {m for ms in truth.values() for m in ms}
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet")
    s1 = s1[s1.entity_id.isin(truth.keys())].reset_index(drop=True)
    qs = []
    for k in (2, 3):
        q = pd.read_parquet(WORK_DIR / f"train_s{k}.parquet")
        is_matched = q.entity_id.isin(matched_all)
        distractor = ~is_matched & q.entity_id.map(keep)
        qs.append(q[q.entity_id.isin(matched_keep) | distractor])
    q = pd.concat(qs, ignore_index=True)
    return s1, q, truth


def candidate_recall(cands: pd.DataFrame, truth: dict) -> dict:
    """Share of true (S1, S2/S3) pairs present in the candidate set, and the best
    macro-F0.5 a perfect matcher could reach on these candidates."""
    from .metrics import macro_f05
    got = cands.groupby("s1_id")["q_id"].agg(set).to_dict()
    oracle = {s: (truth[s] & got.get(s, set())) for s in truth}
    n_true = sum(len(v) for v in truth.values())
    n_hit = sum(len(v) for v in oracle.values())
    m = macro_f05(truth, oracle)
    return {"pair_recall": n_hit / max(n_true, 1), "oracle_f05": m["f05_macro"],
            "n_cand": len(cands), "cand_per_query": len(cands) / max(cands.q_id.nunique(), 1)}
