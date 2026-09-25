"""Stage 4: pair classifier (LightGBM, MIT licence) and match assignment.

Assignment rule: every S2/S3 record is linked to at most one S1 entity -- the candidate
with the highest match probability -- and only if that probability clears a threshold
tuned for macro F0.5. This mirrors the training data, where no S2/S3 record belongs to
two S1 entities, and removes a large class of false merges for free.
"""
from typing import Dict, Iterable, Set

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import N_JOBS, SEED
from .metrics import macro_f05

LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.08, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": N_JOBS, "seed": SEED,
    "verbose": -1,
}


def train_model(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, rounds=1500):
    """Train with early stopping on a 10 % group split of the *training* rows
    (groups = query ids), so the validation fold is never touched."""
    rng = np.random.default_rng(SEED)
    uq = np.unique(groups)
    es = set(rng.choice(uq, size=max(1, len(uq) // 10), replace=False))
    m = np.fromiter((g in es for g in groups), bool, len(groups))
    dtr = lgb.Dataset(X[~m], y[~m])
    dva = lgb.Dataset(X[m], y[m], reference=dtr)
    return lgb.train(LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])


def assign(pairs: pd.DataFrame, prob: np.ndarray, threshold: float) -> Dict[str, Set[str]]:
    """Keep, for each query, only its best S1 and only if prob >= threshold."""
    p = pairs[["q_id", "s1_id"]].copy()
    p["prob"] = prob
    best = p.sort_values("prob", ascending=False, kind="stable").drop_duplicates("q_id")
    best = best[best["prob"] >= threshold]
    return best.groupby("s1_id")["q_id"].agg(set).to_dict()


def sweep(pairs, prob, truth, entities: Iterable[str], thresholds=None) -> pd.DataFrame:
    """Macro F0.5 on `entities` for each threshold."""
    ents = list(entities)
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.2, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = assign(pairs, prob, t)
        r = macro_f05(truth, pred, ents)
        r["threshold"] = t
        rows.append(r)
    return pd.DataFrame(rows)
