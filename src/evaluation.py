"""Fast, exact threshold sweeps for the official macro F0.5.

`sweep` evaluates every threshold of a grid in one vectorised pass:
  * one-to-one: each query keeps only its highest-probability S1 (threshold-independent,
    computed once); a threshold then only filters.
  * no constraint: every pair with prob >= threshold is a match.
Per-entity counts for all thresholds come from np.bincount. `check_against_reference`
re-scores one threshold with src.metrics.macro_f05 (the reference implementation) and
asserts agreement, so the fast path can never silently diverge from the metric.
"""
from typing import Dict, Sequence, Set

import numpy as np
import pandas as pd

from .metrics import macro_f05

GRID = np.round(np.arange(0.05, 0.991, 0.01), 3)


def predictions(pairs: pd.DataFrame, prob: np.ndarray, one_to_one: bool) -> pd.DataFrame:
    """Candidate predictions before thresholding: q_id, s1_id, prob (+ rank if present)."""
    p = pd.DataFrame({"q_id": pairs["q_id"].to_numpy(), "s1_id": pairs["s1_id"].to_numpy(),
                      "prob": np.asarray(prob, dtype=np.float64)})
    if "rank" in pairs:
        p["rank"] = pairs["rank"].to_numpy()
    if one_to_one:
        p = p.sort_values("prob", ascending=False, kind="stable").drop_duplicates("q_id")
    return p


def sweep(pred: pd.DataFrame, truth: Dict[str, Set[str]], entities: Sequence[str],
          grid=GRID) -> pd.DataFrame:
    """Macro F0.5 (and diagnostics) on `entities` for every threshold in `grid`."""
    ents = np.asarray(list(entities))
    eidx = pd.Series(np.arange(len(ents)), index=ents)
    n_true = np.array([len(truth[e]) for e in ents], dtype=np.float64)
    p = pred[pred["s1_id"].isin(eidx.index)]
    ei = eidx.loc[p["s1_id"]].to_numpy()
    owner = {m: e for e in ents for m in truth[e]}
    is_tp = (p["q_id"].map(owner).to_numpy() == p["s1_id"].to_numpy())
    prob = p["prob"].to_numpy()
    single = n_true == 0
    rows = []
    for t in grid:
        keep = prob >= t
        n_pred = np.bincount(ei[keep], minlength=len(ents)).astype(np.float64)
        tp = np.bincount(ei[keep & is_tp], minlength=len(ents)).astype(np.float64)
        prec = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
        rec = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
        den = 0.25 * prec + rec
        f = np.divide(1.25 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
        f[single] = (n_pred[single] == 0).astype(np.float64)
        multi = n_true >= 2
        rows.append({
            "threshold": float(t), "f05_macro": f.mean(),
            "tp": int(tp.sum()), "fp": int((n_pred - tp).sum()), "fn": int((n_true - tp).sum()),
            "pair_precision": tp.sum() / max(n_pred.sum(), 1), "pair_recall": tp.sum() / max(n_true.sum(), 1),
            "singleton_acc": f[single].mean() if single.any() else np.nan,
            "nonsingleton_f05": f[~single].mean(),
            "multi_f05": f[multi].mean() if multi.any() else np.nan,
            "n_pred_nonempty": int((n_pred > 0).sum()),
        })
    return pd.DataFrame(rows)


def check_against_reference(pred, truth, entities, threshold, fast_value, tol=1e-9):
    p = pred[pred["prob"] >= threshold]
    pm = p.groupby("s1_id")["q_id"].agg(set).to_dict()
    ref = macro_f05(truth, pm, entities)["f05_macro"]
    assert abs(ref - fast_value) < tol, f"fast sweep {fast_value} != reference {ref}"
    return ref


def tune_and_report(pairs, prob, truth, tune_ents, eval_ents, one_to_one=True, grid=GRID):
    """Pick the threshold on `tune_ents`, report every metric on the disjoint `eval_ents`."""
    pred = predictions(pairs, prob, one_to_one)
    st = sweep(pred, truth, tune_ents, grid)
    best = st.loc[st["f05_macro"].idxmax()]
    se = sweep(pred, truth, eval_ents, [best["threshold"]]).iloc[0].to_dict()
    check_against_reference(pred, truth, eval_ents, best["threshold"], se["f05_macro"])
    chosen = pred[pred["prob"] >= best["threshold"]]
    if "rank" in chosen:
        se["assign_rank_ge3_share"] = float((chosen["rank"] >= 3).mean()) if len(chosen) else 0.0
    se["tune_f05"] = float(best["f05_macro"])
    return se, st
