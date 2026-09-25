"""End-to-end validation of blocking -> features -> LightGBM -> assignment on a
training universe, with 2-fold cross-fitting so every pair gets an out-of-fold probability.

Threshold is tuned on fold-0 entities and reported on fold-1 entities (and vice versa).
Results are appended to experiments/match_results.jsonl.

Usage: python -m experiments.exp_match 0.2 [top_k]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.blocking import DEFAULT, retrieve_all
from src.config import WORK_DIR
from src.devset import candidate_recall, load_universe
from src.features import pair_features
from src.matching import sweep, train_model
from src.prep import s1_fold


def s1_name_freq(split="train"):
    s1 = pd.read_parquet(WORK_DIR / f"{split}_s1.parquet", columns=["country", "name_core"])
    return s1.value_counts().to_dict()


if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tag = sys.argv[3] if len(sys.argv) > 3 else "baseline"
    t0 = time.time()
    s1, q, truth = load_universe(frac)
    cfg = {**DEFAULT, "top_k": top_k}
    pairs = retrieve_all(s1, q, cfg)
    rec = candidate_recall(pairs, truth)
    print("blocking:", rec, flush=True)

    owner = {m: s for s, ms in truth.items() for m in ms}
    y = (pairs["q_id"].map(owner) == pairs["s1_id"]).to_numpy(np.int8)
    fold = pairs["s1_id"].map(s1_fold).to_numpy()

    t = time.time()
    X = pair_features(pairs, s1.set_index("entity_id"), q.set_index("entity_id"), s1_name_freq())
    print(f"features: {X.shape} in {time.time() - t:.0f}s", flush=True)

    prob = np.zeros(len(pairs), dtype=np.float64)
    models = []
    for f in (0, 1):
        tr = fold == f
        m = train_model(X[tr], y[tr], pairs["q_id"].to_numpy()[tr])
        prob[~tr] = m.predict(X[~tr], num_threads=12)
        models.append(m)
        print(f"fold {f}: trained on {tr.sum()} pairs, best_iter={m.best_iteration}", flush=True)

    ents = np.array(list(truth))
    ef = np.array([s1_fold(e) for e in ents])
    res = {}
    for tune_f, eval_f in ((0, 1), (1, 0)):
        sw = sweep(pairs, prob, truth, ents[ef == tune_f])
        best_t = float(sw.loc[sw.f05_macro.idxmax(), "threshold"])
        ev = sweep(pairs, prob, truth, ents[ef == eval_f], [best_t]).iloc[0].to_dict()
        res[f"eval_fold{eval_f}"] = ev
        print(f"tune fold {tune_f}:\n{sw[['threshold', 'f05_macro', 'singleton_acc', 'pair_precision', 'pair_recall']].to_string(index=False)}")
        print(f"eval fold {eval_f} @ {best_t}: {ev}", flush=True)

    imp = pd.Series(models[0].feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("feature importance (gain, top 20):\n", (imp / imp.sum()).head(20).round(4).to_string())
    line = {"tag": tag, "frac": frac, "top_k": top_k, "blocking": rec, **res,
            "n_features": X.shape[1], "secs": round(time.time() - t0)}
    with open(Path(__file__).with_name("match_results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=float) + "\n")
    # keep artefacts for error analysis
    out = pairs.assign(y=y, prob=prob)
    out.to_parquet(WORK_DIR / f"dev_pairs_{tag}.parquet", index=False)
    X.to_parquet(WORK_DIR / f"dev_X_{tag}.parquet", index=False)
