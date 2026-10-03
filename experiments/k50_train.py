"""K50 Phase E: V-dense K50 matchers (India, leakage-free training pairs from k50_build_train).

Models (same LightGBM params / early-stopping design as src.matching.train_model):
  M7_K10reg   the D regime: the same clean queries, ranks 0-9 only (control, system 1)
  K50_r1.0    all ranks 0-49 (full K50 training)
  K50_r0.5    ranks >= 10 negatives kept with deterministic hash rate 0.5, weight 2
  K50_r0.25   rate 0.25, weight 4
All positives and every rank 0-9 row are always kept. Seed 42 for all; seed 7 for M7_K10reg
and K50_r1.0 (reproducibility). Saved to K50_ROOT/model/<name>_s<seed>.txt + .meta.json.

Usage: python -m experiments.k50_train <bucket> [names...]
"""
import json
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments.k50_common import K50_ROOT, atomic_json, band_counts, fingerprint, mem_ok
from src.matching import LGB_PARAMS

META_COLS = ["q_id", "s1_id", "score", "rank", "y"]
PLAN = [("M7_K10reg", None, 42), ("K50_r1.0", 1.0, 42), ("K50_r0.5", 0.5, 42), ("K50_r0.25", 0.25, 42),
        ("M7_K10reg", None, 7), ("K50_r1.0", 1.0, 7)]


def keep_rate(q, s):
    """Deterministic uniform in [0, 1) per pair."""
    return np.array([(zlib.crc32(f"{a}|{b}".encode()) & 0xFFFFFF) / 0x1000000 for a, b in zip(q, s)])


def train_w(X, y, w, groups, seed):
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    es = set(rng.choice(uq, size=max(1, len(uq) // 10), replace=False))
    m = np.fromiter((g in es for g in groups), bool, len(groups))
    params = {**LGB_PARAMS, "seed": seed}
    dtr = lgb.Dataset(X[~m], y[~m], weight=w[~m])
    dva = lgb.Dataset(X[m], y[m], weight=w[m], reference=dtr)
    return lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


if __name__ == "__main__":
    b = int(sys.argv[1])
    only = set(sys.argv[2:])
    src = K50_ROOT / "train" / f"b{b}"
    assert (src / "DONE").exists(), "run k50_build_train first"
    out = K50_ROOT / "model"
    out.mkdir(parents=True, exist_ok=True)
    D = pd.concat([pd.read_parquet(p) for p in sorted(src.glob("block_*.parquet"))], ignore_index=True)
    feats = [c for c in D.columns if c not in META_COLS]
    print(f"training rows {len(D)} features {len(feats)} {mem_ok(tag='k50_train')}", flush=True)
    u = keep_rate(D.q_id.to_numpy(), D.s1_id.to_numpy())
    rank, y = D["rank"].to_numpy(), D["y"].to_numpy()
    for name, r, seed in PLAN:
        if only and name not in only:
            continue
        f = out / f"{name}_s{seed}.txt"
        if f.exists():
            print(f"{f.name}: exists", flush=True)
            continue
        t = time.time()
        if r is None:
            m = rank < 10
            w = np.ones(int(m.sum()), np.float32)
        else:
            m = (rank < 10) | (y == 1) | (u < r)
            w = np.where((rank[m] >= 10) & (y[m] == 0), 1.0 / r, 1.0).astype(np.float32)
        Xm = D.loc[m, feats]
        model = train_w(Xm, y[m], w, D.q_id.to_numpy()[m], seed)
        model.save_model(str(f))
        meta = {"name": name, "seed": seed, "budget_rank_ge10": r, "E_bucket": b, "rows": int(m.sum()),
                "positives": int(y[m].sum()), "negatives": int((y[m] == 0).sum()), "positive_rate": float(y[m].mean()),
                "by_band": band_counts(rank[m], y[m]), "weights": {"rank_ge10_negative": (1.0 / r) if r else None},
                "by_source": {s: int((D.q_id[m].str[:2] == s).sum()) for s in ("S2", "S3")}, "country": {"India": int(m.sum())},
                "features": feats, "lgb_params": {**LGB_PARAMS, "seed": seed}, "best_iteration": model.best_iteration,
                "fingerprint": fingerprint(D.q_id.to_numpy()[m], D.s1_id.to_numpy()[m]), "train_secs": round(time.time() - t)}
        atomic_json(meta, out / f"{name}_s{seed}.meta.json")
        print(f"{name} s{seed}: rows {meta['rows']} pos {meta['positives']} iters {model.best_iteration} "
              f"{meta['train_secs']}s {mem_ok(tag='k50_train')}", flush=True)
