"""V3 / M6: does more (clean) training data improve the D matcher on V-dense?

Training queries: India train queries with hash bucket < QB_MAX permille, taken from the K50
compact cache restricted to ranks 0-9 (== the K10 lists, parity T1), EXCLUDING every V-dense
evaluation query (ev_pairs + ev_ext_pairs, i.e. owned by E or with an E member in the top 10)
and, as a hard assert, any query owned by E or with an E candidate.  Features: the one
parity-tested path experiments.k50_common.k50_features (49 base + 21 AR-7, top-10 context).
Nested training sizes by query bucket: SIZES permille.  Models: exp3.train (LGB_PARAMS, seed).
Evaluation: identical to exp3c (V-dense E, densify seed 0, D policy adapt 0.5/10, threshold
tuned on half-A, reported on half-B, paired CI vs the exp3c D model M7c_s42 on the same rows).

Checkpoints: C:\\aml2026\\v3_exp\\lc\\block_<i>.parquet (resumable), models in v3_exp\\lc\\models.
Output: C:\\aml2026\\v3_exp\\reports\\learning_curve.json

Usage: python -m experiments.v3_learning_curve
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from experiments import vdense as V
from experiments.ar7_features import NAMES as AR7, _init, s1_counts
from experiments.exp3 import predict, train
from experiments.exp_match2 import REC
from experiments.k50_common import K50_ROOT, atomic_json, atomic_parquet, bucket, k50_features, labels, mem_ok
from experiments.v3_decision import ADAPT
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_records

ROOT = Path(r"C:\aml2026\v3_exp")
LC = ROOT / "lc"
CAND = K50_ROOT / "cand_train"
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"
QB_MAX = 300
SIZES = (50, 150, 300)
BLOCK = 1_000_000
EXP = WORK_DIR / "exp3"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_pairs(owner, E):
    evq = set(pd.read_parquet(V.CACHE / "ev_pairs.parquet", columns=["q_id"]).q_id) | \
        set(pd.read_parquet(EXP / "ev_ext_pairs.parquet", columns=["q_id"]).q_id)
    Es = set(E)
    sel = []
    for p in sorted(CAND.glob("India_*.parquet")):
        c = load_compact_chunk(CAND, p.stem)
        c = c[(c["rank"] < 10).to_numpy()]
        qb = c["q_id"].map(bucket).to_numpy()
        c = c[qb < QB_MAX]
        touch = set(c.loc[c["s1_id"].isin(Es).to_numpy(), "q_id"])
        keep = ~c["q_id"].isin(evq).to_numpy() & ~c["q_id"].isin(touch).to_numpy() & ~c["q_id"].map(owner).isin(Es).to_numpy()
        sel.append(c.loc[keep, ["q_id", "s1_id", "score", "rank"]])
    pairs = pd.concat(sel, ignore_index=True).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
    assert not pairs["s1_id"].isin(Es).any() and not pairs["q_id"].map(owner).isin(Es).any() and not pairs.q_id.isin(evq).any()
    return pairs


if __name__ == "__main__":
    t0 = time.time()
    (LC / "models").mkdir(parents=True, exist_ok=True)
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    from experiments.v3_decision import entities_india
    E = entities_india(None, gt)
    pairs = build_pairs(owner, E)
    y = labels(pairs, owner)
    log(f"clean training pool: {pairs.q_id.nunique()} queries / {len(pairs)} pairs / {int(y.sum())} positives", mem_ok(tag="M6"))
    # ---- features, blocks of whole queries
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    s1c = s1_counts(s1[["country", "name_core", "addr_norm"]])
    freq = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"]).value_counts().to_dict()
    qv = pairs["q_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    bounds, i = [], 0
    while i < len(pairs):
        j = starts[np.searchsorted(starts, i + BLOCK)] if i + BLOCK < starts[-1] else len(pairs)
        bounds.append((i, j))
        i = j
    with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
        for n, (i, j) in enumerate(bounds):
            f = LC / f"block_{n:03d}.parquet"
            if f.exists():
                continue
            t = time.time()
            rows = pairs.iloc[i:j].reset_index(drop=True)
            q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(rows.q_id), REC) for k in (2, 3)]).set_index("entity_id")
            X = k50_features(rows, rows, s1, q, freq, s1c, pool)
            atomic_parquet(pd.concat([rows[["q_id", "s1_id", "rank"]], pd.Series(y[i:j], name="y"), X], axis=1), f)
            log(f"  block {n + 1}/{len(bounds)}: {len(rows)} pairs {time.time() - t:.0f}s {mem_ok(tag='M6')}")
    del s1, s1c, freq, pairs
    # ---- train nested sizes
    D = pd.concat([pd.read_parquet(LC / f"block_{n:03d}.parquet") for n in range(len(bounds))], ignore_index=True)
    feats = [c for c in D.columns if c not in ("q_id", "s1_id", "rank", "y")]
    qb = D["q_id"].map(bucket).to_numpy()
    info = {}
    for size in SIZES:
        for seed in ((42, 7) if size == SIZES[-1] else (42,)):
            name = f"LC{size}_s{seed}"
            mf = LC / "models" / f"{name}.txt"
            m = qb < size
            info[name] = {"queries": int(D.q_id[m].nunique()), "pairs": int(m.sum()), "positives": int(D.y[m].sum())}
            if not mf.exists():
                t = time.time()
                mdl = train(D.loc[m, feats], D.y.to_numpy()[m], D.q_id.to_numpy()[m], seed)
                mdl.save_model(str(mf))
                info[name]["train_secs"] = round(time.time() - t)
            info[name]["trees"] = lgb.Booster(model_file=str(mf)).num_trees()
            log(name, info[name], mem_ok(tag="M6"))
    del D
    # ---- V-dense evaluation, exactly as exp3c
    ev, Xev, truth, A, B, owner = V.load_india(with_ext=True)
    A7 = pd.concat([pd.read_parquet(EXP / "ar7_ev_cache.parquet"), pd.read_parquet(EXP / "ar7_ev_ext.parquet")], ignore_index=True)[AR7]
    X = pd.concat([Xev, A7], axis=1)
    del Xev, A7
    P = pd.read_parquet(EXP / "probs_clean_india.parquet", columns=["q_id", "s1_id", "M7c_s42", "M7c_s7"])
    assert (P.q_id.to_numpy() == ev.q_id.to_numpy()).all() and (P.s1_id.to_numpy() == ev.s1_id.to_numpy()).all()
    out = V.slim(ev, X)
    out["M7c_s42"], out["M7c_s7"] = P["M7c_s42"].to_numpy(), P["M7c_s7"].to_numpy()
    del P
    for name in info:
        out[name] = predict(lgb.Booster(model_file=str(LC / "models" / f"{name}.txt")), X).astype(np.float32)
    del X, ev
    Pd, _ = V.densify(out)
    del out
    res, fv = {}, {}
    for name in ["M7c_s42", "M7c_s7", *info]:
        r, f, _ = V.evaluate(Pd, Pd[name].to_numpy(), truth, A, B, owner, ADAPT, name)
        r2, _, _ = V.evaluate(Pd, Pd[name].to_numpy(), truth, B, A, owner, ADAPT, name + "_sw")
        r["f05_swapped_halves"] = r2["f05"]
        res[name], fv[name] = r, f
        log(name, json.dumps({k: r[k] for k in ("f05", "precision", "recall", "fp", "fn", "threshold", "f05_swapped_halves")}))
    for name in info:
        base = "M7c_s7" if name.endswith("_s7") else "M7c_s42"
        g, lo, hi = V.paired_ci(fv[name], fv[base])
        res[name].update({"base": base, "gain": g, "ci_lo": lo, "ci_hi": hi,
                          "gain_swapped": res[name]["f05_swapped_halves"] - res[base]["f05_swapped_halves"], **info[name]})
    atomic_json({"results": res, "secs": round(time.time() - t0)}, ROOT / "reports" / "learning_curve.json")
    cols = ["f05", "gain", "ci_lo", "ci_hi", "gain_swapped", "precision", "recall", "fp", "fn", "threshold", "pairs", "queries"]
    print(pd.DataFrame(res).T.reindex(columns=cols).to_string())
