"""Exp-3: baseline vs adaptive top-K vs AR-7 vs adaptive top-K + AR-7, one protocol (V-dense).

Stages (each checkpointed under WORK_DIR/exp3; a crash resumes at the missing stage):
  1. AR-7 features for training pairs, India eval pairs (cache + extension) and the US sample
  2. models: M0 = the 49-feature baseline (the saved india_v2 model that produced the 0.9628
     measurement, plus a retrain as a determinism check), M7 = 49 + AR-7 features, both on the
     identical training pairs; second seed for both (reproducibility)
  3. probabilities -> slim frame -> densify -> every system through vdense.evaluate
Results: experiments/exp3_results.jsonl (one line per system) and a printed decision table.

Usage: python -m experiments.exp3
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from experiments import vdense as V
from experiments.ar7_features import NAMES as AR7
from src.config import WORK_DIR
from src.matching import LGB_PARAMS

EXP = WORK_DIR / "exp3"
RES = Path(__file__).with_name("exp3_results.jsonl")
T = {}


def peak_gb():
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ar7_for(name):
    """AR-7 features written by experiments/exp3_ar7.py (stage 1), row-aligned to the pair file."""
    f = EXP / f"ar7_{name}.parquet"
    assert f.exists(), f"run python -m experiments.exp3_ar7 first ({f.name} missing)"
    return pd.read_parquet(f)


def train(X, y, groups, seed):
    """src.matching.train_model with an explicit seed (seed 42 == train_model exactly)."""
    rng = np.random.default_rng(seed)
    uq = np.unique(groups)
    es = set(rng.choice(uq, size=max(1, len(uq) // 10), replace=False))
    m = np.fromiter((g in es for g in groups), bool, len(groups))
    params = {**LGB_PARAMS, "seed": seed}
    dtr = lgb.Dataset(X[~m], y[~m])
    dva = lgb.Dataset(X[m], y[m], reference=dtr)
    return lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def predict(model, X, step=2_000_000):
    cols = model.feature_name()
    return np.concatenate([model.predict(X.iloc[i:i + step][cols], num_threads=12) for i in range(0, len(X), step)])


if __name__ == "__main__":
    t0 = time.time()
    EXP.mkdir(exist_ok=True)
    # ---------------- data
    ev, Xev, truth, A, B, owner = V.load_india(with_ext=True)
    tr = pd.read_parquet(V.CACHE / "tr_pairs.parquet")
    Xtr = pd.read_parquet(V.CACHE / "Xtr.parquet")
    ytr = (tr["q_id"].map(owner) == tr["s1_id"]).to_numpy(np.int8)
    us, Xus, truth_us, A_us, B_us, _ = V.load_us()
    log(f"India eval pairs {len(ev)} (cache {int((ev.part == 'cache').sum())}, ext {int((ev.part == 'ext').sum())}); "
        f"entities A {len(A)} B {len(B)}; train pairs {len(tr)}; US pairs {len(us)} entities B {len(B_us)}")

    # ---------------- 1. AR-7 features
    A7tr = ar7_for("tr")
    A7ev = pd.concat([ar7_for("ev_cache"), ar7_for("ev_ext")], ignore_index=True)
    A7us = ar7_for("us")
    assert len(A7tr) == len(tr) and len(A7ev) == len(ev) and len(A7us) == len(us)

    # ---------------- 2. models
    base49 = list(Xtr.columns)
    Xtr7 = pd.concat([Xtr, A7tr], axis=1)
    models = {}
    t = time.time()
    models["M0_s42"] = lgb.Booster(model_file=str(WORK_DIR / "m2_model_india_v2.txt"))
    assert models["M0_s42"].feature_name() == base49
    models["M0_s42_retrain"] = train(Xtr, ytr, tr.q_id.to_numpy(), 42)
    T["train_M0_secs"] = round(time.time() - t); t = time.time()
    models["M7_s42"] = train(Xtr7, ytr, tr.q_id.to_numpy(), 42)
    T["train_M7_secs"] = round(time.time() - t)
    models["M0_s7"] = train(Xtr, ytr, tr.q_id.to_numpy(), 7)
    models["M7_s7"] = train(Xtr7, ytr, tr.q_id.to_numpy(), 7)
    for k, m in models.items():
        m.save_model(str(EXP / f"model_{k}.txt"))
    log(f"models trained; best iters " + str({k: m.best_iteration for k, m in models.items()}))
    imp = pd.Series(models["M7_s42"].feature_importance("gain"), index=Xtr7.columns)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    log("M7 top features:", imp.head(20).round(4).to_dict())
    log("AR-7 share of gain:", round(float(imp[AR7].sum()), 4))
    del Xtr, Xtr7

    # ---------------- 3. probabilities
    def probs(ev_, X49, A7):
        out = V.slim(ev_, X49)
        X = pd.concat([X49, A7], axis=1)          # one copy; each model selects its columns
        for k, m in models.items():
            out[k] = predict(m, X).astype(np.float32)
        return out
    t = time.time()
    P = probs(ev, Xev, A7ev)
    T["predict_secs_india"] = round(time.time() - t)
    del Xev, A7ev, ev
    cache_ref = pd.read_parquet(WORK_DIR / "m2_eval_india_v2.parquet", columns=["q_id", "s1_id", "prob"])
    pc = P[P.part == "cache"]
    d = np.abs(pc["M0_s42"].to_numpy() - cache_ref["prob"].to_numpy()) if (pc.q_id.to_numpy() == cache_ref.q_id.to_numpy()).all() else None
    log(f"saved-model probs vs cached 0.9628 run: max abs diff {'order mismatch' if d is None else f'{float(d.max()):.2e}'}; "
        f"retrain vs saved: max abs diff {float(np.abs(P['M0_s42_retrain'] - P['M0_s42']).max()):.2e}")
    Pus = probs(us, Xus, A7us)
    del Xus, A7us

    Pd, n_clone = V.densify(P)
    Pusd, n_clone_us = V.densify(Pus)
    log(f"densified: India {len(P)} -> {len(Pd)} rows ({n_clone} clones); US {len(Pus)} -> {len(Pusd)}; peak {peak_gb()} GB")
    del P, Pus

    # ---------------- evaluate every system with the same code
    ADAPT = ("adapt", 0.5, 10)       # pre-registered from the architecture review
    systems = [("A_baseline", "M0_s42", "k3"), ("B_topk", "M0_s42", ADAPT),
               ("C_ar7", "M7_s42", "k3"), ("D_topk_ar7", "M7_s42", ADAPT),
               ("A_baseline_retrain", "M0_s42_retrain", "k3"),
               ("A_seed7", "M0_s7", "k3"), ("B_seed7", "M0_s7", ADAPT),
               ("C_seed7", "M7_s7", "k3"), ("D_seed7", "M7_s7", ADAPT),
               ("menu_M0_k10", "M0_s42", ("k", 10)), ("menu_M0_adapt_0.5_k5", "M0_s42", ("adapt", 0.5, 5)),
               ("menu_M0_adapt_0.5_k6", "M0_s42", ("adapt", 0.5, 6)), ("menu_M0_adapt_0.2_k10", "M0_s42", ("adapt", 0.2, 10)),
               ("menu_M7_adapt_0.5_k5", "M7_s42", ("adapt", 0.5, 5)), ("menu_M7_adapt_0.2_k10", "M7_s42", ("adapt", 0.2, 10))]
    fv, res = {}, {}
    for name, mk, pol in systems:
        r, f, _ = V.evaluate(Pd, Pd[mk].to_numpy(), truth, A, B, owner, pol, name)
        r["model"] = mk
        res[name], fv[name] = r, f
        log(name, json.dumps({k: r[k] for k in ("f05", "precision", "recall", "fp", "fn", "fp_near_copy", "candidate_pairs", "threshold")}))
    # paired CIs against the baseline with the same seed
    for name in res:
        base = "A_seed7" if name.endswith("seed7") else "A_baseline"
        if name != base:
            g, lo, hi = V.paired_ci(fv[name], fv[base])
            res[name].update({"gain_vs": base, "gain": g, "ci_lo": lo, "ci_hi": hi, "rel_gain": g / res[base]["f05"]})
    # swapped halves (tune on B, report on A): must agree in sign
    for name, mk, pol in systems[:4]:
        r, _, _ = V.evaluate(Pd, Pd[mk].to_numpy(), truth, B, A, owner, pol, name + "_swapped")
        res[name]["f05_swapped_halves"] = r["f05"]
    # US spot-check
    for name, mk, pol in systems[:4]:
        r, f, _ = V.evaluate(Pusd, Pusd[mk].to_numpy(), truth_us, A_us, B_us, owner, pol, name + "_US")
        res[name].update({"us_f05": r["f05"], "us_precision": r["precision"], "us_recall": r["recall"], "us_fp": r["fp"], "us_fn": r["fn"]})
        fv[name + "_US"] = f
    for name in ("B_topk", "C_ar7", "D_topk_ar7"):
        g, lo, hi = V.paired_ci(fv[name + "_US"], fv["A_baseline_US"])
        res[name].update({"us_gain": g, "us_ci_lo": lo, "us_ci_hi": hi})
    T["total_secs"] = round(time.time() - t0)
    T["peak_gb"] = peak_gb()
    with open(RES, "a", encoding="utf-8") as f:
        for name, r in res.items():
            f.write(json.dumps({"run": "exp3", "system": name, **r, "timing": T}, default=float) + "\n")
    print("\n" + json.dumps(T, indent=1))
    cols = ["f05", "gain", "ci_lo", "ci_hi", "precision", "recall", "fp", "fn", "fp_distractor", "fp_near_copy",
            "recall_S2", "recall_S3", "precision_S2", "precision_S3", "recall_addr_empty", "precision_addr_empty",
            "candidate_pairs", "threshold", "f05_swapped_halves", "us_f05", "us_gain", "us_ci_lo", "us_ci_hi"]
    print(pd.DataFrame(res).T.reindex(columns=cols).to_string())
