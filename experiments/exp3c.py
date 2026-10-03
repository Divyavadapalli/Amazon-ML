"""Exp-3c (K50 plan, Phase 0): leakage-free re-measurement of systems A-D.

Leak L1 (found in the K50 audit): the V-dense training pairs (m2 cache tr_pairs) exclude only
queries owned by E or touching E at rank < 3, but the eval extension adds queries touching E
at ranks 3..9 -- 54,045 of the 176,425 training queries are also eval queries (62,759
(query, eval-entity) negatives at ranks 3..9 were trained on). A and C (top-3 only) are not
affected; B and D (ranks 3..9) may be optimistic.

Fix: drop every training query that is in the eval set (cache + extension = every query
owned by E or with an E member in its top-10), retrain M0 (49 features) and M7 (+ AR-7) with
the identical training code, and re-score A-D with the identical vdense protocol.
Probabilities -> WORK_DIR/exp3/probs_clean_{india,us}.parquet; results appended to
experiments/exp3_results.jsonl (run "exp3c").

Usage: python -m experiments.exp3c
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from experiments import vdense as V
from experiments.ar7_features import NAMES as AR7
from experiments.exp3 import predict, train
from src.config import WORK_DIR

EXP = WORK_DIR / "exp3"
RES = Path(__file__).with_name("exp3_results.jsonl")
ADAPT = ("adapt", 0.5, 10)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    tr = pd.read_parquet(V.CACHE / "tr_pairs.parquet")
    Xtr = pd.concat([pd.read_parquet(V.CACHE / "Xtr.parquet"), pd.read_parquet(EXP / "ar7_tr.parquet")[AR7]], axis=1)
    evq = set(pd.read_parquet(V.CACHE / "ev_pairs.parquet", columns=["q_id"]).q_id) | \
        set(pd.read_parquet(EXP / "ev_ext_pairs.parquet", columns=["q_id"]).q_id)
    keep = ~tr["q_id"].isin(evq).to_numpy()
    tr, Xtr = tr[keep].reset_index(drop=True), Xtr[keep].reset_index(drop=True)
    from src.io_utils import read_ground_truth
    owner = {m: s for s, ms in read_ground_truth().items() for m in ms}
    ytr = (tr["q_id"].map(owner) == tr["s1_id"]).to_numpy(np.int8)
    assert not tr["q_id"].isin(evq).any()                      # T3-style leakage assert
    log(f"clean training set: {tr.q_id.nunique()} queries / {len(tr)} pairs "
        f"(dropped {int((~keep).sum())} pairs), positives {int(ytr.sum())}")
    base49 = [c for c in Xtr.columns if c not in AR7]
    models = {}
    for seed in (42, 7):
        t = time.time()
        models[f"M0c_s{seed}"] = train(Xtr[base49], ytr, tr.q_id.to_numpy(), seed)
        models[f"M7c_s{seed}"] = train(Xtr, ytr, tr.q_id.to_numpy(), seed)
        log(f"seed {seed} trained in {time.time() - t:.0f}s")
    for k, m in models.items():
        m.save_model(str(EXP / f"model_{k}.txt"))
    del Xtr

    def probs(name, ev, X49, A7):
        f = EXP / f"probs_clean_{name}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        out = V.slim(ev, X49)
        X = pd.concat([X49, A7], axis=1)
        for k, m in models.items():
            out[k] = predict(m, X).astype(np.float32)
        out.to_parquet(EXP / f"probs_clean_{name}.tmp.parquet", index=False)
        (EXP / f"probs_clean_{name}.tmp.parquet").replace(f)
        return out

    ev, Xev, truth, A, B, owner = V.load_india(with_ext=True)
    A7 = pd.concat([pd.read_parquet(EXP / "ar7_ev_cache.parquet"), pd.read_parquet(EXP / "ar7_ev_ext.parquet")], ignore_index=True)[AR7]
    t = time.time()
    P = probs("india", ev, Xev, A7)
    log(f"India probabilities {time.time() - t:.0f}s")
    del ev, Xev, A7
    us, Xus, truth_us, A_us, B_us, _ = V.load_us()
    Pus = probs("us", us, Xus, pd.read_parquet(EXP / "ar7_us.parquet")[AR7])
    del us, Xus
    Pd, _ = V.densify(P)
    Pusd, _ = V.densify(Pus)
    del P, Pus
    res, fv = {}, {}
    for seed in (42, 7):
        for name, mk, pol in (("A_baseline", f"M0c_s{seed}", "k3"), ("B_topk", f"M0c_s{seed}", ADAPT),
                              ("C_ar7", f"M7c_s{seed}", "k3"), ("D_topk_ar7", f"M7c_s{seed}", ADAPT)):
            key = f"{name}_s{seed}"
            r, f, _ = V.evaluate(Pd, Pd[mk].to_numpy(), truth, A, B, owner, pol, key)
            res[key], fv[key] = r, f
            if seed == 42:
                r2, _, _ = V.evaluate(Pd, Pd[mk].to_numpy(), truth, B, A, owner, pol, key + "_swapped")
                r["f05_swapped_halves"] = r2["f05"]
                ru, fu, _ = V.evaluate(Pusd, Pusd[mk].to_numpy(), truth_us, A_us, B_us, owner, pol, key + "_US")
                r.update({"us_f05": ru["f05"], "us_fp": ru["fp"], "us_fn": ru["fn"]})
                fv[key + "_US"] = fu
            log(key, json.dumps({k: r.get(k) for k in ("f05", "precision", "recall", "fp", "fn", "fp_near_copy",
                                                        "threshold", "f05_swapped_halves", "us_f05")}))
    for seed in (42, 7):
        a = f"A_baseline_s{seed}"
        for n in ("B_topk", "C_ar7", "D_topk_ar7"):
            k = f"{n}_s{seed}"
            g, lo, hi = V.paired_ci(fv[k], fv[a])
            res[k].update({"gain": g, "ci_lo": lo, "ci_hi": hi})
            if seed == 42:
                res[k]["gain_swapped"] = res[k]["f05_swapped_halves"] - res[a]["f05_swapped_halves"]
                gu, ulo, uhi = V.paired_ci(fv[k + "_US"], fv[a + "_US"])
                res[k].update({"us_gain": gu, "us_ci_lo": ulo, "us_ci_hi": uhi})
    mi = psutil.Process().memory_info()
    timing = {"total_secs": round(time.time() - t0), "peak_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)}
    with open(RES, "a", encoding="utf-8") as f:
        for k, r in res.items():
            f.write(json.dumps({"run": "exp3c_clean", "system": k, **r, "timing": timing}, default=float) + "\n")
    cols = ["f05", "gain", "ci_lo", "ci_hi", "precision", "recall", "fp", "fn", "fp_near_copy", "threshold",
            "candidate_pairs", "f05_swapped_halves", "gain_swapped", "us_f05", "us_gain", "us_ci_lo", "us_ci_hi"]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(res).T.reindex(columns=cols).to_string())
    print(json.dumps(timing))
