"""Exp-3b: finish Exp-3 after its process was interrupted during the last stage.

Reloads the models Exp-3 saved (M0_s42 = saved india_v2 baseline, M7_s42 = AR-7), recomputes
their probabilities on the same V-dense India set and the US sample (checkpointed to
WORK_DIR/exp3/probs_*.parquet), and produces what was lost: paired bootstrap CIs vs the
baseline, the swapped-halves check (tune on B, report on A) and the US spot-check, for systems
A-D with the same vdense.evaluate code. Appends to experiments/exp3_results.jsonl.

Usage: python -m experiments.exp3b
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

EXP = WORK_DIR / "exp3"
RES = Path(__file__).with_name("exp3_results.jsonl")
MODELS = {"M0_s42": WORK_DIR / "m2_model_india_v2.txt", "M7_s42": EXP / "model_M7_s42.txt"}
ADAPT = ("adapt", 0.5, 10)
SYSTEMS = [("A_baseline", "M0_s42", "k3"), ("B_topk", "M0_s42", ADAPT),
           ("C_ar7", "M7_s42", "k3"), ("D_topk_ar7", "M7_s42", ADAPT)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def probs(name, ev, X49, A7):
    f = EXP / f"probs_{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    out = V.slim(ev, X49)
    X = pd.concat([X49, A7], axis=1)
    for k, path in MODELS.items():
        m = lgb.Booster(model_file=str(path))
        cols = m.feature_name()
        out[k] = np.concatenate([m.predict(X.iloc[i:i + 2_000_000][cols], num_threads=12)
                                 for i in range(0, len(X), 2_000_000)]).astype(np.float32)
    out.to_parquet(EXP / f"probs_{name}.tmp.parquet", index=False)
    (EXP / f"probs_{name}.tmp.parquet").replace(f)
    return out


if __name__ == "__main__":
    t0 = time.time()
    ev, Xev, truth, A, B, owner = V.load_india(with_ext=True)
    A7 = pd.concat([pd.read_parquet(EXP / "ar7_ev_cache.parquet"), pd.read_parquet(EXP / "ar7_ev_ext.parquet")], ignore_index=True)
    assert len(A7) == len(ev)
    t = time.time()
    P = probs("india", ev, Xev, A7[AR7])
    log(f"India probabilities in {time.time() - t:.0f}s")
    del ev, Xev, A7
    us, Xus, truth_us, A_us, B_us, _ = V.load_us()
    Pus = probs("us", us, Xus, pd.read_parquet(EXP / "ar7_us.parquet")[AR7])
    del us, Xus
    Pd, _ = V.densify(P)
    Pusd, _ = V.densify(Pus)
    del P, Pus
    res, fv = {}, {}
    for name, mk, pol in SYSTEMS:
        t = time.time()
        r, f, _ = V.evaluate(Pd, Pd[mk].to_numpy(), truth, A, B, owner, pol, name)
        r["eval_secs"] = round(time.time() - t)
        res[name], fv[name] = r, f
        r2, _, _ = V.evaluate(Pd, Pd[mk].to_numpy(), truth, B, A, owner, pol, name + "_swapped")
        r["f05_swapped_halves"] = r2["f05"]
        ru, fu, _ = V.evaluate(Pusd, Pusd[mk].to_numpy(), truth_us, A_us, B_us, owner, pol, name + "_US")
        r.update({"us_f05": ru["f05"], "us_precision": ru["precision"], "us_recall": ru["recall"],
                  "us_fp": ru["fp"], "us_fn": ru["fn"], "us_fp_near_copy": ru["fp_near_copy"],
                  "us_candidate_pairs": ru["candidate_pairs"]})
        fv[name + "_US"] = fu
        log(name, json.dumps({k: r[k] for k in ("f05", "f05_swapped_halves", "us_f05", "fp", "fn")}))
    for name in ("B_topk", "C_ar7", "D_topk_ar7"):
        g, lo, hi = V.paired_ci(fv[name], fv["A_baseline"])
        gu, ulo, uhi = V.paired_ci(fv[name + "_US"], fv["A_baseline_US"])
        res[name].update({"gain": g, "ci_lo": lo, "ci_hi": hi, "rel_gain": g / res["A_baseline"]["f05"],
                          "gain_swapped": res[name]["f05_swapped_halves"] - res["A_baseline"]["f05_swapped_halves"],
                          "us_gain": gu, "us_ci_lo": ulo, "us_ci_hi": uhi})
    g, lo, hi = V.paired_ci(fv["D_topk_ar7"], fv["C_ar7"])
    res["D_topk_ar7"].update({"gain_vs_C": g, "ci_vs_C": [lo, hi]})
    g, lo, hi = V.paired_ci(fv["D_topk_ar7"], fv["B_topk"])
    res["D_topk_ar7"].update({"gain_vs_B": g, "ci_vs_B": [lo, hi]})
    mi = psutil.Process().memory_info()
    timing = {"total_secs": round(time.time() - t0), "peak_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)}
    with open(RES, "a", encoding="utf-8") as f:
        for name, r in res.items():
            f.write(json.dumps({"run": "exp3b", "system": name, **r, "timing": timing}, default=float) + "\n")
    cols = ["f05", "gain", "ci_lo", "ci_hi", "rel_gain", "precision", "recall", "fp", "fn", "fp_distractor", "fp_near_copy",
            "recall_S2", "recall_S3", "precision_S2", "precision_S3", "recall_addr_empty", "precision_addr_empty",
            "n_accepted_addr_empty", "singleton_acc", "candidate_pairs", "candidate_pairs_orig_queries", "threshold",
            "f05_swapped_halves", "gain_swapped", "us_f05", "us_gain", "us_ci_lo", "us_ci_hi", "us_precision", "us_recall",
            "us_fp", "us_fn", "us_fp_near_copy", "us_candidate_pairs", "gain_vs_C", "ci_vs_C", "gain_vs_B", "ci_vs_B"]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(res).T.reindex(columns=cols).T.to_string())
    print(json.dumps(timing))
