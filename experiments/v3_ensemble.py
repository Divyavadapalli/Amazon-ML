"""V3 / E1: does averaging existing clean matchers beat the single D model? (no new features)

Input: WORK_DIR/exp3/probs_clean_india.parquet (exp3c leakage-free models M7c_s42 = D,
M7c_s7, M0c_s42, M0c_s7), densified exactly as vdense.densify; D candidate policy
('adapt', 0.5, 10) computed from each system's own probability; one-to-one + global threshold
tuned on half-A, reported on half-B; swapped halves; paired bootstrap CI vs D; every F0.5
re-checked with src.metrics.macro_f05 (v3_matcher_eval.evaluate). US spot-check on
probs_clean_us.parquet. Gate: gain >= +0.001, CI lower bound > 0, swapped gain > 0, US >= -0.0005.

Logs: reports/E1_ensemble.log. Output: reports/ensemble.json
Usage: python -m experiments.v3_ensemble
"""
import json
import time

import numpy as np

from experiments import v3_matcher_eval as ME
from experiments.k50_common import atomic_json, bucket, mem_ok
from experiments.v3_common import REPORTS, Log, eval_truth
from experiments.v3_decision import D_REF, Data
from src.config import WORK_DIR

MODELS = ["M7c_s42", "M7c_s7", "M0c_s42", "M0c_s7"]


def systems(d):
    p = d.prob
    return {"D=M7c_s42": p["M7c_s42"],
            "mean(M7c_s42,M7c_s7)": (p["M7c_s42"] + p["M7c_s7"]) / 2,
            "mean(all4)": (p["M7c_s42"] + p["M7c_s7"] + p["M0c_s42"] + p["M0c_s7"]) / 4,
            "geo(M7c_s42,M7c_s7)": np.sqrt(p["M7c_s42"] * p["M7c_s7"]),
            "max(M7c_s42,M7c_s7)": np.maximum(p["M7c_s42"], p["M7c_s7"]),
            "min(M7c_s42,M7c_s7)": np.minimum(p["M7c_s42"], p["M7c_s7"])}


def run(d, ents, truth, isA, label):
    top = lambda prob: (lambda t: t)(None)
    res, base, base_s = {}, None, None
    r = d.rank
    for name, prob in systems(d).items():
        prob = prob.astype(np.float32)
        t3 = np.full(d.nq, -1.0, np.float32)
        s = r < 3
        np.maximum.at(t3, d.q[s], prob[s])
        m = (r < 3) | ((r < 10) & (t3[d.q] < 0.5))
        out = ME.evaluate(d, prob, m, ents, truth, isA)
        sw = ME.evaluate(d, prob, m, ents, truth, ~isA)
        if base is None:
            base, base_s = out, sw
        row = {**{k: v for k, v in out["tunedA"].items() if not k.startswith("_")}, **ME.compare(out, base, "tunedA"),
               "swapped_f05": sw["tunedA"]["f05"], "swapped_gain": ME.compare(sw, base_s, "tunedA")["gain"]}
        res[name] = row
        ME.log(f"  {label} {name}: F={row['f05']:.6f} gain={row['gain']:+.5f} [{row['ci_lo']:+.5f},{row['ci_hi']:+.5f}] "
               f"thr={row['threshold']} P={row['precision']:.4f} R={row['recall']:.4f} fp={row['fp']} fn={row['fn']} "
               f"+TP={row['recovered_true_owners']} +FP={row['new_false_merges']} swapped={row['swapped_gain']:+.5f} {mem_ok(tag='E1')}")
    return res


if __name__ == "__main__":
    t0 = time.time()
    ME.log = Log("E1_ensemble")
    g = eval_truth(us=True)
    E, truth, Eu, truth_u = g["E"], g["truth"], g["Eu"], g["truth_u"]
    d = Data(WORK_DIR / "exp3" / "probs_clean_india.parquet", MODELS)
    rep = {"india": run(d, E, truth, np.array([bucket(e, 2) == 0 for e in E]), "E")}
    assert abs(rep["india"]["D=M7c_s42"]["f05"] - D_REF) < 1e-9, "STOP: D not reproduced"
    ME.log(f"D reproduced exactly: {rep['india']['D=M7c_s42']['f05']:.10f}")
    del d
    du = Data(WORK_DIR / "exp3" / "probs_clean_us.parquet", MODELS)
    rep["us"] = run(du, Eu, truth_u, np.array([bucket(e, 2) == 0 for e in Eu]), "US")
    gate = {}
    for k in rep["india"]:
        if k.startswith("D="):
            continue
        i, u = rep["india"][k], rep["us"][k]
        c = {"gain>=0.001": i["gain"] >= 0.001, "ci_lo>0": i["ci_lo"] > 0, "swapped>0": i["swapped_gain"] > 0,
             "us>=-0.0005": u["gain"] >= -0.0005}
        gate[k] = {"pass": all(c.values()), **c, "gain": i["gain"], "us_gain": u["gain"]}
    rep["gate"] = gate
    rep["secs"] = round(time.time() - t0)
    atomic_json(rep, REPORTS / "ensemble.json")
    ME.log("GATE " + json.dumps(gate))
    ME.log(f"E1 done in {time.time() - t0:.0f}s")
