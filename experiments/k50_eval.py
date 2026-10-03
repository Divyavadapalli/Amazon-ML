"""K50 Phases F/G/H: evaluate every system on the K50 V-dense set (plan section 4).

Same protocol as vdense: entities E' (India fold-1, bucket < b) split into half-A (tune) and
half-B (report), distractors cloned to test density, exact macro-F0.5 via vdense.evaluate
(asserted against src.metrics), paired bootstrap vs system 1, swapped halves, US spot-check.

Systems:
  S1  D regime: M7_K10reg (clean K10-trained), ranks 0-2 + 3-9 if p_top3 < 0.5 (as D), and tuned h
      references: exp3c clean M7 and the leaky exp3 M7 (same policy)
  S2  K50 retrieval + old matcher: M7_K10reg scoring ranks 0-49 (rank out of distribution)
  S3  K50 retrieval + K50 matcher, all ranks scored (every K50 budget)
  S4  K50 + adaptive gates (policies A-E), gates tuned on half-A
Gate grids are capped at h <= 0.95 (k50_build_eval GATE_CAP). Policy E stage s opens when the
best probability so far is in [lo, h). Outputs: K50_ROOT/reports/k50_validation.json and
k50_results.md (the results table) and the oracle-vs-actual table.

Usage: python -m experiments.k50_eval <bucket>
"""
import json
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import psutil

from experiments import vdense as V
from experiments.k50_common import K50_ROOT, atomic_json, bucket, e_prime
from src.evaluation import GRID, predictions, sweep
from src.io_utils import read_ground_truth

H_GRID = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
LO_GRID = [0.0, 0.01, 0.03, 0.1]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def query_max(P, p, rank_lt):
    """Per-row: max of p over the row's query restricted to rank < rank_lt (-1 if none)."""
    v = np.where(P["rank"].to_numpy() < rank_lt, p, -1.0)
    return pd.Series(v).groupby(P["qcode"].to_numpy()).transform("max").to_numpy()


def masks(P, p):
    r = P["rank"].to_numpy()
    p3, p10, p20 = query_max(P, p, 3), query_max(P, p, 10), query_max(P, p, 20)
    out = {"A": r < 3}
    for h in H_GRID:
        out[f"B(h={h})"] = (r < 3) | ((r < 10) & (p3 < h))
        out[f"C(h={h})"] = (r < 3) | ((r < 20) & (p3 < h))
        out[f"D(h={h})"] = (r < 3) | ((r < 50) & (p3 < h))
    for h1, lo, h2 in product([0.35, 0.5, 0.65], LO_GRID, [0.5, 0.8, 0.95]):
        s2 = p3 < h1
        s3 = s2 & (p10 >= lo) & (p10 < h2)
        s4 = s3 & (p20 >= lo) & (p20 < h2)
        out[f"E(h1={h1},lo={lo},h2={h2})"] = (r < 3) | ((r < 10) & s2) | ((r < 20) & s3) | s4
    return out, {"p3": p3, "p10": p10, "p20": p20}


def tuneA(P, p, m, truth, A):
    pr = predictions(P[m], p[m], True)
    return float(sweep(pr, truth, A, GRID).f05_macro.max())


def stage_reach(P, m):
    q = P.loc[m, ["qcode", "rank"]].groupby("qcode")["rank"].max()
    n = P["qcode"].nunique()
    return {">=3": float((q >= 3).sum() / n), ">=10": float((q >= 10).sum() / n), ">=20": float((q >= 20).sum() / n)}


def load(dirname):
    d = K50_ROOT / "eval" / dirname
    assert (d / "DONE").exists(), f"build eval set {dirname} first"
    P = pd.concat([pd.read_parquet(p) for p in sorted(d.glob("*.parquet")) if p.stem != "manifest"], ignore_index=True)
    P["part"] = "k50"
    return P


if __name__ == "__main__":
    b = int(sys.argv[1])
    t0 = time.time()
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    s1c = pd.read_parquet(r"C:\aml2026\work\train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id")["country"]
    E = e_prime(gt, s1c, b)
    A = [e for e in E if bucket(e, 2) == 0]
    B = [e for e in E if bucket(e, 2) == 1]
    truth = {e: gt[e] for e in E}
    P = load(f"b{b}")
    models = [c for c in P.columns if c.startswith(("K50", "M7_K10reg", "ref_"))]
    P, n_clone = V.densify(P)
    P["qcode"] = pd.factorize(P["q_id"])[0]
    log(f"E' bucket<{b}: {len(E)} entities (A {len(A)}, B {len(B)}); rows {len(P)} (clones {n_clone}); models {models}")
    res, fv = {}, {}

    def run(name, mcol, m, extra=None):
        t = time.time()
        r, f, _ = V.evaluate(P, P[mcol].to_numpy(), truth, A, B, owner, ("mask", m), name)
        r["tuneA_f05"] = tuneA(P, P[mcol].to_numpy(), m, truth, A)          # selection statistic (half-A only)
        r.update({"model": mcol, "scored_pairs": int(m.sum()), "stage_reach": stage_reach(P, m),
                  "pairs_per_query": float(m.sum() / P["qcode"].nunique()), "eval_secs": round(time.time() - t), **(extra or {})})
        res[name], fv[name] = r, f
        log(name, json.dumps({k: r[k] for k in ("f05", "precision", "recall", "fp", "fn", "scored_pairs", "threshold")}))
        return r

    def best_gate(mcol, family):
        p = P[mcol].to_numpy()
        ms, _ = masks(P, p)
        cand = {k: v for k, v in ms.items() if k.startswith(family)}
        scores = {k: tuneA(P, p, v, truth, A) for k, v in cand.items()}
        k = max(scores, key=scores.get)
        return k, cand[k], scores

    # ---------------- S1 (D regime) and references
    for mcol in [c for c in models if c.startswith(("M7_K10reg", "ref_"))]:
        p = P[mcol].to_numpy()
        r = P["rank"].to_numpy()
        run(f"S1_D[{mcol}]", mcol, (r < 3) | ((r < 10) & (query_max(P, p, 3) < 0.5)))
    base = "S1_D[M7_K10reg_s42]"
    k, m, sc = best_gate("M7_K10reg_s42", "B(")
    run(f"S1_Dtuned[{k}]", "M7_K10reg_s42", m, {"gate_tuneA": sc})
    # ---------------- S2 old matcher on K50
    run("S2_oldmatcher_all50", "M7_K10reg_s42", np.ones(len(P), bool))
    # ---------------- S3 K50 matcher, all ranks
    for mcol in [c for c in models if c.startswith("K50")]:
        run(f"S3_all50[{mcol}]", mcol, np.ones(len(P), bool))
    s3 = {k: v["f05"] for k, v in res.items() if k.startswith("S3")}
    s3A = {k: v["tuneA_f05"] for k, v in res.items() if k.startswith("S3")}
    best_k50 = res[max(s3A, key=s3A.get)]["model"]              # selected on half-A, never on B
    # ---------------- S4 adaptive policies for the selected K50 model (gates tuned on half-A)
    for fam in ("A", "B(", "C(", "D(", "E("):
        if fam == "A":
            run(f"S4_A[{best_k50}]", best_k50, P["rank"].to_numpy() < 3)
            continue
        k, m, sc = best_gate(best_k50, fam)
        run(f"S4_{k}[{best_k50}]", best_k50, m, {"gate_tuneA": sc})
    # seed-7 reproduction of S1 and the best S4 policy
    s4 = {k: v["tuneA_f05"] for k, v in res.items() if k.startswith("S4")}
    best_s4 = max(s4, key=s4.get)                               # selected on half-A
    pol = best_s4.split("[")[0][3:]
    alt = best_k50.replace("_s42", "_s7")
    if alt in P.columns:
        p = P[alt].to_numpy()
        ms, _ = masks(P, p)
        run(f"S4_{pol}[{alt}]", alt, ms[pol] if pol in ms else P["rank"].to_numpy() < 3)
    if "M7_K10reg_s7" in P.columns:
        p = P["M7_K10reg_s7"].to_numpy()
        r = P["rank"].to_numpy()
        run("S1_D[M7_K10reg_s7]", "M7_K10reg_s7", (r < 3) | ((r < 10) & (query_max(P, p, 3) < 0.5)))
    # ---------------- paired CIs, swapped halves
    for k in list(res):
        ref = "S1_D[M7_K10reg_s7]" if k.endswith("_s7]") and "S1_D[M7_K10reg_s7]" in res else base
        if k != ref:
            g, lo, hi = V.paired_ci(fv[k], fv[ref])
            res[k].update({"gain_vs": ref, "gain": g, "ci_lo": lo, "ci_hi": hi})
    # recompute masks for the swapped-half check
    def mask_of(k):
        mcol = res[k]["model"]
        p = P[mcol].to_numpy()
        r = P["rank"].to_numpy()
        if k.startswith("S1_D["):
            return mcol, (r < 3) | ((r < 10) & (query_max(P, p, 3) < 0.5))
        ms, _ = masks(P, p)
        return mcol, ms.get(k.split("[")[0][3:], r < 3)
    for k in (base, best_s4):
        mcol, m = mask_of(k)
        r2, _, _ = V.evaluate(P, P[mcol].to_numpy(), truth, B, A, owner, ("mask", m), k + "_swapped")
        res[k]["f05_swapped_halves"] = r2["f05"]
    res[best_s4]["gain_swapped"] = res[best_s4]["f05_swapped_halves"] - res[base]["f05_swapped_halves"]
    # ---------------- US spot-check
    usd = K50_ROOT / "eval" / "us"
    if (usd / "DONE").exists():
        U = load("us")
        U, _ = V.densify(U)
        U["qcode"] = pd.factorize(U["q_id"])[0]
        Eus = sorted(s for s in gt if s1c.get(s) == "US" and bucket(s) < 50)
        Aus = [e for e in Eus if bucket(e, 2) == 0]
        Bus = [e for e in Eus if bucket(e, 2) == 1]
        tus = {e: gt[e] for e in Eus}
        usr = {}
        for k in (base, best_s4):
            mcol = res[k]["model"]
            p = U[mcol].to_numpy()
            r = U["rank"].to_numpy()
            if k.startswith("S1_D["):
                m = (r < 3) | ((r < 10) & (query_max(U, p, 3) < 0.5))
            else:
                ms, _ = masks(U, p)
                m = ms.get(k.split("[")[0][3:], r < 3)
            ru, fu, _ = V.evaluate(U, p, tus, Aus, Bus, owner, ("mask", m), k + "_US")
            usr[k] = (ru, fu)
            res[k].update({"us_f05": ru["f05"], "us_fp": ru["fp"], "us_fn": ru["fn"], "us_scored_pairs": int(m.sum())})
        g, lo, hi = V.paired_ci(usr[best_s4][1], usr[base][1])
        res[best_s4].update({"us_gain": g, "us_ci_lo": lo, "us_ci_hi": hi})
    # ---------------- oracle vs actual (Phase G)
    rs = json.loads((K50_ROOT / "reports" / "k50_retrieval_stats.json").read_text())
    orc = json.loads((K50_ROOT / "reports" / "k50_oracle.json").read_text())
    key = f"E_bucket_lt_{b}"
    gap = {"K10_recall": rs[key]["recall_at"]["10"], "K50_recall": rs[key]["recall_at"]["50"],
           "K10_oracle_f05": orc[key]["oracle_f05@10"], "K50_oracle_f05": orc[key]["oracle_f05@50"],
           "K50_actual_f05_best_all_ranks": max(s3.values()), "K50_gated_actual_f05_best": res[best_s4]["f05"],
           "D_regime_actual_f05": res[base]["f05"]}
    gap["K50_oracle_minus_K50_actual"] = gap["K50_oracle_f05"] - gap["K50_actual_f05_best_all_ranks"]
    gap["K50_oracle_minus_K50_gated"] = gap["K50_oracle_f05"] - gap["K50_gated_actual_f05_best"]
    # ---------------- gate G1
    g1 = res[best_s4]
    G1 = {"best_policy": best_s4, "gain": g1.get("gain"), "ci_lo": g1.get("ci_lo"),
          "swapped_gain": g1.get("gain_swapped"), "us_gain": g1.get("us_gain"),
          "pass": bool(g1.get("gain", 0) >= 0.0010 and g1.get("ci_lo", -1) > 0 and g1.get("gain_swapped", -1) > 0
                       and (g1.get("us_gain") is None or g1["us_gain"] >= -0.0005))}
    mi = psutil.Process().memory_info()
    out = {"E_bucket": b, "entities": len(E), "half_A": len(A), "half_B": len(B), "systems": res, "oracle_vs_actual": gap,
           "G1": G1, "secs": round(time.time() - t0), "peak_gb": round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)}
    atomic_json(out, K50_ROOT / "reports" / "k50_validation.json")
    cols = ["model", "f05", "gain", "ci_lo", "ci_hi", "precision", "recall", "fp", "fn", "fp_near_copy", "scored_pairs",
            "pairs_per_query", "threshold", "eval_secs", "f05_swapped_halves", "us_f05", "us_gain"]
    tab = pd.DataFrame(res).T.reindex(columns=cols)
    md = ["# K50 V-dense results", "", f"E' = India fold-1 bucket < {b}: {len(E)} entities (half-B {len(B)}).", "",
          "```", tab.to_string(), "```", "", "## Oracle vs actual", "", "```", pd.Series(gap).to_string(), "```", "",
          "## Gate G1", "", "```", json.dumps(G1, indent=1, default=float), "```"]
    (K50_ROOT / "reports" / "k50_results.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab.to_string())
    print(json.dumps(gap, indent=1))
    print("G1", json.dumps(G1, default=float))
