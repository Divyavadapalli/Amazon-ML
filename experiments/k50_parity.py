"""K50 verification tests T1, T2, T4 (plan section 6). Exit code 1 on any failure.

T1  K50 compact cache, ranks 0-9 == the K10 cache (same S1 set per query, scores <= 1e-6);
    ties at the 10th score are counted and reported, not failed.
T2  ar7_features(ctx_mask=None) reproduces the validated ar7_ev_cache rows bit-for-bit, and
    ctx_mask=all-True gives the same result (default behaviour unchanged).
T4  feature parity:
    a) k50_features on full top-10 lists == cached Xev + ar7_ev_cache (the K10 validation path)
    b) k50_features on a SUBSET of rows (ranks 3-9, as a gated stage computes them) == (a)
    c) deep ranks: k50_features on ranks 10-49 only (inference stage) == on all 50 rows (training
       path) for the same pairs
Usage: python -m experiments.k50_parity [n_queries=3000]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiments.ar7_features import NAMES as AR7, _init, ar7_features, s1_counts
from experiments.k50_common import K50_ROOT, k50_features, atomic_json
from experiments.exp_match2 import REC
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.io_utils import read_records

CACHE = WORK_DIR / "m2cache_v2_blend_ad15_India_50"
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"
TOL = 1e-4

if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    t0 = time.time()
    fails, report = [], {}

    def check(ok, name, detail):
        report[name] = {"pass": bool(ok), **detail}
        print(("PASS " if ok else "FAIL ") + name, json.dumps(detail, default=float), flush=True)
        if not ok:
            fails.append(name)

    # ---- T1
    k50 = load_compact_chunk(K50_ROOT / "cand_train", "India_000")
    k10 = pd.read_parquet(WORK_DIR / "cand_train_blend_ad15" / "India_000.parquet", columns=["q_id", "s1_id", "score", "rank"])
    a = k50[k50["rank"] < 10].merge(k10, on=["q_id", "s1_id"], how="outer", suffixes=("_50", "_10"), indicator=True)
    only50, only10 = int((a._merge == "left_only").sum()), int((a._merge == "right_only").sum())
    both = a[a._merge == "both"]
    dscore = float((both.score_50 - both.score_10).abs().max())
    # a set difference is acceptable only as a tie at the 10th score
    tie_q = set(a.loc[a._merge != "both", "q_id"])
    tie_ok = True
    if tie_q:
        s9 = k10[k10["rank"] == 9].set_index("q_id")["score"]
        diff = a[a._merge != "both"]
        sc = diff["score_50"].fillna(diff["score_10"])
        tie_ok = bool((np.abs(sc.to_numpy() - diff["q_id"].map(s9).to_numpy()) <= 1e-6).all())
    check(dscore <= 1e-6 and tie_ok, "T1_k50_top10_equals_k10",
          {"pairs_both": len(both), "only_k50": only50, "only_k10": only10, "max_score_diff": dscore, "tie_only": tie_ok})

    # ---- sample of cached eval queries (full top-10 lists)
    ev = pd.read_parquet(CACHE / "ev_pairs.parquet")
    Xev = pd.read_parquet(CACHE / "Xev.parquet")
    A7 = pd.read_parquet(WORK_DIR / "exp3" / "ar7_ev_cache.parquet")
    qs = pd.Series(ev.q_id.unique()).sample(nq, random_state=0)
    m = ev.q_id.isin(set(qs)).to_numpy()
    rows, Xc, Ac = ev[m].reset_index(drop=True), Xev[m].reset_index(drop=True), A7[m].reset_index(drop=True)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    s1c = s1_counts(s1[["country", "name_core", "addr_norm"]])
    q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(rows.q_id), REC) for k in (2, 3)]).set_index("entity_id")
    freq = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"]).value_counts().to_dict()
    with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool:
        # ---- T2: deterministic across worker pools; equal to the validated cache except the
        # feature the old code computed hash-seed-dependently (e7_num_q_novel; fixed 2026-09-27)
        A0 = ar7_features(rows[["q_id", "s1_id"]], s1c, q, pool)
        with Pool(8, initializer=_init, initargs=(str(STATS),)) as pool2:
            A1 = ar7_features(rows[["q_id", "s1_id"]], s1c, q, pool2, ctx_mask=np.ones(len(rows), bool))
        d_det = float(np.abs(A1.to_numpy() - A0.to_numpy()).max())
        stable = [c for c in AR7 if c != "e7_num_q_novel"]
        d_cache = float(np.abs(A0[stable].to_numpy() - Ac[stable].to_numpy()).max())
        changed = float((A0["e7_num_q_novel"].to_numpy() != Ac["e7_num_q_novel"].to_numpy()).mean())
        check(d_det == 0.0 and d_cache == 0.0, "T2_ar7_deterministic_and_default_unchanged",
              {"cross_pool_max": d_det, "vs_cache_max_other_20_features": d_cache,
               "e7_num_q_novel_rows_changed_by_fix": changed, "rows": len(rows)})
        # ---- T4a
        ctx = rows[rows["rank"] < 10]
        F = k50_features(rows, ctx, s1, q, freq, s1c, pool)
        ref = pd.concat([Xc, Ac[AR7]], axis=1)
        cols = list(ref.columns)
        cmp_cols = [c for c in cols if c != "e7_num_q_novel"]          # see T2
        diffs = (F[cmp_cols] - ref[cmp_cols]).abs().max()
        bad = diffs[diffs > TOL]
        check(bad.empty and float((F["e7_num_q_novel"] - A0["e7_num_q_novel"]).abs().max()) == 0.0,
              "T4a_k50_features_equal_validated_k10_cache",
              {"rows": len(rows), "worst": diffs.sort_values().tail(3).to_dict(), "failing": bad.to_dict()})
        # ---- T4b: gated-stage subset (ranks 3-9 only) vs full computation
        sub = rows["rank"].between(3, 9).to_numpy()
        Fs = k50_features(rows[sub], ctx, s1, q, freq, s1c, pool)
        ds = (Fs[cols].to_numpy() - F[cols].to_numpy()[sub])
        ds = pd.Series(np.abs(ds).max(axis=0), index=cols)
        check(bool((ds <= TOL).all()), "T4b_stage_subset_equals_full", {"rows": int(sub.sum()), "failing": ds[ds > TOL].to_dict()})
        # ---- T4c: deep ranks on K50 lists, inference stage (10-49 only) vs training path (all 50)
        deep = k50[k50.q_id.isin(set(k50.q_id.drop_duplicates().sample(min(nq, 2000), random_state=1)))].reset_index(drop=True)
        q2 = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(deep.q_id), REC) for k in (2, 3)]).set_index("entity_id")
        ctx2 = deep[deep["rank"] < 10]
        Ftr = k50_features(deep, ctx2, s1, q2, freq, s1c, pool)
        dm = (deep["rank"] >= 10).to_numpy()
        Finf = k50_features(deep[dm], ctx2, s1, q2, freq, s1c, pool)
        dd = pd.Series(np.abs(Finf[cols].to_numpy() - Ftr[cols].to_numpy()[dm]).max(axis=0), index=cols)
        check(bool((dd <= TOL).all()), "T4c_deep_inference_equals_training_path",
              {"rows": int(dm.sum()), "rank_max": int(deep["rank"].max()), "failing": dd[dd > TOL].to_dict()})
    report["secs"] = round(time.time() - t0)
    atomic_json(report, K50_ROOT / "reports" / "k50_parity.json")
    print("ALL PARITY TESTS PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)
