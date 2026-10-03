"""V3 / R2: does the EXISTING D matcher convert R1's extra retrieval recall into real F0.5?

Model: the exact clean D validation model (exp3c M7c_s42; seed-7 replica M7c_s7) -- the
production model final_v2 was trained on these entities (leak L2) and cannot be validated.
Decision rule: D's, i.e. one-to-one (each query -> its best S1) + global threshold; reported
both at D's validation threshold 0.84 (fixed) and re-tuned on half-A; always reported on half-B.
Added candidates are scored as is (no retraining); D's own probabilities are reused unchanged.

Part 1 (E' = India fold-1 bucket < 5, existing K50 eval set k50_exp/eval/b5, already scored by
         ref_exp3c_M7c_s42; plus the US K50 spot-check set eval/us):
   A  D                      ranks 0-2 + ranks 3-9 if best top-3 < 0.5
   C  D + blend top-50       + ranks 10-49 for every query
   X  D + empty-address 50   + ranks 10-49 for empty-address queries
   (the eval set scored ranks 10-49 only where the best top-3 prob < 0.95 -- k50_build_eval)
Part 2 (full V-dense E, 44,360 entities, half-B 22,325 -- the population that predicted the LB):
   X is measured exactly: every empty-address India query in the V-dense eval set, plus every
   empty-address query with an E member at ranks 10-49 (new queries, their ranks 0-9 too), gets
   its ranks 10-49 featurised with the parity-tested k50_features path (top-10 context) and
   scored by M7c_s42 / M7c_s7; appended to probs_clean_india (D rows untouched), densified with
   the same rule (clone draws of the existing rows unchanged), evaluated. D must reproduce
   0.9708919456 on the combined frame.
   B (A5 + N5 passes) is NOT run here: it needs 2 new retrieval passes over all India queries and
   ~68M new test pairs -- it cannot reach a test submission before the deadline.
Every reported F0.5 is re-checked with src.metrics.macro_f05. Paired bootstrap CI vs D.

Logs: reports/R2_matcher_eval.log (heartbeat 60 s). Output: reports/matcher_eval.json
Usage: python -m experiments.v3_matcher_eval [part1|part2|all]
"""
import gc
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from experiments.ar7_features import _init, s1_counts
from experiments.exp3 import predict
from experiments.exp_match2 import REC
from experiments.k50_common import K50_ROOT, atomic_json, atomic_parquet, bucket, k50_features, mem_ok
from experiments.v3_common import CACHE, GT_PATH, REPORTS, Log, eval_truth
from experiments.v3_decision import D_REF, Data
from experiments.vdense import paired_ci
from src.candidates import load_compact_chunk
from src.config import WORK_DIR
from src.evaluation import GRID
from src.io_utils import read_records
from src.metrics import macro_f05
from src.prep import s1_fold

OUT_X = Path(r"C:\aml2026\v3_exp\xe")
STATS = WORK_DIR / "exp3" / "ar7_stats_train_v2.pkl"
EXP = WORK_DIR / "exp3"
BLOCK = 400_000
log = None


def best_per_query(d, prob, mask):
    """One-to-one: stable desc sort by prob (== src.evaluation.predictions) -> best row per query."""
    idx = np.flatnonzero(mask)
    o = idx[np.argsort(-prob[idx].astype(np.float64), kind="stable")]
    first = ~pd.Series(d.q[o]).duplicated().to_numpy()
    return o[first]


def evaluate(d, prob, mask, ents, truth, isA, thr_fixed=0.84, base=None):
    """Tune the threshold on half-A (GRID), report on half-B; also at thr_fixed.
    Returns dict + per-entity F vectors (half-B) + accepted (q,s1) key sets for deltas."""
    brow = best_per_query(d, prob, mask)
    ecode = pd.Index(d.s1_names).get_indexer(ents)
    emap = np.full(len(d.s1_names), -1, np.int64)
    emap[ecode[ecode >= 0]] = np.flatnonzero(ecode >= 0)
    brow = brow[emap[d.s1[brow]] >= 0]
    ei, p = emap[d.s1[brow]], prob[brow].astype(np.float64)
    ok = d.own[brow] == d.s1[brow]
    nt = np.array([len(truth[e]) for e in ents], float)

    def fvec(t):
        acc = p >= t
        npred = np.bincount(ei[acc], minlength=len(ents)).astype(float)
        tp = np.bincount(ei[acc & ok], minlength=len(ents)).astype(float)
        prec = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
        rec = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
        den = 0.25 * prec + rec
        f = np.divide(1.25 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
        f[nt == 0] = (npred[nt == 0] == 0).astype(float)
        return f, tp, npred

    fa = np.array([fvec(t)[0][isA].mean() for t in GRID])
    tA = float(GRID[int(np.argmax(fa))])                              # first max, == sweep().idxmax()
    out = {}
    for tag, t in (("tunedA", tA), ("fixed", thr_fixed)):
        f, tp, npred = fvec(t)
        B = ~isA
        acc = (p >= t) & ~isA[ei]
        qs, ss = d.q_str(d.q[brow][acc]), d.s1_names[d.s1[brow][acc]]
        pm = pd.DataFrame({"q": qs, "s": ss}).groupby("s")["q"].agg(set).to_dict()
        entsB = list(np.asarray(ents)[B])
        ref = macro_f05({e: truth[e] for e in entsB}, pm, entsB)["f05_macro"]
        assert abs(ref - f[B].mean()) < 1e-9, f"fast {f[B].mean()} != macro_f05 {ref}"
        keys_tp = set(zip(qs[ok[acc]], ss[ok[acc]]))
        keys_fp = set(zip(qs[~ok[acc]], ss[~ok[acc]]))
        out[tag] = {"threshold": float(t), "f05": float(f[B].mean()), "f05_verified_macro_f05": float(ref),
                    "tuneA_f05": float(f[isA].mean()), "tp": int(tp[B].sum()), "fp": int((npred - tp)[B].sum()),
                    "fn": int((nt - tp)[B].sum()), "precision": float(tp[B].sum() / max(npred[B].sum(), 1)),
                    "recall": float(tp[B].sum() / max(nt[B].sum(), 1)), "_f": f[B], "_tp": keys_tp, "_fp": keys_fp}
    out["tunedA_swapped"] = None
    return out


def compare(res, base, tag):
    r, b = res[tag], base[tag]
    g, lo, hi = paired_ci(r["_f"], b["_f"])
    return {"gain": g, "ci_lo": lo, "ci_hi": hi, "recovered_true_owners": len(r["_tp"] - b["_tp"]),
            "lost_true_owners": len(b["_tp"] - r["_tp"]), "new_false_merges": len(r["_fp"] - b["_fp"]),
            "removed_false_merges": len(b["_fp"] - r["_fp"])}


def strip(r):
    return {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")} if isinstance(v, dict) else v) for k, v in r.items()}


def run_systems(d, systems, models, ents, truth, isA, label):
    """systems: {name: mask-fn(prob)}; baseline is the first system. Swapped halves + seed replica."""
    res = {}
    for mk in models:
        prob = d.prob[mk]
        base = None
        for name, mfn in systems.items():
            t = time.time()
            m = mfn(prob)
            r = evaluate(d, prob, m, ents, truth, isA)
            rs = evaluate(d, prob, m, ents, truth, ~isA)                      # swapped: tune B, report A
            if base is None:
                base, base_s = r, rs
            row = {"model": mk, "candidates": int(m.sum()), "candidates_per_query": float(m.sum() / d.nq),
                   "secs": round(time.time() - t)}
            for tag in ("tunedA", "fixed"):
                row[tag] = {**{k: v for k, v in r[tag].items() if not k.startswith("_")}, **compare(r, base, tag)}
            row["swapped_tunedB_reportA"] = {"f05": rs["tunedA"]["f05"], **compare(rs, base_s, "tunedA")}
            res[f"{label}|{name}|{mk}"] = row
            log(f"  {label} {name} [{mk}] tunedA: F={row['tunedA']['f05']:.6f} gain={row['tunedA']['gain']:+.5f} "
                f"[{row['tunedA']['ci_lo']:+.5f},{row['tunedA']['ci_hi']:+.5f}] thr={row['tunedA']['threshold']} "
                f"P={row['tunedA']['precision']:.4f} R={row['tunedA']['recall']:.4f} fp={row['tunedA']['fp']} fn={row['tunedA']['fn']} "
                f"+TP={row['tunedA']['recovered_true_owners']} +FP={row['tunedA']['new_false_merges']} | fixed0.84: "
                f"F={row['fixed']['f05']:.6f} gain={row['fixed']['gain']:+.5f} | swapped gain={row['swapped_tunedB_reportA']['gain']:+.5f} "
                f"| cand/q={row['candidates_per_query']:.2f}")
        gc.collect()
    return res


def masks_for(d):
    r = d.rank
    empty = d.x["q_addr_empty"] == 1

    def top3(prob):
        t = np.full(d.nq, -1.0, np.float32)
        s = r < 3
        np.maximum.at(t, d.q[s], prob[s])
        return t[d.q]

    D = lambda prob: (r < 3) | ((r < 10) & (top3(prob) < 0.5))
    return {"A_D": D,
            "C_D+blend50": lambda prob: D(prob) | (r >= 10),
            "X_D+empty50": lambda prob: D(prob) | ((r >= 10) & empty),
            "Xg_D+empty50_gated": lambda prob: D(prob) | ((r >= 10) & empty & (top3(prob) < 0.5))}


def part1(report):
    log.set_stage("part1 E' (existing K50 eval set)")
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
    india = set(s1c.entity_id[s1c.country == "India"])
    us = set(s for s in s1c.entity_id[s1c.country == "US"] if bucket(s) < 50)
    del s1c
    gt = {}
    with open(GT_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            if (s1 in india and bucket(s1) < 5 and s1_fold(s1) == 1) or s1 in us:
                gt[s1] = {x for x in rest.split(",") if x}
    Ep = sorted(s for s in gt if s in india)
    Eu = sorted(us)
    tb = pa.concat_tables([pq.read_table(p) for p in sorted((K50_ROOT / "eval" / "b5").glob("India_*.parquet"))])
    d = Data(tb, ["ref_exp3c_M7c_s42", "M7_K10reg_s42"], extra=("q_addr_empty",))
    del tb
    isA = np.array([bucket(e, 2) == 0 for e in Ep])
    sysm = masks_for(d)
    systems = {k: sysm[k] for k in ("A_D", "C_D+blend50", "X_D+empty50", "Xg_D+empty50_gated")}
    report["part1_Eprime"] = run_systems(d, systems, ["ref_exp3c_M7c_s42", "M7_K10reg_s42"], Ep,
                                         {e: gt[e] for e in Ep}, isA, "E'")
    report["part1_Eprime_entities"] = {"all": len(Ep), "halfB": int((~isA).sum())}
    del d
    gc.collect()
    tb = pa.concat_tables([pq.read_table(p) for p in sorted((K50_ROOT / "eval" / "us").glob("us_*.parquet"))])
    du = Data(tb, ["ref_exp3c_M7c_s42"], extra=("q_addr_empty",))
    del tb
    isAu = np.array([bucket(e, 2) == 0 for e in Eu])
    sysm = masks_for(du)
    report["part1_US"] = run_systems(du, {k: sysm[k] for k in ("A_D", "C_D+blend50", "X_D+empty50", "Xg_D+empty50_gated")},
                                     ["ref_exp3c_M7c_s42"], Eu, {e: gt.get(e, set()) for e in Eu}, isAu, "US")
    del du
    gc.collect()


def build_x_rows(E, member_owner_needed):
    """Rows for X on full E: ranks 10-49 of every empty-address query that is in the V-dense eval
    set or has an E member at ranks 10-49; ranks 0-9 for such queries outside the eval set."""
    OUT_X.mkdir(parents=True, exist_ok=True)
    evq = set(pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["q_id"]).q_id) | \
        set(pd.read_parquet(EXP / "ev_ext_pairs.parquet", columns=["q_id"]).q_id)
    empty = set()
    for k in (2, 3):
        t = ds.dataset(str(WORK_DIR / f"train_s{k}.parquet")).to_table(columns=["entity_id", "addr_norm"],
                                                                        filter=ds.field("country") == "India").to_pandas()
        empty |= set(t.entity_id[t.addr_norm.fillna("").str.strip() == ""])
    Es = set(E)
    sel = []
    for p in sorted((K50_ROOT / "cand_train").glob("India_*.parquet")):
        c = load_compact_chunk(K50_ROOT / "cand_train", p.stem)
        c = c[c.q_id.isin(empty).to_numpy()]
        deepE = set(c.q_id[(c["rank"] >= 10).to_numpy() & c.s1_id.isin(Es).to_numpy()])
        inev = c.q_id.isin(evq).to_numpy()
        keepq = inev | c.q_id.isin(deepE).to_numpy()
        new_q = keepq & ~inev
        rows = c[(keepq & (c["rank"].to_numpy() >= 10)) | (new_q & (c["rank"].to_numpy() < 10))]
        ctx = c[keepq & (c["rank"].to_numpy() < 10)]
        sel.append((rows[["q_id", "s1_id", "score", "rank"]], ctx[["q_id", "s1_id", "score", "rank"]]))
    rows = pd.concat([s[0] for s in sel], ignore_index=True).sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
    ctx = pd.concat([s[1] for s in sel], ignore_index=True)
    return rows, ctx, evq, empty


def part2(report):
    log.set_stage("part2 build X rows (full E)")
    g = eval_truth(us=False)
    E, truth = g["E"], g["truth"]
    fr, fc, fo, fm = OUT_X / "rows.parquet", OUT_X / "ctx.parquet", OUT_X / "own.parquet", OUT_X / "rows_meta.json"
    if fr.exists() and fc.exists() and fo.exists() and fm.exists():             # cached: skip the 9-minute build
        rows, ctx = pd.read_parquet(fr), pd.read_parquet(fc)
        own = pd.read_parquet(fo).set_index("q_id")["own"].to_dict()
        meta = json.loads(fm.read_text())
    else:
        rows, ctx, evq, empty = build_x_rows(E, None)
        new_q = set(rows.q_id) - evq
        tr_q = set(pd.read_parquet(CACHE / "tr_pairs.parquet", columns=["q_id"]).q_id)
        meta = {"rows": len(rows), "queries": int(rows.q_id.nunique()), "new_queries": len(new_q),
                "new_queries_in_exp3c_training": len(new_q & tr_q), "empty_address_india_queries": len(empty)}
        # owners for the new rows' queries
        need = set(rows.q_id)
        own = {}
        with open(GT_PATH, encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, rest = line.rstrip("\r\n").partition("\t")
                for m in rest.split(","):
                    if m in need:
                        own[m] = s1
        Es = set(E)
        assert not any(own.get(q) in Es for q in new_q & tr_q), "LEAKAGE: a training query is owned by E"
        del evq, empty, need, tr_q, new_q
        atomic_parquet(rows, fr)
        atomic_parquet(ctx, fc)
        atomic_parquet(pd.DataFrame({"q_id": list(own), "own": list(own.values())}), fo)
        fm.write_text(json.dumps(meta))
    log(f"X rows {meta['rows']} for {meta['queries']} queries ({meta['new_queries']} new queries outside the eval set; "
        f"{meta['new_queries_in_exp3c_training']} of them were exp3c training queries, none owned by E) {mem_ok(tag='R2')}")
    report["part2_rows"] = meta
    gc.collect()
    log.set_stage("part2 features + scoring")
    models = {k: lgb.Booster(model_file=str(EXP / f"model_{k}.txt")) for k in ("M7c_s42", "M7c_s7")}
    # India S1 only: per-country statistics (s1_counts, name frequency) are identical for India rows
    s1 = ds.dataset(str(WORK_DIR / "train_s1.parquet")).to_table(columns=REC, filter=ds.field("country") == "India").to_pandas().set_index("entity_id")
    s1c = s1_counts(s1[["country", "name_core", "addr_norm"]])
    freq = s1[["country", "name_core"]].value_counts().to_dict()
    qv = rows.q_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, qv[1:] != qv[:-1]])
    bounds, i = [], 0
    while i < len(rows):
        j = starts[np.searchsorted(starts, i + BLOCK)] if i + BLOCK < starts[-1] else len(rows)
        bounds.append((i, j))
        i = j
    ctx_g = ctx.set_index("q_id")
    with Pool(3, initializer=_init, initargs=(str(STATS),)) as pool:
        for n, (i, j) in enumerate(bounds):
            fblk = OUT_X / f"block_{n:03d}.parquet"
            if fblk.exists():
                continue
            t = time.time()
            r = rows.iloc[i:j].reset_index(drop=True)
            cx = ctx_g.loc[ctx_g.index.isin(set(r.q_id))].reset_index()
            q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", set(r.q_id), REC) for k in (2, 3)]).set_index("entity_id")
            X = k50_features(r, cx, s1, q, freq, s1c, pool)
            blk = r[["q_id", "s1_id", "rank"]].copy()
            blk["own"] = r.q_id.map(own)
            blk["y"] = (blk["own"] == blk["s1_id"]).astype(np.int8)
            blk["part"] = "x"
            blk["nc_tsort"], blk["ad_tset"] = X["nc_tsort"].to_numpy(), X["ad_tset"].to_numpy()
            blk["q_addr_empty"] = X["q_addr_empty"].to_numpy()
            for k, m in models.items():
                blk[k] = predict(m, X).astype(np.float32)
            atomic_parquet(blk, fblk)
            log(f"  block {n + 1}/{len(bounds)}: {len(r)} rows {time.time() - t:.0f}s {mem_ok(tag='R2')}")
            del X, q, blk
    del s1, s1c, freq, ctx_g, rows, ctx
    gc.collect()
    log.set_stage("part2 evaluation")
    cols = ["q_id", "s1_id", "rank", "own", "q_addr_empty", "M7c_s42", "M7c_s7"]
    base = pq.read_table(EXP / "probs_clean_india.parquet", columns=cols)
    xb = pa.concat_tables([pq.read_table(p, columns=cols) for p in sorted(OUT_X.glob("block_*.parquet"))])
    xb = xb.cast(base.schema)
    tb = pa.concat_tables([base, xb])
    del base, xb
    d = Data(tb, ["M7c_s42", "M7c_s7"], extra=("q_addr_empty",))
    del tb
    gc.collect()
    isA = np.array([bucket(e, 2) == 0 for e in E])
    sysm = masks_for(d)
    systems = {k: sysm[k] for k in ("A_D", "X_D+empty50", "Xg_D+empty50_gated")}
    res = run_systems(d, systems, ["M7c_s42", "M7c_s7"], E, truth, isA, "E")
    a = res["E|A_D|M7c_s42"]["tunedA"]
    assert a["threshold"] == 0.84 and abs(a["f05"] - D_REF) < 1e-9, f"STOP: D not reproduced on the combined frame: {a}"
    log(f"D reproduced on the combined frame: {a['f05']:.10f}")
    report["part2_E"] = res


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    log = Log("R2_matcher_eval")
    f = REPORTS / "matcher_eval.json"
    report = json.loads(f.read_text()) if f.exists() else {}
    if part in ("part1", "all"):
        part1(report)
        atomic_json(report, f)
    if part in ("part2", "all"):
        part2(report)
        atomic_json(report, f)
    log(f"R2 {part} done in {time.time() - t0:.0f}s")
