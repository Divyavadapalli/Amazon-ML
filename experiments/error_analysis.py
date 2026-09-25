"""Error analysis of a saved validation run (exp_match2 artefacts).

Separates BLOCKING misses (true pair never generated) from MATCHING errors (pair present
but rejected / lost to another S1), buckets false merges by cause, and measures each
bucket's contribution to lost macro F0.5 by counterfactually removing it.

Usage: python -m experiments.error_analysis <run_tag> <threshold>
"""
import json
import sys
import zlib

import numpy as np
import pandas as pd

from src.config import WORK_DIR
from src.evaluation import predictions, sweep
from src.io_utils import read_ground_truth, read_records
from src.prep import s1_fold

REC = ["entity_id", "name_core", "addr_norm", "business_name", "business_address"]


def bucket(x, mod=1000):
    return (zlib.crc32(x.encode()) >> 1) % mod


if __name__ == "__main__":
    tag, thr = sys.argv[1], float(sys.argv[2])
    ev = pd.read_parquet(WORK_DIR / f"m2_eval_{tag}.parquet")
    X = pd.read_parquet(WORK_DIR / f"m2_Xev_{tag}.parquet",
                        columns=["nc_tsort", "ad_tset", "q_addr_empty", "q_nonascii_name", "s1_name_freq",
                                 "num_conflict", "ret_rank"])
    ev = pd.concat([ev.reset_index(drop=True), X], axis=1)
    gt = read_ground_truth()
    ents = sorted(set(ev.s1_id) & {s for s in gt if bucket(s) < 100 and s1_fold(s) == 1})
    ents = [e for e in ents if bucket(e, 2) == 1]            # the reported (eval) half only
    truth = {e: gt[e] for e in ents}
    owner = {m: s for s, ms in gt.items() for m in ms}

    pred = predictions(ev, ev["prob"].to_numpy(), one_to_one=True)
    pred = pred[(pred.prob >= thr) & pred.s1_id.isin(truth)]
    base = sweep(pred.assign(prob=1.0), truth, ents, [0.5]).iloc[0]
    print(f"eval entities {len(ents)}  F0.5={base.f05_macro:.4f} tp={base.tp} fp={base.fp} fn={base.fn}")

    # ---------------- false negatives: blocking vs matching
    pairs_set = set(zip(ev.q_id, ev.s1_id))
    pred_set = set(zip(pred.q_id, pred.s1_id))
    best_s1 = predictions(ev, ev["prob"].to_numpy(), True).set_index("q_id")["s1_id"]
    fn_rows = []
    for e in ents:
        for m in truth[e]:
            if (m, e) in pred_set:
                continue
            if (m, e) not in pairs_set:
                kind = "blocking_miss"
            elif best_s1.get(m) != e:
                kind = "matcher_lost_to_other_s1"
            else:
                kind = "matcher_below_threshold"
            fn_rows.append((m, e, kind))
    fn = pd.DataFrame(fn_rows, columns=["q_id", "s1_id", "kind"])
    print("\nFALSE NEGATIVES by stage:\n", fn.kind.value_counts().to_string())

    # ---------------- false positives: cause buckets
    fp = pred[[(q, s) not in {(m, e) for m in truth[s]} for q, s in zip(pred.q_id, pred.s1_id)]] \
        if False else pred[pred.q_id.map(owner).to_numpy() != pred.s1_id.to_numpy()]
    fp = fp.merge(ev[["q_id", "s1_id", "nc_tsort", "ad_tset", "q_addr_empty", "q_nonascii_name",
                      "s1_name_freq", "num_conflict", "ret_rank"]], on=["q_id", "s1_id"])
    fp["q_owner"] = fp.q_id.map(owner)
    fp["target_is_singleton"] = fp.s1_id.map(lambda e: not truth[e])

    def fp_cat(r):
        if pd.isna(r.q_owner):
            src = "distractor"
        else:
            src = "other_entity"
        if r.q_addr_empty:
            return f"{src}:addr_empty"
        if r.nc_tsort >= 90 and r.ad_tset >= 90:
            return f"{src}:name+addr_both_high"
        if r.nc_tsort >= 90:
            return f"{src}:same_name_diff_addr"
        if r.ad_tset >= 90:
            return f"{src}:same_addr_diff_name"
        return f"{src}:partial_both"
    fp["cat"] = fp.apply(fp_cat, axis=1)
    print("\nFALSE POSITIVES by cause:\n", fp.cat.value_counts().to_string())
    print("FP on true singletons:", int(fp.target_is_singleton.sum()), "of", len(fp))

    # ---------------- F0.5 lost per category (counterfactual: remove that error class)
    def f_with(p):
        return sweep(p.assign(prob=1.0), truth, ents, [0.5]).iloc[0].f05_macro
    out = {"F05": base.f05_macro}
    for c in fp.cat.unique():
        drop = set(zip(fp.q_id[fp.cat == c], fp.s1_id[fp.cat == c]))
        out[f"+fix FP {c}"] = f_with(pred[[(q, s) not in drop for q, s in zip(pred.q_id, pred.s1_id)]]) - base.f05_macro
    for k in fn.kind.unique():
        add = fn[fn.kind == k][["q_id", "s1_id"]]
        p2 = pd.concat([pred[["q_id", "s1_id"]], add]).drop_duplicates()
        out[f"+fix FN {k}"] = f_with(p2) - base.f05_macro
    print("\nF0.5 gain if an error class were fixed (eval half):")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]):
        print(f"  {k:45s} {v:+.4f}")

    # ---------------- samples for inspection
    recs = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
    qids = set(fp.q_id.sample(min(12, len(fp)), random_state=0)) | set(fn[fn.kind != "blocking_miss"].q_id.sample(12, random_state=0))
    qr = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", qids, REC) for k in (2, 3)]).set_index("entity_id")
    print("\n=== sample FALSE POSITIVES (pred S1 | query | true owner)")
    for _, r in fp[fp.q_id.isin(qids)].iterrows():
        a, b = recs.loc[r.s1_id], qr.loc[r.q_id]
        own = recs.loc[r.q_owner].business_name if isinstance(r.q_owner, str) else "-- (distractor)"
        print(f"[{r['cat']}] p={r.prob:.2f}\n   S1 : {a.business_name} | {a.business_address}\n   Q  : {b.business_name} | {b.business_address}\n   own: {own}")
    print("\n=== sample MATCHER FALSE NEGATIVES")
    for _, r in fn[(fn.kind != "blocking_miss") & fn.q_id.isin(qids)].iterrows():
        a, b = recs.loc[r.s1_id], qr.loc[r.q_id]
        pr = ev[(ev.q_id == r.q_id) & (ev.s1_id == r.s1_id)].prob.iloc[0]
        print(f"[{r.kind}] p={pr:.2f}\n   S1 : {a.business_name} | {a.business_address}\n   Q  : {b.business_name} | {b.business_address}")
    fn.to_parquet(WORK_DIR / f"ea_fn_{tag}.parquet", index=False)
    fp.to_parquet(WORK_DIR / f"ea_fp_{tag}.parquet", index=False)
    with open("experiments/error_analysis.jsonl", "a") as f:
        f.write(json.dumps({"run": tag, "threshold": thr, **{k: float(v) for k, v in out.items()},
                            "fn_by_stage": fn.kind.value_counts().to_dict(),
                            "fp_by_cause": fp.cat.value_counts().to_dict()}) + "\n")
