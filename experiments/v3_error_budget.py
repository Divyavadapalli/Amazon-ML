"""V3 / M2: exact, verified error budget of the clean System D on V-dense half-B.

System: exp3c M7c_s42 probabilities (WORK_DIR/exp3/probs_clean_india.parquet, read-only),
densified exactly as vdense.densify, D policy ('adapt', 0.5, 10), one-to-one, threshold 0.84
(tuned on half-A; must reproduce 0.9708919456). Memory-lean (integer codes; truth for E only).

Every half-B true pair that is not predicted gets exactly one stage:
  retrieval_miss   owner not in the query's top-10 (never a candidate)
  truncated        owner at rank 3-9 but the D gate kept ranks 3-9 closed for that query
  lost_to_other    the query's best candidate is another S1
  below_threshold  the query's best candidate is the owner but prob < 0.84
Every accepted wrong pair: fp_distractor (query owned by nobody) / fp_other_entity; near-copy
flag (nc_tsort >= 90 and ad_tset >= 90).
Loss decomposition (exact, sums to 1 - D): fixing, cumulatively, (1) all false merges,
(2) below-threshold misses, (3) lost-to-other misses, (4) truncated misses, (5) retrieval
misses. After (4) the score equals the top-10 oracle on half-B; after (5) it is 1.
Also single-fix counterfactuals from D, overlapping text tags (script, empty address, exact
name, name / address / number conflict, legal-form difference, no shared name token) and
segment breakdowns (S2/S3, script, empty address, core-exact, S1 name frequency).
Exact verification: D itself and every cumulative/stage counterfactual is re-scored with
src.metrics.macro_f05 on string id sets.

Logs: reports/M2_error_budget.log (heartbeat every 60 s).
Output: reports/error_budget_D.json, reports/examples_D.parquet
Usage: python -m experiments.v3_error_budget
"""
import json
import re
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from experiments.k50_common import atomic_json, atomic_parquet, bucket, mem_ok
from experiments.v3_common import REPORTS, Log, eval_truth
from experiments.v3_decision import ADAPT, D_REF, Data
from src.config import WORK_DIR
from src.features import LEGAL_VALS
from src.metrics import macro_f05

THR = 0.84
COLS = ["entity_id", "business_name", "business_address", "name_norm", "name_core", "addr_norm", "addr_digits"]
_NONASCII = re.compile(r"[^\x00-\x7f]")


def records(ids):
    ids = pd.unique(pd.Series(list(ids)))
    parts = []
    for f in ("train_s1", "train_s2", "train_s3"):
        t = ds.dataset(str(WORK_DIR / f"{f}.parquet")).to_table(columns=COLS, filter=ds.field("entity_id").isin(ids))
        parts.append(t.to_pandas())
    return pd.concat(parts).drop_duplicates("entity_id").set_index("entity_id")


def jacc(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if (a or b) else 1.0


def tags(pairs, rec):
    """Overlapping text tags for (q_id, s1_id) pairs (clone prefix ignored)."""
    q = rec.reindex(pairs["q_id"].str.replace("CLONE_", "", regex=False).to_numpy())
    s = rec.reindex(pairs["s1_id"].to_numpy())
    qn, sn = q["name_core"].fillna("").to_numpy(), s["name_core"].fillna("").to_numpy()
    qa, sa = q["addr_norm"].fillna("").to_numpy(), s["addr_norm"].fillna("").to_numpy()
    qd, sd = q["addr_digits"].fillna("").to_numpy(), s["addr_digits"].fillna("").to_numpy()
    qnn, snn = q["name_norm"].fillna("").to_numpy(), s["name_norm"].fillna("").to_numpy()
    raw = q["business_name"].fillna("").to_numpy()
    legal = lambda x: frozenset(t for t in x.split() if t in LEGAL_VALS)
    t = pd.DataFrame(index=pairs.index)
    t["script"] = [bool(_NONASCII.search(r)) for r in raw]
    t["addr_empty"] = [a.strip() == "" for a in qa]
    t["name_exact"] = qn == sn
    nj = np.array([jacc(a, b) for a, b in zip(qn, sn)])
    t["name_conflict"] = nj < 0.5
    t["no_shared_name_token"] = nj == 0
    t["addr_conflict"] = [(a.strip() != "") and jacc(a, b) < 0.5 for a, b in zip(qa, sa)]
    t["number_conflict"] = [bool(a.split()) and bool(b.split()) and set(a.split()) != set(b.split()) for a, b in zip(qd, sd)]
    t["legal_diff"] = [legal(a) != legal(b) for a, b in zip(qnn, snn)]
    return t


def f_vec(tp, npred, nt):
    prec = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    rec = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
    den = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    f[nt == 0] = (npred[nt == 0] == 0).astype(float)
    return f


if __name__ == "__main__":
    t0 = time.time()
    log = Log("M2_error_budget")
    log.set_stage("ground truth (E only)")
    g = eval_truth(us=False)
    E, truth = g["E"], g["truth"]
    B = [e for e in E if bucket(e, 2) == 1]
    truthB = {e: truth[e] for e in B}
    log(f"E {len(E)} entities, half-B {len(B)}, true pairs B {sum(len(v) for v in truthB.values())} {mem_ok(tag='M2')}")

    log.set_stage("load + densify probabilities")
    d = Data(WORK_DIR / "exp3" / "probs_clean_india.parquet", ["M7c_s42"], extra=("nc_tsort", "ad_tset"))
    log(f"{len(d.q)} densified rows {mem_ok(tag='M2')}")
    prob = d.prob["M7c_s42"]
    NS = np.int64(len(d.s1_names))

    log.set_stage("D policy + one-to-one")
    top3 = np.full(d.nq, -1.0, np.float32)
    r3 = d.rank < 3
    np.maximum.at(top3, d.q[r3], prob[r3])
    m = r3 | ((d.rank < ADAPT[2]) & (top3[d.q] < ADAPT[1]))
    idx = np.flatnonzero(m)
    o = idx[np.argsort(-prob[idx].astype(np.float64), kind="stable")]     # == predictions(): stable desc sort
    qs = d.q[o]
    first = ~pd.Series(qs).duplicated().to_numpy()
    brow = o[first]                                                       # best row per query
    best_s1 = np.full(d.nq, -1, np.int64)
    best_p = np.full(d.nq, -1.0)
    best_row = np.full(d.nq, -1, np.int64)
    best_s1[d.q[brow]] = d.s1[brow]
    best_p[d.q[brow]] = prob[brow]
    best_row[d.q[brow]] = brow

    ecode = pd.Index(d.s1_names).get_indexer(B)
    emap = np.full(len(d.s1_names), -1, np.int64)
    emap[ecode[ecode >= 0]] = np.flatnonzero(ecode >= 0)
    nt = np.array([len(truthB[e]) for e in B], float)
    accq = np.flatnonzero((best_p >= THR) & (best_s1 >= 0))
    accq = accq[emap[best_s1[accq]] >= 0]                                  # accepted queries whose S1 is in B
    acc_e = emap[best_s1[accq]]
    own_best = d.own[best_row[accq]]
    ok = own_best == best_s1[accq]
    tp = np.bincount(acc_e[ok], minlength=len(B)).astype(float)
    npred = np.bincount(acc_e, minlength=len(B)).astype(float)
    base = f_vec(tp, npred, nt).mean()
    assert abs(base - D_REF) < 1e-9, f"STOP: base {base} != D {D_REF}"
    pred_sets = pd.DataFrame({"s": d.s1_names[best_s1[accq]], "q": d.q_str(accq)}).groupby("s")["q"].agg(set).to_dict()
    ref = macro_f05(truthB, pred_sets, B)
    assert abs(ref["f05_macro"] - base) < 1e-9
    log(f"D reproduced: {base:.10f} == macro_f05 {ref['f05_macro']:.10f}; tp={int(tp.sum())} fp={int((npred - tp).sum())} "
        f"fn={int((nt - tp).sum())} {mem_ok(tag='M2')}")

    log.set_stage("classify false negatives / false merges")
    orig = d.q < d.nq_orig
    keys_all = np.unique(d.q[orig].astype(np.int64) * NS + d.s1[orig])
    km = m & orig
    keys_mask = np.unique(d.q[km].astype(np.int64) * NS + d.s1[km])
    rk_keys = d.q[orig].astype(np.int64) * NS + d.s1[orig]
    rk_sorted = np.argsort(rk_keys)
    tru = pd.DataFrame([(q, e) for e in B for q in truthB[e]], columns=["q_id", "s1_id"])
    tq = pd.Index(d.q_names).get_indexer(tru.q_id)
    te = pd.Index(d.s1_names).get_indexer(tru.s1_id)
    key = np.where((tq >= 0) & (te >= 0), tq.astype(np.int64) * NS + te, -1)
    retrieved = (key >= 0) & np.isin(key, keys_all)
    in_mask = (key >= 0) & np.isin(key, keys_mask)
    bs = np.where(tq >= 0, best_s1[np.maximum(tq, 0)], -1)
    bp = np.where(tq >= 0, best_p[np.maximum(tq, 0)], -1.0)
    found = in_mask & (bs == te) & (bp >= THR)
    assert int(found.sum()) == int(tp.sum()), "TP count mismatch"
    stage = np.select([found, ~retrieved, retrieved & ~in_mask, in_mask & (bs != te)],
                      ["tp", "retrieval_miss", "truncated", "lost_to_other"], "below_threshold")
    pos = np.searchsorted(rk_keys[rk_sorted], key)
    pos = np.clip(pos, 0, len(rk_sorted) - 1)
    has = retrieved & (rk_keys[rk_sorted][pos] == key)
    tru["rank"] = np.where(has, d.rank[np.flatnonzero(orig)][rk_sorted[pos]], -1)
    tru["stage"], tru["best_prob"] = stage, np.where(np.isin(stage, ["lost_to_other", "below_threshold"]), bp, np.nan)
    fp = pd.DataFrame({"q_id": d.q_str(accq[~ok]), "s1_id": d.s1_names[best_s1[accq[~ok]]], "prob": best_p[accq[~ok]]})
    fp["stage"] = np.where(own_best[~ok] == -1, "fp_distractor", "fp_other_entity")
    br = best_row[accq[~ok]]
    fp["near_copy"] = (d.x["nc_tsort"][br] >= 90) & (d.x["ad_tset"][br] >= 90)
    fn = tru[tru.stage != "tp"].reset_index(drop=True)
    log(f"FN {len(fn)} {fn.stage.value_counts().to_dict()}  FP {len(fp)} {fp.stage.value_counts().to_dict()}")
    del d, prob, m, idx, o, qs, keys_all, keys_mask, rk_keys, rk_sorted, top3

    log.set_stage("text tags for all true pairs + false merges")
    rec = records(set(tru.q_id) | set(tru.s1_id) | set(fp.q_id.str.replace("CLONE_", "", regex=False)) | set(fp.s1_id))
    tru = pd.concat([tru, tags(tru, rec)], axis=1)
    fp = pd.concat([fp.reset_index(drop=True), tags(fp.reset_index(drop=True), rec)], axis=1)
    fn = tru[tru.stage != "tp"].reset_index(drop=True)
    s1n = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["country", "name_core"])
    vc = s1n.name_core[s1n.country == "India"].value_counts()
    del s1n
    tru["s1_name_freq>=5"] = rec["name_core"].reindex(tru.s1_id).map(vc).fillna(0).to_numpy() >= 5
    tru["S3"] = tru.q_id.str.startswith("S3").to_numpy()
    fn = tru[tru.stage != "tp"].reset_index(drop=True)
    log(f"tags done {mem_ok(tag='M2')}")

    log.set_stage("counterfactuals + exact verification")
    eix = pd.Series(np.arange(len(B)), index=B)

    def f_after(add_fn=None, drop_fp=None):
        a = np.bincount(eix.loc[fn.s1_id[add_fn]], minlength=len(B)).astype(float) if add_fn is not None else 0.0
        r = np.bincount(eix.loc[fp.s1_id[drop_fp]], minlength=len(B)).astype(float) if drop_fp is not None else 0.0
        return f_vec(tp + a, npred + a - r, nt).mean()

    def verify(add_fn, drop_fp, fast):
        ps = {k: set(v) for k, v in pred_sets.items()}
        if drop_fp is not None:
            for q, s in zip(fp.q_id[drop_fp], fp.s1_id[drop_fp]):
                ps[s].discard(q)
        if add_fn is not None:
            for q, s in zip(fn.q_id[add_fn], fn.s1_id[add_fn]):
                ps.setdefault(s, set()).add(q)
        r = macro_f05(truthB, ps, B)["f05_macro"]
        assert abs(r - fast) < 1e-9, f"counterfactual mismatch {fast} vs {r}"
        return r

    allfp = np.ones(len(fp), bool)
    cum, add = [], np.zeros(len(fn), bool)
    prev = base
    f1 = f_after(None, allfp); verify(None, allfp, f1)
    cum.append(("false_merges_all", f1 - prev)); prev = f1
    for stg in ("below_threshold", "lost_to_other", "truncated", "retrieval_miss"):
        add = add | (fn.stage == stg).to_numpy()
        f = f_after(add, allfp); verify(add, allfp, f)
        cum.append((stg, f - prev)); prev = f
        if stg == "truncated":
            f_oracle10 = f
    assert abs(prev - 1.0) < 1e-9, f"fixing everything gives {prev}, not 1"
    single = {}
    for stg in ("retrieval_miss", "truncated", "lost_to_other", "below_threshold"):
        mk = (fn.stage == stg).to_numpy()
        single[stg] = {"count": int(mk.sum()), "share_of_fn": float(mk.mean()), "f05_gain_if_fixed": verify(mk, None, f_after(mk, None)) - base}
    for stg in ("fp_distractor", "fp_other_entity"):
        mk = (fp.stage == stg).to_numpy()
        single[stg] = {"count": int(mk.sum()), "share_of_fp": float(mk.mean()), "f05_gain_if_fixed": verify(None, mk, f_after(None, mk)) - base}
    mk = fp.near_copy.to_numpy()
    single["fp_near_copy"] = {"count": int(mk.sum()), "share_of_fp": float(mk.mean()), "f05_gain_if_fixed": f_after(None, mk) - base}
    log("counterfactuals verified with macro_f05")

    tag_cols = ["script", "addr_empty", "name_exact", "name_conflict", "no_shared_name_token", "addr_conflict",
                "number_conflict", "legal_diff"]
    fn_tags, fp_tags = {}, {}
    for c in tag_cols:
        mf, mp = fn[c].to_numpy(bool), fp[c].to_numpy(bool)
        fn_tags[c] = {"count": int(mf.sum()), "share_of_fn": float(mf.mean()), "f05_gain_if_fixed": f_after(mf, None) - base,
                      "by_stage": fn[mf].stage.value_counts().to_dict()}
        fp_tags[c] = {"count": int(mp.sum()), "share_of_fp": float(mp.mean()), "f05_gain_if_fixed": f_after(None, mp) - base}
    segments = {}
    for name, col in (("S2", ~tru.S3), ("S3", tru.S3), ("script", tru.script), ("latin", ~tru.script),
                      ("addr_empty", tru.addr_empty), ("addr_present", ~tru.addr_empty),
                      ("core_exact", tru.name_exact), ("core_nonexact", ~tru.name_exact),
                      ("s1_name_freq>=5", tru["s1_name_freq>=5"]), ("s1_name_freq<5", ~tru["s1_name_freq>=5"])):
        t = tru[col.to_numpy(bool)]
        mf = (fn.q_id.isin(set(t.q_id))).to_numpy()
        segments[name] = {"true_pairs": len(t), "recall": float((t.stage == "tp").mean()),
                          "retrieval_recall@10": float((t.stage != "retrieval_miss").mean()),
                          "fn_by_stage": t[t.stage != "tp"].stage.value_counts().to_dict(),
                          "f05_gain_if_all_fn_fixed": f_after(mf, None) - base}
    out = {"system": "D clean (M7c_s42, adapt 0.5/10, thr 0.84), V-dense half-B", "D_f05": base,
           "D_verified_macro_f05": ref["f05_macro"], "entities_B": len(B), "true_pairs_B": int(nt.sum()),
           "tp": int(tp.sum()), "fp": len(fp), "fn": len(fn),
           "precision": float(tp.sum() / npred.sum()), "recall": float(tp.sum() / nt.sum()),
           "loss_total": 1 - base,
           "cumulative_decomposition": {k: v for k, v in cum},
           "top10_oracle_halfB": f_oracle10,
           "loss_groups": {"retrieval_loss (owner not in top-10)": dict(cum)["retrieval_miss"],
                           "candidate_gate_loss (ranks 3-9 not scored)": dict(cum)["truncated"],
                           "matcher_ranking_loss (owner lost to another S1)": dict(cum)["lost_to_other"],
                           "decision_loss (false merges + owner below threshold)": dict(cum)["false_merges_all"] + dict(cum)["below_threshold"]},
           "single_fix_from_D": single, "fn_tags": fn_tags, "fp_tags": fp_tags, "segments": segments,
           "fn_rank_of_owner": {k: gg["rank"].value_counts().sort_index().to_dict() for k, gg in fn.groupby("stage")},
           "fn_best_prob_quantiles": {k: gg.best_prob.quantile([.1, .25, .5, .75, .9]).to_dict()
                                      for k, gg in fn.groupby("stage") if gg.best_prob.notna().any()},
           "fp_prob_quantiles": fp.prob.quantile([.1, .25, .5, .75, .9]).to_dict(),
           "notes": "cumulative decomposition order: false merges -> below threshold -> lost to other -> truncated -> retrieval; "
                    "every step re-scored with src.metrics.macro_f05; tags overlap", "secs": round(time.time() - t0)}
    atomic_json(out, REPORTS / "error_budget_D.json")
    ex = pd.concat([fn.assign(kind="FN"), fp.assign(kind="FP")], ignore_index=True)
    ex = ex.join(rec[["business_name", "business_address"]], on="s1_id").rename(columns={"business_name": "s1_name", "business_address": "s1_addr"})
    qraw = rec[["business_name", "business_address"]].reindex(ex.q_id.str.replace("CLONE_", "", regex=False).to_numpy())
    ex["q_name"], ex["q_addr"] = qraw.business_name.to_numpy(), qraw.business_address.to_numpy()
    atomic_parquet(ex, REPORTS / "examples_D.parquet")
    log("RESULT " + json.dumps({k: out[k] for k in ("D_f05", "tp", "fp", "fn", "loss_total", "cumulative_decomposition",
                                                     "top10_oracle_halfB", "loss_groups")}, default=float))
    log(f"M2 done in {time.time() - t0:.0f}s")
