"""V3 / M3: entity-aware decision rules on the clean V-dense probabilities (no new inference).

Input: WORK_DIR/exp3/probs_clean_{india,us}.parquet (exp3c, leakage-free M7c models), densified
exactly as vdense.densify (same clone set: distractor queries in first-appearance order, numpy
seed 0, p < 0.94; clones appended after the original rows), D candidate policy ('adapt', 0.5, 10).
Memory-lean: ids are dictionary-encoded integer codes (the first version kept ~27M string rows
and was stopped by the RAM guard).
After the one-to-one query step (each query -> its best S1), rules decide which of an S1's
candidate queries are accepted:
  R0 global threshold t                                   (must reproduce D = 0.970892)
  R1 t1 for the S1's highest-prob query, t2 for its other queries (its best must be accepted)
  R3 query-side margin: p >= t and p - p2 >= m (p2 = the query's 2nd-best candidate prob)
  R2 expected-F0.5 per S1 (prefix search, no subset enumeration): isotonic calibration fitted on
     the tuning half; per S1 candidates sorted by prob, n* = argmax_n 1.25*S_n/(0.25*(sum q+lam)+n)
     vs predicting empty (prod(1-q)*exp(-lam)); only candidates with p >= floor.
Every rule: tuned on half-A, reported once on half-B (+ swapped halves, seed 7 = M7c_s7, US).
Every reported F0.5 is re-checked with src.metrics.macro_f05. Gate G-dec (plan section 11).
Heartbeat log: C:\\aml2026\\v3_exp\\reports\\M3_decision.log (own file handle, flushed).
Output: C:\\aml2026\\v3_exp\\reports\\decision.json

Usage: python -m experiments.v3_decision
"""
import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from experiments.k50_common import atomic_json, bucket, mem_ok
from src.config import WORK_DIR
from src.evaluation import GRID
from src.config import DATA_DIR
from src.metrics import macro_f05
from src.prep import s1_fold

OUT = Path(r"C:\aml2026\v3_exp\reports")
ADAPT = ("adapt", 0.5, 10)
D_REF = 0.9708919456251384          # exp3c D_topk_ar7_s42 half-B (phase0_exp3c.log)
CACHE = WORK_DIR / "m2cache_v2_blend_ad15_India_50"
_LOG = None


def log(*a):
    msg = " ".join([time.strftime("%H:%M:%S"), *map(str, a)])
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def entities_india(_=None, gt=None):
    cache_s1 = set(pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["s1_id"]).s1_id)
    return sorted({s for s in gt if bucket(s) < 100 and s1_fold(s) == 1} & cache_s1)


class Data:
    """Densified candidate rows as integer arrays."""

    def __init__(self, path, models, extra=()):
        t = time.time()
        cols = ["q_id", "s1_id", "rank", "own", *models, *extra]
        if isinstance(path, pa.Table):                      # in-memory table (v3_matcher_eval)
            tb = path.select(cols)
            tb = pa.table({c: (pc.dictionary_encode(tb.column(c)) if c in ("q_id", "s1_id", "own") else tb.column(c))
                           for c in cols})
        else:
            tb = pq.read_table(path, columns=cols, read_dictionary=["q_id", "s1_id", "own"])
        tb = tb.unify_dictionaries()
        q = tb.column("q_id").combine_chunks()
        s = tb.column("s1_id").combine_chunks()
        o = tb.column("own").combine_chunks()
        self.q_names = np.asarray(q.dictionary.to_pylist(), dtype=object)
        self.s1_names = np.asarray(s.dictionary.to_pylist(), dtype=object)
        qc = q.indices.to_numpy(zero_copy_only=False).astype(np.int32)
        sc = s.indices.to_numpy(zero_copy_only=False).astype(np.int32)
        # owner in s1-code space: -1 distractor, -2 owned by an S1 that never appears as a candidate
        own_names = pd.Index(o.dictionary.to_pylist())
        own_map = pd.Index(self.s1_names).get_indexer(own_names)
        own_map = np.where(own_map < 0, -2, own_map).astype(np.int32)
        oi = o.indices
        valid = oi.is_valid().to_numpy(zero_copy_only=False)
        oc = np.full(len(oi), -1, np.int32)
        oidx = pc.fill_null(oi, 0).to_numpy(zero_copy_only=False).astype(np.int64)   # nulls = distractors (masked by `valid`)
        oc[valid] = own_map[oidx[valid]]
        rank = tb.column("rank").to_numpy().astype(np.int16)
        probs = {m: tb.column(m).to_numpy().astype(np.float32) for m in models}
        xcols = {c: tb.column(c).to_numpy().astype(np.float32) for c in extra}
        n0, nq = len(qc), len(self.q_names)
        del tb, q, s, o, oi, oidx, valid
        gc.collect()
        log(f"  loaded {n0} rows, {nq} queries, {len(self.s1_names)} S1 ({time.time() - t:.0f}s) {mem_ok(tag='M3')}")
        # densify exactly as vdense.densify
        dq = pd.unique(qc[oc == -1])                           # distractor queries, first-appearance order
        clone = dq[np.random.default_rng(0).random(len(dq)) < 0.94]
        is_clone_q = np.zeros(nq, bool)
        is_clone_q[clone] = True
        m = is_clone_q[qc]
        self.q = np.concatenate([qc, qc[m] + nq])
        self.s1 = np.concatenate([sc, sc[m]])
        self.own = np.concatenate([oc, oc[m]])
        self.rank = np.concatenate([rank, rank[m]])
        self.prob = {k: np.concatenate([v, v[m]]) for k, v in probs.items()}
        self.x = {k: np.concatenate([v, v[m]]) for k, v in xcols.items()}
        self.nq_orig, self.nq = nq, 2 * nq
        self.n_clone = len(clone)
        log(f"  densified: {len(self.q)} rows ({self.n_clone} cloned queries) {mem_ok(tag='M3')}")

    def q_str(self, codes):
        base = self.q_names[codes % self.nq_orig]
        return np.where(codes >= self.nq_orig, "CLONE_" + base.astype(str), base.astype(str))


class Cands:
    """Per-query best candidate (D policy + one-to-one) restricted to the evaluation S1s."""

    def __init__(self, d, prob, ents):
        # D policy mask (vdense.policy_mask, 'adapt' 0.5 / 10)
        top3 = np.full(d.nq, -1.0, np.float32)
        r3 = d.rank < 3
        np.maximum.at(top3, d.q[r3], prob[r3])
        m = r3 | ((d.rank < ADAPT[2]) & (top3[d.q] < ADAPT[1]))
        q, s1, own, p = d.q[m], d.s1[m], d.own[m], prob[m].astype(np.float64)
        # one-to-one: stable sort by prob desc (== predictions(): sort_values(kind='stable') + drop_duplicates)
        o = np.argsort(-p, kind="stable")
        q, s1, own, p = q[o], s1[o], own[o], p[o]
        k = pd.Series(q).groupby(q).cumcount().to_numpy()      # position of the row within its query
        p2 = np.zeros(d.nq)
        p2[q[k == 1]] = p[k == 1]
        b = k == 0
        q, s1, own, p = q[b], s1[b], own[b], p[b]
        ecode = pd.Index(d.s1_names).get_indexer(ents)        # -1: entity never a candidate (still scored)
        emap = np.full(len(d.s1_names), -1, np.int64)
        emap[ecode[ecode >= 0]] = np.flatnonzero(ecode >= 0)
        self.n_absent = int((ecode < 0).sum())
        keep = emap[s1] >= 0
        self.qc, self.s1c, self.p = q[keep], s1[keep], p[keep]
        self.ei = emap[self.s1c]
        self.p2 = p2[self.qc]
        self.tp = own[keep] == self.s1c
        self.ents = np.asarray(ents)
        self.d = d
        order = np.lexsort((-self.p, self.ei))
        ei_s = self.ei[order]
        new = np.r_[True, ei_s[1:] != ei_s[:-1]]
        start = np.flatnonzero(new)
        grp = np.cumsum(new) - 1
        pos = np.empty(len(order), np.int64)
        pos[order] = np.arange(len(order)) - start[grp]
        self.pos, self.order = pos, order

    def set_truth(self, n_true):
        self.n_true = n_true

    def f_vec(self, acc, sel):
        n_pred = np.bincount(self.ei[acc], minlength=len(self.ents)).astype(np.float64)
        tp = np.bincount(self.ei[acc & self.tp], minlength=len(self.ents)).astype(np.float64)
        nt = self.n_true
        prec = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
        rec = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
        den = 0.25 * prec + rec
        f = np.divide(1.25 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
        f[nt == 0] = (n_pred[nt == 0] == 0).astype(np.float64)
        return f[sel], {"tp": int(tp[sel].sum()), "fp": int((n_pred - tp)[sel].sum()), "fn": int((nt - tp)[sel].sum()),
                        "precision": float(tp[sel].sum() / max(n_pred[sel].sum(), 1)),
                        "recall": float(tp[sel].sum() / max(nt[sel].sum(), 1))}

    def reference(self, acc, truth, ents_sel):
        """Exact re-score with src.metrics.macro_f05 (string ids)."""
        qs = self.d.q_str(self.qc[acc])
        ss = self.d.s1_names[self.s1c[acc]]
        pm = pd.DataFrame({"q": qs, "s": ss}).groupby("s")["q"].agg(set).to_dict()
        return macro_f05(truth, pm, list(ents_sel))["f05_macro"]

    # ---- rules: each returns the accepted-row mask
    def r0(self, t):
        return self.p >= t

    def r1(self, t1, t2):
        first = self.pos == 0
        top_ok = np.zeros(len(self.ents), bool)
        top_ok[self.ei[first]] = self.p[first] >= t1
        return np.where(first, self.p >= t1, (self.p >= t2) & top_ok[self.ei])

    def r3(self, t, m):
        return (self.p >= t) & (self.p - self.p2 >= m)

    def r2(self, q, lam, floor):
        ok = self.p >= floor
        qq = np.where(ok, q, 0.0)
        o = self.order
        ei_s, q_s = self.ei[o], qq[o]
        S = pd.Series(q_s).groupby(ei_s).cumsum().to_numpy()
        tot = np.bincount(ei_s, weights=q_s, minlength=len(self.ents))
        n = self.pos[o] + 1
        En = np.where(q_s > 0, 1.25 * S / (0.25 * (tot[ei_s] + lam) + n), -1.0)
        bestE = np.full(len(self.ents), -1.0)
        np.maximum.at(bestE, ei_s, En)
        E0 = np.exp(np.bincount(ei_s, weights=np.log1p(-np.minimum(q_s, 1 - 1e-9)), minlength=len(self.ents)) - lam)
        cand_n = np.where(En >= bestE[ei_s] - 1e-15, n, np.iinfo(np.int64).max)
        ns = np.full(len(self.ents), np.iinfo(np.int64).max)
        np.minimum.at(ns, ei_s, cand_n)
        ns[(bestE <= E0) | (ns == np.iinfo(np.int64).max)] = 0
        acc = np.empty(len(o), bool)
        acc[o] = n <= ns[ei_s]
        return acc & ok


def rule_grid(c, q):
    tg = [float(t) for t in GRID if t >= 0.3]
    return {"R0": [((t,), (lambda t=t: c.r0(t))) for t in tg],
            "R1": [((a, b), (lambda a=a, b=b: c.r1(a, b))) for a, b in itertools.product(tg, tg) if a >= 0.5 and b >= 0.5],
            "R3": [((t, m), (lambda t=t, m=m: c.r3(t, m))) for t in tg for m in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)],
            "R2": [((lam, fl), (lambda lam=lam, fl=fl: c.r2(q, lam, fl)))
                   for lam in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) for fl in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]}


def run(c, selT, selR, tag, truth, ents_R):
    t = time.time()
    rows = selT[c.ei]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(c.p[rows], c.tp[rows].astype(float))
    q = iso.predict(c.p)
    res = {}
    for name, cands in rule_grid(c, q).items():
        tr = time.time()
        best = None
        for i, (params, fn) in enumerate(cands):
            fT, _ = c.f_vec(fn(), selT)
            if best is None or fT.mean() > best[0] + 1e-12:
                best = (fT.mean(), params, fn)
            if time.time() - tr > 60 and i % 200 == 0:
                log(f"    {tag} {name}: {i + 1}/{len(cands)} parameter sets")
        acc = best[2]()
        fR, st = c.f_vec(acc, selR)
        ref = c.reference(acc, truth, ents_R)
        assert abs(ref - fR.mean()) < 1e-9, f"{name}: fast {fR.mean()} != macro_f05 {ref}"
        res[name] = {"params": best[1], "tune_f05": best[0], "f05": float(fR.mean()), **st, "_fvec": fR}
        log(f"  {tag} {name} done ({len(cands)} sets, {time.time() - tr:.0f}s): params={best[1]} "
            f"tuneF={best[0]:.6f} reportF={fR.mean():.6f} (== macro_f05)")
    base = res["R0"]["_fvec"]
    for k, r in res.items():
        from experiments.vdense import paired_ci
        g, lo, hi = paired_ci(r["_fvec"], base)
        r.update({"gain_vs_R0": g, "ci_lo": lo, "ci_hi": hi})
        log(f"  {tag} {k} {r['params']} f05={r['f05']:.6f} gain={g:+.5f} [{lo:+.5f},{hi:+.5f}] "
            f"P={r['precision']:.4f} R={r['recall']:.4f} fp={r['fp']} fn={r['fn']}")
    log(f"  {tag} finished in {time.time() - t:.0f}s {mem_ok(tag='M3')}")
    return res


def strip(res):
    return {k: {kk: v for kk, v in r.items() if not kk.startswith("_")} for k, r in res.items()}


if __name__ == "__main__":
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _LOG = open(OUT / "M3_decision.log", "a", encoding="utf-8")
    log("M3 start (memory-lean version)")
    # ground truth for the evaluation entities only (the full 2.2M-entity dict costs ~1.5 GB)
    s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
    us_eval = set(s for s in s1c.entity_id[s1c.country == "US"] if bucket(s) < 50)
    del s1c
    cache_s1 = set(pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["s1_id"]).s1_id)
    gt = {}
    with open(DATA_DIR / "train" / "train_ground_truth.tsv", encoding="utf-8") as f:     # == io_utils.read_ground_truth
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            if s1 in us_eval or (s1 in cache_s1 and bucket(s1) < 100 and s1_fold(s1) == 1):
                gt[s1] = {x for x in rest.split(",") if x}
    E = sorted(s for s in gt if s not in us_eval)            # == entities_india(): fold-1, bucket<100, in cache
    Eu = sorted(us_eval)
    truth = {e: gt[e] for e in E}
    truth_u = {e: gt.get(e, set()) for e in Eu}
    del gt, cache_s1
    gc.collect()
    log(f"truth kept for {len(E)} India + {len(Eu)} US evaluation entities {mem_ok(tag='M3')}")
    isA = np.array([bucket(e, 2) == 0 for e in E])
    nt = np.array([len(truth[e]) for e in E], float)
    log(f"India V-dense E: {len(E)} entities (A {int(isA.sum())}, B {int((~isA).sum())})")
    d = Data(WORK_DIR / "exp3" / "probs_clean_india.parquet", ["M7c_s42", "M7c_s7"])
    report = {"population": "V-dense E (India fold-1 bucket<100), exp3c clean models", "d_ref": D_REF,
              "rows_densified": int(len(d.q)), "cloned_queries": d.n_clone, "entities": len(E)}
    EA, EB = list(np.asarray(E)[isA]), list(np.asarray(E)[~isA])
    for mk in ("M7c_s42", "M7c_s7"):
        c = Cands(d, d.prob[mk], E)
        c.set_truth(nt)
        log(f"{mk}: {len(c.p)} one-to-one candidates on E {mem_ok(tag='M3')}")
        res = run(c, isA, ~isA, f"{mk} A->B", truth, EB)
        if mk == "M7c_s42":
            r0 = res["R0"]
            assert r0["params"] == (0.84,) and abs(r0["f05"] - D_REF) < 1e-9, f"STOP: R0 does not reproduce D: {r0['params']} {r0['f05']}"
            log(f"R0 reproduces D half-B exactly: {r0['f05']:.10f} (ref {D_REF:.10f}), threshold {r0['params'][0]}")
            report["swapped"] = strip(run(c, ~isA, isA, f"{mk} swapped B->A", truth, EA))
        report[mk] = strip(res)
        del c
    del d
    # US spot-check (vdense.load_us population: US S1 in bucket < 50)
    gc.collect()
    du = Data(WORK_DIR / "exp3" / "probs_clean_us.parquet", ["M7c_s42"])
    cu = Cands(du, du.prob["M7c_s42"], Eu)
    cu.set_truth(np.array([len(truth_u[e]) for e in Eu], float))
    isAu = np.array([bucket(e, 2) == 0 for e in Eu])
    report["US"] = strip(run(cu, isAu, ~isAu, "US A->B", truth_u, list(np.asarray(Eu)[~isAu])))
    gate = {}
    for k in ("R1", "R2", "R3"):
        m, s7, sw, us = report["M7c_s42"][k], report["M7c_s7"][k], report["swapped"][k], report["US"][k]
        checks = {"gain>=0.0010": m["gain_vs_R0"] >= 0.0010, "ci_lo>0": m["ci_lo"] > 0,
                  "swapped_gain>0": sw["gain_vs_R0"] > 0, "seed7_gain>0": s7["gain_vs_R0"] > 0,
                  "us_gain>=-0.0005": us["gain_vs_R0"] >= -0.0005,
                  "precision>=D-0.001": m["precision"] >= report["M7c_s42"]["R0"]["precision"] - 0.001}
        gate[k] = {"pass": all(checks.values()), **checks, "gain": m["gain_vs_R0"], "params": m["params"]}
    report["gate_G_dec"] = gate
    report["secs"] = round(time.time() - t0)
    atomic_json(report, OUT / "decision.json")
    log("gate G-dec: " + json.dumps(gate, default=str))
    log(f"M3 done in {time.time() - t0:.0f}s")
