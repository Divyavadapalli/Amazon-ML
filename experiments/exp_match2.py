"""Realistic end-to-end validation on cached FULL training candidates.

Uses candidates produced by `python -m src.candidates train <tag>` (every training S2/S3
record retrieved against the full S1 index), so competition and distractor density match
the test set.

  E        ~5 % hash-sample of fold-1 S1 entities (fold 0 trained the translit dictionary) (evaluation; threshold tuned on half, reported on the other)
  Q_eval   all queries owned by E  +  all queries with an E member in their retrieval top-3
           (a query whose E candidate sits at rank >= 3 is almost never assigned to it;
            the share of assignments made from rank >= 3 is reported to verify this)
  Q_train  uniform 10 % of all queries, disjoint from Q_eval

Model variants: rule baseline, logistic regression baseline, LightGBM; assignment with and
without the one-S1-per-query constraint. Appends to experiments/match2_results.jsonl.

Usage: python -m experiments.exp_match2 <cand_tag> [run_tag] [drop_groups] [country]
  country: restrict to one country (retrieval never crosses countries, so a country whose
           chunks are all present is exactly final even while other countries still run).
"""
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src import candidates
from src.config import WORK_DIR
from src.features import pair_features
from src.io_utils import read_ground_truth, read_records
from src.evaluation import tune_and_report
from src.matching import train_model
from src.metrics import macro_f05

REC = ["entity_id", "country", "name_norm", "name_core", "name_alt", "addr_norm", "addr_digits",
       "pobox", "business_name", "business_address"]


def bucket(x, mod=1000):
    return (zlib.crc32(x.encode()) >> 1) % mod


def s1_name_freq(split="train"):
    s1 = pd.read_parquet(WORK_DIR / f"{split}_s1.parquet", columns=["country", "name_core"])
    return s1.value_counts().to_dict()


def featurize(pairs, s1i, qi, freq, chunk=2_000_000):
    """Features in chunks of whole queries (per-query features need all of a query's pairs)."""
    pairs = pairs.sort_values(["q_id", "rank"], kind="stable").reset_index(drop=True)
    starts = np.flatnonzero(np.r_[True, pairs["q_id"].to_numpy()[1:] != pairs["q_id"].to_numpy()[:-1]])
    out, i = [], 0
    while i < len(pairs):
        j = starts[np.searchsorted(starts, i + chunk)] if i + chunk < starts[-1] else len(pairs)
        out.append(pair_features(pairs.iloc[i:j].reset_index(drop=True), s1i, qi, freq))
        i = j
    return pairs, pd.concat(out, ignore_index=True)


def assign(pairs, prob, thr, one_to_one=True):
    p = pd.DataFrame({"q_id": pairs["q_id"].to_numpy(), "s1_id": pairs["s1_id"].to_numpy(),
                      "prob": prob, "rank": pairs["rank"].to_numpy()})
    if one_to_one:
        p = p.sort_values("prob", ascending=False, kind="stable").drop_duplicates("q_id")
    p = p[p["prob"] >= thr]
    return p.groupby("s1_id")["q_id"].agg(set).to_dict(), p


def evaluate(pairs, prob, truth, ents_tune, ents_eval, one_to_one=True, grid=None):
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.99, 0.025), 3)
    best_t, best_f = None, -1
    for t in grid:
        pred, _ = assign(pairs, prob, t, one_to_one)
        f = macro_f05(truth, pred, ents_tune)["f05_macro"]
        if f > best_f:
            best_t, best_f = t, f
    pred, chosen = assign(pairs, prob, best_t, one_to_one)
    r = macro_f05(truth, pred, ents_eval)
    # breakdown: singletons vs multi-match entities
    single = [e for e in ents_eval if not truth[e]]
    multi = [e for e in ents_eval if len(truth[e]) >= 2]
    r["singleton_fp_rate"] = 1 - macro_f05(truth, pred, single)["f05_macro"] if single else None
    m = macro_f05(truth, pred, multi)
    r["multi_f05"], r["multi_pair_p"], r["multi_pair_r"] = m["f05_macro"], m["pair_precision"], m["pair_recall"]
    r["threshold"], r["tune_f05"] = float(best_t), best_f
    r["assign_rank_ge3_share"] = float((chosen["rank"] >= 3).mean()) if len(chosen) else 0.0
    return r


if __name__ == "__main__":
    cand_tag = sys.argv[1]
    run_tag = sys.argv[2] if len(sys.argv) > 2 else cand_tag
    drop_groups = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
    country = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    train_permille = int(sys.argv[5]) if len(sys.argv) > 5 else 100   # share of queries for training
    t0 = time.time()
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() for m in ms}
    # E: evaluation entities. Restricted to fold 1 because the transliteration dictionary is
    # learned from fold-0 pairs (src/prep.py); fold bit and bucket use independent hash bits.
    from src.prep import s1_fold
    E = {s for s in gt if bucket(s) < 100 and s1_fold(s) == 1}
    if country:
        s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
        E &= set(s1c.loc[s1c.country == country, "entity_id"])
    # Filter chunk by chunk: every query's pairs live in exactly one chunk file, so the
    # per-query selection below is exact without ever holding all ~125M pairs in memory.
    cache = WORK_DIR / f"m2cache_v2_{cand_tag}_{country or 'ALL'}_{train_permille}"  # v2: E = fold-1 only
    if (cache / "Xev.parquet").exists():
        ev_pairs = pd.read_parquet(cache / "ev_pairs.parquet")
        tr_pairs = pd.read_parquet(cache / "tr_pairs.parquet")
        Xev = pd.read_parquet(cache / "Xev.parquet")
        Xtr = pd.read_parquet(cache / "Xtr.parquet")
        print(f"loaded cached selection+features from {cache.name}: eval {Xev.shape} train {Xtr.shape}", flush=True)
        from src.features import NUM_LEGAL_NAMES, num_legal_features
        if NUM_LEGAL_NAMES[0] not in Xev:        # augment cache with the v2 number/legal features
            t = time.time()
            s1r = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "addr_norm", "name_norm"]).set_index("entity_id")
            need = set(ev_pairs.q_id) | set(tr_pairs.q_id)
            qr = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", need, ["entity_id", "addr_norm", "name_norm"])
                            for k in (2, 3)]).set_index("entity_id")
            Xev = pd.concat([Xev, num_legal_features(ev_pairs, s1r, qr)], axis=1)
            Xtr = pd.concat([Xtr, num_legal_features(tr_pairs, s1r, qr)], axis=1)
            Xev.to_parquet(cache / "Xev.parquet", index=False)
            Xtr.to_parquet(cache / "Xtr.parquet", index=False)
            print(f"augmented features -> {Xev.shape[1]} cols in {time.time() - t:.0f}s", flush=True)
    else:
        d = WORK_DIR / f"cand_train_{cand_tag}"
        if country:
            import pyarrow.dataset as pads
            n_q = sum(pads.dataset(str(WORK_DIR / f"train_s{k}.parquet")).count_rows(
                filter=pads.field("country") == country) for k in (2, 3))
            files = sorted(d.glob(f"{country}_*.parquet"))
            need_chunks = (n_q + candidates.CHUNK - 1) // candidates.CHUNK
            assert len(files) == need_chunks, f"{country}: {len(files)}/{need_chunks} chunks present"
        else:
            assert (d / "DONE").exists(), "candidate generation incomplete"
            files = sorted(d.glob("*.parquet"))
        ev_parts, tr_parts, n_all = [], [], 0
        for p in files:
            c = pd.read_parquet(p)
            n_all += len(c)
            touch = set(c.loc[c["s1_id"].isin(E).to_numpy() & (c["rank"].to_numpy() < 3), "q_id"])
            touch |= set(c.loc[c["q_id"].map(owner).isin(E).to_numpy(), "q_id"])
            is_eval = c["q_id"].isin(touch).to_numpy()
            ev_parts.append(c[is_eval])
            tr_parts.append(c[~is_eval & (c["q_id"].map(bucket).to_numpy() < train_permille)])
        ev_pairs = pd.concat(ev_parts, ignore_index=True)
        tr_pairs = pd.concat(tr_parts, ignore_index=True)
        del ev_parts, tr_parts
        print(f"candidates {n_all}; eval pairs {len(ev_pairs)} ({ev_pairs.q_id.nunique()} q), "
              f"train pairs {len(tr_pairs)}  {time.time() - t0:.0f}s", flush=True)

        s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=REC).set_index("entity_id")
        need = set(ev_pairs.q_id) | set(tr_pairs.q_id)
        q = pd.concat([read_records(WORK_DIR / f"train_s{k}.parquet", need, REC) for k in (2, 3)])
        q = q.set_index("entity_id")
        freq = s1_name_freq()
        t = time.time()
        tr_pairs, Xtr = featurize(tr_pairs, s1, q, freq)
        ev_pairs, Xev = featurize(ev_pairs, s1, q, freq)
        print(f"features {Xtr.shape} / {Xev.shape} in {time.time() - t:.0f}s  "
              f"rss={psutil.Process().memory_info().rss / 2**30:.1f}GB", flush=True)
        cache.mkdir(exist_ok=True)
        for name, obj in (("ev_pairs", ev_pairs), ("tr_pairs", tr_pairs), ("Xev", Xev), ("Xtr", Xtr)):
            obj.to_parquet(cache / f"{name}.parquet", index=False)
    ytr = (tr_pairs["q_id"].map(owner) == tr_pairs["s1_id"]).to_numpy(np.int8)

    groups = {"retrieval": [c for c in Xtr if c.startswith(("ret_", "blk_"))],
              "name": [c for c in Xtr if c.startswith(("nc_", "nn_", "core_", "s1_missing", "q_extra",
                                                       "legal_", "q_raw_name", "q_nonascii"))],
              "address": [c for c in Xtr if c.startswith(("ad_", "addr_", "num_", "postal", "q_addr", "q_has_pobox"))],
              "freq": ["s1_name_freq"]}
    drop = [c for g in drop_groups for c in groups[g]]
    Xtr, Xev = Xtr.drop(columns=drop), Xev.drop(columns=drop)

    ents = np.array(sorted(E))
    half = np.array([bucket(e, 2) for e in ents]) == 0
    tune_e, eval_e = ents[half], ents[~half]          # threshold chosen on tune_e, reported on eval_e
    truth = {e: gt[e] for e in ents}
    yev = (ev_pairs["q_id"].map(owner) == ev_pairs["s1_id"]).to_numpy(np.int8)
    stats = {"n_train_pairs": len(Xtr), "n_train_pos": int(ytr.sum()), "n_eval_pairs": len(Xev),
             "n_eval_pos": int(yev.sum()), "n_eval_entities": len(eval_e),
             "n_eval_singletons": int(sum(not truth[e] for e in eval_e)),
             "blocking_recall_eval_entities": float(
                 yev[ev_pairs["s1_id"].isin(set(eval_e)).to_numpy()].sum()
                 / max(sum(len(truth[e]) for e in eval_e), 1))}
    print("stats", json.dumps(stats), flush=True)
    results, sweeps = {}, {}

    def run(name, prob, one_to_one=True):
        r, st = tune_and_report(ev_pairs, prob, truth, tune_e, eval_e, one_to_one)
        results[name], sweeps[name] = r, st
        print(name, json.dumps(r, default=float), flush=True)

    run("rule_ret_score", ev_pairs["score"].to_numpy())                  # A: rule baseline
    from sklearn.linear_model import LogisticRegression                   # B: logistic regression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    lr.fit(Xtr.fillna(-1), ytr)
    run("logreg", lr.predict_proba(Xev.fillna(-1))[:, 1])
    t = time.time()                                                        # C: LightGBM
    model = train_model(Xtr, ytr, tr_pairs["q_id"].to_numpy())
    prob = model.predict(Xev, num_threads=12)
    results["lgbm_train_secs"] = round(time.time() - t)
    run("lgbm_1to1", prob, True)
    run("lgbm_no_1to1", prob, False)
    for k, st in sweeps.items():
        st.to_csv(WORK_DIR / f"m2_sweep_{run_tag}_{k}.csv", index=False)
    results["peak_wset_gb"] = round(psutil.Process().memory_info().peak_wset / 2**30, 2)
    imp = pd.Series(model.feature_importance("gain"), index=Xtr.columns)
    line = {"run": run_tag, "cand": cand_tag, "country": country, "dropped": drop_groups, "n_train_pairs": len(Xtr),
            "n_eval_pairs": len(Xev), "stats": stats, "results": results, "best_iter": model.best_iteration,
            "top_features": (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict(),
            "secs": round(time.time() - t0)}
    with open(Path(__file__).with_name("match2_results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=float) + "\n")
    ev_pairs.assign(prob=prob, y=(ev_pairs["q_id"].map(owner) == ev_pairs["s1_id"]).to_numpy(np.int8)) \
        .to_parquet(WORK_DIR / f"m2_eval_{run_tag}.parquet", index=False)
    Xev.to_parquet(WORK_DIR / f"m2_Xev_{run_tag}.parquet", index=False)
    model.save_model(str(WORK_DIR / f"m2_model_{run_tag}.txt"))
