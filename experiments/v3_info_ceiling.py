"""V3 / M1: information ceiling of the supplied fields (text-conflict census) on V-dense E.

Principle: a system that sees only a record's text must give text-identical S2/S3 records the
same output. For a canonical key, the best text-only system therefore assigns every group of
identical-key queries to ONE label (one S1, or none). The macro-F0.5 of the best such group
labelling bounds what any system using only that representation can reach.

Population: every India train S2/S3 record (the competitors of E's records in its country),
distractors (owned by no S1) cloned with p = 0.94, numpy seed 0, in first-appearance order --
the vdense.densify rule applied to all India queries; clones are text-identical to their source.
Scored on the V-dense entity set E (44,360 India fold-1 bucket<100 entities, == vdense.load_india).

Per key:
  lower = macro-F0.5 of an explicit text-only labelling (each group's candidate label is its
          most frequent E-owner; a group is assigned only if that raises its entity's F0.5;
          greedy, purest groups first, iterated to convergence). Achievable => ceiling >= lower.
          Re-scored exactly with src.metrics.macro_f05 on string id sets.
  upper = every E-owned record in a conflict-free group found, only the majority label of each
          conflicting group correct, zero false merges => ceiling <= upper.
  95% CI of `lower`: bootstrap over entities (1000 resamples, seed 1).
Limitation (stated in the report): conflicts between S1 twins (distinct S1 with an identical
key) are not penalised, so for coarse keys the bounds are optimistic; for raw_exact there are
none. Keys are 64-bit hashes of the joined text (pandas hash_array); collisions are
negligible at this scale (~5M items / 2^64) and cannot create conflicts in the ceiling's favour
beyond that probability.

Keys: raw_exact(business_name, business_address), name_norm+addr_norm, name_core+addr_norm,
name_core+addr_digits, name_core.
Logs: reports/M1_info_ceiling.log (heartbeat every 60 s). Output: reports/info_ceiling.json
Usage: python -m experiments.v3_info_ceiling
"""
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from experiments.k50_common import atomic_json, mem_ok
from experiments.v3_common import GT_PATH, REPORTS, Log, eval_truth
from src.config import WORK_DIR
from src.metrics import macro_f05

KEYS = {
    "raw_exact": ["business_name", "business_address"],
    "name_norm+addr_norm": ["name_norm", "addr_norm"],
    "name_core+addr_norm": ["name_core", "addr_norm"],
    "name_core+addr_digits": ["name_core", "addr_digits"],
    "name_core": ["name_core"],
}
TEXT = sorted({c for v in KEYS.values() for c in v})


def h(values):
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False)


def key_hash(df, cols):
    k = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        k = k + "\x1f" + df[c].fillna("").astype(str)
    return h(k.to_numpy())


def f05_vec(tp, npred, nt):
    prec = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    rec = np.divide(tp, nt, out=np.zeros_like(tp), where=nt > 0)
    den = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    return np.where(nt == 0, (npred == 0).astype(float), f)


def ceiling(keyh, lab, nt, log):
    """keyh: uint64 per record; lab: E index or -1. Returns lower vector, upper vector, census, assignment."""
    _, gid = np.unique(keyh, return_inverse=True)
    ng = gid.max() + 1
    size = np.bincount(gid, minlength=ng)
    inE = lab >= 0
    pairs = pd.DataFrame({"g": gid[inE], "e": lab[inE]}).value_counts().rename("n").reset_index()
    pairs = pairs.sort_values(["g", "n", "e"], ascending=[True, False, True], kind="stable")
    top = pairs.drop_duplicates("g")
    choice = np.full(ng, -1)
    choice[top.g.to_numpy()] = top.e.to_numpy()
    hit = np.zeros(ng)
    hit[top.g.to_numpy()] = top.n.to_numpy()
    lab_g = np.unique(np.stack([gid, lab.astype(np.int64)], 1), axis=0)
    n_labels = np.bincount(lab_g[:, 0], minlength=ng)
    conflict = n_labels > 1
    tp, npred = np.zeros(len(nt)), np.zeros(len(nt))
    cand = np.flatnonzero(choice >= 0)
    order = cand[np.argsort(-(hit[cand] / size[cand]), kind="stable")]
    assigned = np.zeros(ng, bool)
    one = lambda a, b, c: f05_vec(np.array([a]), np.array([b]), np.array([c]))[0]
    for it in range(5):
        changed = 0
        for gi in order:
            e = choice[gi]
            if assigned[gi]:
                a_tp, a_np, r_tp, r_np = tp[e], npred[e], tp[e] - hit[gi], npred[e] - size[gi]
            else:
                a_tp, a_np, r_tp, r_np = tp[e] + hit[gi], npred[e] + size[gi], tp[e], npred[e]
            want = one(a_tp, a_np, nt[e]) > one(r_tp, r_np, nt[e])
            if want != assigned[gi]:
                tp[e], npred[e] = (a_tp, a_np) if want else (r_tp, r_np)
                assigned[gi] = want
                changed += 1
        log(f"    greedy pass {it + 1}: {changed} changes")
        if not changed:
            break
    lower = f05_vec(tp, npred, nt)
    ub = np.zeros(len(nt))
    ok = inE & ~conflict[gid]
    np.add.at(ub, lab[ok], 1)
    cg = np.flatnonzero(conflict & (choice >= 0))
    np.add.at(ub, choice[cg], hit[cg])
    upper = f05_vec(ub, ub, nt)
    amb_ents = np.unique(lab[inE & conflict[gid]])
    census = {"groups": int(ng), "records": int(len(keyh)), "groups_size>1": int((size > 1).sum()),
              "conflicting_groups": int(conflict.sum()), "records_in_conflicting_groups": int(size[conflict].sum()),
              "E_owned_records": int(inE.sum()), "E_owned_records_in_conflicting_groups": int((inE & conflict[gid]).sum()),
              "E_entities_with_ambiguous_records": int(len(amb_ents)),
              "max_labels_in_group": int(n_labels.max()),
              "group_size_quantiles": {str(k): float(v) for k, v in pd.Series(size).quantile([.5, .9, .99, 1]).items()}}
    return lower, upper, census, gid, choice, assigned


if __name__ == "__main__":
    t0 = time.time()
    log = Log("M1_info_ceiling")
    log.set_stage("ground truth")
    g = eval_truth(us=False)
    E, truth = g["E"], g["truth"]
    eix = {e: i for i, e in enumerate(E)}
    member_e = {m: eix[e] for e in E for m in truth[e]}
    nt = np.array([len(truth[e]) for e in E], float)
    owned = []                                            # hashes of every owned S2/S3 id (distractor test)
    with open(GT_PATH, encoding="utf-8") as f:
        next(f)
        buf = []
        for line in f:
            rest = line.rstrip("\r\n").partition("\t")[2]
            if rest:
                buf.extend(x for x in rest.split(",") if x)
            if len(buf) > 1_000_000:
                owned.append(h(buf)); buf = []
        owned.append(h(buf))
    owned = np.unique(np.concatenate(owned))
    log(f"E {len(E)} entities, {int(nt.sum())} true pairs; {len(owned)} owned S2/S3 ids hashed {mem_ok(tag='M1')}")

    log.set_stage("hash India S2/S3 keys")
    ids, lab, isd, kh = [], [], [], {k: [] for k in KEYS}
    for k in (2, 3):
        dset = ds.dataset(str(WORK_DIR / f"train_s{k}.parquet"))
        for b in dset.to_batches(columns=["entity_id", *TEXT], filter=ds.field("country") == "India", use_threads=False,
                                 batch_size=500_000):
            df = b.to_pandas()
            ids.append(pa.array(df.entity_id.to_numpy(dtype=object), pa.string()))
            lab.append(df.entity_id.map(member_e).fillna(-1).to_numpy(np.int64))
            isd.append(~np.isin(h(df.entity_id.to_numpy()), owned))
            for name, cols in KEYS.items():
                kh[name].append(key_hash(df, cols))
        log(f"  train_s{k} hashed {mem_ok(tag='M1')}")
    ids = pa.chunked_array(ids).combine_chunks()
    lab, isd = np.concatenate(lab), np.concatenate(isd)
    kh = {k: np.concatenate(v) for k, v in kh.items()}
    del owned
    n0 = len(lab)
    assert not (isd & (lab >= 0)).any(), "an E-owned record is flagged as a distractor"
    # densify: distractors in first-appearance order, seed 0, p < 0.94 (records are unique ids)
    didx = np.flatnonzero(isd)
    clone = didx[np.random.default_rng(0).random(len(didx)) < 0.94]
    lab = np.concatenate([lab, np.full(len(clone), -1)])
    kh = {k: np.concatenate([v, v[clone]]) for k, v in kh.items()}
    src_row = np.concatenate([np.arange(n0), clone])
    log(f"India records {n0} + {len(clone)} distractor clones = {len(lab)}; E-owned {int((lab >= 0).sum())} {mem_ok(tag='M1')}")

    log.set_stage("India S1 key duplicates")
    s1d = ds.dataset(str(WORK_DIR / "train_s1.parquet")).to_table(columns=["entity_id", *TEXT], filter=ds.field("country") == "India").to_pandas()
    s1_in_E = s1d.entity_id.isin(set(E)).to_numpy()
    s1dup = {}
    for name, cols in KEYS.items():
        kk = key_hash(s1d, cols)
        dup = pd.Series(kk).duplicated(keep=False).to_numpy()
        s1dup[name] = {"india_s1_sharing_key_with_another_s1": int(dup.sum()), "E_s1_sharing_key": int((dup & s1_in_E).sum())}
    del s1d
    rng = np.random.default_rng(1)
    boot_idx = [rng.integers(0, len(E), len(E)) for _ in range(1000)]
    res = {"population": "India train S2/S3 records (+ distractor clones x0.94, seed 0) scored on V-dense E",
           "E_entities": len(E), "E_true_pairs": int(nt.sum()), "records": int(len(lab)),
           "oracle_reference": {"top10_oracle_E": 0.9862170941021483, "top50_oracle_E": 0.9923236563063875},
           "method": "see module docstring", "keys": {}}
    for name in KEYS:
        log.set_stage(f"ceiling {name}")
        t = time.time()
        lo, up, census, gid, choice, assigned = ceiling(kh[name], lab, nt, log)
        # exact verification of `lower` with src.metrics.macro_f05 on string ids
        rows = np.flatnonzero(assigned[gid])
        ent = choice[gid[rows]]
        qid = np.asarray(ids.take(pa.array(src_row[rows])).to_pylist(), dtype=object)
        qid = np.where(rows >= n0, "CLONE_" + qid.astype(str), qid.astype(str))
        pred = pd.DataFrame({"s": np.asarray(E, dtype=object)[ent], "q": qid}).groupby("s")["q"].agg(set).to_dict()
        ref = macro_f05(truth, pred, E)["f05_macro"]
        assert abs(ref - lo.mean()) < 1e-9, f"{name}: fast {lo.mean()} != macro_f05 {ref}"
        boots = np.array([lo[b].mean() for b in boot_idx])
        res["keys"][name] = {"ceiling_lower": float(lo.mean()), "ceiling_lower_verified_macro_f05": float(ref),
                             "ceiling_lower_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                             "ceiling_upper": float(up.mean()), **census, **s1dup[name], "secs": round(time.time() - t)}
        log(f"  {name}: lower {lo.mean():.6f} (== macro_f05 {ref:.6f}) CI [{res['keys'][name]['ceiling_lower_ci95'][0]:.6f}, "
            f"{res['keys'][name]['ceiling_lower_ci95'][1]:.6f}] upper {up.mean():.6f} conflicts {census['conflicting_groups']} "
            f"ambiguous E entities {census['E_entities_with_ambiguous_records']} {mem_ok(tag='M1')}")
        del gid, choice, assigned
    res["information_ceiling_raw_exact"] = res["keys"]["raw_exact"]["ceiling_lower"]
    res["secs"] = round(time.time() - t0)
    atomic_json(res, REPORTS / "info_ceiling.json")
    log("RESULT " + json.dumps({k: (v["ceiling_lower"], v["ceiling_lower_ci95"], v["ceiling_upper"], v["E_entities_with_ambiguous_records"])
                                for k, v in res["keys"].items()}))
    log(f"M1 done in {time.time() - t0:.0f}s")
