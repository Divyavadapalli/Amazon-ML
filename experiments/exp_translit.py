"""Held-out audit of transliteration for script-written S2/S3 names (India).

The dictionary is learned on fold-0 pairs only (src/prep.py); everything here is measured
on fold-1 pairs. Variants:
  A  no transliteration (script tokens kept as-is)
  B  anyascii only
  C  learned dictionary only (unmapped script tokens dropped)
  D  learned dictionary, then anyascii for the rest   (current pipeline)
Metrics: dictionary coverage/accuracy, name similarity to the true S1 name, and
name-only retrieval recall@10 against the FULL India S1 index.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from src import translit
from src.blocking import build_views, _topk
from src.config import WORK_DIR
from src.io_utils import read_ground_truth, read_source
from src.normalize import LEGAL, HONORIFIC, STOP, apply_tokmap, name_tokens
from src.prep import s1_fold

LEGAL_V = set(LEGAL.values())
_EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def core_variant(raw, variant, tm):
    if variant == "D":
        return " ".join(name_tokens(raw, tm)[1])
    if variant == "B":
        return " ".join(name_tokens(raw, None)[1])
    if variant == "C":
        s = apply_tokmap(raw, tm)
        s = " ".join(t for t in s.split() if t.isascii())
        return " ".join(name_tokens(s, None)[1])
    # A: lower-case, split on non-word (unicode aware), no romanisation
    toks = [t for t in re.split(r"[\W_]+", raw.lower()) if t]
    toks = [LEGAL.get(t, t) for t in toks]
    core = [t for t in toks if t not in LEGAL_V and t not in HONORIFIC and t not in STOP]
    return " ".join(core or toks)


if __name__ == "__main__":
    tm = translit.load(WORK_DIR / "translit_name.json")
    gt = read_ground_truth()
    owner = {m: s for s, ms in gt.items() if s1_fold(s) == 1 for m in ms}
    s1raw = read_source("train", 1).set_index("entity_id")
    rows = []
    for k in (2, 3):
        df = read_source("train", k)
        df = df[df.entity_id.isin(owner.keys()) & ~df.business_name.map(str.isascii)]
        rows.append(df)
    q = pd.concat(rows, ignore_index=True)
    q["owner"] = q.entity_id.map(owner)
    q = q[q.country == "India"].reset_index(drop=True)
    s1_names = s1raw.loc[q.owner, "business_name"].to_numpy()
    print(f"held-out fold-1 script-name pairs (India): {len(q)}", flush=True)

    # ---- dictionary coverage & accuracy on held-out tokens
    occ = cov = correct = 0
    for scr, lat in zip(q.business_name, s1_names):
        lat_toks = set(re.findall(r"[a-z0-9]+", lat.lower()))
        for t in translit.script_tokens(scr):
            occ += 1
            if t in tm:
                cov += 1
                correct += tm[t] in lat_toks
    amb = collections.Counter(tm.values())
    res = {"token_occurrences": occ, "dict_coverage": cov / occ, "dict_accuracy": correct / max(cov, 1),
           "dict_size": len(tm), "targets_shared_by_>1_script_token": sum(v > 1 for v in amb.values())}

    # ---- similarity + retrieval recall per variant (full India S1 index)
    s1 = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country", "name_core", "name_alt", "addr_norm"])
    s1 = s1[s1.country == "India"].reset_index(drop=True)
    pos = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    true_idx = pos.loc[q.owner].to_numpy()
    s1_core = s1.name_core.to_numpy()[true_idx]
    for v in "ABCD":
        qc = [core_variant(x, v, tm) for x in q.business_name]
        sim = cpdist(list(s1_core), qc, scorer=fuzz.token_sort_ratio, workers=-1)
        qq = pd.DataFrame({"name_core": qc, "name_alt": "", "addr_norm": ""})
        mats = build_views(s1, qq, ["nw", "nc"], {"nw": 0.02, "nc": 0.01})
        import scipy.sparse as sp
        A = sp.hstack([mats["nw"][1], mats["nc"][1]], format="csr") / np.sqrt(2)
        Bt = (sp.hstack([mats["nw"][0], mats["nc"][0]], format="csr") / np.sqrt(2)).T.tocsr()
        r = _topk(A, Bt, 10, 0.01)
        hit = set(zip(r.q_idx, r.s1_idx))
        rec10 = np.mean([(i, t) in hit for i, t in enumerate(true_idx)])
        r1 = r[r["rank"] == 0]
        rec1 = np.mean(pd.Series(r1.s1_idx.values, index=r1.q_idx.values).reindex(range(len(q))).to_numpy() == true_idx)
        res[v] = {"mean_tsort_sim": float(np.mean(sim)), "share_sim>=80": float(np.mean(sim >= 80)),
                  "name_recall@1": float(rec1), "name_recall@10": float(rec10)}
        print(v, res[v], flush=True)
    print(json.dumps(res, indent=1))
    with open(Path(__file__).with_name("translit_results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
