"""V3 / M5: which false candidates make the K50 matcher fail? (K50 E' eval set, no new inference)

Input: K50_ROOT/eval/b5/*.parquet (every scored E'-relevant pair with the probabilities of all
K50 / D-regime models). High-confidence candidates = rows with prob >= 0.5 under a model.
For each rank band (0-2, 3-9, 10-49) and text category (tags from v3_error_budget.tags, plus
'same_name_same_addr_diff_entity', 'same_name_diff_number') report: false count, true count,
true/false ratio, mean prob of false and true rows, for the K50 model (K50_r1.0_s42) and the
D-regime model (M7_K10reg_s42).
Output: C:\\aml2026\\v3_exp\\reports\\hardneg.json
Usage: python -m experiments.v3_hardneg
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.k50_common import K50_ROOT, atomic_json, mem_ok
from experiments.v3_error_budget import records, tags

OUT = Path(r"C:\aml2026\v3_exp\reports")
MODELS = ("K50_r1.0_s42", "M7_K10reg_s42")
BANDS = {"0-2": (0, 3), "3-9": (3, 10), "10-49": (10, 50)}

if __name__ == "__main__":
    t0 = time.time()
    cols = ["q_id", "s1_id", "rank", "y", "own", *MODELS]
    ev = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((K50_ROOT / "eval" / "b5").glob("India_*.parquet"))],
                   ignore_index=True)
    hi = ev[(ev[list(MODELS)] >= 0.5).any(axis=1).to_numpy()].reset_index(drop=True)
    del ev
    print(f"high-confidence rows {len(hi)} ({int((hi.y == 0).sum())} false)", mem_ok(tag="M5"), flush=True)
    rec = records(set(hi.q_id) | set(hi.s1_id))
    T = tags(hi, rec)
    T["same_name_same_addr"] = T.name_exact & ~T.addr_conflict & ~T.number_conflict & ~T.addr_empty
    T["same_name_diff_number"] = T.name_exact & T.number_conflict
    T["same_name_diff_addr"] = T.name_exact & T.addr_conflict
    T["same_addr_diff_name"] = T.name_conflict & ~T.addr_conflict & ~T.addr_empty
    T["legal_and_number"] = T.legal_diff & T.number_conflict
    T["distractor"] = hi["own"].isna()
    T["all"] = True
    res = {}
    for mdl in MODELS:
        m = hi[mdl].to_numpy() >= 0.5
        for band, (lo, up) in BANDS.items():
            b = m & (hi["rank"].to_numpy() >= lo) & (hi["rank"].to_numpy() < up)
            for c in T.columns:
                sel = b & T[c].to_numpy()
                f, t = sel & (hi.y.to_numpy() == 0), sel & (hi.y.to_numpy() == 1)
                if not sel.any():
                    continue
                res[f"{mdl}|{band}|{c}"] = {"false": int(f.sum()), "true": int(t.sum()),
                                            "true_per_false": float(t.sum() / max(f.sum(), 1)),
                                            "mean_prob_false": float(hi[mdl][f].mean()) if f.any() else None,
                                            "mean_prob_true": float(hi[mdl][t].mean()) if t.any() else None,
                                            "false_with_prob>=0.8": int((f & (hi[mdl].to_numpy() >= 0.8)).sum())}
    atomic_json({"rows_prob>=0.5": len(hi), "table": res, "secs": round(time.time() - t0)}, OUT / "hardneg.json")
    df = pd.DataFrame(res).T
    pd.set_option("display.width", 200)
    print(df[df.index.str.contains(r"\|all$|distractor|same_|legal_and|number_conflict|script|addr_empty", regex=True)].to_string())
