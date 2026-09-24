"""Blocking experiments on a training universe. Appends one line per config to
experiments/blocking_results.jsonl so no result is ever overwritten.

Usage: python -m experiments.exp_blocking 0.2
"""
import json
import sys
import time
from pathlib import Path

from src.blocking import DEFAULT, retrieve_all
from src.devset import candidate_recall, load_universe

GRID = [
    {},
    {"top_k": 20},
    {"w_addr": 0.0},
    {"w_name_char": 0.0},
    {"max_df_char": 0.03, "max_df_word": 0.05},
]

if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    s1, q, truth = load_universe(frac)
    print(f"universe frac={frac}: S1={len(s1)} queries={len(q)} true pairs={sum(map(len, truth.values()))}")
    out = Path(__file__).with_name("blocking_results.jsonl")
    for over in GRID:
        cfg = {**DEFAULT, **over}
        t = time.time()
        c = retrieve_all(s1, q, cfg)
        r = candidate_recall(c, truth)
        # recall restricted to top-k' prefixes, to see how many candidates we really need
        ks = {k: candidate_recall(c[c["rank"] < k], truth)["pair_recall"] for k in (1, 2, 3, 5)}
        line = {"frac": frac, "cfg": over, **r, "recall@k": ks, "secs": round(time.time() - t)}
        print(json.dumps(line), flush=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
