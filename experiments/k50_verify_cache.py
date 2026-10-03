"""Verify a compact K50 candidate cache (after Phase A / test retrieval). Exit 1 on failure.

Checks: DONE marker; manifest config == CONFIGS[tag] (K=50, blend weights, max_df, min_score);
expected chunk count per country from the query table (CHUNK); no missing / extra chunks;
per-chunk sha256 and row count == manifest; ranks 0..49 contiguous per query; positions within
the index files; per-query candidate count <= K. Writes <dir>/verify.json.

Usage: python -m experiments.k50_verify_cache <cache_dir> <split> [tag=blend_ad15_k50]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates import CHUNK, config
from src.config import WORK_DIR


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    d, split = Path(sys.argv[1]), sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else "blend_ad15_k50"
    fails, rep = [], {}

    def check(ok, name, detail=""):
        print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
        rep[name] = {"pass": bool(ok), "detail": str(detail)}
        if not ok:
            fails.append(name)

    man = json.loads((d / "manifest.json").read_text())
    check((d / "DONE").exists(), "DONE marker")
    cfg = json.loads(json.dumps(config(tag)))
    check(man["config"] == cfg, "manifest config == CONFIGS[tag]", {"k": man["config"]["passes"][0]["k"]})
    check(man["config"]["passes"][0]["k"] == 50, "K == 50")
    countries = sorted({p.name[len("index_"):-len("_q.parquet")] for p in d.glob("index_*_q.parquet")})
    total_rows = 0
    for c in countries:
        nq = len(pd.read_parquet(d / f"index_{c}_q.parquet"))
        ns = len(pd.read_parquet(d / f"index_{c}_s1.parquet"))
        n_chunks = (nq + CHUNK - 1) // CHUNK
        have = sorted(p.stem for p in d.glob(f"{c}_*.parquet") if not p.stem.endswith(".tmp"))
        want = [f"{c}_{i:03d}" for i in range(n_chunks)]
        check(have == want, f"{c}: chunks {len(have)}/{n_chunks} present, none extra")
        for stem in want:
            m = man["chunks"].get(stem)
            p = d / f"{stem}.parquet"
            if m is None or not p.exists():
                check(False, f"{stem}: in manifest and on disk")
                continue
            if sha(p) != m["sha256"]:
                check(False, f"{stem}: sha256 matches manifest")
            r = pd.read_parquet(p)
            ok = len(r) == m["rows"]
            qp, rk, sp = r["q_pos"].to_numpy(), r["rank"].to_numpy(), r["s1_pos"].to_numpy()
            ok &= bool((qp >= 0).all() and (qp < nq).all() and (sp >= 0).all() and (sp < ns).all())
            first = np.r_[True, qp[1:] != qp[:-1]]
            grp = np.cumsum(first) - 1
            start = np.flatnonzero(first)
            ok &= bool((rk == (np.arange(len(rk)) - start[grp])).all())           # ranks 0..n-1 contiguous
            ok &= bool(rk.max() <= 49)
            if not ok:
                check(False, f"{stem}: rows/positions/ranks valid")
            total_rows += len(r)
        rep[f"{c}_queries"] = nq
    check(not [k for k in fails if ":" in k], "all chunks: sha256, rows, positions, contiguous ranks")
    rep["total_rows"] = total_rows
    (d / "verify.json").write_text(json.dumps(rep, indent=1))
    print(f"total rows {total_rows}; {'ALL CACHE CHECKS PASSED' if not fails else 'FAILED: ' + str(fails)}")
    sys.exit(1 if fails else 0)
