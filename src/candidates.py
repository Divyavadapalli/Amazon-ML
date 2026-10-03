"""Full candidate generation for a split (train or test), resumable.

For every country, the S1 index is built once and ALL S2/S3 records of that country are
queried in chunks. Each chunk is written to
    WORK_DIR/cand_<split>_<tag>/<country>_<chunk>.parquet
and skipped on re-run, so a killed job resumes where it stopped.

Usage: python -m src.candidates <split> <tag>        (tag selects a config in CONFIGS)
       python -m src.candidates <split> <tag> --compact --out <dir>
Compact mode (K50 experiments): instead of string ids, each chunk stores int positions into
per-country index files, so a K=50 test cache (~500M pairs) fits on disk:
    <dir>/<country>_<chunk>.parquet   q_pos:int32 s1_pos:int32 score:float32 rank:int8
                                      (sorted by q_pos, rank; zstd)
    <dir>/index_<country>_q.parquet   entity_id by q_pos (the query order used by run())
    <dir>/index_<country>_s1.parquet  entity_id by s1_pos
    <dir>/manifest.json               config, per-chunk row counts and sha256; DONE marker
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import DEFAULT, VIEW_DF, CountryIndex
from .config import WORK_DIR

CHUNK = 250_000
Q_COLS = ["entity_id", "country", "name_core", "name_alt", "addr_norm"]

_BLEND_AB = {"nw": 1.0, "nc": 1.0, "ad": 0.6, "ab": 0.6}
CONFIGS = {
    "blend10": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 0.8}, "k": 10}]},
    "blend_addrpass": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 0.8}, "k": 10},
                                  {"name": "addr", "w": {"ad": 1.0, "nc": 0.5}, "k": 5}],
                       "exact": True},
    "blend_ad15": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 1.5}, "k": 10}]},
    # identical retrieval, deeper list (K50 experiments; ranks 0-9 equal blend_ad15)
    "blend_ad15_k50": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 1.5}, "k": 50}]},
    "blend_ad20_k15": {"passes": [{"name": "blend", "w": {"nw": 1.0, "nc": 1.0, "ad": 2.0}, "k": 15}]},
    "blend_ab10": {"passes": [{"name": "blend", "w": _BLEND_AB, "k": 10}]},
    "multi_ab": {"passes": [{"name": "blend", "w": _BLEND_AB, "k": 10},
                            {"name": "addr", "w": {"ad": 0.6, "ab": 1.0}, "k": 5}],
                 "exact": True},
}


def config(tag: str) -> dict:
    cfg = {**DEFAULT, **CONFIGS[tag]}
    cfg["max_df"] = {**VIEW_DF, **cfg.get("max_df", {})}
    return cfg


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


PROTECTED = ("cand_train_blend_ad15", "cand_test_blend_ad15")


def run_compact(split: str, tag: str, out_dir, countries=None):
    """Compact, resumable candidate generation (see module docstring)."""
    cfg = config(tag)
    out_dir = Path(out_dir)
    assert not any(out_dir.name == p for p in PROTECTED), "refusing to write into an existing K10 cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"config": cfg, "split": split, "tag": tag, "chunks": {}}
    assert man["config"] == json.loads(json.dumps(cfg)), "config changed since this cache was started"
    s1_all = pd.read_parquet(WORK_DIR / f"{split}_s1.parquet", columns=Q_COLS)
    q_all = pd.concat([pd.read_parquet(WORK_DIR / f"{split}_s{k}.parquet", columns=Q_COLS)
                       for k in (2, 3)], ignore_index=True)
    for c in sorted(set(s1_all["country"]) | set(q_all["country"])):
        if countries and c not in countries:
            continue
        s1 = s1_all[s1_all["country"] == c].reset_index(drop=True)
        q = q_all[q_all["country"] == c].reset_index(drop=True)
        for name, ids in (("q", q["entity_id"]), ("s1", s1["entity_id"])):
            f = out_dir / f"index_{c}_{name}.parquet"
            if not f.exists():
                pd.DataFrame({"entity_id": ids.to_numpy()}).to_parquet(out_dir / f"index_{c}_{name}.tmp.parquet", index=False)
                (out_dir / f"index_{c}_{name}.tmp.parquet").replace(f)
        n_chunks = (len(q) + CHUNK - 1) // CHUNK
        todo = [i for i in range(n_chunks) if not (out_dir / f"{c}_{i:03d}.parquet").exists()]
        print(f"country={c}: S1={len(s1)} queries={len(q)} chunks={n_chunks} todo={len(todo)}", flush=True)
        if not todo or not len(s1):
            continue
        idx = CountryIndex(s1, cfg)
        for i in todo:
            t = time.time()
            qc = q.iloc[i * CHUNK:(i + 1) * CHUNK].reset_index(drop=True)
            r = idx.query(qc)
            out = pd.DataFrame({"q_pos": (r["q_idx"].to_numpy() + i * CHUNK).astype(np.int32),
                                "s1_pos": r["s1_idx"].to_numpy().astype(np.int32),
                                "score": r["score"].to_numpy().astype(np.float32),
                                "rank": r["rank"].to_numpy().astype(np.int8)})
            tmp = out_dir / f"{c}_{i:03d}.tmp.parquet"
            out.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(out_dir / f"{c}_{i:03d}.parquet")
            man["chunks"][f"{c}_{i:03d}"] = {"rows": len(out), "sha256": _sha(out_dir / f"{c}_{i:03d}.parquet")}
            man_path.with_suffix(".tmp").write_text(json.dumps(man, indent=1))
            man_path.with_suffix(".tmp").replace(man_path)
            print(f"  {c} chunk {i + 1}/{n_chunks}: {len(out)} pairs {time.time() - t:.0f}s", flush=True)
        del idx
    if not countries:
        (out_dir / "DONE").write_text("ok")


def load_compact_chunk(out_dir, stem):
    """One compact chunk with string ids rehydrated: q_id, s1_id, score, rank (+ positions)."""
    out_dir = Path(out_dir)
    c = stem.rsplit("_", 1)[0]
    r = pd.read_parquet(out_dir / f"{stem}.parquet")
    qi = pd.read_parquet(out_dir / f"index_{c}_q.parquet")["entity_id"].to_numpy()
    si = pd.read_parquet(out_dir / f"index_{c}_s1.parquet")["entity_id"].to_numpy()
    r.insert(0, "q_id", qi[r["q_pos"].to_numpy()])
    r.insert(1, "s1_id", si[r["s1_pos"].to_numpy()])
    return r


def run(split: str, tag: str):
    cfg = config(tag)
    out_dir = WORK_DIR / f"cand_{split}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    s1_all = pd.read_parquet(WORK_DIR / f"{split}_s1.parquet", columns=Q_COLS)
    q_all = pd.concat([pd.read_parquet(WORK_DIR / f"{split}_s{k}.parquet", columns=Q_COLS)
                       for k in (2, 3)], ignore_index=True)
    for c in sorted(set(s1_all["country"]) | set(q_all["country"])):
        s1 = s1_all[s1_all["country"] == c].reset_index(drop=True)
        q = q_all[q_all["country"] == c].reset_index(drop=True)
        n_chunks = (len(q) + CHUNK - 1) // CHUNK
        todo = [i for i in range(n_chunks) if not (out_dir / f"{c}_{i:03d}.parquet").exists()]
        print(f"country={c}: S1={len(s1)} queries={len(q)} chunks={n_chunks} todo={len(todo)}",
              flush=True)
        if not todo or not len(s1):
            continue
        idx = CountryIndex(s1, cfg)
        for i in todo:
            t = time.time()
            qc = q.iloc[i * CHUNK:(i + 1) * CHUNK].reset_index(drop=True)
            r = idx.query(qc)
            r.insert(0, "q_id", qc["entity_id"].values[r.pop("q_idx").values])
            r.insert(1, "s1_id", s1["entity_id"].values[r.pop("s1_idx").values])
            tmp = out_dir / f"{c}_{i:03d}.parquet.tmp"
            r.to_parquet(tmp, index=False)
            tmp.replace(out_dir / f"{c}_{i:03d}.parquet")   # atomic: no half-written chunks
            print(f"  {c} chunk {i + 1}/{n_chunks}: {len(r)} pairs {time.time() - t:.0f}s", flush=True)
        del idx
    (out_dir / "DONE").write_text("ok")


def load(split: str, tag: str, columns=None) -> pd.DataFrame:
    d = WORK_DIR / f"cand_{split}_{tag}"
    assert (d / "DONE").exists(), f"candidate generation for {split}/{tag} is incomplete"
    return pd.concat([pd.read_parquet(p, columns=columns) for p in sorted(d.glob("*.parquet"))],
                     ignore_index=True)


if __name__ == "__main__":
    if "--compact" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        only = sys.argv[sys.argv.index("--countries") + 1].split(",") if "--countries" in sys.argv else None
        run_compact(sys.argv[1], sys.argv[2], out, only)
    else:
        run(sys.argv[1], sys.argv[2])
