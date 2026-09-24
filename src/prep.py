"""Stage 1: learn the transliteration dictionary, then normalise every source file.

Outputs (in WORK_DIR):
  translit_name.json / translit_addr.json   learned from training fold A only
  {split}_s{k}.parquet                        raw + normalised columns for each source

Run:  python -m src.prep
"""
import time
import zlib
from multiprocessing import Pool

import pandas as pd

from . import translit
from .config import N_JOBS, WORK_DIR
from .io_utils import read_ground_truth, read_source
from .normalize import normalize_record

TRAIN_FOLD_A = 0  # fold 0 trains models / dictionaries; fold 1 is held out for validation


def s1_fold(s1_id: str) -> int:
    """Deterministic 50/50 split of Source-1 entities (their matches follow them)."""
    return zlib.crc32(s1_id.encode()) & 1


def learn_translit():
    gt = read_ground_truth()
    s1 = read_source("train", 1).set_index("entity_id")
    owner = {m: s for s, ms in gt.items() if s1_fold(s) == TRAIN_FOLD_A for m in ms}
    name_pairs, addr_pairs = [], []
    for k in (2, 3):
        df = read_source("train", k)
        df = df[df.entity_id.isin(owner.keys())]
        df = df[~df.business_name.map(str.isascii) | ~df.business_address.map(str.isascii)]
        s1_rows = s1.loc[df.entity_id.map(owner)]
        name_pairs += list(zip(s1_rows.business_name, df.business_name))
        addr_pairs += list(zip(s1_rows.business_address, df.business_address))
    tn = translit.learn(name_pairs)
    ta = translit.learn(addr_pairs)
    translit.save(tn, WORK_DIR / "translit_name.json")
    translit.save(ta, WORK_DIR / "translit_addr.json")
    print(f"translit: {len(name_pairs)} name pairs -> {len(tn)} tokens; "
          f"{len(addr_pairs)} addr pairs -> {len(ta)} tokens", flush=True)


_TN = _TA = None


def _init(tn, ta):
    global _TN, _TA
    # a single merged map: name map wins on conflict
    _TN, _TA = tn, {**ta, **tn}


def _norm_chunk(rows):
    out = []
    for name, addr in rows:
        r = normalize_record(name, "", _TN)
        a = normalize_record("", addr, _TA)
        r.update({k: a[k] for k in ("addr_norm", "addr_digits", "pobox")})
        out.append(r)
    return out


def normalise_all():
    tn = translit.load(WORK_DIR / "translit_name.json")
    ta = translit.load(WORK_DIR / "translit_addr.json")
    with Pool(N_JOBS, initializer=_init, initargs=(tn, ta)) as pool:
        for split in ("train", "test"):
            for k in (1, 2, 3):
                t = time.time()
                df = read_source(split, k)
                rows = list(zip(df.business_name, df.business_address))
                chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
                norm = [r for part in pool.imap(_norm_chunk, chunks) for r in part]
                df = pd.concat([df, pd.DataFrame(norm, index=df.index)], axis=1)
                df.to_parquet(WORK_DIR / f"{split}_s{k}.parquet", index=False)
                print(f"{split} s{k}: {len(df)} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    learn_translit()
    normalise_all()
