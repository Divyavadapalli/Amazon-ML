"""Shared helpers for the V3 measurement scripts (M1, M2): flushed file logging with a
heartbeat, and a memory-lean ground-truth reader (the full 2.2M-entity dict costs ~1.5 GB,
which the machine cannot spare while other applications are open)."""
import os
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

from experiments.k50_common import bucket
from src.config import DATA_DIR, WORK_DIR
from src.prep import s1_fold

REPORTS = Path(r"C:\aml2026\v3_exp\reports")
CACHE = WORK_DIR / "m2cache_v2_blend_ad15_India_50"
GT_PATH = DATA_DIR / "train" / "train_ground_truth.tsv"


class Log:
    """Writes every message to <REPORTS>/<name>.log (flushed) and stdout; a daemon thread
    writes a heartbeat line (PID, elapsed, RSS, CPU seconds, current stage) every `beat` s."""

    def __init__(self, name, beat=60):
        REPORTS.mkdir(parents=True, exist_ok=True)
        self.f = open(REPORTS / f"{name}.log", "a", encoding="utf-8")
        self.t0, self.stage, self.proc = time.time(), "start", psutil.Process()
        self("start pid=%d" % os.getpid())
        threading.Thread(target=self._beat, args=(beat,), daemon=True).start()

    def __call__(self, *a):
        msg = " ".join([time.strftime("%H:%M:%S"), *map(str, a)])
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def set_stage(self, s):
        self.stage = s
        self(f"STAGE {s}")

    def _beat(self, every):
        while True:
            time.sleep(every)
            ct = self.proc.cpu_times()
            self(f"HEARTBEAT pid={os.getpid()} elapsed={time.time() - self.t0:.0f}s "
                 f"rss={self.proc.memory_info().rss / 2**30:.2f}GB cpu={ct.user + ct.system:.0f}s "
                 f"avail={psutil.virtual_memory().available / 2**30:.2f}GB stage={self.stage}")


def eval_truth(us=False):
    """Ground truth for the V-dense India population E (fold-1, bucket < 100, present in the
    V-dense cache; == vdense.load_india) and optionally the US spot-check population (US S1,
    bucket < 50). Same parsing as src.io_utils.read_ground_truth."""
    cache_s1 = set(pd.read_parquet(CACHE / "ev_pairs.parquet", columns=["s1_id"]).s1_id)
    us_eval = set()
    if us:
        s1c = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"])
        us_eval = set(s for s in s1c.entity_id[s1c.country == "US"] if bucket(s) < 50)
    gt = {}
    with open(GT_PATH, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            if s1 in us_eval or (s1 in cache_s1 and bucket(s1) < 100 and s1_fold(s1) == 1):
                gt[s1] = {x for x in rest.split(",") if x}
    E = sorted(s for s in gt if s not in us_eval)
    out = {"E": E, "truth": {e: gt[e] for e in E}}
    if us:
        out["Eu"] = sorted(us_eval)
        out["truth_u"] = {e: gt.get(e, set()) for e in out["Eu"]}
    return out
