"""Paths and global settings.

DATA_DIR must contain train/ and test/ exactly as shipped in student_resource/dataset.
WORK_DIR holds cached intermediate files (normalised parquet, candidates, models).
Both can be overridden with environment variables so the pipeline runs on any machine.
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", r"C:\aml2026\data"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", r"C:\aml2026\work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", r"C:\aml2026\output"))

N_JOBS = int(os.environ.get("ER_N_JOBS", os.cpu_count() or 4))
SEED = 42

for _d in (WORK_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Windows throttles Normal-priority background processes (EcoQoS), especially on battery:
# measured 3.3 cores -> full CPU (4.6x) when raised to High. Applies to pool workers too,
# since they import this module.
try:
    import psutil as _ps
    if hasattr(_ps, "HIGH_PRIORITY_CLASS"):
        _ps.Process().nice(_ps.HIGH_PRIORITY_CLASS)
except Exception as _e:  # priority is a speed optimisation only; never affects results
    print(f"[config] could not raise process priority: {_e}")
