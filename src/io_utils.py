"""Reading the challenge TSVs and writing submission files in the exact required format."""
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import DATA_DIR

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(split: str, source: int) -> pd.DataFrame:
    """Load one source file as strings; empty fields become ''. Quoting is disabled
    because names/addresses contain raw quote characters."""
    path = DATA_DIR / split / f"{split}_source{source}.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    assert list(df.columns) == SOURCE_COLS, df.columns
    return df


def read_ground_truth() -> Dict[str, set]:
    """train_ground_truth.tsv -> {S1 id: set of matched S2/S3 ids} (empty set for singletons)."""
    path = DATA_DIR / "train" / "train_ground_truth.tsv"
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\r\n").partition("\t")
            gt[s1] = {x for x in rest.split(",") if x}
    return gt


def write_id_lists(path: Path, header_col: str, s1_ids: Iterable[str],
                   lists: Dict[str, List[str]]) -> None:
    """Write one row per S1 id, tab-separated, comma-joined ids, empty when none.
    Every id in `s1_ids` gets exactly one row, in the given order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(lists.get(s1, ())))  # de-duplicate, keep order
            f.write(f"{s1}\t{','.join(ids)}\n")
