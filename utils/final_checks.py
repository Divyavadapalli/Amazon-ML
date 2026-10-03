"""Independent checks of a submission directory (run in addition to Amazon's validator).

  1. every test Source-1 entity appears exactly once in each file, and nothing else does
  2. row count == number of rows in test_source1.tsv (header excluded)
  3. every predicted / candidate id exists in test Source 2 or 3
  4. no duplicate id inside a row; no S2/S3 id matched to two S1 entities
  5. every match is present in candidate_pairs.tsv for the same S1
  6. file sizes, absolute paths, SHA-256 of both files

Usage: python utils/final_checks.py <submission_dir> [--test-dir DIR]
Exit code 0 only if every check passes.
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

NL, TAB = chr(10), chr(9)


def ids_of(path, col=0):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        return [row[col] for row in r]


def read_sub(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            rows.append((s1, [x for x in rest.split(",") if x]))
    return header, rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub_dir")
    ap.add_argument("--test-dir", default=r"C:\aml2026\data\test")
    a = ap.parse_args()
    sub, test = Path(a.sub_dir).resolve(), Path(a.test_dir)
    fails = []

    def check(ok, msg):
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            fails.append(msg)

    s1 = ids_of(test / "test_source1.tsv")
    s1set = set(s1)
    q = set(ids_of(test / "test_source2.tsv")) | set(ids_of(test / "test_source3.tsv"))
    # stream both files row by row (same S1 order is NOT assumed; candidates of the
    # current S1 are looked up from a streamed pass keyed by row position when orders agree)
    stats = {}
    for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        p = sub / name
        seen, n_rows, n_ids, bad_ids, dup_rows, header = set(), 0, 0, 0, 0, None
        with open(p, encoding="utf-8", newline="") as f:
            header = f.readline().rstrip(NL).split(TAB)
            for line in f:
                s1id, _, rest = line.rstrip(NL).partition(TAB)
                xs = [x for x in rest.split(",") if x]
                n_rows += 1
                seen.add(s1id)
                n_ids += len(xs)
                bad_ids += sum(1 for x in xs if x not in q)
                dup_rows += len(set(xs)) != len(xs)
        check(header == ["source1_entity_id", col], f"{name}: header {header}")
        check(n_rows == len(s1), f"{name}: {n_rows} rows vs {len(s1)} in test_source1.tsv")
        check(len(seen) == n_rows, f"{name}: every S1 id at most once")
        check(seen == s1set, f"{name}: S1 ids == test_source1 ids exactly")
        check(bad_ids == 0, f"{name}: all {n_ids} ids exist in test S2/S3 ({bad_ids} unknown)")
        check(dup_rows == 0, f"{name}: no duplicate id within a row ({dup_rows} rows with duplicates)")
        stats[name] = n_ids
        print(f"     {name}: {p.stat().st_size / 2**20:.1f} MB  sha256 {sha256(p)}  path {p}", flush=True)
    del q
    matched_seen, dup_matched, not_in_cand, nonempty = set(), 0, 0, 0
    with open(sub / "matching_results.tsv", encoding="utf-8") as fm, open(sub / "candidate_pairs.tsv", encoding="utf-8") as fc:
        fm.readline(); fc.readline()
        same_order = True
        for lm, lc in zip(fm, fc):
            sm, _, rm = lm.rstrip(NL).partition(TAB)
            sc, _, rc = lc.rstrip(NL).partition(TAB)
            if sm != sc:
                same_order = False
                break
            xs = [x for x in rm.split(",") if x]
            nonempty += bool(xs)
            for x in xs:
                dup_matched += x in matched_seen
                matched_seen.add(x)
            if xs:
                cs = set(rc.split(","))
                not_in_cand += sum(1 for x in xs if x not in cs)
    check(same_order, "matching and candidate files list S1 in the same order (needed for the streamed subset check)")
    check(dup_matched == 0, f"matching: no S2/S3 id matched to two S1 ({len(matched_seen)} distinct matches)")
    check(not_in_cand == 0, f"every match is in candidate_pairs.tsv for the same S1 ({not_in_cand} missing)")
    print(f"     candidate pairs {stats['candidate_pairs.tsv']} ({stats['candidate_pairs.tsv'] / len(s1):.2f} per S1); "
          f"matches {stats['matching_results.tsv']}; S1 with >=1 match {nonempty}")
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED")
    sys.exit(0 if not fails else 1)
