"""Sanity statistics vs Submission #1 and a permanent backup of a new submission.

Usage: python utils/backup_submission.py <new_dir> <backup_dir> <config text> [--ref C:\\aml2026\\output]
Copies matching_results.tsv, candidate_pairs.tsv, prediction_stats.json and the validator /
final_checks reports into <backup_dir>, writes config.txt, validation.txt and SHA256SUMS.txt.
"""
import argparse
import hashlib
import shutil
import statistics
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(d):
    counts, cand = [], 0
    with open(Path(d) / "matching_results.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            rest = line.rstrip("\n").split("\t", 1)[1]
            counts.append(len([x for x in rest.split(",") if x]))
    with open(Path(d) / "candidate_pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            rest = line.rstrip("\n").split("\t", 1)[1]
            cand += len([x for x in rest.split(",") if x])
    n = len(counts)
    matched = sum(1 for c in counts if c)
    return {"S1": n, "matched_S1": matched, "unmatched_S1": n - matched, "total_matches": sum(counts),
            "avg_matches_per_S1": round(sum(counts) / n, 4), "median_matches_per_S1": statistics.median(counts),
            "candidate_pairs": cand, "avg_candidates_per_S1": round(cand / n, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir"); ap.add_argument("backup_dir"); ap.add_argument("config")
    ap.add_argument("--ref", default=r"C:\aml2026\output")
    ap.add_argument("--reports", nargs="*", default=[])
    a = ap.parse_args()
    ref, new = stats(a.ref), stats(a.new_dir)
    lines = [f"{'':24s}{'Submission #1':>16s}{'new':>16s}"]
    for k in ref:
        lines.append(f"{k:24s}{ref[k]!s:>16s}{new[k]!s:>16s}")
    table = "\n".join(lines)
    print(table)
    b = Path(a.backup_dir)
    b.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv", "prediction_stats.json"):
        if (Path(a.new_dir) / f).exists():
            shutil.copy2(Path(a.new_dir) / f, b / f)
    val = []
    for r in a.reports:
        val.append(f"===== {r}\n" + Path(r).read_text(encoding="utf-8", errors="replace"))
    (b / "validation.txt").write_text("\n".join(val) + "\n===== sanity vs Submission #1\n" + table + "\n", encoding="utf-8")
    (b / "config.txt").write_text(a.config + "\n", encoding="utf-8")
    sums = [f"{sha(b / f)}  {f}" for f in ("matching_results.tsv", "candidate_pairs.tsv")]
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert sha(b / f) == sha(Path(a.new_dir) / f), f"backup copy differs: {f}"
    (b / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print("\n".join(sums))
    print(f"backup: {b.resolve()}")
