"""Build <team_name>_submission.zip in the exact structure required by the challenge:

    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, README.md, requirements.txt}
    Documentation_template.md

Usage: python utils/make_package.py <team_name> [--from <dir with the two TSVs>] [--dest <dir>]
  --from defaults to OUTPUT_DIR; --dest defaults to the --from directory.
"""
import hashlib
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import OUTPUT_DIR  # noqa: E402

def _arg(name, default):
    return Path(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    team = sys.argv[1]
    src_dir = _arg("--from", OUTPUT_DIR)
    dest = _arg("--dest", src_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dst = dest / f"{team.replace(' ', '_')}_submission.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(src_dir / f, f"output/{f}")
        base = "code/business_entity_resolution"
        for p in sorted((REPO / "src").rglob("*.py")):
            z.write(p, f"{base}/src/{p.relative_to(REPO / 'src').as_posix()}")
        # experiment/validation scripts (incl. exp_blocking2, which produces the US training
        # sample used by src.train_final) ship under src/experiments/
        for p in sorted((REPO / "experiments").glob("*.py")):
            z.write(p, f"{base}/src/experiments/{p.name}")
        for u in ("validate_submission.py", "final_checks.py", "backup_submission.py"):
            z.write(REPO / "utils" / u, f"{base}/src/utils/{u}")
        z.write(REPO / "experiments" / "K50_README.md", f"{base}/src/experiments/K50_README.md")
        z.write(REPO / "README.md", f"{base}/README.md")
        z.write(REPO / "requirements.txt", f"{base}/requirements.txt")
        z.write(REPO / "Documentation_template.md", "Documentation_template.md")
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB) sha256 {h.hexdigest()}")
