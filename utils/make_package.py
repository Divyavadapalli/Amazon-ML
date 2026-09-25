"""Build <team_name>_submission.zip in the exact structure required by the challenge:

    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, README.md, requirements.txt}
    Documentation_template.md

Usage: python utils/make_package.py <team_name>
"""
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.config import OUTPUT_DIR  # noqa: E402

if __name__ == "__main__":
    team = sys.argv[1]
    dst = OUTPUT_DIR / f"{team}_submission.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(OUTPUT_DIR / f, f"output/{f}")
        base = "code/business_entity_resolution"
        for p in sorted((REPO / "src").rglob("*.py")):
            z.write(p, f"{base}/src/{p.relative_to(REPO / 'src').as_posix()}")
        z.write(REPO / "README.md", f"{base}/README.md")
        z.write(REPO / "requirements.txt", f"{base}/requirements.txt")
        z.write(REPO / "Documentation_template.md", "Documentation_template.md")
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
