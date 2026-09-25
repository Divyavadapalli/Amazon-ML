# Business Entity Resolution — Amazon ML Challenge 2026

For each Source-1 business, find every Source-2 / Source-3 record describing the same
real-world business. Scored by per-entity macro F0.5 (singletons included).

## Pipeline

```
raw TSV ─► prep (normalise, romanise scripts) ─► blocking (TF-IDF top-K, same country)
       ─► pair features ─► LightGBM match probability ─► assign each S2/S3 record to its
          best S1 above threshold ─► matching_results.tsv + candidate_pairs.tsv
```

| Stage | Module | Notes |
|---|---|---|
| Normalisation | `src/normalize.py`, `src/translit.py` | anyascii romanisation + script→English token map learned from training pairs |
| Blocking | `src/blocking.py` | S2/S3 → S1 top-K, blended name-word / name-char / address TF-IDF |
| Features | `src/features.py` | rapidfuzz similarities, token/number overlap, retrieval rank & margin |
| Metric | `src/metrics.py` | exact re-implementation of the leaderboard macro F0.5 |

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Data location is configured by environment variables (defaults in `src/config.py`):

| Variable | Default | Content |
|---|---|---|
| `ER_DATA_DIR` | `C:\aml2026\data` | `train/` and `test/` exactly as in `student_resource/dataset` |
| `ER_WORK_DIR` | `C:\aml2026\work` | cached parquet, candidates, models |
| `ER_OUTPUT_DIR` | `C:\aml2026\output` | final TSVs |

## Run end-to-end

```bash
python -m src.prep                                   # translit dictionary (train fold 0) + normalised parquet
python -m src.candidates train blend_ad15            # full-index candidates for training (resumable)
python -m experiments.exp_blocking2 blend_ad15 0.05  # 5 % full-index sample (US training pairs)
# (in the submission zip these scripts live in src/experiments/: python -m src.experiments.exp_blocking2 ...)
python -m src.candidates test blend_ad15             # full-index candidates for test (resumable)
python -m src.train_final final_v1 0.66 100          # final LightGBM, threshold frozen in meta
python -m src.pipeline predict final_v1              # -> output/matching_results.tsv + candidate_pairs.tsv
python utils/validate_submission.py --matching <out>/matching_results.tsv        --candidate <out>/candidate_pairs.tsv --test-dir <data>/test --check-ids
```

## Validation

Training S1 entities are split 50/50 by a hash of their id (`src/prep.py:s1_fold`).
Fold 0 trains models and the transliteration dictionary; fold 1 is scored with
`src/metrics.py`. `src/devset.py` builds closed-world universes that keep the real
share of unmatched distractor records.

## Compliance

* No external data, APIs, registries or geocoding. Everything is learned from the
  provided training files; the only hand-written knowledge is abbreviation/region maps
  in `src/normalize.py`.
* Final model: LightGBM (MIT). No pretrained models.

Experiment log: `experiments/EXPERIMENTS.md`. Submission ledger: `submissions/SUBMISSION_LOG.md`.
