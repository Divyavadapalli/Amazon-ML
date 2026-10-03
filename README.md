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

## Final submission: System D (top-K expansion + AR-7 difference features)

The submitted files come from the V1 pipeline above plus three additions:

- **AR-7 features** (`experiments/ar7_features.py`). These 21 difference-aware features describe *what kind* of difference separates two records, for example a one-digit house-number change versus a formatting change. They also cover IDF-weighted token agreement, how many S1 records share the same address, and whether the record contains the tokens that distinguish this S1 from its retrieved look-alikes.
- **Confidence-gated expansion.** Every S2/S3 record's top-3 candidates are scored. Ranks 4–10 are scored only when the best top-3 probability is below 0.5.
- **Threshold 0.80**, tuned at test distractor density (see Validation).

```bash
# prerequisites: the V1 steps above up to "src.candidates test blend_ad15" and src.train_final's cache
python -m experiments.train_final_v2 0.80 0.5 10     # final_v2 = 49 base + 21 AR-7 features (LightGBM, seed 42)
python -m experiments.predict_v3 <out_dir>          # -> matching_results.tsv + candidate_pairs.tsv (scored pairs only)
python utils/validate_submission.py --matching <out_dir>/matching_results.tsv --candidate <out_dir>/candidate_pairs.tsv --test-dir <data>/test --check-ids
python utils/final_checks.py <out_dir>
```

**Running from the submission zip.** `experiments/` ships inside `src/experiments/`. From `code/business_entity_resolution/`, set `PYTHONPATH=src`; the modules then resolve as both `experiments.*` and `src.*`. For example: `set PYTHONPATH=src && python -m experiments.predict_v3 <out_dir>`.

**Reproducibility note.** On 2026-09-27 a parity test found that one AR-7 feature, `e7_num_q_novel`, depended on Python's per-process hash seed (set iteration order).

- It is now computed deterministically (sorted iteration).
- For the submitted run it differed in about 0.9 % of rows of that single feature.
- A re-run with the current code can therefore differ from the submitted file in a small number of borderline decisions.

## Validation

Training S1 entities are split 50/50 by a hash of their id (`src/prep.py:s1_fold`).
Fold 0 trains the transliteration dictionary; fold-1 entities are the evaluation set.

**V-dense** (`experiments/vdense.py`) is the protocol every decision used:

- India fold-1 entities are split by hash into two halves. Half-A is used to tune thresholds; half-B is used to report.
- Candidates are retrieved from the full training S1 index.
- Unmatched distractor records are cloned from 26 % to the test's measured ~40 % share.
- F0.5 is the exact metric from `src/metrics.py`, with paired bootstrap confidence intervals.

V-dense predicted the first public leaderboard score to within 0.0007 (0.9628 vs 0.9621).

**Leakage-free re-measurement** (`experiments/exp3c.py`, half-B F0.5):

| System | F0.5 | Gain | 95 % CI |
|---|---|---|---|
| V1 | 0.9616 | – | – |
| Top-K expansion | 0.9651 | +0.0035 | – |
| AR-7 | 0.9676 | +0.0060 | – |
| D (top-K + AR-7) | 0.9709 | +0.0093 | [+0.0083, +0.0104] |

- The D gain reproduces with a second seed (+0.0092) and with the halves swapped (+0.0095).
- A US spot-check of the India-only validation model shows a small loss: −0.0010.

## Compliance

* No external data, APIs, registries or geocoding. Everything is learned from the
  provided training files; the only hand-written knowledge is abbreviation/region maps
  in `src/normalize.py`.
* Final model: LightGBM (MIT). No pretrained models.

Experiment log: `experiments/EXPERIMENTS.md`.
