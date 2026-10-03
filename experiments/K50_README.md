# K50 experiment

This phase measures how much of the Top-50 retrieval ceiling converts into real macro-F0.5. The full plan and its rationale are in `C:\Users\Divya\.claude\plans\nested-sniffing-perlis.md`.

All artefacts live under `C:\aml2026\k50_exp\` (override with `AML_K50_ROOT`). Earlier caches, models and submissions are never written; `k50_common.guard` enforces this.

## Design rules

**D1 – Top-10 context at every rank.** Every pair, at any rank, uses the query's top-10 retrieval list as its context:

- `ret_top1`, `ret_margin` and `ret_ncand` are computed from that list.
- The AR-7 contrast features compare against that list.
- As a result, ranks 0–9 are identical to the validated K10 caches (parity tests T4a–c).

**D3 – Labels only for retrieved pairs.** A true owner outside the top 50 creates no row. It is counted as a retrieval miss instead.

**D4 – Leakage rule for V-dense models.** A training query may not be owned by any entity in E′, and may not have any E′ entity anywhere in its top 50. This fixes leak L1 (see Phase 0).

- E′ is the set of India fold-1 entities in hash bucket < b.
- It is nested inside the V-dense set E (bucket < 100).

**D5 – Scored pairs only.** `candidate_pairs.tsv` contains exactly the pairs that were scored.

## Commands

Run from `Amazon-ML\`, one heavy job at a time, on AC power.

| Phase | Command | Output |
|---|---|---|
| 0 – leakage-free A–D | `python -m experiments.exp3c` | `reports/phase0_exp3c.log`, `experiments/exp3_results.jsonl` (run `exp3c_clean`) |
| A – K50 retrieval, train (India) | `python -m src.candidates train blend_ad15_k50 --compact --out C:\aml2026\k50_exp\cand_train --countries India` | `cand_train/` (compact, resumable) |
| B – ceilings and E′ sizing | `python -m experiments.k50_ceiling` | `reports/k50_retrieval_stats.json`, `reports/k50_oracle.json` |
| Tests T1, T2, T4 | `python -m experiments.k50_parity` | `reports/k50_parity.json` |
| C – training data | `python -m experiments.k50_build_train <b>` | `train/b<b>/`, `reports/k50_train_stats_b<b>.json` |
| E – V-dense models | `python -m experiments.k50_train <b>` | `model/*.txt`, `model/*.meta.json` |
| F – eval set | `python -m experiments.k50_build_eval <b>`, then `... <b> --us` | `eval/b<b>/`, `eval/us/` |
| F, G, H – evaluation and gate G1 | `python -m experiments.k50_eval <b>` | `reports/k50_validation.json`, `reports/k50_results.md` |
| A – K50 retrieval, test (only after G1) | `python -m src.candidates test blend_ad15_k50 --compact --out C:\aml2026\k50_exp\cand_test` | `cand_test/` |
| Production model | `python -m experiments.k50_train_final <thr> <r> '<policy json>'` | `C:\aml2026\work\m2_model_final_k50.*` |
| Test inference | `python -m experiments.predict_k50 C:\aml2026\k50_exp\outputs` | `outputs/*.tsv`, `outputs/prediction_stats.json` |
| Validation | `python utils/validate_submission.py --matching ... --candidate ... --test-dir C:\aml2026\data\test --check-ids` and `python utils/final_checks.py C:\aml2026\k50_exp\outputs` | validator logs |
| Backup | `python utils/backup_submission.py C:\aml2026\k50_exp\outputs C:\aml2026\submissions\submission_4_K50 "<config>" --reports ...` | backup folder with SHA256SUMS |

## Compact cache format

Candidate chunks are stored as `<country>_<i>.parquet` (zstd) with these columns:

- `q_pos` (int32)
- `s1_pos` (int32)
- `score` (float32)
- `rank` (int8)

Rows are sorted by `q_pos`, then `rank`.

Supporting files:

- `index_<country>_q.parquet` and `index_<country>_s1.parquet` map positions back to entity ids.
- `manifest.json` holds the config plus a row count and SHA-256 for each chunk.
- A `DONE` marker is written when the run completes.

Use `src.candidates.load_compact_chunk(dir, stem)` to read a chunk with entity ids restored.

## Gate G1

Test inference runs only if the best K50 policy clears all four conditions:

- F0.5 gain ≥ +0.0010 over system 1 (the D regime) on half-B;
- paired 95% CI entirely above 0;
- the gain holds when the tune and report halves are swapped;
- US gain ≥ −0.0005.

Model and policy are selected on half-A only. Otherwise the best earlier validated system stays final.
