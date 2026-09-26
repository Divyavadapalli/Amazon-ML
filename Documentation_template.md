# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** mog squad
**Team Members:** Divya Sri Vadapalli (team leader) and team
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
A block-then-match pipeline. Records are normalised, including learned transliteration of Indic scripts. An address-weighted TF-IDF retrieval (per country, top-3 Source-1 candidates per Source-2/3 record) builds the candidate set. A LightGBM classifier on 49 string, number, legal-form and retrieval features scores each pair. Each S2/S3 record is assigned to at most its best Source-1 entity above a precision-tuned threshold. Held-out macro F0.5 (India validation) is **0.9651** with top-3 candidates (0.9681 with top-10).

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data:
- **Size:** 2.2M S1, 5.0M S2 and 5.3M S2/S3 records in training.
- **Matches:** 3.46 matches per S1 on average; 5.6 % of S1 entities are singletons.
- **Ownership:** no S2/S3 record matches more than one S1 (0 of 7.64M), and 100 % of matches share the S1 country.
- **Distractors:** about 26 % of S2/S3 records match nothing.
- **Noise:**
  - 18–28 % of Indian S2/S3 names are written in Indic scripts (Devanagari, Telugu, Tamil, Kannada, Gujarati, Bengali).
  - Indian S1 names are heavily duplicated: a median of 54 S1 share a name among blocking misses.
  - Junk prefixes, domain-style names, `M/s`, `trading as`, the literal `null` in addresses, mojibake.
  - State written as a code, a full name, or in a native script.
- **Distractor pattern:** hard distractors are near-duplicates with one house number changed and/or a different legal form.

### 2.2 Solution Strategy
**Approach Type:** Blocking + pairwise classifier + constrained assignment.
**Core Innovation:**
- Address-weighted retrieval that overcomes name duplication.
- A script→English token dictionary learned from training matches.
- Number-agreement features (zero-stripped digit sets, "all S1 numbers present") that separate true matches from near-duplicate distractors.

---

## 3. Candidate Generation (Blocking)
- **Normalisation before blocking:**
  - Romanisation with AnyAscii plus a learned dictionary of 3,225 script tokens (98.5 % accurate on held-out tokens).
  - Legal-form canonicalisation, street-type and state abbreviations.
- **Blocking:** per country (country treated as an open set).
  - Three TF-IDF views fitted on S1: name-core words, name character 3-grams, address tokens.
  - Views blended with weights 1 / 1 / 1.5.
  - Features with document frequency above 1–2 % are excluded.
  - Each S2/S3 record retrieves its top-K S1 by cosine (`sparse_dot_topn`, Apache-2.0).
- **Final candidate set:**
  - top-3 per S2/S3 record: about 29M pairs for 1.73M test S1, versus 1.73M × 9.97M possible pairs.
  - Per-query retrieval features are still computed from the top-10 list.
- **How true matches were kept:** measured against the full training S1 index.

  | Configuration | Pair recall | Oracle F0.5 |
  | --- | --- | --- |
  | Name-heavy blend | 0.951 | 0.982 |
  | Address-weighted blend (chosen) | 0.977 at K=10; 0.968 at K=3 | 0.992 |

- **Rejected with evidence:**
  - Address bigrams: +0.2 pp for 3.7× the cost.
  - Lower max_df: −2.6 pp recall.
  - Address weight 2.0: lower recall.
  - State partitioning: 2–5 % of true pairs disagree on state.

---

## 4. Matching Model

**Features used (49):**
- Name features:
  - RapidFuzz ratio, token-sort, token-set, partial, Jaro-Winkler and Levenshtein on core and full names.
  - Best similarity over DBA parts.
  - Token Jaccard / overlap, missing and extra tokens.
  - Raw-name ratio; non-ASCII flag.
- Address features:
  - ratio, token-sort, token-set, partial, Jaccard, overlap.
  - Empty-address flag, PO box flag.
- Number features:
  - Digit-set Jaccard, first-number equality, conflict, postcode equality.
  - S1 numbers missing in the query, extra numbers, "all S1 numbers present", longest shared number, share of shared numbers.
- Legal form: agreement, conflict, `pvt` mismatch, symmetric difference.
- Other:
  - Retrieval score, rank, gap to top-1, margin over runner-up, candidate count.
  - Frequency of the S1 name in its country; source (S2/S3).
  - Country is not a feature, so France (unseen in training) is handled like any other country.

**Language-invariant address signals.** French addresses reorder and rename street syntax ("12 Rue de la Paix" vs "Paix Street 12"), so token- and character-similarity on address text degrades under the unseen-French shift. The number features sidestep this: `_nums()` extracts zero-stripped digit runs and `postal_eq` compares 5–6-digit postal codes, both independent of surrounding words or language. Because these are structural rather than lexical, they carry the same discriminative power on French records as on the training distribution — which is why country is deliberately excluded as a feature. The conflict signals (`num_conflict`, `legal_conflict`) then catch the dominant false-merge mode — matching names with clashing house numbers — that pure name similarity cannot. A gradient-boosted tree learns these interactions directly, which is why it lifts macro F0.5 from 0.722 (retrieval-score rule) to 0.9651.

**Model type:** LightGBM (MIT licence), binary objective, 127 leaves, learning rate 0.08, early stopping (935 trees). Trained on 7.24M labelled pairs from training data only.

**Threshold selection method:**
- F0.5 is optimised on one half of the held-out entities (tune half) and reported on the other half.
- Threshold 0.73 for top-3 candidates.
- Assignment: each S2/S3 record goes to its single highest-probability S1, if that probability is at least the threshold. S1 entities can receive zero, one or many records.

---

## 5. Results & Error Analysis

Held-out India validation (22,331 entities, 1,272 singletons). Candidates come from the full training S1 index, and every query that can reach an evaluated entity is included.

| Model | Macro F0.5 | Pair precision | Pair recall |
| --- | --- | --- | --- |
| Rule (retrieval score) | 0.722 | 0.743 | 0.866 |
| Logistic regression | 0.925 | 0.961 | 0.885 |
| LightGBM, 42 features | 0.9618 | 0.988 | 0.923 |
| LightGBM, 49 features, top-10 | 0.9681 | 0.991 | 0.933 |
| **LightGBM, 49 features, top-3 (submitted)** | **0.9651** | **0.992** | **0.922** |

- **F0.5 Score (macro):** 0.9651 (submitted top-3 configuration). Singleton accuracy about 0.96; multi-match entities F0.5 about 0.97.
- **Common false positives (wrong merges):**
  - Near-duplicate distractors: same name, one house or unit number changed (e.g. `626/48` vs `626/49`), often a different legal form.
  - Different businesses sharing one address.
  - The number and legal-form features cut these by 20 %.
- **Common false negatives (missed matches):**
  - About half are blocking misses: empty address with a generic name, weak transliteration of rare script names.
  - The rest are below-threshold matches with heavily reformatted house numbers or script names.

---

## 6. Conclusion
- Address-aware retrieval and features designed from error analysis gave most of the gains. Learned transliteration and number agreement mattered more than model choice.
- Precision was prioritised throughout (pair precision about 0.99), in line with F0.5.
- Next steps:
  - Validate US matching as rigorously as India.
  - Recover below-threshold matches (about +0.012 potential).
  - Recover blocking misses on empty-address records.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — see `README.md` for exact commands.
- `src/prep.py`: learn transliteration dictionary; normalise all sources to parquet.
- `src/normalize.py`, `src/translit.py`: normalisation and transliteration.
- `src/blocking.py`, `src/candidates.py`: resumable, chunked TF-IDF candidate generation.
- `src/features.py`: pairwise features.
- `src/matching.py`, `src/train_final.py`: LightGBM training.
- `src/pipeline.py`: inference, assignment, hard checks, writes both TSVs.
- `src/metrics.py`, `src/evaluation.py`: exact macro F0.5 (reproduces the 0.714 worked example).
- `src/experiments/`: blocking, transliteration, validation and error-analysis experiments.
- `src/utils/validate_submission.py`: the official validator.

Entry points: `python -m src.pipeline predict final_v1 3 0.73` produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
- Transliteration ablation (held-out, mean name similarity to the true S1): none 24.7, AnyAscii 71.9, dictionary 62.8, dictionary + AnyAscii 85.3.
- Compliance: no external data, APIs, registries or geocoding. Libraries: scikit-learn (BSD-3, TF-IDF only), rapidfuzz (MIT), sparse-dot-topn (Apache-2.0), anyascii (ISC), LightGBM (MIT).
