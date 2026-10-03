# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** mog squad
**Team Members:** Divya Sri Vadapalli (team leader) and team
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

The pipeline works in five steps:

1. Normalise every record and romanise Indic scripts, using a transliteration dictionary learned from training matches.
2. Retrieve Source-1 candidates per country with address-weighted TF-IDF (top-10).
3. Score each candidate pair with LightGBM on 70 features. These are 49 string/number/legal/retrieval features plus 21 "difference-aware" features that describe *what kind* of difference separates two near-identical records.
4. Always score each record's top-3 candidates. Score ranks 4–10 only when the best top-3 probability is below 0.5.
5. Assign each Source-2/3 record to its single best Source-1 entity if the probability is at least 0.80.

**Validation.** Measured on held-out Indian entities at the test set's distractor density, with leakage-free retraining:

| System | Macro F0.5 | Gain | 95 % CI |
|---|---|---|---|
| First submission (V1) | 0.9616 | – | – |
| Final system (D) | **0.9709** | +0.0093 | [+0.0083, +0.0104] |

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA of the training data:

- **Size:** 2.2M S1, 5.0M S2 and 5.3M S3 records.
- **Matches:** 3.46 per S1 on average; 5.6 % of S1 entities are singletons.
- **Structure:** no S2/S3 record matches more than one S1, and every match shares the S1's country.
- **Distractors:** 26 % of training S2/S3 records match nothing. The test set has about 40 %: its best-probability distribution is exactly a 0.397 distractor mixture, and it has 5.8 vs 4.7 S2/S3 records per S1. Validation therefore clones distractors to test density; this protocol predicted the first public score to within 0.0007.
- **Noise:**
  - 18–28 % of Indian S2/S3 names are in Indic scripts;
  - Indian S1 names are heavily duplicated;
  - junk prefixes and domain-style names occur;
  - states appear as codes, full names or native script.
- **Hard distractors:** near-copies of a real S1 with one house number changed, or a different legal form. 87 % of false merges are records that belong to no S1.

### 2.2 Solution Strategy

**Approach type:** blocking + pairwise classifier + query-side assignment. Each S2/S3 record gets at most one S1; an S1 can receive any number of records.

**Core innovations:**

- Address-weighted retrieval that overcomes name duplication.
- A learned script→English token dictionary.
- Number-agreement features.
- **Difference-aware (AR-7) features**, which classify the residual differences between two records:
  - one-digit substitution, transposition, extension or segmentation of numbers;
  - typo, abbreviation or joined-word vs novel tokens, with IDF weight;
  - how many S1 share this S1's address;
  - whether the record contains the tokens that distinguish this S1 from the other retrieved look-alikes.
- Confidence-gated candidate expansion.

---

## 3. Candidate Generation (Blocking)

- **Normalisation before blocking:** AnyAscii plus the learned dictionary (3,225 tokens, 98.5 % held-out accuracy), legal-form canonicalisation, street and state abbreviations.
- **Blocking:**
  - per country;
  - TF-IDF views fitted on S1 only: name-core words, name character 3-grams, address tokens;
  - blend weights 1 / 1 / 1.5;
  - top-10 S1 per S2/S3 record (`sparse_dot_topn`, Apache-2.0).
- **Final candidate set:**
  - ranks 1–3 for every record, plus ranks 4–10 for records whose best top-3 probability is < 0.5;
  - 59,505,769 pairs (34.4 per S1). `candidate_pairs.tsv` is exactly the set of pairs the model scored.
- **Recall on held-out Indian entities, full S1 index:**

| Recall @K | @3 | @10 | @50 |
|---|---|---|---|
| Measured | 0.949 | 0.963 | 0.980 |

- **Perfect-matcher oracle F0.5:** 0.9806 at top-3, 0.9862 at top-10.
- **Why not top-50:** it was measured (0.9923 oracle), but a top-50 matcher could not be trained and validated in the time available.

---

## 4. Matching Model

**Features (70):**

- **Base features (49):**
  - name and address similarities (RapidFuzz ratio, token-sort, token-set, partial, Jaro-Winkler, Levenshtein);
  - token and digit overlap;
  - number agreement (S1 numbers missing or extra, all S1 numbers present, postcode);
  - legal-form agreement or conflict;
  - retrieval score, rank, margin and gap;
  - S1 name frequency; source flag.
- **AR-7 features (21):** described in Section 2.2.
- **No country feature**, so France (absent from training) is handled by the same model.

**Model type:**

- LightGBM (MIT), binary objective, 127 leaves, learning rate 0.08, seed 42.
- Early stopping on a 10 % query-group split: 623 trees.
- Trained on 7.24M labelled training pairs: 10 % of Indian queries plus a 5 % US full-index sample.

**Threshold selection:**

- F0.5 is maximised on half-A of the held-out entities at test distractor density: threshold 0.80.
- Everything is reported on the disjoint half-B.
- The gate for ranks 4–10 (0.5) was fixed before testing.

---

## 5. Results & Error Analysis

Held-out Indian entities at test distractor density (half-B, 22,325 entities; leakage-free retraining):

| System | Macro F0.5 | Precision | Recall | False merges | Misses |
|---|---|---|---|---|---|
| V1: top-3, 49 features (first submission) | 0.9616 | 0.990 | 0.912 | 734 | 6,740 |
| + gated ranks 4–10 | 0.9651 | 0.989 | 0.921 | 770 | 6,108 |
| + AR-7 features, top-3 | 0.9676 | 0.992 | 0.924 | 541 | 5,832 |
| **D: both (submitted)** | **0.9709** | **0.992** | **0.932** | **557** | **5,247** |

- **Robustness of D's gain:**
  - D's gain is +0.0093 (95 % CI +0.0083 to +0.0104);
  - it reproduces with a second seed (+0.0092) and with the tune/report halves swapped (+0.0095).
- **Public leaderboard:** the first submission (V1) scored 0.9621 against a predicted 0.9628.
- **Common false positives:**
  - near-copy distractors (one house or unit number changed, legal form changed). AR-7 cut them by 26 %;
  - different businesses sharing one address.
- **Common false negatives:**
  - true S1 outside the top-10: empty address with a generic name, rare script names, trade names;
  - heavily reformatted numbers below the threshold.
- **Known caveat:** on a US spot-check, the India-only validation model with AR-7 is slightly worse (−0.0010). The submitted model also trains on US pairs.

---

## 6. Conclusion

- Most of the gains came from evidence-driven features and candidate handling rather than model choice:
  - address-aware retrieval;
  - learned transliteration;
  - number agreement;
  - difference-aware features;
  - confidence-gated expansion.
- A validation protocol that reproduces the test's distractor density made the public leaderboard predictable. Every change was accepted only on paired, held-out evidence.
- **Next steps:**
  - a top-50 matcher: +0.006 oracle headroom, already measured;
  - better handling of empty-address and script-name records;
  - direct validation on US and French data.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`; see `README.md` for exact commands.

- `src/`: normalisation, transliteration, blocking, features, LightGBM, metrics, inference.
- `experiments/`:
  - AR-7 features (`ar7_features.py`);
  - final model (`train_final_v2.py`) and inference (`predict_v3.py`);
  - the validation protocol (`vdense.py`, `exp3*.py`);
  - K50 research scripts (not used for the submission).
- `utils/`: official validator and independent output checks.

### B. Additional Results

- **Transliteration ablation** (held-out mean name similarity to the true S1):

| Method | Similarity |
|---|---|
| None | 24.7 |
| AnyAscii | 71.9 |
| Dictionary | 62.8 |
| Dictionary + AnyAscii | 85.3 |

- **Compliance:**
  - no external data, APIs, registries or geocoding;
  - libraries: scikit-learn (BSD-3), rapidfuzz (MIT), sparse-dot-topn (Apache-2.0), anyascii (ISC), LightGBM (MIT);
  - no pretrained models.
