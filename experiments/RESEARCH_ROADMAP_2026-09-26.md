# Entity-Resolution Research Audit and Roadmap — 2026-09-26

Scope: a full audit of the current Business Entity Resolution system, the challenge rules, the error and
ceiling analysis, a literature review, and a prioritised, evidence-gated roadmap.
Every number is labelled **M** (measured on our held-out, test-density validation), **E** (oracle or
counterfactual bound, never a score claim) or **H** (hypothesis, untested).
Deadline: 27 Sep 2026 23:59 IST (≈34 h after this audit started). Hardware: 12-thread laptop CPU,
16 GB RAM, **no GPU** (Intel integrated graphics).

---

## 1. Executive summary

**Where we are.** Public LB **0.9621** (V1). Our test-density validation (V-dense, India half-B) predicted
0.9628, so it tracks the leaderboard within 0.0007 and is the only selection criterion used here.

**What was measured today (M, V-dense half-B, 22,325 India S1 entities at test distractor density):**

| System | F0.5 seed 42 | F0.5 seed 7 | Gain | False merges | Misses | Candidate pairs |
|---|---|---|---|---|---|---|
| A: submitted configuration (49 features, top-3) | 0.9628 | 0.9631 | – | 807 | 6,332 | 1.00× |
| B: + gated top-K (ranks 3–9 if best top-3 p < 0.5) | 0.9661 | 0.9664 | +0.0033 | 818 | 5,761 | 2.01× |
| C: + AR-7 difference-aware features (70 features), top-3 | 0.9683 | 0.9684 | +0.0055 | 613 | 5,502 | 1.00× |
| **D: AR-7 + gated top-K** | **0.9718** | **0.9717** | **+0.0090** | 633 | 4,877 | 2.01× |
| D-lite: AR-7 + gate only to rank 4 | 0.9700 | – | +0.0072 | 621 | 5,191 | 1.29× |

Both seeds agree to within 0.0003. Paired-bootstrap CIs, the swapped-halves check and the US spot-check for
A–D are being produced by `experiments/exp3b.py` (running at the time of writing; see section 21). Done so
far: B replicates on the swapped halves (0.9614 → 0.9649, +0.0034) and does not regress on the US sample
(0.9839 → 0.9847, +0.0008); C and D pending.

**Where the remaining 0.028 (India, test density) sits:** ≈ 0.0135 is true pairs that retrieval never
returns in the top 10 (oracle with a perfect matcher on top-10 = 0.9865), and ≈ 0.0147 is matcher error
inside the top 10 (false merges on near-copy distractors, true records rejected below threshold).

**Ceilings.** A perfect matcher on today's top-10 candidates scores 0.9921 (mixed India+US sample) and on
top-50 0.9956 (E). So **0.999 and above cannot be reached with lexical retrieval at any practical K**, and
0.995 needs a near-perfect matcher on K≈50 candidates. The data is synthetic with typed corruptions; some
records are information-limited (a distractor textually identical to its S1 with an empty address;
invented trade names with no address; namesakes sharing name and city). No method separates those.

**Verdict on the targets.** 0.99 — *unsupported by current evidence* (would need both ≈ +0.010 from
retrieval beyond K10 and ≈ +0.008 from the matcher, neither available on this hardware by the deadline).
0.995 — *difficult; bounded by the K50 oracle 0.9956 on the mixed sample*. 0.999 / 0.9999 / 0.99999 —
*theoretically possible only with perfect information; information-limited on this data*.

**Recommendation (deadline: 27 Sep 23:59 IST).**
1. Ship **D = final_v2 (AR-7) + gated top-K** — the only large, replicated, measured gain (+0.0090);
   D-lite if the team rejects a 2× candidate set (decision D1, open since the prior review).
2. Before shipping: hold the V-dense evaluation queries out of final_v2's training so the shipped model
   and its threshold are measured, not assumed; fix README line 45; commit the Exp-3 code.
3. Only then try cheap post-processing experiments that need no new test features (entity-aware decision
   rule, seed-averaging) — each must pass the same gate.
4. Everything that needs new retrieval or a neural model is post-challenge.

**Expected private-LB score if D ships:** ≈ **0.970–0.972** (V-dense 0.9718 minus the observed
0.0007 LB offset, ± the ±0.001 run-to-run spread; France is unvalidated).

---

## 2. Current system audit (verified against code and artefacts, not summaries)

| # | Item | Current state | Where |
|---|---|---|---|
| 1 | Architecture | normalise + romanise → per-country TF-IDF retrieval (S2/S3 → S1, top-10) → top-3 scored → 49 pair features → LightGBM → each query assigned to its best S1 if p ≥ t → TSVs | `src/prep.py`, `src/candidates.py`, `src/pipeline.py` |
| 2 | Model | LightGBM binary, 127 leaves, lr 0.08, 935 trees, trained on 7.24M pairs (India 10 % of queries + US 5 % full-index sample), 7.2 % positives | `src/train_final.py`, `m2_model_final_v1.*` |
| 3 | Candidate generation | 3 TF-IDF views fitted on S1 per country: name-core words (`nw`), name char-3-grams (`nc`), address tokens (`ad`); blend weights 1 / 1 / 1.5; `max_df` 0.02 / 0.01 / 0.02; `min_score` 0.05; sparse top-K matmul | `src/blocking.py` (`blend_ad15`) |
| 4 | K | retrieve K=10; the submitted run scores ranks 0–2 only (29.9M pairs, 17.3 per S1); retrieval features use all 10 | `src/pipeline.py predict … 3 0.73` |
| 5 | Features | 49: retrieval (score, rank, top1, gap, margin, n-cand), 9 name similarities (rapidfuzz on core/norm/DBA parts), 5 address similarities, 12 token-set/number/legal/postal features, 7 number & legal-form features (M2), S1 name frequency, flags (empty addr, PO box, non-ASCII name, S3) | `src/features.py` |
| 6 | Training | labels from ground truth; early stopping on a 10 % query-group split; transliteration dictionary learned from fold-0 pairs only | `src/matching.py`, `src/translit.py` |
| 7 | Validation | **V-dense**: India fold-1 entities (hash bucket < 100, ≈ 44 k, halves A/B), candidates from the full training S1 index, distractor queries cloned ×0.94 to reach the test's 2.35 distractors per S1; threshold tuned on A, reported on B; paired bootstrap; gate = +0.0010 with CI > 0 | `experiments/vdense.py` |
| 8 | Threshold | 0.73 (submitted); 0.79 is the V-dense-tuned value, within 0.0001 | CLI override |
| 9 | Output | one row per S1; 5.68M matches; 94.0 % of S1 matched; 3.49 matches per matched S1 (truth: 3.46); validator PASS | `C:\aml2026\output` |
| 10 | Public LB | **0.9621** (V1) | Amazon portal |
| 11 | Validation score | V-dense **0.9628** (P 0.9887, R 0.9178); training density 0.9650 | M |
| 12 | Biggest errors | not retrieved in top-10 (+0.0152 E), best candidate below threshold (+0.0086 E), top-3 cut (+0.0058 E), false merges (+0.0063 E, 87 % distractors) | `error_analysis.jsonl`, prior review |
| 13 | Bottlenecks | Python-loop features 4–8 K pairs/s; test retrieval ≈ 70 min; 16 GB RAM (US retrieval paged at 2 jobs); Windows spawn start-up; no GPU | logs |
| 14 | Experiments | B0–B7 blocking, T1 translit, M1/M2 matcher, E1 errors, F1 final, EXP-1 gated top-K (+0.0034 M), **EXP-3 (today)** | `EXPERIMENTS.md`, `exp3_results.jsonl` |

**Audit findings that need fixing regardless of any new modelling:**

1. **README reproduces the wrong file.** `python -m src.pipeline predict final_v1` runs top-10 at the meta
   threshold 0.66; the submitted file came from `predict final_v1 3 0.73` (`C:\aml2026\predict_only.sh`).
   `Documentation_template.md` is right; `README.md:45` is not. The reviewers regenerate from the README.
2. **The measured Exp-3 code is uncommitted** (`experiments/vdense.py`, `exp3.py`, `exp3_ar7.py`,
   `ar7_features.py`, `build_eval_ext.py`, `predict_v2.py`). Commit before anything depends on it.
3. `experiments/BEST_KNOWN.md` (still M1 = 0.9618) and `submissions/SUBMISSION_LOG.md` (empty) are stale;
   the 0.9621 upload is recorded only in memory notes.
4. **The shipped model has never been measured.** `train_final` samples India queries by query-id hash,
   so final_v1's training rows overlap the V-dense evaluation queries; its threshold was transferred from
   the validation model. The US spot-check in `vdense.load_us` reuses final_v1's US *training* pairs, so it
   is only valid for models that did not train on them (true for the Exp-3 models, false for final_v1/v2).
5. `pipeline.predict` hard-codes `num_threads=12` (harmless here; not portable).

---

## 3. Challenge-rule audit (authoritative: problem statement PDF + guidelines PDF)

| Constraint | Source text (paraphrased) | Consequence for us |
|---|---|---|
| No external lookup | No external databases, APIs or services to look up or resolve entities; no commercial ER APIs; no government registries; **no geocoding**; **no external data augmentation from internet sources**; violation = disqualification | Everything must be learned from the supplied files. Pretrained models are *allowed as models* (rule 5 below) but anything that brings in external *reference data* (gazetteers, libpostal's OSM-trained address model, registries) is at best grey: avoid |
| Model licence and size | "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters" | LightGBM (MIT) is compliant. Any neural component must be MIT/Apache-2.0 **and** ≤ 8B |
| Output: matching_results.tsv | one row per test S1; header `source1_entity_id	matched_entity_ids`; comma list; empty for singletons; no duplicates; only existing S2/S3 ids | enforced by `pipeline._write` hard checks + official validator |
| Output: candidate_pairs.tsv | "the exact set of records you feed into your matching model … the last [stage]: whatever your model actually runs inference over"; matches must be a subset | with a gated top-K the file = ranks 0–2 of every query ∪ ranks 3–9 of gated queries — exactly the scored pairs |
| Candidate-set review | not scored; used "to analyse blocking quality (recall ceiling, reduction ratio)"; organisers' later update (per prior review): smaller candidate sets rank higher, magnitude unstated | a 2× larger candidate set is a judgement call for the team (open decision D1) |
| Metric | per-S1 macro F0.5, singletons included (1 if empty prediction, else 0) | singletons are worth a full point each; decision rules should be entity-aware (section 7.5) |
| Leaderboards | public = subset of test; **final ranking = private**; shortlisting uses both | select on V-dense, never on public LB |
| Submissions | ≤ 5 uploads per day, 25–27 Sep | ≤ 2 more uploads are needed |
| Package | `output/` (2 TSVs) + `code/business_entity_resolution/{src/, README.md, requirements.txt}` + filled `Documentation_template.md`; must regenerate outputs from data | README fix (finding 1); package new modules |
| Country | open set; France unseen in training; no hard-coding, filtering or one-hot on {US, India} | no country features; per-country index is a data partition, not a feature — compliant |
| Training data | "using only the provided training data" | the transliteration dictionary and every statistic come from training/test files only |

Classification used below: **LEGAL / SAFE**, **LEGAL BUT NEEDS VERIFICATION**, **NOT ALLOWED**.

---

## 4. What currently limits the score

Ranked by measured or bounded F0.5 at stake (India, test density):

| # | Limit | Evidence | At stake | Nature |
|---|---|---|---|---|
| 1 | True S1 not in the retrieved top-10 (namesakes crowd it out; empty address; script names; invented trade names) | 2,821 of 76,985 half-B true pairs (3.7 %); K10 recall India 0.962 | ≈ 0.0135–0.0152 (E) | architectural: retrieval |
| 2 | False merges on near-copy distractors (S1 copy + legal form added + one number changed) | 87 % of A's false merges are distractor records; test has 1.94× the training distractor density | ≈ 0.006–0.008 (E) in A; AR-7 removed 24 % of them (807 → 613, M) | matcher features |
| 3 | True records rejected (best candidate but p < threshold): heavy corruption looks like distractor edits | 1,854 pairs in the M2 analysis; reduced by AR-7 and the gate (FN 6,332 → 4,877 in D) | ≈ 0.006–0.009 (E) | matcher features |
| 4 | Top-3 cut of already-retrieved candidates | +0.0035 M, removed by the gate | solved in D | policy |
| 5 | One global threshold for a per-entity metric | a candidate's accept/reject should depend on its S1's other candidates (section 7.5) | unknown — probe queued | decision rule |
| 6 | Training volume | validation models trained on 1.76M pairs (5 % of India queries); final models on 7.24M | H: +0.000–0.002 | data |
| 7 | Process risk | shipped model never measured; README reproduces the wrong file | could silently cost more than any feature | engineering |

---

## 5. Error budget

### 5.1 Submitted system A (M; counterfactual gain if a whole error class vanished, half-B, training density)

| Error class | Pairs | F0.5 gain if fixed | Stage responsible | Fix type |
|---|---|---|---|---|
| True pair not in top-10 | 2,821 | +0.0152 | retrieval | architectural |
| True S1 is best candidate, p < threshold | 1,854 | +0.0086 | matcher | local (features) |
| True pair at ranks 3–9, cut by top-3 | 1,100 | +0.0058 | candidate policy | local (gate) — **done in D** |
| False merges, all | 542 | ≈ +0.0063 (≈ ×1.3 at test density) | matcher | local (features) |
| – of which distractor records (no owner in S1) | 471 (87 %) | – | matcher | – |
| – same address, different name | – | +0.0031 | matcher | local |
| True S1 lost to another S1 | 250 | +0.0011 | matcher / ranking | local |
| Singletons | 5.7 % of S1; accuracy 0.958 | each false merge on a singleton costs the full 1.0 of that S1 | decision rule | local |

These are upper bounds per class and overlap; they are not additive predictions.

### 5.2 Current best system D (M totals; split E)

| Quantity | Value | How derived |
|---|---|---|
| F0.5 | 0.97175 | M |
| Missing F0.5 to 1.0 | 0.0282 | M |
| True pairs (half-B) | 76,985 | M |
| Misses (FN) | 4,877 | M |
| – not retrieved in top-10 | 2,821 (58 % of misses) | M (retrieval funnel; unchanged by any matcher) |
| – inside top-10 (below threshold, lost to another S1, gate-excluded) | 2,056 | M (= 4,877 − 2,821) |
| False merges (FP) | 633, of which 259 near-copies (name and address ≥ 90 similar) | M |
| F0.5 lost to retrieval beyond top-10 | ≈ 0.0135 | E (1 − oracle top-10 0.9865) |
| F0.5 lost inside top-10 (matcher + decision) | ≈ 0.0147 | E (0.9865 − 0.9718) |

The exact per-class counterfactuals for D (`budget.py M7 adapt`) are queued behind exp3b; they split the
0.0147 between false merges and in-top-10 misses and separate singleton-target false merges.

### 5.3 Where the gap from 0.962 to 0.99+ is hiding

| From → to | Needs | Available by the deadline? |
|---|---|---|
| 0.9628 → 0.9718 | AR-7 features + gate | **yes, measured** |
| 0.9718 → ≈ 0.975 | halve the remaining matcher error inside top-10 (≈ 0.007 of 0.0147) | partly H (decision rule, bagging, more data) |
| ≈ 0.975 → ≈ 0.980 | recover ~half the rank 11–50 misses (K50 retrieval + retrained model) | no (8–10 h) |
| ≈ 0.980 → 0.99 | retrieval recall ≥ 0.995 **and** a near-perfect matcher; needs learned retrieval for DBA / script / empty-address records | no; unproven even post-challenge |
| 0.99 → 0.999+ | resolving records that carry no distinguishing information | not possible (section 7.3) |

---

## 6. Retrieval / blocking ceiling

### 6.1 Recall and oracle F0.5 by K (M: 5 % entity sample, India + US, full 2.21M-record S1 index, training density)

| K | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|
| True-pair recall | 0.9524 | 0.9629 | 0.9676 | 0.9725 | 0.9775 | 0.9817 | 0.9842 | 0.9877 |
| Oracle macro F0.5 (perfect matcher on these candidates) | – | – | – | – | 0.9921 | – | – | 0.9956 |
| Candidates per query | 1 | 2 | 3 | 5 | 10 | 20 | 30 | 50 |
| Retrieval time (5 % sample) | | | | | 575 s | | | 783 s |
| Peak RSS | | | | | 6.1 GB | | | 8.7 GB |

Oracle on the V-dense India half-B entities (E, test density; the oracle rejects every distractor, so density
does not change it): top-1 **0.9723**, top-3 **0.9810**, top-5 **0.9836**, top-10 **0.9865**.
India is the hard country: recall@10 0.962 vs US 0.9875. India is 46.8 % of test S1, against 40 % of training S1.

### 6.2 Where retrieval fails (recall K10 → K50, M)

| Segment | Share of true pairs | Recall K10 | Recall K50 | Comment |
|---|---|---|---|---|
| India | 40 % | 0.9622 | 0.9792 | namesakes: median 54 S1 share a name core; addresses decide |
| US | 60 % | 0.9875 | 0.9934 | |
| Query address empty | 4.4 % | **0.7549** | 0.8805 | name-only retrieval against thousands of namesakes |
| Non-Latin (script) name | 13.8 % | 0.9489 | 0.9674 | dictionary covers 53 % of script tokens |
| Name core not identical to S1 | 44 % | 0.9611 | 0.9757 | typos, DBA, domain names, invented trade names |
| Name core identical to S1 | 56 % | 0.9903 | 0.9972 | |
| S2 / S3 | 48 / 52 % | 0.9779 / 0.9769 | 0.9874 / 0.9880 | no source effect |

### 6.3 Candidate volume against realised F0.5 (M, V-dense half-B, AR-7 model M7)

| Candidate policy | Pairs vs submitted (×) | F0.5 | Gain vs top-3 |
|---|---|---|---|
| top-3 (as submitted) | 1.00 | 0.9683 | – |
| top-3 + ranks 3–4 when best top-3 p < 0.5 | 1.29 | 0.9700 | +0.0017 |
| top-3 + ranks 3–9 when best top-3 p < 0.2 | 1.99 | 0.97179 | +0.0035 |
| top-3 + ranks 3–9 when best top-3 p < 0.5 (**D**) | 2.01 | 0.97175 | +0.0035 |
| M0 (submitted features) all top-10 | 3.33 | 0.9661 | (M0 top-3 0.9628) |

The gate is what makes deeper K cheap: queries that already have a confident top-3 candidate almost
never have their owner below rank 3 — the whole top-10 benefit (+0.0035 on M0) is reached at 2.0× instead
of 3.3× the pairs. Test volumes: top-3 = 29.9M pairs (17.3 per S1); full top-10 ≈ 99.7M; gated ≈ 60M.

### 6.4 Answer: how much can retrieval still give?

* Ranks 4–10 (already retrieved): **+0.0035 M**, captured by the gate.
* Ranks 11–50: oracle +0.0035 E on the mixed sample (0.9921 → 0.9956); India-heavy test ≈ +0.004–0.006 E.
  Realising it needs K50 test retrieval (≈ 1.5 h), a model trained with rank 11–50 negatives, and a
  deeper gate. Realised share H: ~50 % of the oracle, i.e. **+0.002–0.003**; not feasible before the deadline
  on this laptop (≈ 8–10 h end-to-end).
* Beyond K50 (1.23 % of true pairs): lexical retrieval is exhausted; section 7.3 shows how much of it is
  information-limited.

---

## 7. Matcher ceiling and the information floor

### 7.1 Ceilings (India half-B unless stated)

| Ceiling | Value | Label |
|---|---|---|
| A. Perfect matcher, current candidates (top-3) | 0.9810 | E |
| A'. Perfect matcher, top-10 (what D can see) | 0.9865 | E |
| B. Perfect retrieval, current matcher | ≈ 0.9718 + 0.9 × 0.0135 ≈ 0.984 (the matcher recovers most retrieved easy pairs; retrieval misses are harder than average, so this is optimistic) | E/H |
| C. Perfect retrieval + perfect matcher | 1.0 minus the information floor (7.3) | E |
| D. Current system as K grows | 0.9683 (K3) → 0.9700 (gate K5) → 0.9718 (gate K10); K50 needs retraining | M |
| E–H. By segment (retrieval ceiling at K50) | empty address 0.88 recall, script names 0.967, India 0.979, US 0.993 | M |
| Mixed India+US, perfect matcher | K10 0.9921, K50 0.9956 | E |

### 7.2 What the remaining false merges look like (M: raw text of sampled errors)

The data is generated: a clean S1 record is copied into S2/S3 with typed corruptions. Observed operator
families on **true** pairs: character typos ("Wilblims", "Enterpires", "ENRTPRMISES"), wrong abbreviation
expansion ("Street" → "SAINT"), ordinal corruption ("45th" → "45ND"), zero padding ("Af-684" → "AF-0684"),
number ranges and suffixes ("1056" → "1056-1060", "1056c"), literal "null" tokens, dropped house numbers,
state names in native script, full-name transliteration (Tamil, Devanagari), domain-style names
("maurewilliamscolombier.com"), invented trade names ("Dréxkor"), accents added ("Énterprises"),
component reordering, empty addresses.

**Distractors** that the model merges are near-copies of one S1 with a different operator mix:

| Pattern (India samples) | Example (S1 → distractor) |
|---|---|
| legal form **added** to an S1 that has none, plus one number changed | "Seven Enterprises, Plot-4-B …" → "Seven Enterprises Limited, PLOT-25-B …" |
| one number changed, legal form spelled differently | "Kaveraksh Estate Private Limited, 6-3-1239/2" → "Caveraksh Estate Private Ltd, 6-3-1239/6" (also K→C) |
| an extra leading "No NN" / "House No" number | "Star Engineering … O Dhermainder Jha" → "Shri Star Engineering Limited, No 95 O Dhermainder Jha …" |
| a distinctive name token replaced | "Tech Business Private Limited, No.423 …" → "Prime Business Private Limited, No 423 …" |
| exact name, **empty address** | "Alka Academy, <long address>" → "alka academy", no address (p = 0.98) |

Records owned by *another* S1 that get merged are mostly namesakes ("Noida Solar Private Limited" at two
addresses in the same city) or invented trade names sharing a building ("Mr K0rdovaxylo, #20 Phars Lane").

### 7.3 The information floor (why 0.999+ is information-limited)

Evidence that some errors cannot be removed by any method using these four fields:

1. **Exact-name, empty-address distractors.** A distractor "alka academy" with no address is textually a
   perfect (case-insensitive) copy of S1 "Alka Academy". A true S2 record of the same S1 can look exactly
   the same. The label is not a function of the text → irreducible error on both sides.
2. **Empty-address namesakes.** 4.4 % of true pairs have an empty query address; their K50 recall is 0.88.
   When the name core is shared by n S1 of the country, the best any rule can do is 1/n or abstain.
3. **Invented trade names without a usable address** ("Dréxkor", "Nexavinovi"): no textual link to the S1.
4. **Namesakes with the same name and city**, where the S2/S3 record carries only the shared part.

The exact size of cases 1–2 over the full training split (records whose normalised text is identical but
whose labels differ; the P(correct | exact copy) rate) is computed by `info_ceiling.py`, queued. Until it
lands, the defensible statement is: K50 lexical recall is 0.9877; the 1.23 % beyond K50 is dominated by
empty-address and different-name records; so a perfect *text-based* system is bounded well below 0.9999.

### 7.4 Corruption-channel analysis: P(transform | match) vs P(transform | distractor)

Literature: learned edit channels (Ristad & Yianilos [4]; Bilenko & Mooney [5]) and discriminative edit
models trained on matches **and** non-matches (McCallum et al. [6]). The AR-7 feature group is a compact,
GBDT-friendly version of this: it classifies each residual difference as typo / abbreviation / join /
novel token, weights it by IDF, and types each number difference (one-digit substitution, transposition,
extension, other). Result: +0.0055 M, false merges −24 %, near-copy false merges 315 → 256.

Transform types that AR-7 does **not** see because normalisation erases them, and that the samples suggest
are generator signatures: raw casing (all-lower "alka academy"), double spaces from token deletion
("Private  Limited"), parentheses around legal forms, a leading "No NN" prefix, legal form added to an S1
that has none, first-letter swaps (K→C). `channel.py` measures each one's likelihood ratio overall and —
more importantly — inside the model's uncertain band (p 0.3–0.97), comparing the true-match rate with the
model's mean probability: a large difference means the model is not using that transform yet. Queued.

### 7.5 Decision rule: one global threshold vs entity-aware rules

The metric is F0.5 per S1. With calibrated probabilities, expected F says:

* a single candidate for an S1 (possible singleton) should be accepted only if p > 0.5 — a false merge
  scores 0 and costs the singleton's full point;
* an extra candidate for an S1 that already has confident matches should be accepted from p ≈ 0.27
  (worked example: one sure match + one candidate: accept iff 0.444 p > 0.167 (1 − p)).

A single threshold (0.79–0.80 on V-dense) cannot express both. It also sits well above 0.5, which means
the probabilities at test density are over-confident for the first candidate; calibration has to come
before any expected-F rule. `dec_rule.py` compares, on the same V-dense protocol: G (global threshold,
must reproduce 0.9718), R1 (separate thresholds for an S1's first and additional candidates, tuned on half
A) and R2 (isotonic calibration on half A + per-S1 expected-F0.5 maximisation, Jansche / Ye et al. [19]),
with paired bootstrap CIs and a swapped-halves check. It changes no test features — only post-processing
of stored probabilities — so if it passes it costs seconds on test. Queued.

---

## 8. Web / literature research findings

Each row: what it does · why it matters **for this data** · compliance · expected benefit · cost · complexity ·
false-positive risk · verdict. Sources are listed in full in the bibliography.

| Technique (primary source) | What it does | Why it matters here | Compliance | Benefit (evidence) | Compute | Complexity | FP risk | Test? |
|---|---|---|---|---|---|---|---|---|
| Fellegi–Sunter linkage [1] | log-likelihood-ratio of field agreement patterns, m/u probabilities, 3-way decision | the theoretical frame for "match weight"; our LightGBM already learns a discriminative version with 49 comparison features | SAFE | as the *matcher*: negative (logistic regression on the same features trails LightGBM by 0.043, M proxy) | low | medium | medium | NO-GO as matcher |
| Term-frequency adjustment (Splink [2], fastLink [3]) | u-probability of an exact-value match = its frequency → rare-value agreement counts more | India names are heavily duplicated (median 54 S1 per name); the AR-7 IDF residual features are exactly this idea and passed (section 5) | SAFE | captured by AR-7 (M) | low | low | low | done via AR-7 |
| Learned edit distance (Ristad & Yianilos [4]; Bilenko & Mooney [5]; McCallum et al. CRF edit distance [6]) | learn per-operation edit costs from matched pairs (a "corruption channel") | the data is synthetic with typed corruptions; true pairs and near-copy distractors differ in *which* edits occur (section 7.4) | SAFE | typed-edit features (AR-7) +0.0055 M; a full stochastic transducer is H | medium–high | high | low | features: GO (done); full transducer: DEFER |
| Synthetic ER data generators (GeCo [7]) | generate records and corrupt them with typed operators (typos, OCR, phonetic, missing values, value swaps) | our data shows the same operator families (typos "Wilblims", ordinal "45ND", St→"SAINT", zero padding, "null" tokens, invented DBA names); the operator mix is learnable from labels | SAFE | frames the corruption-channel features | – | – | – | informs design |
| TF-IDF top-K blocking (Sparkly [8]) | per-record top-K by TF-IDF/BM25 | shows that well-tuned TF-IDF top-K is a very strong blocker, often better than deep blockers — consistent with our B-series | SAFE | already our design | – | – | – | keep |
| Deep blocking (DeepBlocker [9], SC-Block [10]) | learned embeddings + ANN top-K | could reach DBA / renamed / script-name misses lexical retrieval cannot | model needs MIT/Apache, ≤ 8B: VERIFY | +0.001–0.006 H (bound: blocking misses +0.0152 E) | CPU embed of 22M records: many hours | high | medium (blurs digits) | DEFER (post-deadline) |
| Blocking surveys (Christen [11]; Papadakis et al. [12]) | taxonomy: multi-pass, sorted neighbourhood, q-gram, meta-blocking | multi-pass union already measured: no recall gain over the address-weighted blend (B5 = B6) | SAFE | ≈0 M | – | – | – | NO-GO |
| Ditto / transformer cross-encoders [13] | fine-tuned BERT-family pair classifier with augmentation | reads characters, scripts, digit edits jointly | VERIFY licence (e.g. MIT/Apache checkpoints) | +0.001–0.005 H, uncertain band only | CPU fine-tune ≈ a day; inference on 1–2M uncertain pairs ≈ 2 h | high | low–medium | DEFER |
| LLM matchers (Peeters et al. [14]) | prompt / fine-tune LLMs for pair matching | strong zero-shot on hard product pairs; our volume is 60M pairs | ≤ 8B + MIT/Apache only (e.g. some Mistral/Qwen variants): VERIFY | H, impractical on CPU | prohibitive | high | – | NO-GO |
| Multilingual encoders (multilingual-e5-small, MIT, 118M [15]) | shared embedding space for 100 languages | possible retrieval view for script names | SAFE if MIT verified | H | ≈ 18 GB float32 for 12M vectors → needs fp16/PQ | high | medium | DEFER |
| Learned Indic transliteration (IndicXlit, MIT, 11M params [16]) | seq2seq romanisation for 21 Indic scripts | our learned dictionary covers 53 % of script tokens; the rest fall back to AnyAscii | model is MIT, but it was trained on external corpora (Aksharantar) → "external data augmentation" is arguable: VERIFY with organisers | H: script-name recall 0.896 → ? | re-normalise + re-retrieve all: 8–10 h | medium | low | DEFER; ask organisers first |
| Bipartite Bayesian linkage (Sadinle [17]) | one-to-one constraint as a prior, Bayes estimate under a chosen loss incl. abstention | the *query* side is one-to-one in our data (0 of 7.64M violations); S1 side is many — we already enforce exactly the valid half | SAFE | done (M1: +0.0005) | – | – | – | keep |
| Expected-F maximisation (Lewis; Jansche; Dembczyński et al. [18]; Ye et al. [19]) | choose the predicted set that maximises *expected* F given calibrated marginals | our metric is F0.5 **per S1**: the right accept/reject rule for a candidate depends on the S1's other candidates; one global threshold cannot express that | SAFE | H; measurement queued (section 7.5) | seconds | low | controlled by A-half tuning | measured below |
| Label-shift estimation (BBSE, Lipton et al. [20]) | estimate test class prior from black-box score histograms | exactly how V-dense derived the 40 % test distractor share; justifies density cloning | SAFE | enables trustworthy validation (V-dense predicted LB within 0.0007, M) | – | – | – | keep |
| AWS Entity Resolution [21] | rule-based waterfall (Levenshtein, cosine, Soundex) + ML matching workflow | production pattern: explicit blocking + explainable rules + ML; confirms our retrieval → scoring → threshold layering | the *service* is prohibited (external ER service) | design reference only | – | – | – | NOT ALLOWED as a tool |
| Hard / corner-case benchmarks (WDC Products [22]) | EM benchmark with corner cases and unseen entities | shows pairwise matchers degrade on near-duplicates and unseen entities: our distractor near-copies and France | SAFE | motivates density-aware validation | – | – | – | informs design |
| Gradient-boosted trees (LightGBM [23]) | histogram GBDT, native missing values | 49–70 heterogeneous similarity features; fast on CPU | MIT: SAFE | current matcher | – | – | – | keep |
| Learning to rank (LambdaMART [24]) | listwise ranking loss | only fixes ordering errors; "lost to another S1" is worth ≤ +0.0011 E | SAFE | ≤ +0.001 H | low | low | low | DEFER |

---

## 9. Challenge-compliant approaches worth pursuing

VALUE: HIGH / MEDIUM / LOW. STATUS: GO / TEST / DEFER / NO-GO. Gains are V-dense half-B F0.5.

| Approach | Problem addressed | Evidence from our errors | Source | Expected gain | Recall | Precision | Test runtime | RAM | Complexity | Compliance | Confidence | Value / Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR-7 difference-aware features | near-copy false merges, below-threshold misses | 87 % of FP are near-copy distractors | [4][5][6][2] | **+0.0055 M** (seed 7: +0.0053) | + | + | AR-7 on 100M test pairs ≈ 1 h | 3 GB | medium (code exists) | SAFE | high | HIGH / **GO** |
| Gated top-K (ranks 3–9 if best top-3 p < 0.5) | top-3 cut | 1,100 true pairs at ranks 3–9 | – | **+0.0035 M** on top of AR-7 | + | ≈ | +≈30M pairs of 49 features ≈ 1–2 h | 4 GB | low (code exists) | SAFE; candidate set 2× (D1) | high | HIGH / **GO** (or D-lite) |
| Hold V-dense queries out of final training; tune threshold on the shipped model | unmeasured shipped model | final_v1 trained on queries overlapping V-dense | – | protects ± 0.001–0.003 | – | – | none | – | low | SAFE | high | HIGH / **GO** |
| Entity-aware decision rule (R1 / R2) | global threshold vs per-S1 metric | singletons cost 1.0 per false merge; threshold 0.79 ≫ 0.5 | [18][19] | must be measured (probe queued); plausible +0.000–0.002 | + | ± | seconds | – | low | SAFE | medium | MEDIUM / **TEST** |
| Seed / bagging ensemble (average 2–3 GBDTs) | variance | seeds 42 vs 7 differ by up to 162 FP at equal F | – | H +0.000–0.001 | ± | + | ×2–3 predict only (features shared) | – | low | SAFE | medium | MEDIUM / **TEST** |
| More training data (India 10 → 30 %, US 5 → 15 %) | variance, rare corruptions | untested learning curve | – | H +0.000–0.002 | + | + | none at test; train features 1–2 h | 8 GB | low | SAFE | low | MEDIUM / TEST (learning curve first) |
| Raw-format features (casing, double spaces, parentheses, "No NN" prefix) | generator signatures erased by normalisation | samples (7.2, 7.4) | [6][7] | must be measured (channel probe) | ± | + | cheap, same pass | – | low | SAFE | low | MEDIUM / TEST |
| Transliteration dictionary from all training folds (production) | 47 % script tokens fall back to AnyAscii | script-name recall 0.949 at K10 | – | H small; not measurable without leakage | + | ≈ | re-normalise + re-retrieve test 1.5–2 h + rescore | 4 GB | medium | SAFE | low | LOW / DEFER |
| K50 gated retrieval + model trained on ranks 11–50 | blocking misses | 45 % of India top-10 misses sit at ranks 11–50 | [8][11] | E +0.004–0.006; realised H +0.002–0.003 | + | − (more distractors) | K50 retrieval 1.5 h + train 3–4 h + rescore 3 h | 9 GB | medium | SAFE | medium | HIGH / **DEFER** (time) |
| Learned rewrite normalisation (Latin + script, from aligned training pairs) | abbreviations, legal-form spellings, landmark words | dictionary precedent: name sim 71.9 → 85.3 | [5] | H +0.001–0.004 | + | ± | re-normalise + re-retrieve everything 8–10 h | 6 GB | medium–high | SAFE | low | MEDIUM / DEFER |
| Cross-encoder on the uncertain band only | residual hard pairs | uncertain band ≈ 5–8 % of queries | [13] | H +0.001–0.005 | + | + | CPU fine-tune ≈ a day | 4–8 GB | high | licence VERIFY | low | MEDIUM / DEFER |
| Learned bi-encoder retrieval unioned with TF-IDF | DBA / script / empty-address misses | 1.23 % beyond K50 | [9][10][15] | H +0.001–0.006 | + | − | embed 22M records: many CPU hours | 18 GB fp32 | high | licence VERIFY | low | MEDIUM / DEFER |

## 10. Approaches to reject (or that look impressive but are low-value here)

| Approach | Why rejected (quantitative where possible) | Value / Status |
|---|---|---|
| Fellegi–Sunter / EM as the matcher | conditional-independence scorer; a supervised linear model on richer features already trails LightGBM by 0.043 (0.925 vs 0.968, M); its useful part (TF adjustment) is inside AR-7 | LOW / NO-GO |
| Replacing LightGBM (XGBoost, CatBoost, deep tabular) | model class is not the bottleneck; its inputs were (AR-7 +0.0055 with the same model) | LOW / NO-GO |
| Giant LLM matcher | 60M test pairs on a CPU; ≤ 8B + MIT/Apache only; no evidence it separates generator near-copies better than typed-edit features | LOW / NO-GO |
| Graph clustering / correlation clustering / GNN | S1s are fixed anchors and queries are partitioned (0 of 7.64M queries have two owners), so clustering collapses to per-query assignment; sibling evidence does not separate false merges (4 % vs 4 %, M) | LOW / NO-GO |
| Bipartite one-to-one on both sides | S1s take 3.46 records on average; destroys recall | NO-GO |
| Segment / country / source-specific thresholds or models | segment thresholds −0.0001 to 0.0000 (M); France has no labels; S2 vs S3 precision equal (0.9922 vs 0.9926) | LOW / NO-GO |
| Multi-pass blocking union | B5 union = single blend recall (0.9774) at 1.4× cost (M) | LOW / NO-GO |
| Adding many more generic fuzzy metrics | the 49 features already include 14 rapidfuzz scores; the win came from difference *types*, not more similarity scores | LOW / NO-GO |
| Self-training / pseudo-labelling on test | labels on test would come from our own errors; near-copy distractors would be pseudo-labelled as matches | LOW / NO-GO |
| Probability re-calibration alone | a monotone recalibration only moves the global threshold (+0.0001, M) — it matters only inside an entity-aware rule | LOW / NO-GO alone |
| External enrichment (geocoding, registries, business APIs, scraped data, libpostal/OSM gazetteers) | prohibited or grey under the rules | NOT ALLOWED |

---

## 11. Top 20 experiments

Common protocol for all: V-dense (India fold-1 bucket < 100 entities; half A tunes, half B reports; test
density via distractor cloning seed 0 rate 0.94; exact metric asserted against `src/metrics.py`).
**Gate:** half-B gain ≥ +0.0010, paired bootstrap 95 % CI > 0, half A moves the same way, US spot-check does
not regress by more than 0.0005. Status: ✅ passed · ⏳ running/queued · ○ not started.

| # | Hypothesis | Baseline → change | Metrics | Success threshold | Runtime / RAM | Expected failure mode | Decision rule | Status |
|---|---|---|---|---|---|---|---|---|
| E1 | ranks 3–9 contain recoverable owners | A → gated top-K | F0.5, FN, pairs | gate | cached, 3 min / 8 GB | more distractor FPs | ship if gate passes | ✅ +0.0033 |
| E2 | typed residual differences separate near-copies | A → +AR-7 (k3) | F0.5, FP, near-copy FP | gate + FP falls | train 7 min | overfits India generator | ship if gate + US flat | ✅ +0.0055 (CI ⏳ exp3b) |
| E3 | E1 and E2 are complementary | A → D | as above | gain > max(E1, E2) | – | overlap | ship D | ✅ +0.0090 (CI ⏳) |
| E4 | gains replicate across seeds | seed 42 → seed 7 | F0.5 | same sign, within 0.001 | 7 min | seed luck | trust E2/E3 | ✅ 0.9684 / 0.9717 |
| E5 | smaller gate keeps most of the gain | D → gate to rank 4 | F0.5 vs pairs | report curve | 3 min | – | D-lite if D1 says "small" | ✅ 0.9700 at 1.29× |
| E6 | AR-7 holds on US | US sample A → D | US F0.5 | no regression | 10 min | India-specific signatures | block ship if US drops | ⏳ exp3b |
| E7 | halves are exchangeable | tune B, report A | swapped F0.5 | same sign | 3 min | half-specific threshold | – | ⏳ exp3b |
| E8 | shipped final_v2 matches M7 | final_v2 trained **without** V-dense queries, evaluated on V-dense | F0.5, threshold | ≥ 0.9718 − 0.001 | train 25 min, eval 15 min / 8 GB | threshold shifts with 4× data | re-tune threshold on half A of final_v2 itself | ○ |
| E9 | entity-aware first/extra thresholds beat one threshold | G → R1 | F0.5, singleton acc | gate | 5 min / 3 GB | overfit on A | ship as post-processing | ⏳ dec_rule |
| E10 | calibrated expected-F0.5 per S1 beats thresholds | G → R2 | F0.5 | gate | 10 min | calibration shift at test density | ship if > R1 | ⏳ dec_rule |
| E11 | error diversity across seeds | M7 s42 vs s7 disagreement; average | F0.5, disagreement rate | gate | 10 min (both already trained) | seeds make the same mistakes | ship 2-model average | ○ |
| E12 | learning curve | M7 on 50 % vs 100 % of training pairs | F0.5 | 100 % beats 50 % by ≥ 0.001 | 20 min | flat curve | only then scale data | ○ |
| E13 | raw-format transforms are unused signal | channel probe → add 4–6 raw features | uncertain-band LR, then F0.5 | LR ≥ 2 inside the band, then gate | probe 10 min; retrain 10 min | the generator applies them symmetrically | add to final_v3 only if gate | ⏳ channel |
| E14 | information floor | exact-text label conflicts over training | irreducible pairs, P(correct│exact copy) | report | 10 min / 6 GB | – | sets the ceiling statement | ⏳ info_ceiling |
| E15 | D's error budget | per-class counterfactual | F0.5 per class | report | 5 min | – | chooses next target | ⏳ budget |
| E16 | deeper gated retrieval (K50) | D → K50 gate + retrained model | recall, F0.5, pairs | gate at ≤ 3× pairs | 8–10 h / 9 GB | distractors at ranks 11–50 | post-challenge | ○ DEFER |
| E17 | learned rewrite normalisation | D → rewrites learned on fold 0 | recall@10, F0.5 | gate | 8–10 h | over-merging rewrites | post-challenge | ○ DEFER |
| E18 | all-fold translit dictionary | fold-0 dict → all folds | test-only | cannot be validated without leakage | 2 h | – | post-challenge | ○ DEFER |
| E19 | cross-encoder on uncertain band | D → D + reranker on p ∈ [0.05, 0.95] | F0.5 | gate | ~1 day CPU | slow; calibration of the blend | post-challenge (GPU) | ○ DEFER |
| E20 | bi-encoder retrieval view | TF-IDF → TF-IDF ∪ ANN | recall@10 at equal pairs | +0.5 pp recall | many hours | digit blurring → FPs | post-challenge (GPU) | ○ DEFER |

**Top 5 highest-value experiments:** E3 (done), E8, E9/E10, E11, E13.
**Top 5 highest-value architectural changes:** AR-7 difference features (done), confidence-gated
candidate expansion (done), measured final model (E8), K50 gated retrieval (E16), learned rewrite
normalisation (E17).
**Top 5 that look impressive but are probably low-value here:** LLM matcher, graph/GNN clustering,
Fellegi–Sunter EM matcher, swapping GBDT for a deep tabular model, self-training on test.

---

## 12. Prioritised roadmap (phases 0–8, mapped to the deadline)

| Phase | Content | Before 27 Sep 23:59? | Exit criterion |
|---|---|---|---|
| 0 Reproducibility + baseline | fix README:45; commit Exp-3 code; update BEST_KNOWN / SUBMISSION_LOG; record V1 hash (done by the other session: `output_v1_sha256.txt`) | **yes, 30 min** | a clean checkout regenerates V1 |
| 1 Retrieval ceiling | K-sweep and segment recall (done); gated top-K (done) | done | – |
| 2 Error-specific features | AR-7 (done); raw-format transforms (E13) | AR-7 yes; E13 only if the probe shows LR ≥ 2 | gate |
| 3 Hard-negative / density-aware training | V-dense (done); train final_v2 without V-dense queries (E8) | **yes, ≈ 1 h** | E8 |
| 4 Probabilistic / hybrid | TF adjustment via AR-7 (done); calibrated expected-F decision (E10) | yes if the probe passes | gate |
| 5 Specialist reranking | gated top-K (done); listwise / LambdaRank | no (≤ +0.0011 E) | – |
| 6 Optional deep model | cross-encoder on uncertain band; bi-encoder retrieval | **no** (no GPU) | post-challenge |
| 7 Ensemble | seed average (E11) | yes if gate passes | gate |
| 8 Final calibration + output | threshold on final_v2's own half A; validator; package; documentation | **yes** | validator PASS, V-dense of the shipped model reported |

---

## 13. Proposed final architecture (derived from the evidence)

```
RAW TSV (4 fields)
  │  prep: mojibake fix · AnyAscii + learned script→English dictionary · legal-form, honorific, stop-word maps
  │        · DBA split · address abbreviations · state → code            [unchanged: src/prep.py, normalize.py]
  ▼
PER-COUNTRY TF-IDF RETRIEVAL, S2/S3 → S1, top-10                         [unchanged: blend_ad15, cached]
  │  views: name-core words 1.0 · name char-3-grams 1.0 · address tokens 1.5
  ▼
PAIRWISE EVIDENCE, stage 1 (ranks 0–2 of every query)
  │  49 similarity / number / legal / retrieval features
  │  + 21 AR-7 difference features: residual-token type and IDF mass, typed number edits,
  │    S1 address sharing, rival contrast over the query's full top-10 list
  ▼
LightGBM final_v2 (MIT) → p for ranks 0–2
  ▼
CONFIDENCE GATE: best top-3 p < 0.5 → also score ranks 3–9 (same features)       [D; D-lite: ranks 3–4]
  ▼
ONE-TO-ONE ON THE QUERY SIDE (each S2/S3 record → its best S1 only)
  ▼
DECISION: global threshold tuned on V-dense half A of the *shipped* model
          (entity-aware rule R1/R2 only if E9/E10 pass)
  ▼
OUTPUT + HARD CHECKS: matches ⊂ scored candidates; one row per S1; validator
```

What was deliberately **not** added: country features, S1-side capacity limits, clustering, a second
retrieval pass, neural components. Each was measured as ≈ 0 or ruled out by the data's structure.

---

## 14. Detailed implementation plan

The production path already exists in the other session's uncommitted scripts; the plan below reuses them
and adds only what the evidence requires.

### 14.1 Phase 0 — reproducibility (≈ 30 min)

| File | Change | Test | Rollback |
|---|---|---|---|
| `README.md:45` | `python -m src.pipeline predict final_v1 3 0.73`; add the final_v2 commands once shipped | run the README commands on the France smoke subset; compare hashes with `output_v1_sha256.txt` | git revert |
| git | commit `experiments/{vdense,exp3,exp3b,exp3_ar7,ar7_features,build_eval_ext,predict_v2,predict_v3,train_final_v2}.py` | `python -m py_compile` on each | – |
| `experiments/BEST_KNOWN.md`, `submissions/SUBMISSION_LOG.md` | add the V1 upload (0.9621, commit, threshold, top-3) and the Exp-3 table | – | – |

### 14.2 Phase 3 — a measured final model (E8, ≈ 1 h)

* **File:** `experiments/train_final_v2.py` → promote to `src/train_final.py` (keep one entry point).
* **Change:** before training, drop every training query that V-dense uses:
  `exclude = set(pd.read_parquet(CACHE_V/"ev_pairs.parquet").q_id) | set(pd.read_parquet(EXP/"ev_ext_pairs.parquet").q_id)`,
  `pairs = pairs[~pairs.q_id.isin(exclude)]` (and the aligned rows of `X` and the AR-7 frame).
  This removes ≈ 5 % of India training queries.
* **Then:** score the V-dense set with the new model (`exp3b.probs` pattern), run `vdense.evaluate` for
  policy `("adapt", 0.5, 10)`, tune the threshold on half A, and freeze that threshold and the policy into
  `m2_model_final_v2.meta.json`.
* **Accept** if half-B F0.5 ≥ 0.9708 (M7 − 0.001). **Rollback:** keep M7's threshold 0.80 and report the gap.
* **Cost:** AR-7 on 7.24M pairs ≈ 4 min (cached after the first run), LightGBM ≈ 20–30 min, evaluation 15 min.

### 14.3 Test inference (`experiments/predict_v3.py`, ≈ 3–4 h detached)

* **Inputs:** `cand_test_blend_ad15/*.parquet` (full top-10 lists), `test_s{1,2,3}.parquet`, final_v2 model + meta.
* **Per chunk:** AR-7 on all 10 ranks with **test** S1 statistics (`build_stats(test_s1)`, pickled once) →
  49 features for ranks 0–2 (`ret_ncand` from the full list) → predict → gate → features for ranks 3–9 of
  gated queries (`predict_v2.extra_features`, full-list retrieval features) → predict → atomic checkpoint
  `pred_final_v2_adapt0.5_k10/<chunk>.npz`.
* **Parity test (must pass before the full run):** on one *training* chunk, compute features with the
  predict_v3 code path and compare them with the cached V-dense features for the same (query, S1) pairs:
  max |Δ| < 1e-4 for all 70 columns. It guards against the AR-7 rival-contrast features being computed
  on a truncated candidate list (the scored ranks only) instead of the full top-10, as in validation.
* **Outputs:** a NEW directory (`C:\aml2026\output_v2`), both TSVs from exactly the scored pairs;
  `prediction_stats.json` with gated share, pairs per S1, matches by country.
* **Sanity gates:** matches per matched S1 within 3.3–3.6 per country (V1: India 3.23, US 3.33, France
  3.29); France share of matched S1 not lower than V1's; validator PASS with `--check-ids`.
* **Cache strategy:** checkpoints are resumable; the V1 checkpoints and output stay untouched.

### 14.4 Post-processing experiments (only after 14.3 is running or done)

| Experiment | New module | Inputs → outputs | Algorithm | Test cost | Rollback |
|---|---|---|---|---|---|
| E9/E10 decision rule | `src/decision.py` with `assign_entity_aware(sel, t1, t2)` and `assign_expected_f(sel, calibrator, lam)` | per-query best (S1, p) → accepted pairs | R1: first candidate ≥ t1, extra candidates ≥ t2; R2: isotonic calibration fitted on V-dense half A (pickled, from training data only), per-S1 top-k maximising E[F0.5] via Poisson-binomial DP | seconds on stored probabilities | global threshold |
| E11 seed average | `src/matching.py` accept a list of boosters, average p | same features | mean of 2 models | +1 predict per chunk (≈ 10 % of runtime) | single model |
| E13 raw-format features | extend `src/features.py` with `raw_format_features(pairs, s1, q)` | raw name/address strings | casing class, double-space count, parentheses, leading-number prefix, legal-form-added flag | same pass, ≈ +5 % | drop the columns |

---

## 15. Hardware / compute plan

**No GPU is available** (Intel integrated graphics, 2 GB shared). All plans are CPU-first.

| Job | Size | Measured / estimated time | Peak RAM |
|---|---|---|---|
| Test retrieval top-10 (existing, reusable) | 9.97M queries | ≈ 70 min (M) — **not needed again** | ≈ 4 GB |
| AR-7 features | 32 K pairs/s on 8 workers (M, Exp-3) | 100M test pairs (all 10 ranks, needed for rival contrast) ≈ 55 min | ≈ 3 GB |
| 49 base features | 4.6–8 K pairs/s (M) | top-3 (29.9M) ≈ 1.0–1.8 h; gated ranks 3–9 (≈ 25–30M) ≈ 1.0–1.8 h | ≈ 4 GB |
| LightGBM training | 7.24M × 70 features | ≈ 20–30 min | ≈ 6 GB |
| V-dense evaluation | 21M densified rows | ≈ 2.5 min per system (M) | ≈ 8 GB |

A GPU would matter only for the deferred neural items (cross-encoder on the uncertain band, bi-encoder
retrieval). If one becomes available post-challenge: a single 16–24 GB card (T4/L4/A10 class) fine-tunes a
~110M-parameter MIT/Apache encoder on ~1M pairs in ~1–2 h and scores 2M uncertain pairs in minutes
(batch 256, fp16, < 6 GB VRAM). The AWS free-tier SageMaker hours in the guidelines PDF (m5.xlarge = CPU)
do not include a GPU.

Operational rules (from the 25 Sep failures): one heavy job at a time, detached via `Start-Process`,
High priority, AC power, resumable atomic chunks.

---

## 16. Licence / compliance audit

| Component | Role | Licence | Size | In final pipeline? | Status |
|---|---|---|---|---|---|
| LightGBM 4.7.0 | final matcher | MIT | – | yes | LEGAL / SAFE |
| scikit-learn 1.9.1 | TF-IDF vectoriser, isotonic calibration | BSD-3-Clause | – | yes (not "the model") | SAFE; the rule names only the *model* licence — BSD is permissive; note it in the documentation |
| rapidfuzz 3.14.6 | string similarities | MIT | – | yes | SAFE |
| sparse-dot-topn 1.2.0 | top-K sparse matmul | Apache-2.0 | – | yes | SAFE |
| anyascii 0.3.3 | romanisation | ISC | – | yes | SAFE (permissive, MIT-equivalent); state in documentation |
| numpy / scipy / pandas / pyarrow / psutil | infrastructure | BSD / Apache-2.0 | – | yes | SAFE |
| Learned translit dictionary, AR-7 IDF tables | derived from supplied data | ours | 3 MB | yes | SAFE (training/test files only) |
| multilingual-e5-small | deferred retrieval view | MIT (model card) | 118M | no | VERIFY LICENSE file before any use |
| IndicXlit | deferred transliteration | MIT (code/models) | 11M | no | VERIFY (trained on external corpus → possible "external data augmentation") |
| libpostal | address parsing | MIT, but trained on OpenStreetMap | – | no | avoid (external reference data) |
| Any geocoder / registry / ER service | – | – | – | never | NOT ALLOWED |

---

## 17. Validation strategy

**Why a random split is not enough.** The test has 1.94× the training distractor density (S2+S3 per S1:
India 5.82 vs 4.68; US 5.76 vs 4.67), and 87 % of false merges are distractors — so a random held-out split
overstates precision (training-density 0.9650 vs test-density 0.9628 for the same system).

**V-dense (in use, keep):**
1. Entities: India fold-1 S1 with hash bucket < 100 (fold 1 never trained the transliteration dictionary),
   halves A/B by an independent hash bit.
2. Candidates: retrieved against the **full** 2.21M-record training S1 index, so namesake competition is real.
3. Density: every distractor query is cloned with probability 0.94 under a fresh id → 2.35 distractors per S1.
   Justified by label-shift estimation [20]: the test's best-probability histogram equals a mixture of the
   validation distractor and owned histograms at a 39.7 % distractor share.
4. Metric: exact re-implementation of the leaderboard macro F0.5, asserted against the reference code.
5. Decisions: threshold on A, report on B; paired bootstrap over entities; swapped-halves sign check; US
   spot-check on a US sample the model did not train on.
6. Calibration against reality: V-dense predicted the public LB within 0.0007.

**Gaps and fixes:** the shipped model must not train on V-dense queries (E8); the US check must use a US
sample outside the final model's training set (for final_v2: a fresh US hash bucket); France has no labels —
monitor prediction statistics (matches per S1, matched share) against V1 and against India/US.

**Stress tests worth adding post-challenge:** density ×1.5 and ×3 (clone rates 0.5 and 2.0) to check that
the threshold and ranking are stable; an orphan simulation (remove 20 % of S1 from the index so their
records become realistic distractors) to test behaviour on entities whose S1 is absent.

---

## 18. Expected score range

| Submission | V-dense (India) | Expected public / private LB | Basis |
|---|---|---|---|
| V1 (submitted) | 0.9628 | 0.9621 (observed public) | M |
| D-lite (AR-7, gate to rank 4) | 0.9700 | ≈ 0.969 | M − 0.0007 offset |
| **D (AR-7 + gate to rank 10)** | **0.9718** | **≈ 0.970–0.972** | M − offset ± spread |
| D + any passing post-processing | 0.972–0.974 | ≈ 0.971–0.973 | H |
| Post-challenge: + K50 gated retrieval | 0.974–0.977 | – | E/H |
| Post-challenge: + learned retrieval and a reranker | 0.977–0.985 | – | H |

The LB mixes India (47 % of test S1), US (38 %, easier: the US spot-check scores the submitted feature set at 0.984) and France (15 %, unvalidated).
The observed public 0.9621 ≈ India V-dense 0.9628, so France appears to cost about what the US gains; this
can shift on the private split.

---

## 19. What 0.99 / 0.995 / 0.999 / 0.9999 / 0.99999 would require

| Target | Verdict | What it would take | Why |
|---|---|---|---|
| **0.99** | **UNSUPPORTED BY CURRENT EVIDENCE** (for this deadline and hardware); difficult but conceivable long-term | retrieval recall ≈ 0.995 (today 0.9775 at K10, 0.9877 at K50) **and** matcher error inside the candidates cut from 0.0147 to ≈ 0.004 | India oracle at top-10 is only 0.9865 — even a perfect matcher on today's candidates misses 0.99; K50 raises the mixed-sample oracle to 0.9956, so 0.99 needs *both* deep retrieval and a near-perfect matcher |
| **0.995** | **DIFFICULT; bounded by the K50 oracle** | perfect matching on K50 candidates (oracle 0.9956 on the mixed sample; India lower) plus learned retrieval for DBA / script / empty-address records | lexical retrieval at K50 still misses 1.23 % of true pairs |
| **0.999** | **THEORETICALLY POSSIBLE BUT INFORMATION-LIMITED** | resolving empty-address namesakes and exact-copy distractors | a distractor can be textually identical to a true record (7.3); no field separates them |
| **0.9999** | **INFORMATION-LIMITED** | fewer than ≈ 2 wrong entities per 22 k S1 | the floor cases in 7.3 alone exceed this (exact size: `info_ceiling.py`) |
| **0.99999** | **INFORMATION-LIMITED** | fewer than 1 wrong entity per 100 k S1 | same; also each singleton false merge costs 1/N of the score |

Could hidden structure beat this? We looked for it: file order and id numbers carry no signal (position
spread 2.04M vs 1.68M random; id correlation −0.003, M) and would be illegitimate anyway; sibling records
of the same S1 do not separate false merges (4 % vs 4 %, M). The one structural rule the data does obey —
each S2/S3 record has at most one owner — is already enforced. The exploitable structure is the
*corruption generator*, and AR-7 is the first step at modelling it; raw-format transforms (E13) are the next.

---

## 20. Risks and failure modes

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Shipped final_v2 behaves differently from the validation model (4× data, threshold transferred) | medium | ± 0.001–0.003 unseen | E8: hold V-dense queries out, tune the threshold on the shipped model |
| AR-7 feature parity bug at test time (rival contrast on truncated lists; test vs train IDF tables) | medium | silent loss | parity test on a training chunk before the full run (14.3) |
| Candidate-set size judged against us (2× pairs) | unknown | unknown | decision D1: D (2.01×, +0.0035) vs D-lite (1.29×, +0.0017) |
| India-specific signatures hurt US / France | low–medium | −0.00x on private | US spot-check (exp3b); France prediction statistics vs V1 |
| Laptop: battery / throttling / two heavy jobs at once | **high right now** (on battery; exp3b running) | hours lost | AC power; one heavy job at a time; detached, resumable chunks |
| Running out of time before packaging | medium | invalid final zip | freeze 27 Sep 18:00; V1 files kept as fallback; validator on every candidate output |
| Chasing the public LB | real | private shake-up | select on V-dense only; LB is a sanity check (expected LB ≈ V-dense − 0.001) |
| Over-reading hypotheses in this document | – | wasted time | only M-labelled gains ship |

---

## 21. Exact next actions

Status when this report was written (26 Sep, ≈ 16:00 IST): `experiments.exp3b` (other session) is
producing the CIs, swapped halves and US check for A–D; the four research probes (`channel.py`,
`dec_rule.py`, `budget.py`, `info_ceiling.py`, in this session's scratchpad) are queued behind it and fill
sections 5.2, 7.3, 7.4 and 7.5 when they finish.

1. **Plug in the laptop** before any further heavy job.
2. Let exp3b finish; accept D only if its CI excludes 0, the swapped halves agree and US does not regress.
3. **Decide D1** (candidate-set size): D (2×) or D-lite (1.29×).
4. Phase 0 fixes (README:45, commits, logs) — 30 min.
5. E8: retrain final_v2 with the V-dense queries excluded; tune its threshold on half A; record its half-B score.
6. Parity test, then `predict_v3` detached into `output_v2`; validator; compare statistics with V1; upload once.
7. While it runs: read the probe results; run E9–E11 on stored probabilities; add any passing rule as
   post-processing (seconds on test) and re-upload only if it passes the gate.
8. Freeze on 27 Sep 18:00 IST: package (`utils/make_package.py`), documentation (methodology, AR-7, gate,
   V-dense), validator, final upload.

---

## Bibliography

| # | Title | Authors / organisation | Year | URL | Takeaway for this project |
|---|---|---|---|---|---|
| 1 | A Theory for Record Linkage | I. P. Fellegi, A. B. Sunter (JASA 64(328)) | 1969 | https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501049 | match weight = likelihood ratio of agreement patterns; 3-way decision (link / possible / non-link) |
| 2 | Splink: term-frequency adjustments | UK Ministry of Justice | 2024–26 | https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html | u-probability of an exact match = the value's frequency → rare agreements weigh more |
| 3 | Using a Probabilistic Model to Assist Merging of Large-scale Administrative Records (fastLink) | T. Enamorado, B. Fifield, K. Imai (APSR) | 2019 | https://imai.fas.harvard.edu/research/files/linkage.pdf | scalable Fellegi–Sunter with missing data and TF adjustment |
| 4 | Learning String-Edit Distance | E. S. Ristad, P. N. Yianilos (IEEE TPAMI 20) | 1998 | https://www.researchgate.net/publication/3192848_Learning_String_Edit_Distance | edit-operation probabilities learned by EM from matched pairs = a corruption channel |
| 5 | Adaptive Duplicate Detection Using Learnable String Similarity Measures | M. Bilenko, R. J. Mooney (KDD) | 2003 | https://www.cs.utexas.edu/~ml/papers/marlin-kdd-03.pdf | per-field learned edit distance beats fixed metrics for record linkage |
| 6 | A Conditional Random Field for Discriminatively-trained Finite-state String Edit Distance | A. McCallum, K. Bellare, F. Pereira (UAI) | 2005 | https://arxiv.org/pdf/1207.1406 | discriminative edit model trained on matches **and** non-matches — the LR(transform) idea |
| 7 | Flexible and Extensible Generation and Corruption of Personal Data (GeCo) | P. Christen, D. Vatsalan (CIKM) | 2013 | https://dmm.anu.edu.au/geco/ | typed corruption operators used to build ER benchmarks; our data shows the same families |
| 8 | Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching | D. Paulsen, Y. Govind, A. Doan (PVLDB 16) | 2023 | https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf | TF-IDF top-K blocking is competitive with or better than deep blockers |
| 9 | Deep Learning for Blocking in Entity Matching: A Design Space Exploration | S. Thirumuruganathan et al. (PVLDB 14) | 2021 | https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf | learned blocking helps most on dirty / textual data; costly |
| 10 | SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines | A. Brinkmann, R. Shraga, C. Bizer (ESWC) | 2024 | https://arxiv.org/pdf/2303.03132 | supervised contrastive embeddings + ANN shrink candidate sets at equal recall |
| 11 | A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication | P. Christen (IEEE TKDE 24(9)) | 2012 | https://dl.acm.org/doi/10.1109/TKDE.2011.127 | pairs completeness vs reduction ratio; multi-pass indexing |
| 12 | Blocking and Filtering Techniques for Entity Resolution: A Survey | G. Papadakis, D. Skoutas, E. Thanos, T. Palpanas (ACM CSUR 53(2)) | 2020 | https://arxiv.org/pdf/1905.06167 | taxonomy of blocking / meta-blocking / filtering |
| 13 | Deep Entity Matching with Pre-Trained Language Models (Ditto) | Y. Li, J. Li, Y. Suhara, A. Doan, W.-C. Tan (PVLDB 14) | 2020 | https://arxiv.org/abs/2004.00584 | transformer pair classifier with domain knowledge injection and augmentation |
| 14 | Entity Matching using Large Language Models | R. Peeters, A. Steiner, C. Bizer (EDBT) | 2025 | https://arxiv.org/pdf/2310.11244 | LLMs are strong but expensive; small fine-tuned models competitive |
| 15 | multilingual-e5-small model card | intfloat / Microsoft | 2023 | https://huggingface.co/intfloat/multilingual-e5-small | MIT, 118M parameters, 384-d — the only encoder size that is plausible on this CPU |
| 16 | Aksharantar / IndicXlit | Madhani et al., AI4Bharat (Findings of EMNLP) | 2023 | https://github.com/AI4Bharat/IndicXlit | 11M-param MIT transliteration for 21 Indic languages; trained on external corpus |
| 17 | Bayesian Estimation of Bipartite Matchings for Record Linkage | M. Sadinle (JASA 112(518)) | 2017 | https://www.tandfonline.com/doi/abs/10.1080/01621459.2016.1148612 | matching constraints + loss-based decisions with abstention |
| 18 | An Exact Algorithm for F-Measure Maximization | K. Dembczyński, W. Waegeman, W. Cheng, E. Hüllermeier (NeurIPS) | 2011 | https://papers.nips.cc/paper/4389-an-exact-algorithm-for-f-measure-maximization | the expected-F-optimal prediction set is not a fixed threshold |
| 19 | Optimizing F-measures: A Tale of Two Approaches | N. Ye, K. M. A. Chai, W. S. Lee, H. L. Chieu (ICML) | 2012 | https://arxiv.org/pdf/1206.4625 | decision-theoretic (calibrated probability + expected-F) vs empirical threshold tuning |
| 20 | Detecting and Correcting for Label Shift with Black Box Predictors | Z. Lipton, Y.-X. Wang, A. Smola (ICML) | 2018 | https://arxiv.org/abs/1802.03916 | estimate the test prior from score histograms — basis of V-dense |
| 21 | AWS Entity Resolution user guide | Amazon Web Services | 2026 | https://docs.aws.amazon.com/entityresolution/latest/userguide/what-is-service.html | rule-based waterfall + ML matching workflows; design reference only (service use prohibited) |
| 22 | WDC Products: A Multi-Dimensional Entity Matching Benchmark | R. Peeters, R. C. Der, C. Bizer (EDBT) | 2024 | https://arxiv.org/html/2301.09521 | matchers degrade on corner cases and unseen entities |
| 23 | LightGBM: A Highly Efficient Gradient Boosting Decision Tree | G. Ke et al. (NeurIPS) | 2017 | https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree | the matcher; MIT licence |
| 24 | From RankNet to LambdaRank to LambdaMART: An Overview | C. J. C. Burges (MSR-TR-2010-82) | 2010 | https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/ | listwise ranking objective (deferred) |
