# Experiment log

Never edit past rows; append new ones. Raw machine-written results live next to this file
(`blocking_results.jsonl`, …).

| ID | Date | Blocking | Features | Model | Threshold | Val F0.5 | Cand. recall | Runtime | Notes |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 2026-09-25 | blend nw1/nc1/ad0.8 K=10, **20 % index (optimistic)** | – | – | – | – | 0.9703 (oracle F0.5 0.989) | 459 s | S1 index shrunk to 20 % → 5× fewer confusers than test; superseded by B-series v2 |
| B1 | 2026-09-25 | same, K=20, 20 % index | – | – | – | – | 0.9780 | 1091 s | recall@1..5 identical to B0 |
| B2 | 2026-09-25 | blend10, **full S1 index**, 5 % entity sample | – | – | – | – | **0.9510** (oracle 0.982) | 619 s, peak 6.2 GB | India 0.927, US 0.967; addr-empty 0.755; script names 0.896; non-exact core 0.900; r@1 0.914 |
| T1 | 2026-09-25 | translit audit, fold-1 held out (360K script pairs) | – | – | – | – | – | – | dict acc 98.5 %, coverage 53 %; mean name sim A 24.7 / B 71.9 / C 62.8 / **D 85.3** → keep D (dict+anyascii) |
| B3 | 2026-09-25 | blend + address bigram view (ab), K=10, full index | – | – | – | – | 0.9531 (oracle 0.983) | 2311 s (US paging) | +0.21 pp for 3.7× cost → rejected for main blend |
| B4 | 2026-09-25 | blend10, max_df 0.005 all views (speed variant) | – | – | – | – | 0.9252 (oracle 0.970) | 206 s | 3× faster but −2.6 pp (India −4.5 pp) → **rejected** (common tokens carry signal) |
| D1 | 2026-09-25 | miss analysis of B2 (18,743 misses) | – | – | – | – | – | – | ~60 % of misses have addr token-set sim 100 but lose slots to S1 namesakes (India median 54 S1 per name); 18.5 % DBA/different name; 17 % empty addr + generic name (unrecoverable) |
| B5 | 2026-09-25 | blend10 ∪ addr-pass(ad1.0,nc0.5,K5) ∪ exact-name | – | – | – | – | 0.9774 (oracle 0.992) | 1179 s, 14 cand/q | India 0.962, US 0.988, script 0.947 — address-dominant evidence recovers namesake losses |
| B6 | 2026-09-25 | **single blend nw1/nc1/ad1.5, K=10** | – | – | – | – | **0.9774 (oracle 0.992)** | **575 s, 10 cand/q** | same recall as B5 at half cost; r@1 0.952 (vs 0.914 B2) → current choice |
| B7 | 2026-09-25 | blend nw1/nc1/ad2.0, K=15 | – | – | – | – | 0.9759 (r@10 0.9722) | 778 s | address over-weighted; ad1.5/K10 remains best → **locked: blend_ad15** |
| P1 | 2026-09-25 | feature throughput profile | 42 feats | – | – | – | – | 4.6K pairs/s warm (under full CPU contention) | first measurement 271 pairs/s was pool spawn (28 s/pool under Windows spawn) → pool now created once per process |
| M1 | 2026-09-25 | blend_ad15 full train cands (India) | 42 feats | rule / logreg / **LightGBM** | tuned on half-A (0.52 / 0.62 / **0.69**) | **0.9618** (LGBM, half-B) | 0.963 (eval ents) | sel 479 s, feat 486 s | India, E=fold-1 5 %; P 0.988 R 0.923 FP 861 FN 5964; singleton acc 0.944; multi F0.5 0.967; rule 0.722, logreg 0.888 |
| M1b | 2026-09-25 | same | same | LightGBM **without** 1-to-1 | 0.69 | 0.9613 | – | – | FP 918 vs 861 → keep 1-to-1 (+0.0005, also better on tune half) |
| E1 | 2026-09-25 | error analysis of M1 | – | – | – | – | – | – | FN: blocking 2821 (+0.0147 if fixed), below-thr 2653 (+0.0124), lost-to-other 483; FP: distractor near-duplicates (one house number changed, legal form changed) 657 (+0.008) |
| M2 | 2026-09-25 | blend_ad15 (India) | 42 + 7 number/legal feats | LightGBM 1-to-1 | 0.66 (half-A) | **0.9680** (half-B) | 0.963 | augment 512 s | +0.0062 vs M1; P 0.991 R 0.933 FP 688 FN 5141; singleton acc 0.958; multi 0.972; logreg 0.888→0.925; num_s1_missing #3 feature |
| M2b | 2026-09-25 | same | same | LightGBM no 1-to-1 | 0.73 | 0.9680 | – | – | tie on half-B (0.96801 vs 0.96798); half-A favours 1-to-1 (0.9677 vs 0.9673) → keep 1-to-1 |
| F1 | 2026-09-25 | final_v1: India 10 % queries + US 5 % full-index sample | 49 | LightGBM (935 it) | 0.66 frozen | – | – | 1020 s | 7.24M pairs, 7.2 % pos; France smoke: 60 % queries matched, manual sample correct; official validator PASS (--check-ids) on smoke output |
| P0 | 2026-09-27 | blend_ad15 K10 | 49 / 70 (AR-7) | LightGBM, leakage-free retrain (exp3c) | half-A (0.82/0.81/0.84/0.84) | A 0.9616, B 0.9651, C 0.9676, **D 0.9709** (+0.0093, CI +0.0083..+0.0104; seed7 +0.0092; swapped +0.0095) | – | 25 min | leak L1 fixed (54k train queries overlapped eval ext.); D gain holds; US (India-only model) D -0.0010 CI excl. 0 |
| K50 | 2026-09-27 | blend_ad15 **K50** (compact cache, 206.7M train pairs, verified) | 70 (AR-7 deterministic fix) | M7_K10reg vs K50_r{1,0.5,0.25} (clean, E' excluded within top-50) | half-A, E'=bucket<5 (921 half-B ents) | D-regime 0.9801; best K50 (policy D h=0.35) 0.9796, **gain -0.0005 [-0.0030,+0.0019]**, seed7 +0.0018, swapped +0.0023, US -0.0014 -> **G1 FAIL** | E' recall@10 0.9698 / @50 0.9854; oracle 0.9898 / 0.9949 | ~2 h compute | K50 oracle - actual = 0.0146: the matcher, not retrieval, is the binding constraint; K50 not shipped, D stays final |
