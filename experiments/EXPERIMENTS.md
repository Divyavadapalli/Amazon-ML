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
