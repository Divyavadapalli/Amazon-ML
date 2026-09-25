# Experiment log

Never edit past rows; append new ones. Raw machine-written results live next to this file
(`blocking_results.jsonl`, …).

| ID | Date | Blocking | Features | Model | Threshold | Val F0.5 | Cand. recall | Runtime | Notes |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 2026-09-25 | blend nw1/nc1/ad0.8 K=10, **20 % index (optimistic)** | – | – | – | – | 0.9703 (oracle F0.5 0.989) | 459 s | S1 index shrunk to 20 % → 5× fewer confusers than test; superseded by B-series v2 |
| B1 | 2026-09-25 | same, K=20, 20 % index | – | – | – | – | 0.9780 | 1091 s | recall@1..5 identical to B0 |
| B2 | 2026-09-25 | blend10, **full S1 index**, 5 % entity sample | – | – | – | – | **0.9510** (oracle 0.982) | 619 s, peak 6.2 GB | India 0.927, US 0.967; addr-empty 0.755; script names 0.896; non-exact core 0.900; r@1 0.914 |
| T1 | 2026-09-25 | translit audit, fold-1 held out (360K script pairs) | – | – | – | – | – | – | dict acc 98.5 %, coverage 53 %; mean name sim A 24.7 / B 71.9 / C 62.8 / **D 85.3** → keep D (dict+anyascii) |
