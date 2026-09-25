# BEST_KNOWN_VERSION

## Validation baseline (M1, 2026-09-25) — commit recorded below
- Blocking: blend_ad15 (TF-IDF name-words 1.0 / name-char3 1.0 / address 1.5, K=10, per country, full S1 index)
- Blocking recall (India eval entities): 0.963; 5 % full-index sample (India+US): 0.9774
- Features: 42 (src/features.py @ M1), Model: LightGBM (MIT), params src/matching.py LGB_PARAMS
- Assignment: one S2/S3 → at most one S1 (best prob), threshold 0.69 (tuned on half-A of eval entities)
- India held-out macro F0.5 = **0.9618** (P 0.988, R 0.923, FP 861, FN 5964, singleton acc 0.944)

## Leaderboard submissions
(none yet)
