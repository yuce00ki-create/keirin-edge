# KEIRIN EDGE v7 — Selective race-gating experiment (2026-11-17)

## Objective
Seek trifecta top-5 hit rate >=45% AND realized ROI >=101%; do not use 2026 for strategy selection.

## Method
- Source: already acquired public historical CSVs, validated races with payout matching recorded actual trifecta.
- Predictions: previous **race-score-only** Plackett–Luce top-five candidates; NOT the CLAR v6 conditional model.
- 2025: strategy search / development; 2026 Jan–Apr: independent holdout (no tuning).
- Each qualifying race: five trifecta tickets, 100 yen each. Return = actual payout if one of five tickets wins.
- Search: 1,540 eligible combinations among segment (all/boys/girls/7/9), predicted top-5 probability thresholds, top-1 probability thresholds, and score gaps. Filters with <150 validation or <75 test races excluded.
- IMPORTANT: 2025 model-selection results are optimistically biased by scanning 1,540 strategies. Never select a strategy by viewing the 2026 test performance.

## Results
- Across 1,540 candidates: **0** strategies met both target metrics on 2025 validation.
- One most favorable **2025 joint-score** selection is girls racing, top two riders' score gap >=4:
  - 2025: 254 races, 146/254 hits = **57.48%**; payout 108,760 JPY / stake 127,000 JPY = **85.64%** ROI.
  - 2026 Jan–Apr: 92 races, 50/92 hits = **54.35%**; payout 50,330 JPY / stake 46,000 JPY = **109.41%** ROI.
- 15,000-race bootstrap draws by year:
  - 2025 hit rate 95% percentile CI ~51.18–63.39%; ROI CI ~69.31–105.50%.
  - 2026 hit rate CI ~44.57–64.13%; ROI CI ~65.20–165.63%.
- The 2026 test result clearing both thresholds **does not** establish a robust profitable strategy: 2025 failed ROI target, small test sample, broad ROI uncertainty, selection bias on 2025.

## Guardrails
1. Exclude invalid racing result rows and require matching payout and trifecta outcome.
2. Use only info known before start; never choose conditions from holdout metrics.
3. Report all races, abstentions, purchases, losses and hit counts.
4. Keep girls' races separate from men's line-based races.
5. Evaluate entire 2025/2026 blocks and further future samples before deployment.
6. CLAR v6 must be benchmarked against this selection gate with genuine line data and calibrated probabilities; not yet done.

Code and full 1,540-row grid are distributed in the downloadable local research bundle in conversation. No actual future return is guaranteed.
