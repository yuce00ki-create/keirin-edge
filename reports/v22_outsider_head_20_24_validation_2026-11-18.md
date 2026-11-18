# KEIRIN EDGE v22 — outsider-head hedge experiment (2026-11-18)

## Outcome
A 3rd-line rider winning (松山3R 6-3-1, user reported) motivated an experiment to include additional first-place winners. It does **not** establish that v19 itself predicted the race: the prior race pick was manually generated, not model-run.

Same v18/v19 trained probability model: training 8,000 races in 2022–2024; 2025 200-race calibration; ranker alpha=1.0. Tested same 1,000 races of Jan–Apr 2026 plus remaining 360 2025 races (excluding 200 calibration and 140 tune samples).

| Selection rule | 2026 top20 | 2026 top24 | 2025 360 top20 | 2025 360 top24 |
|---|---:|---:|---:|---:|
| 1st restricted to 3 candidates | **481/1000 (48.1%)** | 518/1000 (51.8%) | 164/360 (45.6%) | 174/360 (48.3%) |
| 1st unrestricted, probability top k | 479/1000 (47.9%) | 522/1000 (52.2%) | 164/360 (45.6%) | 174/360 (48.3%) |
| Force >=1 ticket with winner outside top 3 | 479/1000 (47.9%) | **523/1000 (52.3%)** | 163/360 (45.3%) | 173/360 (48.1%) |
| Force >=3 outside-top-3 winner tickets | 455/1000 (45.5%) | 513/1000 (51.3%) | 164/360 (45.6%) | 175/360 (48.6%) |

2026 winner outside top three winner marginals: 210/1000. Forcing 1 outsider in 24 tickets compared with first3 added 20 exclusive wins but lost 15 previously won races: net +5 only; not reliable evidence.

## Decision
- Reject forced outsider as default: exploratory +0.5 percentage points on 2026 did not replicate on the separate 2025 set, and forced 3 worsened 2026 by 0.5 pp at 24 or 2.6 pp at 20.
- Maintain current v19 first3 20-ticket selection (48.1% on 2026; 45.6% on 2025 supplementary).
- Add unrestricted 24-ticket mode as an optional probability-maximizing strategy; +0.4 pp on previously explored 2026 and no measurable difference on 2025.
- The trained probability model was not improved in this experiment. All retrospective results remain exploratory; 2026 was repeatedly reused and not a genuine holdout. True line positions and race-timestamped features remain missing.
- No payouts/odds: ROI unknown. A single race loss cannot justify forcing third-line winners to the top.

## Implementation
[src/coverage_v22.py](../src/coverage_v22.py) provides `pick_tickets(joint,k,first_limit)`, allowing `first_limit=None` to pick unrestricted top k tickets. Tested scripts, 1,000-race side-by-side CSV, 2025 360-race side-by-side CSV, JSON in conversation attachment keirin_edge_v22_improvement_validation.zip.
