# KEIRIN EDGE top-5 accuracy and leakage audit — 2026-11-17 19:00 JST

## Scope
Audited conversation artifacts `keirin_edge_v8_precision_prototype.zip`, `keirin_edge_v9_validation.zip`, `keirin_integrity_audit.zip`, `keirin-public-data.zip` and current GitHub repository files. These historical artifacts were available in the conversation runtime but the v8/v9 experiment code and source data are **not yet checked into this GitHub repo**.

## Paired results (the exact same 120 race rows in v9 results.csv)
- v8 top-5: **15/120 = 12.50%**.
- v9 top-5: **15/120 = 12.50%**. Improvement **0.00 percentage points**.
- Both correct **11**, v8 only **4**, v9 only **4**, neither **101**.
- Exact McNemar p-value **1.0**. No evidence of improvement.
- Note: the CSV contains **119 races with 2026 in race_id and one with 2025** (race_id 5620251231030004). Prior wording “2026 Jan–Apr 120 races” is inaccurate; this appears to be a selection boundary and must be repaired for strict evaluation.
- Historical v8 prototype's 60-race report: 10/60 top-5 = 16.67%; not directly comparable to the 120-race paired evaluation.
- v9 top-1 2/120 = 1.67%, versus v8 4/120 = 3.33%. Top-10 v9 28/120 = 23.33% vs v8 30/120 = 25.00%.

## Real line / data leakage
- Public data archive columns include `lineup`, but this is the **start list**, not the verified tactical group/leader/chaser order. **0 verified historical tactical line records**; do not train a line-aware model from lineup as if it were true line data.
- Fields `rank`, `agari`, `finish_type`, `margin`, `win_lose`, `ni_sha_tan`, `san_ren_tan`, `san_ren_fuku`, `wide` are outcome/post-race and must be excluded from inference.
- `race_score` and other current race-card features have no independently verified pre-start snapshots; their historical point-in-time integrity is **unproven**, even if historical cumulative stats are computed using strictly earlier races.
- Current 2026 holdout has been reused repeatedly, so it is not a pristine untouched evaluation set.

## Implemented change
- `src/top5_integrity_gate.py`: fail-closed source timestamp and post-race feature checks, 5-unique-ticket verification, same-race paired comparison, exact McNemar significance, year boundary counts. It improves **experimental correctness**, not demonstrated prediction accuracy.
- Tested the gate against v9's 120 saved predictions. Results above are reproduced.
- No new model is promoted: v9 fails to beat v8 on top-5, and line coefficients are not trained from genuine line history.

## Next experiment (pre-registered protocol)
1. Acquire immutable pre-start snapshots with source URL and capture timestamp, plus verified historical leader/follower tactical lines; store hashes and race IDs.
2. Select a new time-ordered future window **after 2026-11-17**, without looking at its results for tuning. Use earlier period for training, subsequent period for calibration, and later period for test; no same-day leakage.
3. Train v8 and new candidate with exactly identical training races and candidate features; optionally ablate genuine line information.
4. Publish per-race top-5, top-1, top-10, NLL, paired discordance counts and bootstrap intervals. Do not promote without a positive unseen-window result.
5. 45% top-5 hit rate and 101% ROI **not achieved or validated**.

## Caveat
No newly trained challenger or new genuinely out-of-time historical paired run was completed in this audit. The confirmed result is **no top-5 improvement on the archived same-race 120 sample**.
