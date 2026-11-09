# Independent frozen-V10 replication: 20 new random seeds

Frozen policy: V10 (same 78 runtime files as the first holdout test). A second set of 20 distinct seeds was sampled from 620000–620999 with Python `random.Random(20261108)` before the run. Both tasks × three official variation levels were tested on source column 1, for 120 episodes. The non-strict, 5-step success hold and 500/3200-step budgets were unchanged. The run used CPU OSMesa/llvmpipe.

- Mean completion: **96.62%**, computed as the equally weighted mean over the six task/variation settings (exact unrounded value 96.625%).
- Official success: **94.17%**, or 113/120.
- Task 2×5: 585/600 valid cookies, 58/60 successes.
- Task 2×10: 1,149/1,200 valid cookies, 55/60 successes.

Files: `episodes.csv` (all 120 final scores), `native_episodes.jsonl.gz` (the native evaluator's original JSONL, compressed without rewriting rows), `summary.json` and `complete.json` (original summary/receipt), `seeds.json` (presampled seed list), `manifest.json` (source and hash information), `original_run_grid.py` (the original shard wrapper). The portable equivalent is `../../scripts/run_retest_grid.py`.

Verification from the repo root:

```bash
python3 scripts/verify_retest.py
```

The data have no selected successful-only subset: all seven unsuccessful episodes are retained. Scores are the official evaluator's final values. The earlier, independent seed batch is in `../v10_holdout_episodes.csv` and `../v10_official_grid_summary.json`.

To re-run this exact independent seed grid using the repository's pinned evaluator (requires the official simulator environment), invoke the portable shard wrapper from the repository root:

```bash
python3 scripts/run_retest_grid.py --seeds results/retest/seeds.json \
    --output outputs/new_retest --workers 2
```

The script does not alter official scoring; each episode is evaluated by `evaluation/evaluate_policy.py` using the frozen policy. It is separate from the cheap offline `verify_retest.py` check.
