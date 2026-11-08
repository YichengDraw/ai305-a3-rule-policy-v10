# Frozen V10 official evaluation results

- Dataset: 20 preregistered holdout seeds (24000–24019) × 2 tasks × 3 official profiles = 120 episodes.
- `source_column=1` was fixed **before evaluation**; randomization applies to the other configured factors. This does **not** claim results for source columns 2–4.
- Each row is a genuine final outcome; failures are retained. Scores are the official final `cookies_in_target` and original `success`, not an episode's peak count.
- `v10_holdout_episodes.csv` contains one row per episode and the SHA256 of the native evaluation episode JSONL from which it was extracted.
- `v10_official_grid_summary.json` contains per-cell and equal-cell averages. Recompute via `python3 scripts/verify_release.py`.
- The original run was performed on a CPU OSMesa/llvmpipe renderer. Other physics/rendering versions may produce different stochastic results.
