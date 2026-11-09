# First implementation-efficiency experiment — reviewed, not a replacement release

**Status: replay speed gain verified; optimized candidate is not approved for deployment.**

The first efficiency candidate removes unused visual work, rounds visible triangles in batches, and caches convex geometry under exact input-byte keys. It retains the original algorithm's image size, observation frequency, solver parameters and collision criteria. An independent review verified its measurements but also found a conditional initialization error that the two replay inputs did not exercise. The submitted `frozen_policy_v10/` remains unchanged.

## Measured progress

Both versions ran on the **same Intel i5-10500 CPU**, using four CPU cores on the RTX 3090 host. The GPU was not used for this V10-to-V10 comparison. Each complete public-observation recording was replayed three times in alternating parent/candidate order. The interval contains the whole native `policy.act`, excluding input loading, rendering and physics.

| Workload | Observations per repetition | Original V10 mean | Candidate mean | Latency reduction | Original / candidate P95 |
|---|---:|---:|---:|---:|---:|
| 2×5 Basic | 316 | 166.44 ms | 150.98 ms | **9.29%** | 494.89 / 453.79 ms |
| 2×10 Advanced | 671 | 238.10 ms | 213.19 ms | **10.46%** | 598.43 / 544.08 ms |

There are **987 distinct recorded observations**, repeated three times: 2,961 paired calls, not 2,961 independent tasks. The serialized actions recover to identical float64 bytes for every pair; phase and finished flags also match. The 2×10 policy finishes after the last input frame, with no post-finish calls diluting the mean. Each of the three rounds shows a reduction in mean latency.

These numbers use a fresh paired V10 parent run. They are not computed by mixing this candidate with the earlier SmolVLA timing experiment.

## Why the quality gate is still open

1. **A conditional error was reproduced.** When a confirmed source-box position is retained while the target still needs fitting, the lazy generic distance map is absent. Three candidate implementations nevertheless access its shape and raise `TypeError`. The original functions return normally. An isolated in-memory correction of the shape reference removes that exception. This is a synthetic branch test, not evidence that the error occurred in either measured replay.
2. **The cache is not actually read-only.** It returns the same writable geometric object. An isolated mutation test shows that a caller can alter later cached results. No current production mutation has been demonstrated; this is an ownership contract that should be fixed before expansion.
3. **The available trace does not establish full internal-state equality.** Batch-index fields are empty and decision-state fingerprints were not recorded. Small closed-loop regression, the 240-episode protection set, and new-seed quality tests have not been run for this candidate.

The correct conclusion is **a reproducible 9–10% reduction in computation on these two replay workloads, with a known branch defect still to fix**. The original V10 completion/success scores are not automatically inherited by this candidate. This is a development record, not a new “lossless V10” submission.

## Inspect or verify

- `raw/parent/` and `raw/cand/`: all 12 original per-call CSVs, gzip-compressed without rewriting rows.
- `run_metadata.json`: selected original run metadata for all 12 runs; numeric values unchanged.
- `reviewed_summary.json`: independent calculations, exact-action checks and missing-evidence flags.
- `manifest.json`: parent and changed-file SHA256 values, raw-file hashes and qualification status.
- `candidate_review_only.patch`: the nine-file tested change, **for review of this unapproved experiment**.
- `review/branch_probe.json`: results of the isolated conditional-branch and cache-ownership tests.
- `review/REVIEW_zh.md`: plain-language review in Chinese.
- `code/`: independent verifier and original-function probe source.

From the repository root, using standard Python (no simulator):

```bash
python3 results/efficiency_round1/code/verify_saved_calls.py \
  --raw results/efficiency_round1/raw \
  --metadata results/efficiency_round1/run_metadata.json \
  --output outputs/efficiency_round1_recheck.json
```

This verifies the saved replay measurements, not the candidate's closed-loop quality. The separate function probe requires NumPy, OpenCV and SciPy and the original/candidate source directories; its default assertion demonstrates the known defect, rather than certifying a repaired candidate.

## Next engineering step

Repair the lazy-map branch and cache ownership first. Then profile the repaired candidate and reduce repeated preparation inside visual fitting: compute fixed observed-contour edges once per fit, and share identical object geometry across camera views at the same solver parameter. Evaluate within-call gripper-image reuse next. Preserve the original solver, observations and all collision checks, and compare control-state behavior before accepting further speed improvements.

The first-round profile belongs to the original policy, so it is guidance for where to look—not a prediction of how much the next candidate will save. Detailed execution instructions are maintained separately in the local course project.
