# Second implementation-efficiency experiment — fixes verified, quality regression pending

**Status: experimental, not a replacement for the submitted frozen V10.**

Round 2 repairs the conditional image-map error found in the first review, makes cached polygon arrays read-only, and prepares the fixed observed polygon once per numerical fit. An independent review confirms the saved timing results and checks the actual code on the test host. The full-task quality checks remain to be run.

## Paired computation results

Both versions use the same Intel i5-10500 CPU on the RTX 3090 host, four CPU cores, the same complete public-observation inputs, and three alternating-order repetitions. The GPU is not used for this rule-versus-rule comparison. Timing contains the full native `policy.act`, excluding input loading, rendering and physics.

| Workload | Observations / repetition | Original V10 mean | R2 candidate mean | Latency reduction | Original / candidate P95 |
|---|---:|---:|---:|---:|---:|
| 2×5 Basic | 316 | 163.92 ms | 149.78 ms | **8.63%** | 491.39 / 444.24 ms |
| 2×10 Advanced | 671 | 236.36 ms | 212.95 ms | **9.90%** | 596.83 / 532.38 ms |

These reductions are **relative to the original V10**, not an additional 9% on top of round 1. The same-stage comparison between round 1 and round 2 was not separately isolated. Round 2's clearest contribution is repairing correctness while retaining the earlier speed improvement.

All 2,961 paired calls recover to identical float64 action bytes; actual phases, finished flags and batch indices also match. The data contain **987 distinct observations repeated three times**, not 2,961 independent tasks. Batch indices are now populated and were checked. The 2×10 replay finishes only after its final input; no post-finish calls dilute the average.

## What the independent review confirmed

- All 78 candidate runtime files and all 12 raw CSVs on the Mac matched the actual test-host files. Eleven runtime files differ from the frozen parent.
- The previously failing lazy-image-map branch returns normally in all three corrected implementations. The test uses isolated original functions and controlled image-component inputs, not new robot episodes.
- Element mutation of the cached polygon arrays is rejected.
- The new prepared-polygon distance helper produces byte-identical outputs to its parent on 120 synthetic input cases in each of its two implementations.

## A correction to the executor's summary

The tested polygon cache **does not yet implement a 64 MiB LRU**. It uses `OrderedDict`, but `_poly` still clears the entire cache after exceeding 8,192 entries, and cache hits do not update recency. A separate path-result cache in the same file does implement an LRU; it is not this polygon cache.

The independent probe checks this behavior directly. Polygon fields can still be rebound, although no production caller has been shown to do that. These observations clarify what was actually implemented rather than attributing a new task failure to the cache.

## Next optimization target

Two particularly slow calls per repetition account for **20.68% of the 2×5 candidate's total call time** and **8.13% of the 2×10 total**. They occur at frames 94/95 and 77/78 respectively, in the target-retreat stage. Their specific cost should be profiled in the current candidate before selecting another optimization.

The next proposed changes preserve all observations, candidate choices and collision criteria: reuse invariant preparation inside a single view search, avoid rebuilding fixed collision geometry, and only then consider a small compiled sampling kernel or shared multi-view geometry. These are plans, not measured additional gains.

## Quality status

The 8-episode protection test, the 240-episode regression, new-seed checks, and comparison of decision-relevant geometry/history have **not** been completed for this candidate. The original V10 success/completion scores are not assigned to the faster candidate. The official `frozen_policy_v10/`, its evaluation table and the published paper remain unchanged.

## Files and verification

- `raw/parent/`, `raw/cand/`: 12 original per-call CSVs, gzip-compressed without rewriting their bytes.
- `run_metadata.json`: selected original runtime metadata, with paths omitted and numerical values preserved.
- `reviewed_summary.json`: independently recalculated timing, action/phase/batch checks and stage costs.
- `manifest.json`: original/candidate file hashes, raw CSV hashes and pending quality checks.
- `candidate_review_only.patch`: exact 11-file change against the frozen policy, for review rather than deployment.
- `review/independent_contract_probes.json`: isolated branch, cache and polygon-helper results.
- `review/REVIEW_zh.md`: detailed review in Chinese.
- `code/`: independent raw-data verifier and the function-level probe used in the review.

From the repository root, without running a policy or simulator:

```bash
python3 results/efficiency_round2/code/verify_saved_calls.py \
  --raw results/efficiency_round2/raw \
  --metadata results/efficiency_round2/run_metadata.json \
  --output /tmp/v10_efficiency_round2_recomputed.json
```

The function probe additionally needs the project's NumPy/OpenCV/SciPy environment and a separately reconstructed candidate. It does not import the policy or run physics. The independent review itself performed no new full-policy timing or physical rollouts.
