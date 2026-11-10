# R6 efficiency review — retain R4 and correct the cache experiment

Review date: 2026-11-10. **Official frozen V10 and the paper's success/completion tables are unchanged. Neither R6 candidate was adopted.**

## One-pass screening, independently recomputed

Same PRO CPU, four threads, affinity 8–11; complete 316-frame 2×5 and 671-frame 2×10 observation tapes; one run per version/task. The measurement covers complete policy calls and excludes new physics/rendering.

| Version | 2×5 mean ms | 2×10 mean ms |
|---|---:|---:|
| R4 development parent | 85.6012521 | 125.1317837 |
| R6 shared 3D body | 84.1412932 | 125.2691019 |
| R6 world/mask cache | 85.9949101 | 127.3377505 |

The shared-body candidate used 1.71% less time on 2×5 and 0.11% more on 2×10. The mask candidate used 0.46% and 1.76% more time. These single-pass differences do not establish repeatability or a general slowdown.

All six means were recomputed from original CSV rows. Independent byte comparisons found identical 14-dimensional actions, before/after phases, batch indices and finished flags for the candidate/parent pairs. Full internal-state snapshots, complete-fit contracts and new closed-loop quality tests were not performed for R6. Retaining R4 is appropriate.

## Substantive review finding

The R6 mask cache is queried **after** projection, triangle sorting and all `fillConvexPoly` calls. Even a cache hit redraws the entire mask before returning a cached copy. Therefore the existing negative result does not test the intended early-return optimization.

A pure-function probe extracted the actual R6 method. With the same small synthetic geometry called twice, drawing occurred once on the first call and once again on the second, despite an existing cache entry. An illustrative guard moved before computation reduced the second call to zero drawing calls, with identical pixel bytes. Returned-array independence and a miss following a pose change also passed.

Actual task-level cache hit rates and component costs remain to be measured. The previous explanation that low repetition or copying cost caused the regression is not established by the one-pass timings alone.

## Next experiment

Keep R4 as the parent. First test a correctly positioned, exact-input, within-control-step mask lookup with actual hit/cost counters. Separately test a strict native bilinear sampling kernel that preserves the existing numerical expression and falls back to the reference for unsupported cases.

An isolated CPython-buffer prototype was compiled using the existing PRO toolchain, without dependency upgrades. It matched the extracted R4 reference on 2,022 synthetic contract cases: 2,016 native calls and 6 reference fallbacks, covering read-only/noncontiguous arrays, negative strides and boundary coordinates. **It has not been integrated into a policy or measured for task speed.**

Acceptance still requires real recorded-input and full-fit checks, identical sequential actions/decision state, repeatable same-host timing, and per-episode task protection. Existing successes may not be lost and final cookie count may not decrease in any protected episode. The final accelerated release still needs the 240-episode regression and new-seed validation.

## Evidence and local execution package

The full original six CSV/six metadata records, current source hashes, candidate patches, independent verifiers and probe sources are saved in the local project under `results/efficiency_round6/` in local commit `2ae1b6625ce7398bb34bf648eaac5a3f3981f20c`. The ordinary Git transport encountered an SSL connection error, so that full commit is **not** represented as published here. This standalone review is the successfully published summary.

The executable local handoff is `DEEPSEEK_V10_效率优化_R7_执行入口.md`, with the detailed package in `deepseek_v10_efficiency_r7_20261110/`. It includes the reviewed evidence, two probes, 12 groups of primary research/documentation references, exact code locations, test sequence and acceptance conditions. No new policy rollouts or simulation were started during this review.

**中文结论：**本轮没有采用新加速版，但发现遮挡缓存放错位置：先画完再查，命中也省不掉绘图。下一轮先验证真正的提前复用，再独立评估精确原生像素采样。正式V10与论文原成绩不变。
