# Third efficiency round: eight complete protection cases passed

**Efficiency-development parent: ABC_fixed_r2. Official submission: unchanged frozen V10.**

This round adds real simulator quality checks for the existing faster candidate. It does not add a newly accepted speed optimization. The two exploratory patches were not merged.

## Complete-task protection

Eight preselected cases include four original successes and four original failures. Independent review matched each candidate result against the original V10 native record, including final target count, source count, success flag, stop step, initialization, and task instruction. All eight comparisons matched. All 78 candidate runtime hashes and 24 saved result/protocol/summary files were also checked against the PRO server.

| Case | Parent = candidate target / source | Parent = candidate success | Stop step |
|---|---:|---|---:|
| 2×5 Basic, 19000 | 10 / 70 | Yes | 316 |
| 2×5 Basic, 620335 | 5 / 75 | No | 500 |
| 2×5 Advanced, 620335 | 0 / 75 | No | 500 |
| 2×5 Medium, 620015 | 10 / 70 | Yes | 312 |
| 2×10 Advanced, 19000 | 20 / 60 | Yes | 671 |
| 2×10 Medium, 24014 | 20 / 58 | No | 708 |
| 2×10 Basic, 620000 | 20 / 60 | Yes | 730 |
| 2×10 Advanced, 620780 | 10 / 70 | No | 3200 |

**8/8 passed means eight protection comparisons passed, not a 100% task success rate.** This deliberately selected set is not a new random benchmark. Full internal-state comparison, the 240-episode regression, and unseen-seed validation remain pending.

## Speed remains at the previously measured R2 level

The existing three-repeat paired study gave 163.92→149.78 ms for 2×5 and 236.36→212.95 ms for 2×10: 8.63% and 9.90% less call time than frozen V10. These are the **round-2** measurements, not newly repeated or additional improvements. See [the paired raw evidence](../efficiency_round2/README.md).

## Two exploratory changes were not selected

| Change | One-pass 2×5 mean | One-pass 2×10 mean | Decision |
|---|---:|---:|---|
| Reuse the float occlusion map inside a view search | 152.74 ms | 212.53 ms | Not merged; no clear incremental benefit |
| Cache fixed local box-wall polygons | 149.69 ms | 212.19 ms | Not merged; improvement too small to establish with one pass |

Their recorded actions, phases, finished flags and batch indices match the R2 reference tapes. One pass does not establish a repeatable speed gain or a general slowdown. The remaining optimization work should address the cost of dynamic collision geometry rather than continue adding fixed-box caches.

## Meaning and next step

The existing 9%-level speed improvement now has more useful quality evidence: it preserved both successful and difficult full episodes, including a 3,200-step failure case. The official policy is still not replaced. Next candidates should preserve existing geometry and decisions while reducing internal work, then pass the same quality checks.

The local next-round plan includes exact integer edge deduplication and demand-driven geometric auxiliary data, with checks for numerical identity and original failure behavior. These are proposed implementations, not additional measured policy gains.

## Inspectable evidence

- `reviewed_summary.json`: independently recomputed case comparisons and exploratory timings.
- `raw/native_quality_selected.json.gz`: unchanged selected fields from parent and candidate native results, public initialization, protocol settings, and all runtime hashes. Private filesystem paths and unrelated diagnostic reports are omitted.
- `raw/P21_test_*.csv.gz`, `raw/P22_test_*.csv.gz`: original per-call exploratory records compressed without rewriting rows.
- `manifest.json`: source-record hashes, candidate identity, and remaining validation scope.
- `code/verify_saved_results.py`: standalone verifier using this folder and previously published R2 reference calls.

From the repository root:

```bash
python3 results/efficiency_round3/code/verify_saved_results.py
```

No simulator or policy inference is needed to run the verifier. The review itself did not execute new policy rollouts or timing experiments. The frozen policy, published paper, and original two success/completion tables remain unchanged.

**中文概述：**这一轮补齐了8局真实任务保护，终态数量、成功和终止步数都与原V10一致；没有新的额外提速版本被采用。下一步聚焦动态碰撞几何内部的重复计算，正式V10暂不替换。
