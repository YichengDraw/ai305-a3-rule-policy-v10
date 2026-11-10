# AI305 — A3 Cookie Transfer: Frozen Rule Policy V10

**Reproducible policy submission for the AI305 A3 dual-arm MuJoCo cookie-transfer benchmark.**

This is the **frozen V10** rule-based RGB + robot-state policy and a stand-alone evaluator that uses the original A3 task and official terminal score/success conditions. The `frozen_policy_v10/` submission is unchanged. A later **A1 development candidate** is now published separately under [`candidates/a1/`](candidates/a1/), with its own evidence and report section; it does not replace V10. No pretrained weights are included.

## Verified holdout evaluation (source column 1)

| Task | Holdout episodes | Mean completion rate | Official success rate |
|---|---:|---:|---:|
| 2×5 | 60 | **100.00%** | **100.00% (60/60)** |
| 2×10 | 60 | **96.83%** | **91.67% (55/60)** |
| **Equal-weight mean across 6 cells** | **120** | **98.42%** | **95.83% (115/120)** |

**Scope:** two tasks × basic / medium / advanced × 20 test seeds (24000–24019), `source_column=1`, as fixed prior to the evaluation. The official default six-cell grid fixes source column 1; the upstream environment supports four source columns and random source selection. **The above numbers are not claims of 90%+ success for arbitrary source columns.** The 2×10 first-grasp count was randomly sampled under the official setup. Failed episodes are included. Completion is the per-episode final `cookies_in_target / required_count` averaged equally across the six equally sized task/profile cells. Success is the original official `success` flag, with no extra criteria added.

**Protocol:** official non-strict mode; stable-success hold=5; 2×5 max=500 control steps; 2×10 max=3200. Keep the task's native release and placement conditions. The reported run used CPU MuJoCo rendering with OSMesa/llvmpipe; other hardware/software configurations may differ.

## Paper and complete development history

**[Read the Group 3 research report (PDF)](report/group3_rule_policy_v10.pdf)** · [LaTeX source](report/main.tex) · [Code history V1–V9](history/README.md)

The paper explains the camera/force-based rule controller, the GPT-6 advisor ↔ separate coding-executor iteration loop, and the main decisions from V1 to V10 using plain language. The public history includes V1–V4 development snapshots, V5–V9 frozen policies, retained experimental variants, and 1,320 official episode results from the milestone comparisons.

**Independent V10 replication:** an additional set of 20 randomly selected seeds (620xxx, 120 episodes) achieved **96.62% mean completion** and **94.17% official success** (113/120). This uses the same frozen code and benchmark grid, and all failures are included. [Full reproduction records and seed list](results/retest/README.md). The first test above remains a separate 120-episode holdout; the pooled mean across both tests is 97.52% completion and 95.00% success.

## A1 development update (10 November 2026)

[Exact A1 snapshot](candidates/a1/) · [240 native results, protocol and verifier](results/a1_regression_20261110/README.md) · [Updated report PDF](report/group3_rule_policy_v10.pdf)

On the two **previously seen** source-column-1 grids, A1 achieves **98.8333% completion / 115 of 120 successes** and **97.6667% / 116 of 120**. Pooled: **98.25% / 96.25% (231/240)**. Relative to V10, three additional episodes succeed, no original success is lost, and no final count decreases. Nine official failures remain.

These are development regressions, **not independent A1 holdouts**. The release includes 79 byte-verified runtime files, all 240 native terminal records, complete recorded-initialization comparisons, fixed-count replay equivalence, and invalid-attempt/retry disclosure. The completed W1b auxiliary comparison is documented separately in the same evidence. Frozen V10, its original headline tables and its timing results remain unchanged. Provenance metadata uses relative identifiers and omits local machine information; runtime code and all terminal records remain byte-preserved. Verify with `python3 scripts/verify_a1.py` and `python3 scripts/verify_a1_privacy.py`.

## Policy computation timing

A separate synchronized replay benchmark measures the full native policy call on one Basic 2×5 trajectory (316 observations) and one Advanced 2×10 trajectory (671 observations), repeated three times on an Intel i5-10500 + RTX 3090 host. V10 runs on CPU; the teacher SmolVLA checkpoint uses CUDA and executes 8 actions per generated block. The intervals exclude rendering, physics, file reading and model loading.

| Recorded workload | V10 CPU mean | SmolVLA CUDA mean | SmolVLA new block | SmolVLA queue call |
|---|---:|---:|---:|---:|
| 2×5 Basic | 165.0 ms | 70.5 ms | 532.2 ms | 3.57 ms |
| 2×10 Advanced | 236.7 ms | 68.7 ms | 523.5 ms | 3.61 ms |

These are computation costs on recorded observations, not new closed-loop success trials. The 2×10 replay has six stage differences per 671-step repetition; the detailed record includes all action differences and a sensitivity check. **The measured rule implementation is not faster on average than the GPU model**; model generation remains much slower than its inexpensive queued calls. [Full timing methods, raw CSVs and review](results/latency/README.md) · Verify with `python3 scripts/verify_latency.py`.

## Implementation-efficiency experiment (not merged)

A separate V10-to-V10 paired replay reduced average CPU call time from **166.44 to 150.98 ms (2×5)** and **238.10 to 213.19 ms (2×10)**, with identical recorded actions and phases across three repetitions. Review also reproduced a conditional error in an unexercised lazy-initialization branch. **The candidate is experimental: full quality validation is pending and the submitted frozen V10 is unchanged.** [Measurements, code changes, known issue and independent verifier](results/efficiency_round1/README.md).

### Reviewed second efficiency round

The second candidate repairs the previously found conditional error and retains a **163.92→149.78 ms (2×5)** / **236.36→212.95 ms (2×10)** paired CPU improvement, with identical recorded actions, phases, finished flags and actual batches. This is **8.63% / 9.90% relative to original V10**, not an additional gain over round 1. Review corrected an implementation claim: polygon arrays are read-only, but the reported 64 MiB LRU is not present in the tested code. Full-task quality regression is still pending; frozen V10 remains the official submission. [Round 2 raw evidence, corrections and next bottlenecks](results/efficiency_round2/README.md).

### Reviewed third efficiency round

The existing R2 speed candidate has now passed **eight complete protection cases**: target/source counts, official success, stop steps, initializations, and task instructions match frozen V10 in every case. The set deliberately contains four successes and four failures; **8/8 means protection checks passed, not 100% task success**. Two one-pass implementation probes showed no clear additional speed benefit and were not merged. The 240-episode and new-seed quality gates remain pending. [Round 3 native evidence and independent verifier](results/efficiency_round3/README.md).

## Files

- [`frozen_policy_v10/`](frozen_policy_v10/): complete original frozen routing policy and all **78 verified runtime dependencies**; `policy.py` exposes `make_policy_b()` and action mode `cartesian_delta`.
- [`evaluation/`](evaluation/): standalone `evaluate_policy.py` and its three helper modules. Uses the upstream benchmark/evaluator; does not replace the environment's scorer.
- [`scripts/run_official_grid.sh`](scripts/run_official_grid.sh): reproduce the six-cell test under the pinned upstream revision.
- [`scripts/verify_release.py`](scripts/verify_release.py): verify all frozen SHA256 files and recompute the published metrics without MuJoCo.
- [`results/v10_holdout_episodes.csv`](results/v10_holdout_episodes.csv): full **120-row** test log extract (task/profile/seed/n/final count/official success/steps and original result-file hashes).
- [`results/v10_official_grid_summary.json`](results/v10_official_grid_summary.json): summary and reproduction protocol.

## Quickstart

Requires **Python 3.12+**, Git, and an environment able to run MuJoCo with the three RGB cameras. The simulator is an external, upstream MIT-licensed dependency. Clone the exact revision used for the reported results:

```bash
git clone https://github.com/Eter0109/a3_dual_arm_sim.git
cd a3_dual_arm_sim
git checkout 0729068ee1e75a562cfbc4f2c83ab9539cf75081
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
cd ..
git clone https://github.com/YichengDraw/ai305-a3-rule-policy-v10.git
cd ai305-a3-rule-policy-v10
export SIM_REPO="$(cd ../a3_dual_arm_sim && pwd)"
python3 scripts/verify_release.py
bash scripts/run_official_grid.sh
```

For a fast configuration/import check without simulation, append `--dry-run`. On Linux, OSMesa may require system packages such as `libosmesa6` and `libgl1-mesa-dri`. The runner outputs `protocol.json`, `episodes.jsonl`, `summary.json` and `summary.md` in a new timestamped folder under `outputs/` (ignored by Git). For additional source columns, pass `--source-column 2`, etc., and report the resulting metrics separately; do not conflate those results with the table above.

## How the policy is evaluated

The evaluator loads:

```text
--policy policy:make_policy_b
--policy-code frozen_policy_v10
```

The policy's public task instruction selects the source-column branch (1, inner 2–3, outer 4), while its actions are computed from permitted RGB/state inputs. All scoring is performed by the upstream simulator. The bundled `evaluation/` files are the exact local benchmark helpers used for V10; the full simulator is not vendored here. Three pinned upstream source files are included only as A1 mode-equivalence audit evidence, under the upstream MIT license. This release preserves the frozen code rather than cleaning or rewriting it, to keep hashes and behavior reproducible.

## Interpretation and attribution

These scores are **empirical results** on a preregistered holdout seed window, not a mathematical guarantee for all randomized scenes. In particular, the 20 source-column-1 seeds for 2×10 did not cover every possible first-grasp count, and other source columns have separate, harder behavior. Upstream simulator: [Eter0109/a3_dual_arm_sim](https://github.com/Eter0109/a3_dual_arm_sim) (MIT). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This is an independent student policy submission, not an official upstream release.

**中文说明：**这是 AI305 作业用的冻结 V10 规则策略评测提交包。98.42% 完成率、95.83% 成功率对应**来源列1**、三档随机化、每档20个预登记测试种子；其他列需要单独评测。
