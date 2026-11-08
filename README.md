# AI305 — A3 Cookie Transfer: Frozen Rule Policy V10

**Reproducible policy submission for the AI305 A3 dual-arm MuJoCo cookie-transfer benchmark.**

This is the **frozen V10** rule-based RGB + robot-state policy and a stand-alone evaluator that uses the original A3 task and official terminal score/success conditions. It does **not** contain later W1b/DSH development changes, pretrained weights, or privileged ground-truth policy inputs.

## Verified holdout evaluation (source column 1)

| Task | Holdout episodes | Mean completion rate | Official success rate |
|---|---:|---:|---:|
| 2×5 | 60 | **100.00%** | **100.00% (60/60)** |
| 2×10 | 60 | **96.83%** | **91.67% (55/60)** |
| **Equal-weight mean across 6 cells** | **120** | **98.42%** | **95.83% (115/120)** |

**Scope:** two tasks × basic / medium / advanced × 20 test seeds (24000–24019), `source_column=1`, as fixed prior to the evaluation. The official default six-cell grid fixes source column 1; the upstream environment supports four source columns and random source selection. **The above numbers are not claims of 90%+ success for arbitrary source columns.** The 2×10 first-grasp count was randomly sampled under the official setup. Failed episodes are included. Completion is the per-episode final `cookies_in_target / required_count` averaged equally across the six equally sized task/profile cells. Success is the original official `success` flag, with no extra criteria added.

**Protocol:** official non-strict mode; stable-success hold=5; 2×5 max=500 control steps; 2×10 max=3200. Keep the task's native release and placement conditions. The reported run used CPU MuJoCo rendering with OSMesa/llvmpipe; other hardware/software configurations may differ.

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

The policy's public task instruction selects the source-column branch (1, inner 2–3, outer 4), while its actions are computed from permitted RGB/state inputs. All scoring is performed by the upstream simulator. The bundled `evaluation/` files are the exact local benchmark helpers used for V10; the original simulator files are not vendored here. This release preserves the frozen code rather than cleaning or rewriting it, to keep hashes and behavior reproducible.

## Interpretation and attribution

These scores are **empirical results** on a preregistered holdout seed window, not a mathematical guarantee for all randomized scenes. In particular, the 20 source-column-1 seeds for 2×10 did not cover every possible first-grasp count, and other source columns have separate, harder behavior. Upstream simulator: [Eter0109/a3_dual_arm_sim](https://github.com/Eter0109/a3_dual_arm_sim) (MIT). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This is an independent student policy submission, not an official upstream release.

**中文说明：**这是 AI305 作业用的冻结 V10 规则策略评测提交包。98.42% 完成率、95.83% 成功率对应**来源列1**、三档随机化、每档20个预登记测试种子；其他列需要单独评测。
