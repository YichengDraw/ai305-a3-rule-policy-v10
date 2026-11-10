# A1 paired development regression, 10 November 2026

**A1 is published separately. Frozen V10 remains the formal submission.** Both seed collections were already seen during development, so these are matched development regressions, not independent A1 holdouts.

| Grid (120 episodes each) | V10 completion / success | W1b completion / success | A1 completion / success |
|---|---:|---:|---:|
| 24000–24019 | 98.4167% / 115/120 | 98.4167% / 115/120 | **98.8333% / 115/120** |
| Same 20 seeds in 620xxx | 96.6250% / 113/120 | 97.0417% / 114/120 | **97.6667% / 116/120** |

A1 pooled: **98.25% completion and 96.25% official success (231/240)**. Against V10 on the same 240 rows this is +0.7292 percentage points completion and +1.25 points success. No V10 official success is lost and no final cookie count decreases. This does not assert identical trajectories or timings.

- Basic 2×10 seed 24001 improves from 10 to 20 cookies but remains officially unsuccessful.
- Advanced 2×10 seeds 620382 and 620780 improve 10→20, and 620461 improves 15→20; all three become officially successful.
- Nine official failures remain. This is not a 100% result or evidence for arbitrary source columns.
- W1b is an auxiliary, completed comparison, confirmed by the final 14:45 UTC audit. Relative to W1b, A1 adds the successes at Advanced 620461/620780 and the count improvement at Basic 24001, with no count/success regressions. W1b code and its full native logs are not released in this A1 update; its terminal summary and source-result hashes are included for context.

## Metric and protocol

Two tasks × basic/medium/advanced × 20 seeds; source column 1. Completion is the average of each episode's **final** `cookies_in_target / max_score`, equivalently the equal-weight mean of the six equally sized cells. It is not the pooled raw-cookie fraction. Success is the native `success` boolean, not a custom diagnostic or an AND with extra release requirements.

Official non-strict mode; stable success for 5 steps; 2×5 budget 500, 2×10 budget 3200. The task's native placement conditions remain intact: 2×5 non-strict does not add release, while 2×10 retains native `require_released=true`. All runs use CPU physics with OSMesa/llvmpipe software rendering. The simulator is pinned to `0729068ee1e75a562cfbc4f2c83ab9539cf75081`; the final audit confirms its 156 tracked source/config/model files and 121 V10 native result files unchanged.

## Reuse, equivalence and excluded attempts

The selected A1 data comprise 131 existing results and 109 new valid results. All 120 old-grid records come from one earlier A1 run; 60 old-grid 2×10 episodes replay the original randomly sampled first-grasp count as a fixed value. The other 60 old-grid episodes are 2×5. The selected 620xxx records retain the native random mode where applicable.

The published verifier compares every A1 initialization and task instruction against a 240-row extract from the original V10 records. Exactly 60 differences remain, solely the `first_grasp_mode` metadata field. Each actual first-grasp count matches both the old V10 CSV and the seed's native independent random stream. [Mode-equivalence source evidence](mode_equivalence.md) explains why the same count preserves batch plan, prompt and policy inputs. The extracted initialization file is derived evidence, with original-file hashes; the original 620xxx native file is also available in `../retest/` and is cross-checked directly.

An early new-run attempt serialized the color dictionary in sorted order, changing advanced palette sampling; its 2×10 configuration also omitted a task-specific first-grasp setting and those jobs failed before simulation. These were configuration/launch-invalid attempts, not policy failures removed for poor performance. The original evidence was retained unchanged and only affected jobs were retried with the original configuration order and native random mode. The sanitized source-run completion receipts and `configuration_repair_audit.json` preserve this history. Every valid matched episode, including all nine official A1 failures, is included in the final 240.

Six source roots contribute 120, 7, 1, 3, 40 and 69 selected records respectively. A root's failed status can therefore coexist with valid selected records; the verifier checks the selected job's zero exit code and protocol individually. Raw invalid-attempt terminal/trace files, large policy traces and video are not included in this public export.

## Files

- `../../candidates/a1/`: exact 79-file runtime snapshot and independent SHA256 manifest.
- `episodes.csv`: all 240 final rows; `summary.json` and `configuration_table.csv`: aggregate results.
- `native_results.tar.gz`: all 240 original `result.json` files, byte-preserved, including native diagnostics. `result_manifest.json` records each original SHA256, source-root mapping and reuse status. Offline/oracle fields in these logs are outer diagnostics, not policy inputs or replacement success criteria.
- `baseline_initializations.json.gz`: original V10 initialization/task-text extract plus source-file hashes.
- `protocols/`: sanitized copies of the six protocol/progress/completion receipts; relative source identifiers preserve the history of invalid attempts without exposing machine locations.
- `FINAL_VERIFICATION.json`: final complete A1/W1b audit and original result hashes. The portable verification implementation is `../../scripts/verify_a1.py`.
- `paired_results.csv` / `paired_changes.json`: terminal comparisons including W1b; no claim of full native W1b reproduction from this update alone.
- `environment_snapshot.json`: simulator revision and source-file hashes; machine identity is removed.
- `upstream_mode_evidence/`: three unmodified MIT-licensed upstream files, pinned to the tested commit; the existing `../../third_party/A3_SIM_LICENSE.txt` applies.

## Privacy-preserving evidence export

The 79 runtime files and all 240 native terminal records are byte-preserved. Protocols, final-audit metadata and the baseline initialization extract are sanitized copies: absolute source paths are replaced by relative identifiers, and machine names and machine-specific library paths are removed. `sanitization_manifest.json` records original and public hashes separately. No scores, initializations, task instructions or code are changed.

`native_results.tar.gz#member` locates a bundled archive member. `source_runs/rootN` and `unbundled_sources/` are provenance identifiers, not executable paths or promises of additional files. The actual code is under `candidates/a1/` from the repository root. The offline checks need only this repository and Python; rerunning simulations needs the public upstream dependency described below. Original private evidence remains unchanged.

## Offline verification

From the repository root:

```bash
python3 scripts/verify_release.py
python3 scripts/verify_retest.py
python3 scripts/verify_a1.py
python3 scripts/verify_a1_privacy.py
```

These commands do not run physics. They check the frozen V10 release, the baseline replication, A1 code/results, initialization equivalence and the reported comparisons. They do not recreate long trajectories or independently attest to data collection on the source server.

## Running A1 again

Use the pinned official simulator and environment described in the repository Quickstart, with CPU physics and OSMesa software rendering. The portable wrapper uses native random first-grasp sampling, the unchanged bundled evaluator and fresh output directories:

```bash
export SIM_REPO=../a3_dual_arm_sim
python3 scripts/run_a1_grid.py --grid grid24000 --output outputs/a1_grid24000
python3 scripts/run_a1_grid.py --grid grid620xxx --output outputs/a1_grid620xxx
```

This reproduces the task grid, not the original recorder's large diagnostics. The wrapper has been checked without starting a new simulation during publication. Freeze any next candidate before choosing genuinely unseen seeds; the present two collections cannot establish A1 generalization.
