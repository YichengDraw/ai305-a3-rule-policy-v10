# Why fixed sampled n is an equivalent replay here

This argument applies only when actual first-grasp count, complete recorded initialization and task text match. All 240 pairs are checked by `scripts/verify_a1.py`; the only differences are 60 old-grid `first_grasp_mode` metadata entries.

Source: official simulator commit `0729068ee1e75a562cfbc4f2c83ab9539cf75081`. The three unmodified source files under `upstream_mode_evidence/` were fetched from that exact public revision and their SHA256 values match `environment_snapshot.json` from the actual run.

1. `tasks/cookie_2x10_plan.py`, `Cookie2x10Plan.from_seed`: the native first count comes from `Random(f"a3-cookie-2x10-first-grasp-v1:{seed}").randrange(10)`, a stream separate from scene randomization. Fixed mode substitutes the same count when that value was supplied.
2. `batch_sizes`, `target_rows` and `prompt` depend on `first_grasp`, not `mode`. The mode is emitted by `metadata()` as `first_grasp_mode`.
3. `workflows/cookie_2x10_execution.py`: `_episode_task` uses that prompt. `_episode_policy` passes first-grasp mode to an expert only for the named `variable_grasp` expert; A1 uses `policy:make_policy_b` and does not enter that branch. `_episode_metadata` appends metadata separately.
4. `workflows/episode_execution.py`: reset uses `EpisodeContext(seed, task, action_mode)` and actions receive observation/task. The mode is not an added A1 policy input.
5. The 79 published A1 runtime files contain no `first_grasp_mode` reference. Their seed references are diagnostic trace filenames, not seed-based action selection. The external recorder may log simulator truth for offline inspection; the source protocols explicitly record `runtime_truth_access=false`, and this does not change native scoring.

This proves a narrow equivalence of the retained replay configuration. It does not turn already-seen scenes into an independent holdout, or justify ignoring any other initialization difference.
