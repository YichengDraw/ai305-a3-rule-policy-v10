"""Record an independent development episode; traces stay outside the policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def numeric_json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("2x5", "2x10"), default="2x5")
    parser.add_argument("--profile", choices=("basic", "medium", "advanced"), default="basic")
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--first-grasp", default="5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from a3_dual_arm_sim.paths import resource_root
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic
    from policy import make_policy

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["A3_POLICY_TRACE_DIR"] = str(output / "policy_trace")
    if args.task == "2x10":
        settings, _ = load_cookie_2x10_settings(resource_root() / "configs/randomization_2x10.yaml")
    else:
        settings = load_randomization_config(resource_root() / "configs/randomization.yaml")
    settings["source_column"] = "1"
    extra = {"first_grasp": args.first_grasp} if args.task == "2x10" else {}
    benchmark = make_cookie_benchmark(
        task=args.task, profile=args.profile, source_column="1",
        randomization_settings=settings, max_steps=args.max_steps,
        success_hold_steps=5, strict_policy_completion=False, **extra,
    )
    policy = make_policy()
    env = benchmark.create_env(render_cameras=True)
    trace = EpisodeDiagnostic(output / "diagnostic")
    try:
        result = benchmark.run_episode(policy, seed=args.seed, env=env, diagnostic=trace)
        payload = result.to_dict()
        payload.update(policy_reports=policy.controller.batch_reports,
                       policy_transitions=policy.controller.transition_history,
                       policy_failure_reason=policy.controller.failure_reason,
                       policy_recovery_events=getattr(policy.controller, "recovery_events", []))
        (output / "result.json").write_text(json.dumps(payload, indent=2, default=numeric_json) + "\n")
        print(json.dumps({key: payload[key] for key in (
            "score", "max_score", "success", "steps", "wall_seconds", "phase",
            "policy_failure_reason", "policy_reports", "policy_transitions",
        )}, indent=2, default=numeric_json), flush=True)
    finally:
        trace.close()
        env.close()
        policy.close()


if __name__ == "__main__":
    main()
