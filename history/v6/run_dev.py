"""Record an independent development episode; traces stay outside the policy."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import importlib
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
    parser.add_argument("--code-dir", "--policy-code", "--code-directory", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--policy", default="policy:make_policy")
    parser.add_argument("--oracle-source-target", action="store_true",
                        help="Non-formal diagnostic: replace only source-group and target-pose estimates with truth")
    parser.add_argument("--no-video", action="store_true", help="Keep numerical measurements without full camera videos")
    args = parser.parse_args()

    from a3_dual_arm_sim.paths import resource_root
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from a3_dual_arm_sim.workflows.diagnostics import EpisodeDiagnostic
    from diagnostic_support import DiagnosticPolicy, diagnostic_summary

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["A3_POLICY_TRACE_DIR"] = str(output / "policy_trace")
    os.environ["A3_POLICY_MAX_STEPS"] = str(args.max_steps)
    code = args.code_dir.resolve()
    sys.path.insert(0, str(code))
    module, factory = args.policy.rsplit(":", 1)
    make_policy = getattr(importlib.import_module(module), factory)
    protocol = {"scope": "development_oracle" if args.oracle_source_target else "development_rgb_force",
                "task": args.task, "profile": args.profile, "seed": args.seed,
                "max_steps": args.max_steps, "first_grasp": args.first_grasp,
                "policy_code": str(code), "policy": args.policy,
                "oracle_source_target": args.oracle_source_target,
                "oracle_changes": ["selected source-group pose/width", "target-bin pose"] if args.oracle_source_target else [],
                "controller_changed_by_evaluator": False,
                "offline_ground_truth_audit": True, "argv": sys.argv,
                "started_unix": time.time()}
    (output / "development_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
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
    policy = DiagnosticPolicy(make_policy(), oracle_source_target=args.oracle_source_target,
                              audit=True, output=output / "offline_truth")
    env = benchmark.create_env(render_cameras=True)
    trace = None if args.no_video else EpisodeDiagnostic(output / "diagnostic")
    try:
        result = benchmark.run_episode(policy, seed=args.seed, env=env, diagnostic=trace)
        payload = result.to_dict()
        payload.update(policy_reports=policy.controller.batch_reports,
                       policy_transitions=policy.controller.transition_history,
                       policy_failure_reason=policy.controller.failure_reason,
                       policy_recovery_events=getattr(policy.controller, "recovery_events", []),
                       policy_batch_ledger=getattr(policy.controller, "batch_ledger", []),
                       policy_stage_reports=getattr(policy.controller, "stage_reports", []),
                       policy_current_report=getattr(policy.controller, "current_report", getattr(policy.controller, "_report", None)),
                       offline_batch_audit=policy.finalize_audit(),
                       oracle_source_target=args.oracle_source_target,
                       development_scope=True, task=args.task, profile=args.profile)
        payload["diagnostic_summary"] = diagnostic_summary([payload])
        (output / "result.json").write_text(json.dumps(payload, indent=2, default=numeric_json) + "\n")
        print(json.dumps({key: payload[key] for key in (
            "score", "max_score", "success", "steps", "wall_seconds", "phase",
            "policy_failure_reason", "policy_reports", "policy_transitions",
        )}, indent=2, default=numeric_json), flush=True)
    finally:
        if trace is not None:
            trace.close()
        env.close()
        policy.close()


if __name__ == "__main__":
    main()
