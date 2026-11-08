"""Evaluate a zero-argument module:factory policy through the official benchmark.

Run with the official simulator environment, for example:
    MUJOCO_GL=egl python evaluate_policy.py --policy my_policy:make_policy \
        --output outputs/my_policy_grid
Add --render-cameras when the policy reads RGB observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path


TASKS = ("2x5", "2x10")
PROFILES = ("basic", "medium", "advanced")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def numeric_json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, tasks, profiles, episodes):
    """Completion uses final counts; success uses the unchanged official flag."""
    cells = []
    for task in tasks:
        for profile in profiles:
            selected = [row for row in rows if row["task"] == task and row["profile"] == profile]
            counts = [row["cookies_in_target"] for row in selected]
            capacity = 10 if task == "2x5" else 20
            cells.append({
                "task": task,
                "profile": profile,
                "completed": len(selected),
                "planned": episodes,
                "successes": sum(bool(row["success"]) for row in selected),
                "success_rate": ratio(sum(bool(row["success"]) for row in selected), len(selected)),
                "completion_rate": ratio(sum(counts), capacity * len(selected)),
                "mean_score": ratio(sum(counts), len(selected)),
                "total_score": sum(counts),
                "max_possible_score": capacity * len(selected),
                "score_distribution": dict(sorted(Counter(counts).items())),
                "mean_steps": ratio(sum(row["steps"] for row in selected), len(selected)),
                "mean_wall_seconds": ratio(sum(row["wall_seconds"] for row in selected), len(selected)),
            })
    present = [cell for cell in cells if cell["completed"]]
    total_score = sum(row["cookies_in_target"] for row in rows)
    total_capacity = sum(row["max_score"] for row in rows)
    overall = {
        "completed": len(rows),
        "planned": len(tasks) * len(profiles) * episodes,
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": ratio(sum(bool(row["success"]) for row in rows), len(rows)),
        "total_score": total_score,
        "max_possible_score": total_capacity,
        "micro_completion_rate": ratio(total_score, total_capacity),
        "macro_completion_rate": ratio(sum(cell["completion_rate"] for cell in present), len(present)),
        "mean_episode_completion_rate": ratio(
            sum(row["cookies_in_target"] / row["max_score"] for row in rows), len(rows)
        ),
        "cells_with_results": len(present),
    }
    return cells, overall


def save_summary(output, rows, args, status, error=None):
    cells, overall = summarize(rows, args.tasks, args.profiles, args.episodes)
    summary = {"status": status, "results": cells, "overall": overall}
    if error is not None:
        summary["error"] = error
    atomic_json(output / "summary.json", summary)

    def percent(value):
        return "待评测" if value is None else f"{100 * value:.2f}%"

    lines = [
        f"状态：{status}；已评测 {overall['completed']}/{overall['planned']} 回合。",
        "",
        "完成率 = 终态入盒数 / 要求数量；成功率使用官方 success 标志。",
        "",
        "| 任务 | 难度 | 已评测/计划 | 完成率 | 成功率 | 成功次数 | 平均入盒 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        mean = "待评测" if cell["mean_score"] is None else f"{cell['mean_score']:.2f}"
        lines.append(
            f"| {cell['task']} | {cell['profile']} | {cell['completed']}/{cell['planned']} "
            f"| {percent(cell['completion_rate'])} | {percent(cell['success_rate'])} "
            f"| {cell['successes']} | {mean} |"
        )
    lines += [
        "",
        f"总成功率：{percent(overall['success_rate'])}（{overall['successes']}/{overall['completed']}）。",
        f"微平均完成率：{percent(overall['micro_completion_rate'])}（所有入盒数 / 所有要求数量）。",
        f"宏平均完成率：{percent(overall['macro_completion_rate'])}（已有结果的各任务难度格等权平均）。",
        f"逐回合平均完成率：{percent(overall['mean_episode_completion_rate'])}。",
        "",
        "六格各完成相同回合数时，宏平均与逐回合平均相同。未完成评测的汇总使用已保存回合。",
    ]
    if error is not None:
        lines += ["", f"错误：{error}"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def code_fingerprints(policy_spec):
    """Only fingerprint the small Python entry files; never scan model assets."""
    module, _factory = policy_spec.rsplit(":", 1)
    found = importlib.util.find_spec(module)
    if found is None:
        raise ModuleNotFoundError(module)
    paths = {"evaluation_script": Path(__file__)}
    if found.origin and Path(found.origin).is_file():
        paths["policy_module"] = Path(found.origin)
        for sibling in ("perception.py", "rule_controller.py"):
            dependency = Path(found.origin).parent / sibling
            if dependency.is_file():
                paths[sibling] = dependency
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Importable module:zero_argument_factory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task/profile")
    parser.add_argument("--seed-start", type=int, default=13000)
    parser.add_argument("--max-steps", type=int, default=500, help="2x5 control-step budget")
    parser.add_argument("--max-steps-2x10", type=int, default=3200)
    parser.add_argument("--source-column", choices=("1", "2", "3", "4", "random"), default="1")
    parser.add_argument("--prompt-style-2x5", choices=("training", "task", "short"), default="training")
    parser.add_argument("--prompt-style-2x10", choices=("training", "task", "short"), default="task")
    parser.add_argument("--first-grasp-2x10", default=None, help="Override official configured first grasp")
    parser.add_argument("--randomization-config-2x5", type=Path)
    parser.add_argument("--randomization-config-2x10", type=Path)
    parser.add_argument("--success-hold-steps", type=int, default=None,
                        help="Default: 5 in non-strict mode, 20 in strict mode")
    parser.add_argument("--strict-policy-completion", action="store_true")
    parser.add_argument("--render-cameras", action="store_true", help="Render all three RGB cameras")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print protocol without running simulation")
    args = parser.parse_args(argv)
    if args.success_hold_steps is None:
        args.success_hold_steps = 20 if args.strict_policy_completion else 5
    if ":" not in args.policy or not all(args.policy.rsplit(":", 1)):
        parser.error("--policy must be module:factory")
    if min(args.episodes, args.max_steps, args.max_steps_2x10, args.success_hold_steps) < 1:
        parser.error("Episode, step, and hold counts must be positive")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.profiles)) != len(args.profiles):
        parser.error("Tasks and profiles must not contain duplicates")

    from a3_dual_arm_sim.paths import resource_root
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.tasks.cookie_2x10_settings import load_cookie_2x10_settings
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from a3_dual_arm_sim.workflows.policy_adapters import make_policy_adapter
    import a3_dual_arm_sim.workflows.benchmark as benchmark_module

    root = resource_root()
    settings5 = load_randomization_config(args.randomization_config_2x5 or root / "configs/randomization.yaml")
    settings10, first_grasp = load_cookie_2x10_settings(
        args.randomization_config_2x10 or root / "configs/randomization_2x10.yaml"
    )
    settings = {"2x5": settings5, "2x10": settings10}
    for value in settings.values():
        value["source_column"] = args.source_column
    first_grasp = args.first_grasp_2x10 if args.first_grasp_2x10 is not None else first_grasp
    budgets = {"2x5": args.max_steps, "2x10": args.max_steps_2x10}
    prompts = {"2x5": args.prompt_style_2x5, "2x10": args.prompt_style_2x10}
    source_root = Path(benchmark_module.__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, capture_output=True, check=False
    )
    protocol = {
        "policy": args.policy,
        "code_sha256": code_fingerprints(args.policy),
        "source_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "tasks": args.tasks,
        "profiles": args.profiles,
        "episodes_per_cell": args.episodes,
        "seed_start": args.seed_start,
        "max_steps_by_task": budgets,
        "source_column": args.source_column,
        "prompt_styles": prompts,
        "first_grasp_2x10": first_grasp,
        "randomization_settings": settings,
        "strict_policy_completion": args.strict_policy_completion,
        "success_hold_steps": args.success_hold_steps,
        "render_cameras": args.render_cameras,
        "metrics": {
            "completion_rate": "sum(final cookies_in_target) / sum(required cookies)",
            "success_rate": "sum(official success) / number of evaluated episodes",
            "macro_completion_rate": "equal-weight mean of task/profile completion rates with results",
        },
    }
    if args.dry_run:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0

    output = args.output.expanduser().resolve()
    if args.resume:
        if json.loads((output / "protocol.json").read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Protocol differs from saved run; choose a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "protocol.json", protocol)
    rows_path = output / "episodes.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()] if rows_path.exists() else []
    done = {(row["task"], row["profile"], row["seed"]) for row in rows}
    expected = {
        (task, profile, seed) for task in args.tasks for profile in args.profiles
        for seed in range(args.seed_start, args.seed_start + args.episodes)
    }
    if len(done) != len(rows) or not done.issubset(expected):
        raise RuntimeError("Saved episodes contain duplicate or out-of-protocol keys")

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (output / "pid").write_text(str(os.getpid()), encoding="utf-8")
    save_summary(output, rows, args, "running")
    policy = None
    try:
        if done != expected:
            policy = make_policy_adapter(args.policy)
        for task in args.tasks:
            for profile in args.profiles:
                pending = [seed for seed in range(args.seed_start, args.seed_start + args.episodes)
                           if (task, profile, seed) not in done]
                if not pending:
                    continue
                extra = {"first_grasp": first_grasp} if task == "2x10" else {}
                benchmark = make_cookie_benchmark(
                    task=task, profile=profile, source_column=args.source_column,
                    prompt_style=prompts[task], randomization_settings=dict(settings[task], profile=profile),
                    max_steps=budgets[task], strict_policy_completion=args.strict_policy_completion,
                    success_hold_steps=args.success_hold_steps, **extra,
                )
                env = benchmark.create_env(render_cameras=args.render_cameras)
                try:
                    for seed in pending:
                        current = {"task": task, "profile": profile, "seed": seed, "max_steps": budgets[task]}
                        atomic_json(output / "current.json", current)
                        print(f"START {task} {profile} seed={seed}", flush=True)
                        result = benchmark.run_episode(
                            policy=policy, seed=seed, episode_idx=seed - args.seed_start + 1, env=env
                        )
                        row = {
                            **asdict(result), **current, "benchmark": task,
                            "task_instruction": benchmark._episode_task(result.source_column, seed),
                            "prompt_style": prompts[task], "strict_policy_completion": args.strict_policy_completion,
                            "success_hold_steps": args.success_hold_steps,
                        }
                        controller = getattr(policy, "controller", None)
                        if controller is not None:
                            row["policy_reports"] = getattr(controller, "batch_reports", [])
                            row["policy_transitions"] = getattr(controller, "transition_history", [])
                            row["policy_failure_reason"] = getattr(controller, "failure_reason", None)
                            row["policy_recovery_events"] = getattr(controller, "recovery_events", [])
                        with rows_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, ensure_ascii=False, default=numeric_json) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        rows.append(row)
                        done.add((task, profile, seed))
                        save_summary(output, rows, args, "running")
                        print(
                            f"DONE {task} {profile} seed={seed} final={result.cookies_in_target}/{result.max_score} "
                            f"success={result.success} steps={result.steps} wall={result.wall_seconds}s", flush=True
                        )
                finally:
                    env.close()
        save_summary(output, rows, args, "completed")
        print("ALL SELECTED EPISODES COMPLETE", flush=True)
    except KeyboardInterrupt:
        save_summary(output, rows, args, "interrupted")
        return 130
    except Exception as exc:
        save_summary(output, rows, args, "error", repr(exc))
        raise
    finally:
        if policy is not None and hasattr(policy, "close"):
            policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
