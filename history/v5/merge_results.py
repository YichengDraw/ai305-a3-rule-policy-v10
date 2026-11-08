"""Merge fixed disjoint evaluation shards and print measured task metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_policy import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["2x5", "2x10"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=13000)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    profiles = ["basic", "medium", "advanced"]
    rows = []
    protocols = []
    for root in args.inputs:
        protocol = json.loads((root / "protocol.json").read_text())
        if protocol["strict_policy_completion"] or protocol["success_hold_steps"] != 5:
            raise ValueError(f"Unexpected success contract: {root}")
        if protocol["source_column"] != "1" or not protocol["render_cameras"]:
            raise ValueError(f"Unexpected observation/source configuration: {root}")
        protocols.append(protocol)
        for line in (root / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["max_steps"] != args.max_steps:
                raise ValueError(f"Unexpected step budget in {root}")
            rows.append(row)
    fingerprints = {json.dumps(p["code_sha256"], sort_keys=True) for p in protocols}
    commits = {p["source_commit"] for p in protocols}
    if len(fingerprints) != 1 or len(commits) != 1:
        raise ValueError("Shards use different evaluator/policy entry code or simulator commits")
    expected = {(task, profile, seed) for task in args.tasks for profile in profiles
                for seed in range(args.seed_start, args.seed_start + args.episodes)}
    actual = [(row["task"], row["profile"], row["seed"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Incomplete or duplicate evaluation: {len(actual)}/{len(expected)}")
    rows.sort(key=lambda r: (args.tasks.index(r["task"]), profiles.index(r["profile"]), r["seed"]))
    cells, overall = summarize(rows, args.tasks, profiles, args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    payload = {"status": "completed", "step_budget": args.max_steps, "source_column": 1,
               "observation": "three RGB cameras, public robot state and force feedback",
               "cells": cells, "overall": overall,
               "sources": [str(root.resolve()) for root in args.inputs]}
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# 视觉与力反馈规则策略：{args.max_steps} 步评测", "",
             "三档各20个种子，来源列1，非严格官方结算。完成率按终态入盒比例，成功率按官方success字段。", "",
             "| 任务 | 难度 | 回合数 | 完成率 | 成功率 | 平均入盒 | 完整成功 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(f"| {c['task']} | {c['profile']} | {c['completed']} | "
                     f"{c['completion_rate']:.2%} | {c['success_rate']:.2%} | "
                     f"{c['mean_score']:.2f} | {c['successes']}/{c['completed']} |")
    lines += ["", f"按回合等权的总体完成率：{overall['mean_episode_completion_rate']:.2%}。",
              f"总体成功率：{overall['success_rate']:.2%}（{overall['successes']}/{overall['completed']}）。",
              f"按饼干总数加权的完成率：{overall['micro_completion_rate']:.2%}"
              f"（{overall['total_score']}/{overall['max_possible_score']}）。", "",
              "运行时policy通过标准观测接口读取图像、机器人状态与力反馈；评测器单独读取真值计算成绩。",
              "种子为13000–13019；开发运行使用12000系列种子。"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
