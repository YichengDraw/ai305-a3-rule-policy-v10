"""Merge disjoint shards with task-specific budgets and honest batch metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_metrics import diagnostic_summary
from evaluate_policy import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["2x5", "2x10"])
    parser.add_argument("--profiles", nargs="+", default=["basic", "medium", "advanced"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=16000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-steps-2x10", type=int, default=3200)
    args = parser.parse_args()
    budgets = {"2x5": args.max_steps, "2x10": args.max_steps_2x10}
    rows, protocols = [], []
    for root in args.inputs:
        protocol = json.loads((root / "protocol.json").read_text())
        if protocol["strict_policy_completion"] or protocol["success_hold_steps"] != 5:
            raise ValueError(f"Unexpected success contract: {root}")
        if protocol["source_column"] != "1" or not protocol["render_cameras"]:
            raise ValueError(f"Unexpected observation/source configuration: {root}")
        if json.loads((root / "summary.json").read_text())["status"] != "completed":
            raise ValueError(f"Incomplete shard: {root}")
        protocols.append(protocol)
        for line in (root / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["max_steps"] != budgets[row["task"]]:
                raise ValueError(f"Unexpected step budget in {root}")
            if row.get("oracle_source_target") or row.get("development_scope"):
                raise ValueError(f"Development/oracle result cannot enter formal merge: {root}")
            if row.get("source_column") != 1:
                raise ValueError(f"Unexpected episode source column: {root}")
            if row["score"] != row["cookies_in_target"]:
                raise ValueError(f"Official score/count mismatch: {root}")
            rows.append(row)
    fingerprints = {json.dumps(p["code_sha256"], sort_keys=True) for p in protocols}
    commits = {p["source_commit"] for p in protocols}
    policies = {(p["policy"], p.get("policy_code_directory")) for p in protocols}
    audit_contracts = {p.get("offline_ground_truth_audit", False) for p in protocols}
    if len(fingerprints) != 1 or len(commits) != 1 or len(policies) != 1 or len(audit_contracts) != 1:
        raise ValueError("Shards differ in policy/evaluator code, simulator commit, or audit contract")
    expected = {(task, profile, seed) for task in args.tasks for profile in args.profiles
                for seed in range(args.seed_start, args.seed_start + args.episodes)}
    actual = [(row["task"], row["profile"], row["seed"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Incomplete or duplicate evaluation: {len(actual)}/{len(expected)}")
    rows.sort(key=lambda r: (args.tasks.index(r["task"]), args.profiles.index(r["profile"]), r["seed"]))
    cells, overall = summarize(rows, args.tasks, args.profiles, args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    payload = dict(status="completed", max_steps_by_task=budgets, source_column=1,
                   observation="three RGB cameras, public robot state and force feedback",
                   cells=cells, overall=overall, batch_diagnostics=diagnostic_summary(rows),
                   sources=[str(root.resolve()) for root in args.inputs],
                   code_sha256=protocols[0]["code_sha256"], source_commit=next(iter(commits)))
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 视觉与力反馈规则策略评测", "",
             f"2×5 {args.max_steps}步、2×10 {args.max_steps_2x10}步；每格{args.episodes}个种子，来源列1，非严格官方结算。",
             "完成率按终态入盒比例，成功率按官方success字段。", "",
             "| 任务 | 难度 | 回合数 | 完成率 | 成功率 | 平均入盒 | 完整成功 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for cell in cells:
        lines.append(f"| {cell['task']} | {cell['profile']} | {cell['completed']} | "
                     f"{cell['completion_rate']:.2%} | {cell['success_rate']:.2%} | "
                     f"{cell['mean_score']:.2f} | {cell['successes']}/{cell['completed']} |")
    lines += ["", f"逐回合等权完成率：{overall['mean_episode_completion_rate']:.2%}。",
              f"成功率：{overall['success_rate']:.2%}（{overall['successes']}/{overall['completed']}）。",
              f"数量加权完成率：{overall['micro_completion_rate']:.2%}（{overall['total_score']}/{overall['max_possible_score']}）。",
              "", f"种子{args.seed_start}–{args.seed_start+args.episodes-1}；所有失败、超时和重试计入。",
              "运行policy只读取标准观测；离线真值审计独立保存于offline_batch_audit字段。"]
    diag = payload["batch_diagnostics"]
    lines += ["", "| 离线批次诊断 | 通过/有观测 | 比率 |", "|---|---:|---:|"]
    labels = (("有效抓取", "offline_truth_grasp"), ("有效放置", "offline_truth_placement"),
              ("最终仍有效放置", "offline_truth_final_retention"),
              ("此前各批有效条件下的后续批次放置", "subsequent_batch_placement_given_previous_valid"))
    for label, key in labels:
        item = diag[key]
        text = "记录不足" if item["rate"] is None else f"{item['rate']:.2%}"
        lines.append(f"| {label} | {item['passed']}/{item['observed']} | {text} |")
    recovery = diag["recovery_success_rate"]
    lines += ["", f"恢复事件{diag['recovery_events']}次，具有明确成功标注{diag['explicitly_labeled_recovery_observations']}次；"
              + (f"标注样本恢复成功率{recovery:.2%}。" if recovery is not None else "恢复成功率记录不足。"),
              f"控制器DONE且官方失败：{diag['done_but_official_failure']}/{diag['done_episodes']}次DONE。",
              "批次真值诊断在阶段切换与每5控制步采样；放置检查包含入盒几何、落座、静止和释放，采样判据与官方整局判定分别保留。"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "cells", "overall", "batch_diagnostics")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
