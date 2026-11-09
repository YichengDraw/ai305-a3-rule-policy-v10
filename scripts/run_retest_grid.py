#!/usr/bin/env python3
"""Run frozen V10 on a pre-registered random seed sample, one process per episode.

All simulation/scoring stays in the published evaluator. This wrapper only
shards independent episodes across CPU workers and merges the untouched
terminal rows afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation" / "evaluate_policy.py"
POLICY = ROOT / "frozen_policy_v10"
TASKS = ("2x5", "2x10")
PROFILES = ("basic", "medium", "advanced")
BUDGETS = {"2x5": 500, "2x10": 3200}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_ok(directory: Path) -> bool:
    summary = directory / "summary.json"
    episodes = directory / "episodes.jsonl"
    if not (summary.is_file() and episodes.is_file()):
        return False
    try:
        if json.loads(summary.read_text(encoding="utf-8")).get("status") != "completed":
            return False
        return len([x for x in episodes.read_text(encoding="utf-8").splitlines() if x.strip()]) == 1
    except Exception:
        return False


def run_job(job, output: Path, logs: Path, status_path: Path):
    task, profile, seed = job
    name = f"{task}_{profile}_{seed}"
    directory = output / "shards" / name
    log = logs / f"{name}.log"
    if shard_ok(directory):
        return dict(job=name, status="resumed", returncode=0)
    last_rc = None
    for attempt in (1, 2):
        if attempt > 1 and directory.exists():
            shutil.rmtree(directory)
        command = [
            sys.executable, str(EVAL),
            "--policy", "policy:make_policy_b",
            "--policy-code", str(POLICY),
            "--output", str(directory),
            "--tasks", task,
            "--profiles", profile,
            "--episodes", "1",
            "--seed-start", str(seed),
            "--max-steps", str(BUDGETS["2x5"]),
            "--max-steps-2x10", str(BUDGETS["2x10"]),
            "--source-column", "1",
            "--success-hold-steps", "5",
            "--render-cameras",
        ]
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== attempt {attempt} {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===\n")
            stream.flush()
            process = subprocess.run(command, cwd=EVAL.parent, env=os.environ.copy(),
                                     stdout=stream, stderr=subprocess.STDOUT)
        last_rc = process.returncode
        if process.returncode == 0 and shard_ok(directory):
            return dict(job=name, status="completed", returncode=0, attempt=attempt)
        if process.returncode == 0:
            last_rc = 90  # evaluator exited without a complete one-row shard
    return dict(job=name, status="failed", returncode=last_rc)


def _task_lines(cells):
    out = []
    for task in ("2x5", "2x10"):
        selected = [c for c in cells if c["task"] == task]
        total = sum(c["total_score"] for c in selected)
        capacity = sum(c["max_possible_score"] for c in selected)
        successes = sum(c["successes"] for c in selected)
        episodes = sum(c["completed"] for c in selected)
        out.append(f"按任务汇总：{task} = {total}/{capacity} ({total/capacity:.2%}) 完成率，"
                   f"{successes}/{episodes} ({successes/episodes:.2%}) 成功率。")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))["seeds"]
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise SystemExit("Expected 20 distinct seeds")
    output = args.output.resolve()
    if (output / "complete.json").exists():
        print("already complete:", output)
        return
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    status_path = output / "status.json"
    jobs = [(task, profile, seed) for task in ("2x10", "2x5")
            for profile in PROFILES for seed in seeds]
    started = time.time()
    results = []
    status_path.write_text(json.dumps({"state": "running", "started": started,
                                       "planned": len(jobs), "finished": 0}, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, output, logs, status_path): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
            done = sum(1 for r in results if r["status"] in ("completed", "resumed"))
            status_path.write_text(json.dumps({
                "state": "running", "started": started, "planned": len(jobs),
                "finished": len(results), "completed": done,
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "last": results[-1], "updated": time.time(),
            }, indent=2) + "\n")
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        (output / "failed.json").write_text(json.dumps(failed, indent=2) + "\n")
        raise SystemExit(f"{len(failed)} shard(s) failed; rerun with the same output to resume")

    # Merge through the published evaluator's own summarize().
    sys.path.insert(0, str(EVAL.parent))
    from evaluate_policy import summarize  # noqa: E402
    from batch_metrics import diagnostic_summary  # noqa: E402

    rows, protocols = [], []
    shard_dirs = sorted((output / "shards").iterdir(), key=lambda p: p.name)
    for directory in shard_dirs:
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        protocols.append(protocol)
        for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 120:
        raise SystemExit(f"Expected 120 terminal rows, found {len(rows)}")
    fingerprints = {json.dumps(p["code_sha256"], sort_keys=True) for p in protocols}
    commits = {p["source_commit"] for p in protocols}
    policies = {(p["policy"], p["policy_code_directory"]) for p in protocols}
    if len(fingerprints) != 1 or len(commits) != 1 or len(policies) != 1:
        raise SystemExit("Shard provenance differs")
    for p in protocols:
        if p["strict_policy_completion"] or p["success_hold_steps"] != 5:
            raise SystemExit("Unexpected success contract")
        if p["source_column"] != "1" or not p["render_cameras"]:
            raise SystemExit("Unexpected observation/source configuration")
        if p["max_steps_by_task"] != BUDGETS:
            raise SystemExit("Unexpected step budgets")
    expected = {(task, profile, seed) for task in TASKS for profile in PROFILES for seed in seeds}
    actual = [(r["task"], r["profile"], r["seed"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise SystemExit("Incomplete or duplicate episode set")
    for r in rows:
        if r["max_steps"] != BUDGETS[r["task"]] or r.get("source_column") != 1:
            raise SystemExit("Unexpected per-row budget/column")
        if r["score"] != r["cookies_in_target"]:
            raise SystemExit("Official score/count mismatch")
    rows.sort(key=lambda r: (TASKS.index(r["task"]), PROFILES.index(r["profile"]), seeds.index(r["seed"])))
    cells, overall = summarize(rows, TASKS, PROFILES, 20)
    finished = time.time()
    summary = {
        "status": "completed",
        "protocol": {
            "policy": "policy:make_policy_b",
            "policy_release": "YichengDraw/ai305-a3-rule-policy-v10",
            "policy_release_commit": "df9fe79269050c3082ee8dd4745072de41e1427f",
            "official_simulator_commit": next(iter(commits)),
            "source_column": 1,
            "seeds": seeds,
            "seed_sampling": json.loads(args.seeds.read_text(encoding="utf-8")),
            "episodes_per_cell": 20,
            "tasks": list(TASKS),
            "profiles": list(PROFILES),
            "max_steps_by_task": BUDGETS,
            "success_hold_steps": 5,
            "strict_policy_completion": False,
            "render_cameras": True,
            "backend": "CPU OSMesa/llvmpipe",
            "host": socket.gethostname(),
            "workers": args.workers,
            "wall_seconds": finished - started,
            "code_sha256": protocols[0]["code_sha256"],
        },
        "cells": cells,
        "overall": overall,
        "batch_diagnostics": diagnostic_summary(rows),
        "sources": [str(p.resolve()) for p in shard_dirs],
    }
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    lines = [
        "# 冻结 V10：随机新种子复测（source_column=1）", "",
        f"种子：{', '.join(map(str, seeds))}（{summary['protocol']['seed_sampling']['method']}）",
        "2×5 500 步；2×10 3200 步；官方非严格结算；success hold=5。", "",
        "| 任务 | profile | 平均完成率 | 数量（片） | 平均成功率 |",
        "|---|---|---:|---:|---:|",
    ]
    for c in cells:
        lines.append(f"| {c['task']} | {c['profile']} | {c['completion_rate']:.1%} "
                     f"| {c['total_score']}/{c['max_possible_score']} "
                     f"| {c['success_rate']:.1%} ({c['successes']}/{c['completed']}) |")
    lines += [
        "",
        f"合计：{overall['macro_completion_rate']:.2%}（数量加权 {overall['micro_completion_rate']:.2%}），"
        f"{overall['total_score']}/{overall['max_possible_score']}，"
        f"{overall['success_rate']:.2%}（{overall['successes']}/{overall['completed']}）。",
    ]
    lines += _task_lines(cells)
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "complete.json").write_text(json.dumps({
        "state": "completed", "started": started, "finished": finished,
        "wall_seconds": finished - started, "episodes": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "overall": overall, "wall_seconds": finished - started},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
