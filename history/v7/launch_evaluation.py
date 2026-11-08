"""Run disjoint formal seed shards with task-specific budgets, then merge.

The caller decides when a policy is frozen and launches this program. Defaults
are the registered new test set: 16000--16019, 2x5/500 and 2x10/3200.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-code", "--code-directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--policy", default="policy:make_policy")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--egl-devices", type=int, nargs="+", default=[4])
    parser.add_argument("--seed-start", type=int, default=16000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", choices=("2x5", "2x10"), default=["2x5", "2x10"])
    parser.add_argument("--profiles", nargs="+", choices=("basic", "medium", "advanced"), default=["basic", "medium", "advanced"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-steps-2x10", type=int, default=3200)
    parser.add_argument("--default-budgets-only", action="store_true", help="Compatibility flag: default is already only 500/3200")
    parser.add_argument("--extra-2x10-500", action="store_true", help="Optional separate 500-step 2x10 table")
    parser.add_argument("--audit-ground-truth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.episodes, args.shard_size, args.max_steps, args.max_steps_2x10) < 1:
        parser.error("worker, episode, shard, and step counts must be positive")
    if args.default_budgets_only and args.extra_2x10_500:
        parser.error("--default-budgets-only conflicts with --extra-2x10-500")
    root, code, policy_code = args.output.resolve(), Path(__file__).resolve().parent, args.policy_code.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    environment = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = []
    budgets = {"2x5": args.max_steps, "2x10": args.max_steps_2x10}
    groups = [("default", args.tasks, budgets)]
    if args.extra_2x10_500:
        groups.append(("extra_2x10_500", ["2x10"], {"2x5": args.max_steps, "2x10": 500}))
    for group, tasks, group_budgets in groups:
        for task in tasks:
            for profile in args.profiles:
                for offset in range(0, args.episodes, args.shard_size):
                    seed = args.seed_start + offset
                    name = f"{task}_{profile}_{seed}"
                    destination = root / group / name
                    jobs.append(dict(name=name, group=group, budget=group_budgets[task], task=task,
                                     profile=profile, seed=seed, episodes=min(args.shard_size, args.episodes-offset),
                                     output=str(destination),
                                     egl_device=args.egl_devices[len(jobs) % len(args.egl_devices)]))
    protocol = {"evaluator_directory": str(code), "policy_code_directory": str(policy_code),
                "policy": args.policy, "tasks": args.tasks, "profiles": args.profiles,
                "seed_start": args.seed_start, "episodes_per_cell": args.episodes,
                "max_steps_by_task": budgets, "offline_ground_truth_audit": args.audit_ground_truth,
                "jobs": jobs, "seed_selection": "preassigned disjoint consecutive shards; all failures retained"}
    protocol_path = root / "launch_protocol.json"
    if args.resume:
        saved = json.loads(protocol_path.read_text())
        # Device placement can change while seed allocation and numerical budgets stay fixed.
        strip = lambda p: {k: ([{jk:jv for jk,jv in j.items() if jk != 'egl_device'} for j in v]
                              if k == 'jobs' else v) for k,v in p.items()}
        if strip(saved) != strip(protocol):
            raise RuntimeError("Launch protocol changed; select a new output directory")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()

    def run(job):
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        existing = output / "summary.json"
        if args.resume and existing.exists() and json.loads(existing.read_text()).get("status") == "completed":
            return dict(job, returncode=0, resumed_completed=True)
        command = [sys.executable, str(code / "evaluate_policy.py"), "--policy", args.policy,
                   "--policy-code", str(policy_code), "--output", str(output),
                   "--tasks", job["task"], "--profiles", job["profile"],
                   "--episodes", str(job["episodes"]), "--seed-start", str(job["seed"]),
                   "--max-steps", str(args.max_steps), "--max-steps-2x10",
                   str(job["budget"] if job["task"] == "2x10" else args.max_steps_2x10), "--render-cameras"]
        if args.audit_ground_truth:
            command.append("--audit-ground-truth")
        if args.resume and output.exists():
            command.append("--resume")
        job_environment = dict(environment, MUJOCO_EGL_DEVICE_ID=str(job["egl_device"]),
                               CUDA_VISIBLE_DEVICES=str(job["egl_device"]))
        print(f"START {job['group']} {job['name']}", flush=True)
        with output.with_suffix(".log").open("a" if args.resume else "w") as log:
            process = subprocess.run(command, cwd=code, env=job_environment, stdout=log, stderr=subprocess.STDOUT)
        print(f"FINISH {job['group']} {job['name']} rc={process.returncode}", flush=True)
        return dict(job, returncode=process.returncode, command=command)

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run, job) for job in jobs]):
            completed.append(future.result())
            (root / "launch_status.json").write_text(json.dumps(dict(
                finished_shards=len(completed), planned_shards=len(jobs),
                wall_seconds=time.monotonic()-started, completed=completed), indent=2) + "\n")
    failures = [job for job in completed if job["returncode"]]
    if failures:
        raise RuntimeError(f"Evaluation shard errors: {failures}")
    for group, tasks, group_budgets in groups:
        inputs = [job["output"] for job in jobs if job["group"] == group]
        subprocess.run([sys.executable, str(code / "merge_results.py"), "--inputs", *inputs,
                        "--output", str(root / f"final_{group}"), "--tasks", *tasks,
                        "--profiles", *args.profiles, "--episodes", str(args.episodes),
                        "--seed-start", str(args.seed_start), "--max-steps", str(group_budgets["2x5"]),
                        "--max-steps-2x10", str(group_budgets["2x10"])], cwd=code, check=True)
    (root / "complete.json").write_text(json.dumps(dict(
        status="completed", episodes=sum(job["episodes"] for job in jobs),
        wall_seconds=time.monotonic()-started)) + "\n")


if __name__ == "__main__":
    main()
