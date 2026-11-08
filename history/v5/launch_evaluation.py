"""Run fixed, disjoint seed shards, then automatically merge both budgets."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--egl-devices", type=int, nargs="+", default=[4])
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    environment = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = []
    for budget, tasks in ((500, ("2x5", "2x10")), (3200, ("2x10",))):
        for task in tasks:
            for profile in ("basic", "medium", "advanced"):
                for seed in range(13000, 13020, 5):
                    name = f"{task}_{profile}_{seed}"
                    destination = root / f"budget_{budget}" / name
                    jobs.append({"name": name, "budget": budget, "task": task,
                                 "profile": profile, "seed": seed, "output": str(destination),
                                 "egl_device": args.egl_devices[len(jobs) % len(args.egl_devices)]})
    (root / "launch_protocol.json").write_text(json.dumps({
        "code_directory": str(code), "workers": args.workers, "jobs": jobs,
        "seed_selection": "preassigned disjoint consecutive shards; all failures retained",
        "primary_budget": 500, "additional_2x10_budget": 3200,
    }, indent=2) + "\n")
    started = time.monotonic()

    def run(job):
        output = Path(job["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(code / "evaluate_policy.py"), "--policy", "policy:make_policy",
                   "--output", str(output), "--tasks", job["task"], "--profiles", job["profile"],
                   "--episodes", "5", "--seed-start", str(job["seed"]), "--max-steps", "500",
                   "--max-steps-2x10", str(job["budget"]), "--render-cameras"]
        with output.with_suffix(".log").open("w") as log:
            print(f"START {job['budget']} {job['name']}", flush=True)
            job_environment = dict(environment, MUJOCO_EGL_DEVICE_ID=str(job["egl_device"]),
                                   CUDA_VISIBLE_DEVICES=str(job["egl_device"]))
            process = subprocess.run(command, cwd=code, env=job_environment, stdout=log, stderr=subprocess.STDOUT)
        print(f"FINISH {job['budget']} {job['name']} rc={process.returncode}", flush=True)
        return {**job, "returncode": process.returncode}

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run, job) for job in jobs]):
            completed.append(future.result())
            (root / "launch_status.json").write_text(json.dumps({
                "finished_shards": len(completed), "planned_shards": len(jobs),
                "wall_seconds": time.monotonic() - started, "completed": completed,
            }, indent=2) + "\n")
    failures = [job for job in completed if job["returncode"]]
    if failures:
        raise RuntimeError(f"Evaluation shard errors: {failures}")
    for budget, tasks in ((500, ("2x5", "2x10")), (3200, ("2x10",))):
        inputs = [job["output"] for job in jobs if job["budget"] == budget]
        subprocess.run([sys.executable, str(code / "merge_results.py"), "--inputs", *inputs,
                        "--output", str(root / f"final_{budget}"), "--tasks", *tasks,
                        "--max-steps", str(budget)], cwd=code, check=True)
    (root / "complete.json").write_text(json.dumps({"status": "completed", "episodes": 180,
                                                     "wall_seconds": time.monotonic() - started}) + "\n")


if __name__ == "__main__":
    main()
