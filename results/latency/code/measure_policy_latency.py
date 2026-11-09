#!/usr/bin/env python3
"""Measure native policy.act latency on stored public observations.

No environment, simulator step, renderer, video decode, or success scoring.
V10 uses its original CPU implementation; SmolVLA uses the teacher checkpoint
with the native adapter (n_action_steps=8, num_steps=25, batch=1).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

CAMS = ("front", "left_wrist", "right_wrist")


def set_thread_env(n: int):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENCV_FOR_THREADS_NUM", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(n)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def sha256(path: Path) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tape(path: Path, task: str):
    # Load all compressed arrays once; do not repeatedly decompress inside act/timing.
    with np.load(path, allow_pickle=False) as archive:
        z = {k: np.array(archive[k], copy=True) for k in archive.files}
    manifest = json.loads((path.parent / "tape_manifest.json").read_text(encoding="utf-8"))
    meta = manifest["tasks"][task]
    return z, meta


def build_obs(z, i):
    return {
        "observation.images.front": z["front"][i].copy(),
        "observation.images.left_wrist": z["left_wrist"][i].copy(),
        "observation.images.right_wrist": z["right_wrist"][i].copy(),
        "observation.state": z["state"][i].copy(),
        "observation.velocity": z["velocity"][i].copy(),
        "observation.eef_pose": z["eef_pose"][i].copy(),
        "observation.force": z["force"][i].copy(),
        "time": float(z["time"][i]),
        "safety_stop": bool(z["safety_stop"][i]),
    }


class ChunkCounter:
    def __init__(self, plugin):
        self.plugin = plugin
        self.count = 0
        self.original = getattr(plugin._policy, "_get_action_chunk")

    def install(self):
        def counted(*args, **kwargs):
            self.count += 1
            return self.original(*args, **kwargs)
        self.plugin._policy._get_action_chunk = counted

    def restore(self):
        self.plugin._policy._get_action_chunk = self.original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("v10", "smolvla"), required=True)
    ap.add_argument("--task", choices=("2x5", "2x10"), required=True)
    ap.add_argument("--tape", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--affinity", type=int, nargs="*", default=None)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--policy-code", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--max-calls", type=int, default=0)
    args = ap.parse_args()

    set_thread_env(args.threads)
    if args.affinity:
        os.sched_setaffinity(0, set(args.affinity))
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    import cv2
    cv2.setNumThreads(args.threads)

    source = args.source.expanduser().resolve()
    sys.path.insert(0, str(source / "src" if (source / "src").is_dir() else source))
    if args.policy_code:
        sys.path.insert(0, str(args.policy_code.expanduser().resolve()))
    from a3_dual_arm_sim.contracts import EpisodeContext

    z, tape_meta = load_tape(args.tape.resolve(), args.task)
    n = int(len(z["state"]))
    if args.max_calls:
        n = min(n, args.max_calls)
    episodes = tape_meta["episodes"]
    episode_id = np.asarray(z["episode_id"][:n])
    bounds = []
    for ep in range(len(episodes)):
        idx = np.flatnonzero(episode_id == ep)
        if len(idx):
            bounds.append((ep, int(idx[0]), int(idx[-1]) + 1))

    started = time.time()
    load_t0 = time.perf_counter()
    if args.policy == "v10":
        import policy as rules
        plugin = rules.make_policy_b()
        device = "cpu"
        mode = "cartesian_delta"
        chunk = None
    else:
        if args.device != "cuda":
            raise SystemExit("main SmolVLA run requires --device cuda")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        from a3_dual_arm_sim.policies.smolvla import SmolVLAPolicyPlugin
        plugin = SmolVLAPolicyPlugin(args.checkpoint.expanduser().resolve(), device="cuda",
                                     num_steps=25, n_action_steps=8, inference_seed=13000)
        device = "cuda"
        mode = "joint_position"
        chunk = ChunkCounter(plugin)
        chunk.install()
    load_s = time.perf_counter() - load_t0

    cuda_sync = device == "cuda"

    def context(ep):
        m = episodes[ep]
        return EpisodeContext(seed=int(m["seed"]), task=str(m["task_instruction"]), action_mode=mode)

    # Warm-up outside the timed samples: model 8 real chunks (64 calls); V10 16 calls.
    warm_calls = 64 if args.policy == "smolvla" else 16
    ep0, s0, e0 = bounds[0]
    plugin.reset(context(ep0))
    warm_end = min(e0, s0 + warm_calls)
    for i in range(s0, warm_end):
        plugin.act(build_obs(z, i), episodes[ep0]["task_instruction"])
    plugin.reset(context(ep0))
    warm_done = warm_end - s0

    # 16-call self-test: exactly two new chunks for SmolVLA, none for V10.
    if args.self_test:
        if args.policy == "smolvla":
            plugin.reset(context(ep0))
            before = chunk.count
            flags = []
            for i in range(s0, min(s0 + 16, e0)):
                b = chunk.count
                plugin.act(build_obs(z, i), episodes[ep0]["task_instruction"])
                flags.append(chunk.count > b)
            n_new = chunk.count - before
            if n_new != 2 or [j + 1 for j, x in enumerate(flags) if x] != [1, 9]:
                raise SystemExit(f"self-test failed: new chunks={n_new} at {[j+1 for j,x in enumerate(flags) if x]}")
        else:
            plugin.reset(context(ep0))
            for i in range(s0, min(s0 + 16, e0)):
                plugin.act(build_obs(z, i), episodes[ep0]["task_instruction"])

    rows = []
    call = 0
    stopped_episodes = []
    for ep, start, end in bounds:
        if start >= n:
            break
        end = min(end, n)
        plugin.reset(context(ep))
        instr = episodes[ep]["task_instruction"]
        for i in range(start, end):
            obs = build_obs(z, i)
            reference_phase = str(z["phase"][i])
            policy_phase = str(getattr(plugin, "phase_name", ""))
            finished_before = bool(getattr(plugin, "finished", False))
            batch_index = getattr(plugin, "batch_index", None)
            before = chunk.count if chunk else 0
            if cuda_sync:
                torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            action = plugin.act(obs, instr)
            if cuda_sync:
                torch.cuda.synchronize()
            dt = (time.perf_counter_ns() - t0) / 1e9
            arr = np.asarray(action)
            if not np.all(np.isfinite(arr)):
                raise SystemExit(f"non-finite action at frame {int(z['frame_id'][i])}")
            new_chunk = (chunk.count > before) if chunk else False
            finished_after = bool(getattr(plugin, "finished", False))
            rows.append(dict(
                round=args.round, task=args.task, policy=args.policy, device=device,
                episode=ep, call=call, frame_id=int(z["frame_id"][i]),
                reference_phase=reference_phase, policy_phase_before=policy_phase,
                policy_phase_after=str(getattr(plugin, "phase_name", "")),
                actual_batch_index=batch_index,
                finished_before=finished_before, finished_after=finished_after,
                new_chunk=new_chunk, latency_ms=dt * 1000.0,
                action_dim=int(arr.size), action_norm=float(np.linalg.norm(arr)),
                action_json=json.dumps(np.asarray(arr, dtype=float).tolist()),
            ))
            call += 1
            if finished_after:
                stopped_episodes.append(dict(episode=ep, frame_id=int(z["frame_id"][i]),
                                             call=call - 1, reason="policy_finished"))
                break
        if args.max_calls and call >= args.max_calls:
            break

    if chunk:
        chunk.restore()
    policy_obj = getattr(plugin, "_policy", None)
    if hasattr(plugin, "close"):
        plugin.close()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.policy}_{device}_r{args.round}"
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(key):
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        if not len(vals):
            return {}
        return dict(n=int(len(vals)), mean_ms=float(vals.mean()), median_ms=float(np.median(vals)),
                    p95_ms=float(np.quantile(vals, 0.95)), max_ms=float(vals.max()),
                    fraction_under_50ms=float((vals <= 50).mean()))

    meta = {
        "policy": args.policy, "task": args.task, "round": args.round, "device": device,
        "host": socket.gethostname(), "cpu": platform.processor(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
        "tape": str(args.tape), "tape_sha256": sha256(args.tape),
        "tape_frames": n, "tape_episodes": len(bounds),
        "warmup_calls": warm_done, "self_test": bool(args.self_test),
        "simulation_steps": 0, "render_calls": 0,
        "load_s": load_s,
        "chunk_size": int(policy_obj.config.chunk_size) if args.policy == "smolvla" else None,
        "n_action_steps": int(policy_obj.config.n_action_steps) if args.policy == "smolvla" else None,
        "num_steps": int(policy_obj.config.num_steps) if args.policy == "smolvla" else None,
        "parameter_count": int(sum(p.numel() for p in policy_obj.parameters())) if args.policy == "smolvla" else None,
        "parameter_dtype": str(next(policy_obj.parameters()).dtype) if args.policy == "smolvla" else None,
        "calls": len(rows), "stopped_episodes": stopped_episodes,
        "all_calls": stats("latency_ms"),
        "actual_phase_counts": {p: sum(1 for r in rows if r["policy_phase_before"] == p)
                               for p in sorted({r["policy_phase_before"] for r in rows})},
        "reference_phase_counts": {p: sum(1 for r in rows if r["reference_phase"] == p)
                                  for p in sorted({r["reference_phase"] for r in rows})},
        "wall_s": time.time() - started,
    }
    if args.policy == "smolvla":
        refresh = [r["latency_ms"] for r in rows if r["new_chunk"]]
        cache = [r["latency_ms"] for r in rows if not r["new_chunk"]]
        meta["refresh"] = dict(n=len(refresh), **{k: float(v) for k, v in
                              zip(("mean_ms", "median_ms", "p95_ms", "max_ms"),
                                  (np.mean(refresh), np.median(refresh), np.quantile(refresh, .95), np.max(refresh)))})
        meta["cache"] = dict(n=len(cache), **{k: float(v) for k, v in
                           zip(("mean_ms", "median_ms", "p95_ms", "max_ms"),
                               (np.mean(cache), np.median(cache), np.quantile(cache, .95), np.max(cache)))})
        cycles = []
        idx = [i for i, r in enumerate(rows) if r["new_chunk"]]
        for a, b in zip(idx, idx[1:]):
            cycles.append((b - a, sum(r["latency_ms"] for r in rows[a:b])))
        if idx:
            cycles.append((len(rows) - idx[-1], sum(r["latency_ms"] for r in rows[idx[-1]:])))
        full = [(n, c) for n, c in cycles if n == 8]
        meta["cycles"] = len(cycles)
        meta["full_cycles"] = len(full)
        meta["amortized_ms_per_action"] = float(np.mean([c / 8.0 for _, c in full])) if full else None
        meta["cycle_actions"] = [n for n, _ in cycles]
    (out / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"task": args.task, "policy": args.policy, "calls": len(rows),
                      "mean_ms": meta["all_calls"]["mean_ms"],
                      "refresh_n": len(refresh) if args.policy == "smolvla" else None}, ensure_ascii=False), flush=True)
    # Avoid a driver/teardown stall seen after CUDA runs; all files are closed and flushed.
    os._exit(0)


if __name__ == "__main__":
    main()
