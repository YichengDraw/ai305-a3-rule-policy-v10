#!/usr/bin/env python3
"""Build in-memory-equivalent public observation tapes from verified lossless keyframes.

No simulation, no rendering, no video decode. Images are the original PNG bytes
decoded once, outside the timed region; public state is the recorded public_before
row at the same frame_id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

CAMS = ("front", "left_wrist", "right_wrist")


def load_episode(root: Path, task: str):
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    keyframes = [json.loads(x) for x in (root / "diagnostic/lossless_keyframes.jsonl")
                 .read_text(encoding="utf-8").splitlines() if x.strip()]
    frames = {}
    for line in (root / "diagnostic/frames.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            frames[row["frame_id"]] = row
    images = {cam: [] for cam in CAMS}
    state, velocity, eef, force, times, safety = [], [], [], [], [], []
    episode_ids, frame_ids, phases = [], [], []
    image_hash = hashlib.sha256()
    used = []
    for kf in keyframes:
        fid = kf["frame_id"]
        if fid not in frames:
            raise KeyError(f"frame {fid} missing in {root}")
        pub = frames[fid]["public_before"]
        for cam in CAMS:
            path = root / "diagnostic" / kf["png_paths"][cam]
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"unexpected image shape {rgb.shape} at {path}")
            images[cam].append(rgb)
            image_hash.update(rgb.tobytes())
        state.append(np.asarray(pub["observation.state"], dtype=np.float32))
        velocity.append(np.asarray(pub["observation.velocity"], dtype=np.float32))
        eef.append(np.asarray(pub["observation.eef_pose"], dtype=np.float32))
        force.append(np.asarray(pub["observation.force"], dtype=np.float32))
        times.append(float(pub["time"]))
        safety.append(bool(pub["safety_stop"]))
        episode_ids.append(0)
        frame_ids.append(int(fid))
        phases.append(str(kf["phase_before"]))
        used.append(fid)
    phases_count = {}
    for p in phases:
        phases_count[p] = phases_count.get(p, 0) + 1
    meta = {
        "task": task,
        "source": str(root),
        "profile": result.get("profile"),
        "seed": result.get("seed"),
        "success": bool(result.get("success")),
        "cookies_in_target": result.get("cookies_in_target"),
        "max_score": result.get("max_score"),
        "steps": result.get("steps"),
        "task_instruction": result.get("task_instruction"),
        "keyframes": len(used),
        "frame_ids": [used[0], used[-1]],
        "phases": phases_count,
        "code_policy_sha256": hashlib.sha256((root.parent / "code" / "policy.py").read_bytes()).hexdigest()
        if (root.parent / "code" / "policy.py").is_file() else None,
        "image_rgb_sha256": image_hash.hexdigest(),
    }
    return dict(images={c: np.stack(images[c]) for c in CAMS},
                state=np.stack(state), velocity=np.stack(velocity), eef=np.stack(eef),
                force=np.stack(force), time=np.asarray(times, np.float64),
                safety_stop=np.asarray(safety, np.bool_), episode_id=np.zeros(len(used), np.int32),
                frame_id=np.asarray(frame_ids, np.int32), phase=np.asarray(phases)), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"tasks": {}, "cams": list(CAMS), "image_shape": [256, 256, 3]}
    for task, roots in config.items():
        all_data = {k: [] for k in ("state", "velocity", "eef", "force", "time", "safety_stop",
                                    "frame_id", "phase")}
        images = {c: [] for c in CAMS}
        episode_id = []
        metas = []
        for i, root in enumerate(roots):
            root = Path(root)
            data, meta = load_episode(root, task)
            for c in CAMS:
                images[c].append(data["images"][c])
            for k in all_data:
                all_data[k].append(data[k])
            episode_id.append(np.full(len(data["state"]), i, np.int32))
            meta["episode_id"] = i
            metas.append(meta)
        out = {
            "front": np.concatenate(images["front"]),
            "left_wrist": np.concatenate(images["left_wrist"]),
            "right_wrist": np.concatenate(images["right_wrist"]),
            "state": np.concatenate(all_data["state"]),
            "velocity": np.concatenate(all_data["velocity"]),
            "eef_pose": np.concatenate(all_data["eef"]),
            "force": np.concatenate(all_data["force"]),
            "time": np.concatenate(all_data["time"]),
            "safety_stop": np.concatenate(all_data["safety_stop"]),
            "episode_id": np.concatenate(episode_id),
            "frame_id": np.concatenate(all_data["frame_id"]),
            "phase": np.concatenate(all_data["phase"]),
        }
        path = args.out / f"tape_{task}.npz"
        np.savez_compressed(path, **out)
        manifest["tasks"][task] = {
            "frames": int(len(out["state"])),
            "episodes": metas,
            "npz": path.name,
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(task, len(out["state"]), path, path.stat().st_size, flush=True)
    (args.out / "tape_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
