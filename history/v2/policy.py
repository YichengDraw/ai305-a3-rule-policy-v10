"""RGB and public force feedback policy; simulator ground truth is not bound.

The evaluator owns the environment. This module receives only EpisodeContext
and the public observation dictionary through the standard policy interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.paths import resource_root

from perception import VisionEstimator
from rule_controller import RuleController


class VisualForcePolicy:
    action_mode = "cartesian_delta"
    name = "visual_force_rules"

    def __init__(self):
        self.controller = None
        self.vision = None
        self.step = 0
        self.trace = None

    def reset(self, context):
        self.close()
        task = context.task.lower()
        config_name = "cookie_2x10.yaml" if "20 cookies" in task else "cookie_batch.yaml"
        config = load_config(resource_root() / "configs" / config_name)
        self.controller = RuleController(config)
        self.vision = VisionEstimator(config)
        self.controller.reset(context)
        self.vision.reset(context)
        self.step = 0
        directory = os.environ.get("A3_POLICY_TRACE_DIR")
        if directory:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            self.trace = (root / f"seed_{context.seed}.jsonl").open("w", encoding="utf-8")

    def act(self, observation, task=""):
        if self.controller is None:
            raise RuntimeError("Policy must be reset before acting")
        request = self.controller.perception_request()
        estimated = self.vision.estimate(observation, request)
        action = np.asarray(self.controller.act(observation, estimated), dtype=np.float64)
        if self.trace is not None and (self.step % 5 == 0 or request.get("stage") != self.phase_name):
            def encode(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                return str(value)

            row = {
                "step": self.step,
                "time": float(observation["time"]),
                "request": request,
                "phase": self.phase_name,
                "perception": {key: estimated.get(key) for key in (
                    "source_box", "target_box", "group", "held_count",
                    "held_count_confidence", "target_count", "target_count_confidence",
                    "camera_calibration", "slip_m", "position_error_m",
                )},
                "eef": observation["observation.eef_pose"],
                "force": observation["observation.force"],
                "action": action,
            }
            self.trace.write(json.dumps(row, ensure_ascii=False, default=encode) + "\n")
            self.trace.flush()
        self.step += 1
        return np.clip(action, -1.0, 1.0)

    @property
    def finished(self):
        return bool(self.controller is not None and self.controller.finished)

    @property
    def phase_name(self):
        return str(getattr(self.controller, "phase_name", "UNINITIALIZED"))

    def close(self):
        if self.trace is not None:
            self.trace.close()
            self.trace = None


def make_policy():
    return VisualForcePolicy()
