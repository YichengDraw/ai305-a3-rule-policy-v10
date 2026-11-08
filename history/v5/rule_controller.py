"""Force-feedback Cartesian state machine driven by public observations and vision.

No simulator instance, object state, or contact graph is accepted by this module.
The fingertip offset and canonical wrist orientation are fixed robot CAD geometry.
Visual counts may be unavailable; that is recorded explicitly in batch_reports.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np


CANONICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
# Robot CAD: fingertip midpoint is at flange y=0.130324 m, while
# the public EEF site is at flange y=0.145 m (robot_builder.py).
PAD_OFFSET = np.array([0., 0.130324 - 0.145, 0.])


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
    if axis == "y":
        return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class RuleController:
    action_mode = "cartesian_delta"
    name = "visual_force_rule"

    def __init__(self, config):
        self.config = config
        self.geometry = _get(config, "cookie_transfer", config)
        self.translation_scale = float(_get(config, "cartesian_translation_scale_m", 0.025))
        self.rotation_scale = float(_get(config, "cartesian_rotation_scale_rad", 0.1))
        self.half_size = np.asarray(_get(self.geometry, "cookie_half_size_m", [.025, .019 / 6, .0125]))
        self.source_layout = np.asarray(_get(self.geometry, "cookie_source_positions_m", []), dtype=float)
        self.slots = np.asarray(_get(self.geometry, "target_slots_local_m", []), dtype=float)
        self.reset()

    def reset(self, context=None):
        self.task = str(_get(context, "task", "transfer 10 cookies into target box"))
        self.is_2x10 = "20 cookies" in self.task.lower() or "2x10" in self.task.lower()
        found = re.search(r"source column\s+(\d+)", self.task, re.I)
        self.source_column = int(found.group(1)) if found else 1
        first = re.search(r"grasp\s+(\d+)\s+cookies?\s+first", self.task, re.I)
        n = int(first.group(1)) if first else (0 if "skip grasp 1" in self.task.lower() else 5)
        self.batch_sizes = [n, 10 - n, n, 10 - n] if self.is_2x10 else [5, 5]
        self.batch_index = 0
        self.stage = "SELECT"
        self.phase_steps = 0
        self.total_steps = 0
        self.opening = 1.
        self.finished = False
        self.failed = False
        self.failure_reason = None
        self.batch_reports = []
        self.transition_history = []
        self.completed_indices = []
        self.retries = 0
        self._push_attempts = {}
        self.recovery_events = []
        self._stable = 0
        self._jam = 0
        self._low_force = 0
        self._command_position = None
        self._command_quaternion = None
        self._last_position = None
        self._last_scene = None
        self._report = None
        self._put_rotation = CANONICAL.copy()
        self._held_cookie_local = CANONICAL.T.copy()
        self._return_pending = False
        self._skip_zero_batches()

    @property
    def phase_name(self):
        return self.stage

    @property
    def count(self):
        return self.batch_sizes[self.batch_index] if self.batch_index < len(self.batch_sizes) else 0

    def perception_request(self):
        return {
            "task": self.task, "batch_index": self.batch_index,
            "source_column": self.source_column, "count": self.count,
            "source_start_index": sum(self.batch_sizes[:self.batch_index]),
            "stage": self.stage,
            "completed_indices": list(self.completed_indices),
            "straighten_attempts": self._push_attempts.get(self.batch_index, 0),
        }

    def _skip_zero_batches(self):
        while self.batch_index < len(self.batch_sizes) and self.batch_sizes[self.batch_index] == 0:
            self.batch_reports.append({"batch_index": self.batch_index, "planned_count": 0, "skipped": True})
            self.batch_index += 1
        if self.batch_index >= len(self.batch_sizes):
            self.finished = True
            self.stage = "DONE"

    def _advance(self, stage):
        self.transition_history.append({"step": self.total_steps, "from": self.stage, "to": stage})
        self.stage, self.phase_steps = stage, 0
        self._stable = self._jam = self._low_force = 0
        self._command_position = None
        self._command_quaternion = None

    def _hold(self):
        action = np.zeros(14, dtype=float)
        action[6], action[13] = 2 * np.clip(self.opening, 0, 1) - 1, 1.
        return action

    def _servo(self, observation, target, rotation, speed, tolerance=.0025):
        """Same normalized translation/rotation limits as the official _servo."""
        pose = np.asarray(observation["observation.eef_pose"], dtype=float)[:7]
        position = pose[:3]
        # Cartesian IK projects each requested delta onto robot joint limits.
        # Its achieved command pose is not the ideal integral of our deltas.
        # Close the loop with public measured EEF feedback so this projection
        # cannot leave zero commands while a real pose error remains (dev v2).
        target = np.asarray(target, dtype=float)
        error = target - position
        distance = float(np.linalg.norm(error))
        effective = max(.0012, speed * distance / .020) if distance < .020 and speed > .002 else speed
        delta = error * min(.35, effective / max(distance, 1e-9))
        desired_q = np.empty(4)
        mujoco.mju_mat2Quat(desired_q, np.asarray(rotation, dtype=float).ravel())
        rotation_error = np.empty(3)
        mujoco.mju_subQuat(rotation_error, desired_q, pose[3:7])
        angle = float(np.linalg.norm(rotation_error))
        action = self._hold()
        action[:3] = np.clip(delta / self.translation_scale, -1, 1)
        action[3:6] = rotation_error * min(.35, min(.05, self.rotation_scale) / max(angle, 1e-9)) / self.rotation_scale
        velocity = np.asarray(observation.get("observation.velocity", np.zeros(16)), dtype=float)
        reached = (np.linalg.norm(target - position) < tolerance and angle < .07
                   and np.linalg.norm(velocity[:7]) < .4)
        return action, reached

    def _touch(self, observation):
        force = np.asarray(observation.get("observation.force", np.zeros(18)), dtype=float)
        return np.maximum(force[12:14], 0.) if len(force) >= 14 else np.zeros(2)

    def _group(self, scene):
        group = scene.get("group")
        if group is not None and group.get("center") is not None:
            return group
        cookies = scene.get("cookies", {})
        centers = np.asarray(cookies.get("centers", []), dtype=float)
        if len(centers) == 0 or len(self.source_layout) == 0:
            return None
        xs = np.unique(self.source_layout[:, 0])
        indices = np.flatnonzero(np.isclose(self.source_layout[:, 0], xs[self.source_column - 1]))
        start = sum(self.batch_sizes[:self.batch_index])
        indices = indices[start:start + self.count]
        if len(indices) != self.count:
            return None
        confidence = np.asarray(cookies.get("confidence", np.ones(len(centers)) * .3))
        rotations = np.asarray(cookies.get("rotations", np.tile(np.eye(3), (len(centers), 1, 1))))
        return {"indices": indices.tolist(), "center": centers[indices].mean(axis=0),
                "rotation": rotations[indices[len(indices) // 2]], "tilt_deg": 0.,
                "width_m": self.count * 2 * self.half_size[1] + (self.count - 1) * .0025,
                "confidence": float(np.min(confidence[indices])), "position_error_m": .002}

    def _select(self, scene):
        group = self._group(scene)
        target = scene.get("target_box", {})
        if group is None or target.get("center") is None or float(group.get("confidence", 0)) < .15:
            return False
        self.group = group
        center = np.asarray(group["center"], dtype=float)
        cookie_rotation = np.asarray(group.get("rotation", np.eye(3)), dtype=float).reshape(3, 3)
        tilt = float(group.get("tilt_deg", 0.))
        # Follow a visually estimated common tilt. A varying tilt cannot be
        # reconstructed from a single group rotation and stays unverified.
        up = cookie_rotation[:, 2]
        fall = np.arctan2(up[1], max(up[2], .1)) if tilt > 5 else 0.
        self._pick_rotation = _rotation("y", np.deg2rad((0., -15., -35., -35.)[self.source_column - 1])) @ _rotation("x", -np.clip(fall, -.6, .6)) @ CANONICAL
        self._held_cookie_local = self._pick_rotation.T @ cookie_rotation
        self._group_offset = PAD_OFFSET - self._pick_rotation.T @ np.array([0., 0., .010])
        self._pick = center + [0., 0., .010] - self._pick_rotation @ PAD_OFFSET
        if self.retries:
            self._pick += self._pick_rotation[:, 0] * (.0006 if self.retries == 1 else -.0006)
        # Source walls stand 20 mm above the floor top in the course scenes.
        # A 40 mm object rise leaves about 20 mm above the rim; taller fixed
        # wall configurations retain 14 mm clearance beyond their wall height.
        self._clearance = max(.040, float(_get(self.geometry, "source_bin_wall_height_m", .026)) + .014)
        self._high = self._pick + [0., 0., self._clearance]
        self._trial = self._pick + [0., 0., .012]
        width = float(group.get("width_m", self.count * 2 * self.half_size[1] + (self.count - 1) * .0025))
        self.opening = float(np.clip((width - .0038) / .085, 0, 1))
        delta = (self.count - 5) * 2 * self.half_size[1] / .085
        self._min_opening = max(0., .28 + delta)
        self._release_opening = min(1., max(.18, .49 + delta))
        self._target_force = .45 if self.count == 1 else 1.3
        self._stable_force = .35 if self.count == 1 else 1.0
        self._report = {
            "batch_index": self.batch_index, "planned_count": self.count,
            "indices": [int(i) for i in group.get("indices", [])],
            "retry": self.retries, "group_confidence": float(group.get("confidence", 0)),
            "position_error_m": float(group.get("position_error_m", 0)),
            "tilt_deg": tilt, "count_verified": False, "placement_verified": False,
        }
        attempts = self._push_attempts.get(self.batch_index, 0)
        if self.batch_index > 0 and bool(group.get("tilt_verified", False)) and tilt > 10. and attempts < 2:
            # The official low push pose is an EEF pose, not a pad midpoint:
            # canonical tool geometry puts the pad bottom 2.074 mm below EEF.
            # EEF=floor+4 mm therefore leaves the pad bottom 1.926 mm above it.
            indices = group.get("indices", [])
            centers = np.asarray(scene.get("cookies", {}).get("centers", []), dtype=float)
            if len(indices) and len(centers) > int(indices[0]):
                first_center = centers[int(indices[0])].copy()
            else:
                first_center = center - cookie_rotation[:, 1] * (self.count - 1) * (.006333333333 + .0025) / 2
            base_y = first_center[1] - cookie_rotation[1, 2] * self.half_size[2]
            floor_top = float(_get(self.geometry, "source_floor_z_m", .753)) + .003
            self._push_start = np.array([first_center[0], base_y - .012, floor_top + .004])
            self._push_end = self._push_start + [0., .016, 0.]
            self._push_high = self._push_start + [0., 0., .050]
            self._push_rotation = self._pick_rotation if self.source_column == 4 else CANONICAL.copy()
            self._push_attempts[self.batch_index] = attempts + 1
            self._push_event = {
                "batch_index": self.batch_index, "attempt": attempts + 1,
                "before_tilt_deg": tilt, "tilt_verified": True,
                "start_eef": self._push_start.tolist(), "end_eef": self._push_end.tolist(),
            }
            self.opening = 1.
            self._report = None
            self._advance("PUSH_APPROACH")
            return True
        self._advance("APPROACH")
        return True

    def _place_pose(self, scene, clearance=0.):
        box = scene["target_box"]
        box_rotation = _rotation("z", float(box.get("yaw", 0.)))
        column = self.batch_index // 2 if self.is_2x10 else self.batch_index
        slot_x = np.unique(self.slots[:, 0])[column]
        ys = self.slots[np.isclose(self.slots[:, 0], slot_x), 1]
        if self.is_2x10:
            n = self.batch_sizes[0]
            rows = np.arange(n) if self.batch_index % 2 == 0 else np.arange(n, 10)
            local_y = float(np.mean(ys[rows]))
            if n > 0:
                local_y += -.004 if self.batch_index % 2 == 0 else .004
        else:
            local_y = float(np.mean(ys))
        # floor center + its 3 mm half thickness + cookie half-height,
        # plus the official 2 mm drop allowance.
        floor_top = float(_get(self.geometry, "target_floor_z_m", .003)) + .003
        cookie_center = np.asarray(box["center"], dtype=float) + box_rotation @ np.array([
            slot_x, local_y,
            floor_top + self.half_size[2] + .002 + clearance,
        ])
        rotation = box_rotation @ self._held_cookie_local.T
        return cookie_center - rotation @ self._group_offset, rotation

    def _retry(self, reason, observation, carrying=False):
        if self._report is not None:
            self._report.update(outcome="retry", reason=reason)
            self.batch_reports.append(self._report)
            self._report = None
        self.retries += 1
        if self.retries > 2:
            self.failed = self.finished = True
            self.failure_reason = reason
            self._advance("FAILED")
            return
        self._return_pending = carrying
        self._retry_high = np.asarray(observation["observation.eef_pose"], dtype=float)[:3] + [0., 0., .040]
        if not carrying:
            self.opening = min(1., self.opening + .08)
        self._advance("RECOVER_UP")

    def act(self, observation, scene):
        self.total_steps += 1
        self.phase_steps += 1
        self._last_scene = scene
        if self.finished:
            return self._hold()
        pose = np.asarray(observation["observation.eef_pose"], dtype=float)[:7]
        touch = self._touch(observation)
        if self.phase_steps > 110 and self.stage not in ("SELECT", "CLOSE", "VERIFY"):
            self._retry(f"{self.stage.lower()} did not reach its target", observation,
                        carrying=self.stage in ("TRIAL_LIFT", "LIFT", "TRANSPORT", "PLACE"))
        if self.stage == "SELECT":
            if not self._select(scene) and self.phase_steps > 30:
                self._retry("visual group unavailable", observation)
            return self._hold()
        if self.stage == "APPROACH":
            action, reached = self._servo(observation, self._high, self._pick_rotation, .018)
            if reached:
                self._advance("DESCEND")
            return action
        if self.stage == "DESCEND":
            self._jam = self._jam + 1 if max(touch) > 20. else 0
            if self._jam >= 3:
                self._retry("insertion blocked by sustained fingertip force", observation)
                return self._hold()
            remaining = np.linalg.norm(pose[:3] - self._pick)
            # Dev v4's first follow-up descent struck an upright leading slab
            # at 54--67 N and toppled it before CLOSE. The v3 contact approach
            # succeeded on that same development scene. Retain fast empty-space
            # motion, and use the measured slower profile for follow-up/retry.
            slow_contact = sum(self.batch_sizes[:self.batch_index]) > 0 or self.retries > 0
            if slow_contact:
                descend_speed = .0025 if remaining > .015 else .0012
            else:
                descend_speed = .0035 if remaining > .006 else .0012
            action, reached = self._servo(observation, self._pick, self._pick_rotation, descend_speed, .0017)
            if reached:
                self._advance("CLOSE")
            return action
        if self.stage == "CLOSE":
            if min(touch) < self._target_force:
                self.opening = max(self._min_opening, self.opening - (.004 if max(touch) > .1 else .015))
            self._stable = self._stable + 1 if min(touch) >= self._stable_force else 0
            action, _ = self._servo(observation, self._pick, self._pick_rotation, .0008)
            if self._stable >= 3:
                self._report["grip_touch_n"] = touch.tolist()
                state = np.asarray(observation.get("observation.state", []), dtype=float)
                if len(state) > 7:
                    self._report["measured_gripper_opening"] = float(state[7])
                    self._report["opening_count_estimate"] = None
                    self._report["opening_count_note"] = "fixed-pad width relation requires contact calibration"
                self._advance("TRIAL_LIFT")
            elif self.phase_steps > 45:
                self._retry("two-finger grip did not stabilize", observation)
            return action
        if self.stage in ("TRIAL_LIFT", "LIFT"):
            target = self._trial if self.stage == "TRIAL_LIFT" else self._high
            action, reached = self._servo(observation, target, self._pick_rotation, .0024 if self.stage == "TRIAL_LIFT" else .0045)
            self._low_force = self._low_force + 1 if min(touch) < .1 else 0
            if self._low_force >= 5:
                self._retry("grip force lost during lift", observation, carrying=True)
            elif reached:
                held = scene.get("held_count")
                confidence = float(scene.get("held_confidence", scene.get("held_count_confidence", 0.)))
                if held is not None and confidence >= .6:
                    self._report.update(observed_held_count=int(held), count_verified=int(held) == self.count)
                    if int(held) != self.count:
                        self._retry("visual trial-lift count differs from requested count", observation, carrying=True)
                        return action
                if self.stage == "TRIAL_LIFT":
                    self._advance("LIFT")
                else:
                    self._advance("TRANSPORT")
            return action
        if self.stage in ("TRANSPORT", "PLACE"):
            clearance = self._clearance if self.stage == "TRANSPORT" else 0.
            target, rotation = self._place_pose(scene, clearance)
            speed = .015 if self.stage == "TRANSPORT" else (.0045 if np.linalg.norm(pose[:3] - target) > .012 else .0012)
            action, reached = self._servo(observation, target, rotation, speed)
            if float(scene.get("slip_m", 0.) or 0.) > .004:
                self._retry("visual group slip exceeds 4 mm", observation, carrying=True)
            elif reached:
                self._put_rotation = rotation
                self._put_position = target
                if self.stage == "TRANSPORT":
                    self._advance("PLACE")
                else:
                    self._advance("OPEN")
            return action
        if self.stage == "OPEN":
            release = self._release_opening
            if self.is_2x10 and self.batch_index % 2:
                # Compress width is known CAD; 6 mm extra leaves room for the
                # neighboring group while releasing the inner fingertip.
                release = min(release, max(self.opening + .015, (self.count * 2 * self.half_size[1] + .006) / .085))
            self.opening = min(release, self.opening + .025)
            action, _ = self._servo(observation, self._put_position, self._put_rotation, .0008)
            if self.opening >= release - 1e-6 and self.phase_steps >= 5:
                self._retract = pose[:3] + [0., 0., .045]
                self._advance("RETRACT")
            return action
        if self.stage == "RETRACT":
            action, reached = self._servo(observation, self._retract, self._put_rotation, .006)
            if reached:
                self._advance("VERIFY")
            return action
        if self.stage == "VERIFY":
            if self.phase_steps >= 6:
                target_count = scene.get("target_count")
                target_confidence = float(scene.get("target_count_confidence", 0.))
                expected = sum(self.batch_sizes[:self.batch_index + 1])
                released = max(touch) < .1
                self._report.update(
                    outcome="transfer_attempted", released_by_touch=released,
                    target_count=target_count,
                    placement_verified=(target_count is not None and target_confidence >= .6 and int(target_count) >= expected
                                        and bool(scene.get("target_placement_valid", False))),
                )
                self.batch_reports.append(self._report)
                self.completed_indices.extend(self._report["indices"])
                self._report = None
                self.batch_index += 1
                self.retries = 0
                self._skip_zero_batches()
                if not self.finished:
                    self._advance("SELECT")
            return self._hold()
        if self.stage == "PUSH_APPROACH":
            action, reached = self._servo(observation, self._push_high, self._push_rotation, .018)
            if reached:
                self.opening = .10
                self._advance("PUSH_PRE_CLOSE")
            return action
        if self.stage == "PUSH_PRE_CLOSE":
            action, _ = self._servo(observation, self._push_high, self._push_rotation, .001)
            state = np.asarray(observation.get("observation.state", np.zeros(16)), dtype=float)
            if abs(float(state[7]) - self.opening) < .02:
                self._advance("PUSH_DESCEND")
            elif self.phase_steps > 30:
                self._push_event["outcome"] = "preclose_not_reached"
                self._push_retract = pose[:3] + [0., 0., .040]
                self._advance("PUSH_RETRACT")
            return action
        if self.stage in ("PUSH_DESCEND", "PUSH_SWEEP"):
            self._jam = self._jam + 1 if max(touch) > 12. else 0
            if self._jam >= 3:
                self._push_event["outcome"] = "push_blocked_by_force"
                self._push_retract = pose[:3] + [0., 0., .050]
                self._advance("PUSH_RETRACT")
                return self._hold()
            target = self._push_start if self.stage == "PUSH_DESCEND" else self._push_end
            action, reached = self._servo(observation, target, self._push_rotation, .0018 if self.stage == "PUSH_DESCEND" else .0012)
            if reached:
                if self.stage == "PUSH_DESCEND":
                    self._advance("PUSH_SWEEP")
                else:
                    self._push_event["outcome"] = "sweep_completed_requires_visual_recheck"
                    self._push_retract = pose[:3] + [0., 0., .050]
                    self._advance("PUSH_RETRACT")
            return action
        if self.stage == "PUSH_RETRACT":
            action, reached = self._servo(observation, self._push_retract, self._push_rotation, .006)
            if reached:
                self._push_event["step"] = self.total_steps
                self.recovery_events.append(self._push_event)
                self._advance("SELECT")
            return action
        if self.stage == "RECOVER_UP":
            action, reached = self._servo(observation, self._retry_high, self._pick_rotation if hasattr(self, "_pick_rotation") else CANONICAL, .005)
            if reached:
                self._advance("RECOVER_RETURN" if self._return_pending else "SELECT")
            return action
        if self.stage == "RECOVER_RETURN":
            action, reached = self._servo(observation, self._high, self._pick_rotation, .010)
            if reached:
                self._advance("RECOVER_LOWER")
            return action
        if self.stage == "RECOVER_LOWER":
            action, reached = self._servo(observation, self._pick, self._pick_rotation, .0024)
            if reached:
                self.opening = min(1., self._release_opening + .06)
                self._advance("RECOVER_RELEASE")
            return action
        if self.stage == "RECOVER_RELEASE":
            action, reached = self._servo(observation, self._high, self._pick_rotation, .005)
            if reached and self.phase_steps >= 5:
                self._advance("SELECT")
            return action
        return self._hold()
