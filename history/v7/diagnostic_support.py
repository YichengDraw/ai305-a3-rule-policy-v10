"""Evaluator-owned truth measurements and an explicitly non-formal oracle ablation.

The ordinary policy only receives the public observation. The optional oracle
replaces selected-group geometry and target-bin pose after visual estimation;
controller code, image counts, forces, deadlines, and recovery remain unchanged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class DiagnosticPolicy:
    def __init__(self, policy, *, oracle_source_target=False, audit=True, output=None):
        self.policy = policy
        self.oracle_source_target = bool(oracle_source_target)
        self.audit_enabled = bool(audit)
        self.output = Path(output) if output else None
        self.env = None
        self.audit = {}
        self.step = 0
        self.stream = None
        self._last_stage = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def bind_env(self, env):
        self.env = env
        geom_cookie = {int(g): i for i, geoms in enumerate(env._cookie_collision_geoms) for g in geoms}
        self._contact_nodes = {}
        for geom in range(env.model.ngeom):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if geom in geom_cookie:
                self._contact_nodes[geom] = ("cookie", geom_cookie[geom])
            elif name.startswith(("L_", "R_")):
                self._contact_nodes[geom] = ("robot", geom)
        if hasattr(self.policy, "bind_env"):
            self.policy.bind_env(env)

    def reset(self, context):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.policy.reset(context)
        self.audit, self.step, self._last_stage = {}, 0, None
        self.context = context
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            self.stream = (self.output / f"seed_{context.seed}_measurements.jsonl").open("w")
        if self.oracle_source_target:
            original = getattr(self.policy.vision, "_diagnostic_original_estimate", self.policy.vision.estimate)
            self.policy.vision._diagnostic_original_estimate = original

            def estimate(observation, request=None):
                visual = original(observation, request)
                return self._oracle(visual, request or {})

            self.policy.vision.estimate = estimate

    def _oracle(self, scene, request):
        env = self.env
        scene = copy.deepcopy(scene)
        positions = env.cookie_positions
        rotations = env.data.xmat[list(env._cookie_bodies)].reshape(-1, 3, 3)
        group = dict(scene.get("group", {}))
        ids = np.asarray(group.get("indices", []), dtype=int)
        if len(ids):
            rotation = Rotation.from_matrix(rotations[ids]).mean().as_matrix()
            row = rotation[:, 1]
            half = np.asarray(env.COOKIE_HALF_SIZE)
            extent = np.abs(np.einsum("j,njk->nk", row, rotations[ids])) @ half
            coordinate = positions[ids] @ row
            width = float(np.max(coordinate + extent) - np.min(coordinate - extent))
            tilt = float(np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), 0, 1))))
            group.update(center=positions[ids].mean(0), rotation=rotation, width_m=width,
                         tilt_deg=tilt, tilt_verified=True, confidence=1., position_error_m=0.,
                         geometry_source="offline_oracle_selected_group_truth")
            scene["group"] = group
            cookies = dict(scene.get("cookies", {}))
            if "centers" in cookies:
                cookies["centers"] = np.asarray(cookies["centers"]).copy()
                cookies["centers"][ids] = positions[ids]
            if "rotations" in cookies:
                cookies["rotations"] = np.asarray(cookies["rotations"]).copy()
                cookies["rotations"][ids] = rotations[ids]
            scene["cookies"] = cookies
            scene["source_tilt_deg"] = tilt
        body = env._target_bin_body
        rotation = env.data.xmat[body].reshape(3, 3)
        scene["target_box"] = dict(scene.get("target_box", {}),
                                   center=env.data.xpos[body].copy(),
                                   yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                                   confidence=1., position_error_m=0.,
                                   geometry_source="offline_oracle_target_pose_truth")
        return scene

    def _robot_connected(self):
        env, graph = self.env, {}
        wrench = np.zeros(6)
        for index in range(env.data.ncon):
            contact = env.data.contact[index]
            if int(contact.efc_address) < 0:
                continue
            a, b = self._contact_nodes.get(int(contact.geom1)), self._contact_nodes.get(int(contact.geom2))
            if a is None or b is None:
                continue
            mujoco.mj_contactForce(env.model, env.data, index, wrench)
            if float(wrench[0]) < .005:
                continue
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        seen, queue = set(), [n for n in graph if n[0] == "robot"]
        while queue:
            n = queue.pop()
            if n not in seen:
                seen.add(n)
                queue.extend(graph.get(n, set()) - seen)
        return {n[1] for n in seen if n[0] == "cookie"}

    def _selected_ids(self, controller, batch_index):
        sizes = list(controller.batch_sizes)
        config = controller.geometry
        layout = np.asarray(config.cookie_source_positions_m)
        columns = np.unique(layout[:, 0])
        column = min(max(int(controller.source_column), 1), len(columns))
        ids = np.flatnonzero(np.isclose(layout[:, 0], columns[column - 1]))
        ids = ids[np.argsort(layout[ids, 1])]
        start = sum(sizes[:batch_index])
        return ids[start:start + sizes[batch_index]]

    def _measure(self, controller, *, final=False):
        env = self.env
        positions = env.cookie_positions
        rotations = env.data.xmat[list(env._cookie_bodies)].reshape(-1, 3, 3)
        supported = self._robot_connected()
        bottom = positions[:, 2] - np.abs(rotations)[:, 2, :] @ np.asarray(env.COOKIE_HALF_SIZE)
        lifted = bottom > float(env.SOURCE_FLOOR_TOP_Z) + .010
        target_rotation = env.data.xmat[env._target_bin_body].reshape(3, 3)
        local = (positions - env.data.xpos[env._target_bin_body]) @ target_rotation
        relative = np.einsum("ij,njk->nik", target_rotation.T, rotations)
        local_bottom = local[:, 2] - (np.abs(relative) @ np.asarray(env.COOKIE_HALF_SIZE))[:, 2]
        floor = float(env._target_cookie_center_z - env.COOKIE_HALF_SIZE[2])
        seated = np.abs(local_bottom - floor) <= .004
        inside = np.array([env.privileged_cookie_in_target(i) for i in range(len(positions))])
        released = np.array([i not in supported for i in range(len(positions))])
        valid = inside & seated & released
        current = int(getattr(controller, "batch_index", 0))
        stage = str(getattr(controller, "stage", ""))
        for batch, count in enumerate(controller.batch_sizes):
            if count == 0:
                continue
            ids = self._selected_ids(controller, batch)
            item = self.audit.setdefault(batch, {
                "batch_index": batch, "planned_count": int(count), "indices": ids.tolist(),
                "attempted": False, "valid_grasp": None, "valid_placement": None,
                "grasp_observations": 0, "placement_observations": 0,
                "final_valid_placement": None, "first_valid_grasp_step": None,
                "first_valid_placement_step": None,
                "first_attempt_step": None,
            })
            if batch > current:
                continue
            if batch == current:
                item["attempted"] = item["attempted"] or stage not in ("SELECT", "DONE")
                if item["attempted"] and item["first_attempt_step"] is None:
                    item["first_attempt_step"] = self.step
                if stage in ("LIFT", "TRANSPORT", "PLACE"):
                    item["grasp_observations"] += 1
                    exact = set(np.flatnonzero(lifted).tolist()) & supported
                    grasp = bool(len(ids) == count and set(ids.tolist()) == exact)
                    if item["valid_grasp"] is None:
                        item["valid_grasp"] = False
                    if grasp:
                        item["valid_grasp"] = True
                        if item["first_valid_grasp_step"] is None:
                            item["first_valid_grasp_step"] = self.step
            if batch < current or stage in ("OPEN", "RETRACT", "VERIFY", "DONE") or final:
                item["placement_observations"] += 1
                placed = bool(len(ids) == count and np.all(valid[ids]))
                item["valid_placement"] = bool(item["valid_placement"] or placed)
                if placed and item["first_valid_placement_step"] is None:
                    item["first_valid_placement_step"] = self.step
            if final:
                item["final_valid_placement"] = bool(len(ids) == count and np.all(valid[ids]))
                if item["attempted"] and item["valid_grasp"] is None:
                    item["valid_grasp"] = False
        if self.stream is not None:
            row = {"step": self.step, "phase": stage, "batch_index": current,
                   "oracle_source_target": self.oracle_source_target,
                   "offline_truth_only": True, "cookie_positions": positions.tolist(),
                   "cookie_rotations": rotations.tolist(), "robot_supported_ids": sorted(supported),
                   "lifted_ids": np.flatnonzero(lifted).tolist(), "valid_placed_ids": np.flatnonzero(valid).tolist(),
                   "controller_group": getattr(controller, "group", None)}
            self.stream.write(json.dumps(row, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")

    def act(self, observation, task=""):
        action = self.policy.act(observation, task)
        self.step += 1
        controller = getattr(self.policy, "controller", None)
        if self.audit_enabled and self.env is not None and controller is not None:
            stage = (getattr(controller, "stage", ""), getattr(controller, "batch_index", 0))
            if self.step % 5 == 0 or stage != self._last_stage:
                self._measure(controller)
                self._last_stage = stage
        return action

    def finalize_audit(self):
        if self.audit_enabled and self.env is not None and getattr(self.policy, "controller", None) is not None:
            self._measure(self.policy.controller, final=True)
        return sorted(self.audit.values(), key=lambda item: item["batch_index"])

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.policy.close()



from batch_metrics import diagnostic_summary
