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
from v10_leaves_17e0363838a9.two5_inner.perception import VisionEstimator
from v10_leaves_17e0363838a9.two5_inner.rule_controller import RuleController

class VisualForcePolicy:
    action_mode = 'cartesian_delta'
    name = 'visual_force_rules_v7'

    def __init__(self, variant='c'):
        self.variant = variant
        self.controller = None
        self.vision = None
        self.step = 0
        self.trace = None
        self.config_name = None

    def reset(self, context):
        self.close()
        task = context.task.lower()
        config_name = 'cookie_2x10.yaml' if '20 cookies' in task else 'cookie_batch.yaml'
        if config_name != self.config_name:
            config = load_config(resource_root() / 'configs' / config_name)
            self.controller = RuleController(config)
            self.vision = VisionEstimator(config)
            self.config_name = config_name
        self.controller.reset(context)
        if hasattr(self.controller, 'set_variant'):
            self.controller.set_variant(self.variant)
        if hasattr(self.controller, 'set_budget'):
            default_budget = 3200 if self.controller.is_2x10 else 500
            self.controller.set_budget(int(os.environ.get('A3_POLICY_MAX_STEPS', default_budget)))
        self.vision.reset(context)
        self._bootstrap_planner = None
        self._source_return_planner = None
        self._inner_metric_target = None
        self.step = 0
        directory = os.environ.get('A3_POLICY_TRACE_DIR')
        if directory:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            self.trace = (root / f'seed_{context.seed}.jsonl').open('w', encoding='utf-8')

    def act(self, observation, task=''):
        if self.controller is None:
            raise RuntimeError('Policy must be reset before acting')
        request = self.controller.perception_request()
        estimated = self.vision.estimate(observation, request)
        group = estimated.get('group') or {}
        if not self.controller.is_2x10 and self.controller.stage == 'SELECT' and (self.controller.phase_steps >= 12) and (self.controller._last_safe_pose is None) and (not any((entry['attempts'] for entry in self.controller.batch_ledger))) and (group.get('center') is None or float(group.get('confidence', 0.0) or 0.0) < 0.15 or (estimated.get('target_box') or {}).get('center') is None):
            if self._bootstrap_planner is None:
                from v10_leaves_17e0363838a9.two5_inner.source_bootstrap import InitialSourceView
                self._bootstrap_planner = InitialSourceView(self.vision)
            estimated['source_observation'] = self._bootstrap_planner.plan(observation, estimated, request)
        if not self.controller.is_2x10 and self.controller.source_column in (2, 3) and (self.controller._last_safe_pose is not None) and (self.controller.stage == 'SELECT') and (self.controller.phase_steps >= 12) and (float((estimated.get('group') or {}).get('confidence', 0.0) or 0.0) < 0.15):
            if self._source_return_planner is None:
                from v10_leaves_17e0363838a9.two5_inner.inner_source_view import SourceViewPlanner
                self._source_return_planner = SourceViewPlanner(self.vision)
            cameras = self.vision._robot_cameras(observation)
            cameras['left_wrist'] = self.vision._v6_wrist_camera(cameras['left_wrist'], observation)
            cameras = {name: self.vision._adjust_camera(camera, self.vision.camera_adjustments.get(name, np.zeros(7))) for name, camera in cameras.items()}
            estimated['source_observation'] = self._source_return_planner.plan(observation, cameras, estimated, request)
        if not self.controller.is_2x10 and self.controller.source_column in (2, 3) and (self.controller.stage in ('LIFT', 'TRANSPORT', 'INNER_TARGET_VIEW')):
            if self._inner_metric_target is None:
                from v10_leaves_17e0363838a9.two5_inner.inner_target_observation import InnerMetricTarget
                self._inner_metric_target = InnerMetricTarget(self.vision)
            estimated = self._inner_metric_target.update(observation, estimated, self.controller.stage)
        action = np.asarray(self.controller.act(observation, estimated), dtype=np.float64)
        if self.trace is not None and (self.step % 5 == 0 or request.get('stage') != self.phase_name):

            def encode(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                return str(value)
            row = {'step': self.step, 'time': float(observation['time']), 'request': request, 'phase': self.phase_name, 'perception': {key: estimated.get(key) for key in ('source_box', 'target_box', 'group', 'held_count', 'held_count_confidence', 'target_count', 'target_count_confidence', 'camera_calibration', 'wrist_calibration', 'slip_m', 'position_error_m', 'local_align', 'held', 'held_group', 'placement', 'target_occupancy', 'target_placement_valid', 'target_placement_confidence', 'source_inventory', 'source_presence', 'object_location', 'target_pose_diagnostics', 'source_observation')}, 'eef': observation['observation.eef_pose'], 'force': observation['observation.force'], 'action': action}
            self.trace.write(json.dumps(row, ensure_ascii=False, default=encode) + '\n')
            self.trace.flush()
        self.step += 1
        return np.clip(action, -1.0, 1.0)

    @property
    def finished(self):
        return bool(self.controller is not None and self.controller.finished)

    @property
    def phase_name(self):
        return str(getattr(self.controller, 'phase_name', 'UNINITIALIZED'))

    def close(self):
        if self.trace is not None:
            self.trace.close()
            self.trace = None

def make_policy():
    return VisualForcePolicy('b')

def make_policy_a():
    return VisualForcePolicy('a')

def make_policy_b():
    return VisualForcePolicy('b')

def make_policy_c():
    return VisualForcePolicy('c')
