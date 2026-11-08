"""Public-data interpretation of a held-pose alarm; official scores are external."""
from __future__ import annotations
import copy
import numpy as np

class B2Supervisor:

    def _b2_reset(self):
        self._b2_reference_epoch = 0
        self._b2_samples = []
        self._b2_seen = set()
        self._b2_empty_samples = 0
        self._b2_reason = None
        self._b2_view_index = 0
        self._b2_view_anchor = None

    def _b2_begin(self, reason):
        if self.stage == 'SLIP_RECHECK':
            return
        self._b2_resume_stage = self.stage
        self._b2_view_index = 0
        self._b2_view_anchor = None
        self._b2_hold_pose = self._snapshot_pose(self._obs)
        self._b2_samples = []
        self._b2_seen = set()
        self._b2_empty_samples = 0
        self._b2_reason = reason
        self._b2_closing_command = float(self.opening)
        self._request_held_refresh = True
        self._rule_reason = 'held-pose alarm: retain clamp and inspect current region before choosing a return'
        self._event('held_pose_alarm', reason)
        self._advance('SLIP_RECHECK')

    def _b2_return_allowed(self):
        scene = self._scene or {}
        held = scene.get('held') or {}
        source = scene.get('source_presence') or {}
        count = held.get('count')
        known = count is not None and int(count) > 0 and (float(held.get('count_confidence', 0.0) or 0.0) >= 0.6)
        pose_known = held.get('center_tool') is not None and float(held.get('pose_confidence', 0.0) or 0.0) >= 0.6
        vacancy = source.get('state') == 'absent' and float(source.get('confidence', 0.0) or 0.0) >= 0.6
        path = bool(self._carry_path and self._carry_source_high_pose is not None)
        width = held.get('width_m')
        old_width = getattr(self, '_b2_departure_width', None)
        same_envelope = width is not None and old_width is not None and (float(width) <= float(old_width) + 0.001)
        return bool(known and pose_known and vacancy and path and same_envelope and (min(self._touch(self._obs)) >= 0.1))

    def _b2_known_target_transfer(self, scene):
        target = scene.get('target_box') or {}
        held = scene.get('held') or {}
        count = scene.get('target_count')
        confidence = float(scene.get('target_count_confidence', 0.0) or 0.0)
        expected = sum(self.batch_sizes[:self.batch_index + 1])
        if count is None or confidence < 0.6 or int(count) < expected:
            return False
        if not self._b2_samples:
            return False
        world = np.asarray(self._b2_samples[-1]['world_center'])
        if target.get('center') is None:
            return False
        yaw = float(target.get('yaw', 0.0))
        c, s = (np.cos(yaw), np.sin(yaw))
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = rotation.T @ (world - np.asarray(target['center']))
        inner = np.asarray(self.geometry.target_bin_half_size_m, float)
        return bool(np.all(abs(local[:2]) < inner + 0.004))

    def _b2_step(self, obs, scene):
        self.opening = self._b2_closing_command
        self._request_held_refresh = True
        held = scene.get('held') or {}
        touch = self._touch(obs)
        identifier = held.get('measurement_id')
        if identifier is not None and identifier not in self._b2_seen:
            self._b2_seen.add(identifier)
            count = held.get('count')
            count_conf = float(held.get('count_confidence', 0.0) or 0.0)
            pose_conf = float(held.get('pose_confidence', 0.0) or 0.0)
            if count is not None and int(count) > 0 and (count_conf >= 0.6) and (pose_conf >= 0.6) and (held.get('center_tool') is not None):
                center = np.asarray(held['center_tool'], float)
                rotation = np.asarray(held['rotation_tool'], float)
                eef, tool = self._snapshot_pose(obs)
                width = held.get('width_m')
                width_ok = width is not None and abs(float(width) - int(count) * 2 * self.half_size[1]) <= 0.0025
                self._b2_samples.append(dict(id=identifier, center=center, rotation=rotation, count=int(count), width=width, width_ok=width_ok, world_center=eef + tool @ center, time=float(obs.get('time', 0.0)), minimum_touch=float(min(touch))))
                self._b2_samples = self._b2_samples[-4:]
        self._b2_empty_samples = self._b2_empty_samples + 1 if max(touch) < 0.12 else 0
        placement = scene.get('placement') or {}
        support = bool(self._b2_resume_stage == 'PLACE' and placement.get('inside') is True and (placement.get('upright') is True) and (placement.get('seated') is True) and (placement.get('stationary') is True) and (float(placement.get('confidence', 0.0) or 0.0) >= 0.6) and (placement.get('count') is not None) and (placement.get('count') == self.execution_count))
        if support and self._b2_samples:
            width = float(held.get('width_m') or self._held_width or 2 * self.execution_count * self.half_size[1])
            self._release_goal = float(np.clip((width + 0.006 - 0.0053616) / 0.085, 0, 1))
            self.opening = self._release_goal
            self._request_held_refresh = False
            self._put_position, self._put_rotation = self._snapshot_pose(obs)
            self._rule_reason = 'fresh measured target support and actual quantity: controlled release at the current supported pose'
            self._advance('OPEN')
            return self._hold()
        if len(self._b2_samples) >= 2:
            a, b = self._b2_samples[-2:]
            angle = float(np.arccos(np.clip((np.trace(a['rotation'].T @ b['rotation']) - 1) / 2, -1, 1)))
            stable = bool(b['time'] - a['time'] >= 0.049 and np.linalg.norm(b['center'] - a['center']) < 0.0015 and (angle < 0.08) and (a['count'] == b['count']) and a['width_ok'] and b['width_ok'] and (min(a['minimum_touch'], b['minimum_touch']) >= 0.1))
            if stable:
                self._held_center_local = (a['center'] + b['center']) / 2
                self._held_cookie_local = b['rotation'].copy()
                self._held_width = float(b['width'])
                self._held_pose_confidence = float(held.get('pose_confidence', 0.0))
                self._b2_reference_epoch += 1
                if self._b2_resume_stage != 'PLACE':
                    self._placement_goal = None
                self._slip_exceedances = 0
                self._last_slip_measurement = None
                self._request_held_refresh = False
                self._rule_reason = 'two fresh stationary payload fits with actual count, width and continuing bilateral contact: rebase then continue forward'
                resume = self._b2_resume_stage
                self._advance(resume)
                return self._hold()
        if self._b2_empty_samples >= 2:
            self._request_held_refresh = False
            source = scene.get('source_presence') or {}
            if self._b2_resume_stage in ('TRIAL_LIFT', 'LIFT') and source.get('state') == 'present' and (source.get('confidence', 0.0) >= 0.6):
                self._rule_reason = 'hand empty above a still observed source group: empty insertion retreat, not a carrying return'
                self._start_source_recovery(self._rule_reason, carrying=False)
            elif self._b2_known_target_transfer(scene) and self.current_report is not None:
                self._rule_reason = 'tracked payload over target, hand clear and matching new target count: retain transfer association'
                self.ledger['transfer_association'] = 'observed_target_region_quantity_candidate'
                self._commit(placement, False)
                if self.finished:
                    self.finished = False
                    self._advance('COMPLETION_CANDIDATE')
            else:
                self._rule_reason = 'hand empty or unlocated payload: observe current target region before any source return'
                self._start_target_observation(self._rule_reason)
            return self._hold()
        latest_fresh = bool(self._b2_samples and self._b2_samples[-1]['time'] >= float(obs.get('time', 0.0)) - 0.051)
        if self.phase_steps >= 8 and (not latest_fresh or self.phase_steps >= 14):
            if self._b2_resume_stage == 'TRANSPORT' and min(touch) >= 0.1 and (self._b2_view_index < 3):
                self._b2_start_view(obs)
                return self._hold()
            self._request_held_refresh = False
            self._rule_reason = 'fresh held evidence remains unresolved; inspect current/target region with load state retained'
            self._start_target_observation(self._rule_reason)
            return self._hold()
        return self._servo(obs, *self._b2_hold_pose, 0.003)[0]

    def _b2_start_view(self, obs):
        if self._b2_view_anchor is None:
            self._b2_view_anchor = self._snapshot_pose(obs)
        center, rotation = self._b2_view_anchor
        self._b2_view_index += 1
        dx = (0.0, 0.01, -0.01)[self._b2_view_index - 1]
        self._b2_view_rotation = rotation.copy()
        rise = center + np.array([0.0, 0.0, 0.015])
        self._b2_view_nodes = [rise, rise + np.array([dx, 0.0, 0.0])]
        self._b2_view_node = 0
        self._request_held_refresh = True
        self._rule_reason = 'uncertain transported payload: retain clamp, reach a distinct high held-observation view'
        self._event('held_reobserve', self._rule_reason)
        self._advance('SLIP_VIEW')

    def _b2_view_step(self, obs, scene):
        self.opening = self._b2_closing_command
        self._request_held_refresh = True
        if max(self._touch(obs)) < 0.12:
            self._b2_hold_pose = self._snapshot_pose(obs)
            self._b2_samples = []
            self._b2_seen = set()
            self._advance('SLIP_RECHECK')
            return self._hold()
        target = self._b2_view_nodes[self._b2_view_node]
        action, reached = self._servo(obs, target, self._b2_view_rotation, 0.002)
        if reached:
            self._b2_view_node += 1
            if self._b2_view_node == len(self._b2_view_nodes):
                self._b2_hold_pose = self._snapshot_pose(obs)
                self._b2_samples = []
                self._b2_seen = set()
                self._advance('SLIP_RECHECK')
        return action
