"""V6 visual/force controller. Inputs are public observations and RGB estimates.

Fixed CAD supplies the thin-pad geometry. Scene estimates stay hypotheses until
supported by observations; placement execution and placement commitment have
separate records. No episode simulator, object truth, or contact graph is bound.
"""
from __future__ import annotations
import copy
import re
import mujoco
import numpy as np
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.source1.b2_supervisor import B2Supervisor
CANONICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
PAD_OFFSET = np.array([0.0, 0.130324 - 0.145, 0.0])
GAP_ZERO_M = 0.0053616
GAP_STROKE_M = 0.085

def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

def _rotation(axis, angle):
    c, s = (np.cos(angle), np.sin(angle))
    if axis == 'x':
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 'y':
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def gap_from_opening(opening):
    return GAP_ZERO_M + GAP_STROKE_M * float(np.clip(opening, 0.0, 1.0))

def opening_from_gap(gap_m):
    return float(np.clip((float(gap_m) - GAP_ZERO_M) / GAP_STROKE_M, 0.0, 1.0))

def _mat(quaternion):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    return rotation.reshape(3, 3)

class RuleController(B2Supervisor):
    action_mode = 'cartesian_delta'
    name = 'visual_force_rule_v8_motion_m1_v7'

    def __init__(self, config):
        self.config = config
        self.geometry = _get(config, 'cookie_transfer', config)
        self.translation_scale = float(_get(config, 'cartesian_translation_scale_m', 0.025))
        self.rotation_scale = float(_get(config, 'cartesian_rotation_scale_rad', 0.1))
        self.half_size = np.asarray(_get(self.geometry, 'cookie_half_size_m', [0.025, 0.019 / 6, 0.0125]), dtype=float)
        self.source_layout = np.asarray(_get(self.geometry, 'cookie_source_positions_m', []), dtype=float)
        self.slots = np.asarray(_get(self.geometry, 'target_slots_local_m', []), dtype=float)
        self.variant = 'c'
        self.reset()

    def set_variant(self, variant):
        if variant not in ('a', 'b', 'c'):
            raise ValueError('variant must be a, b, or c')
        self.variant = variant

    def set_budget(self, max_steps):
        if int(max_steps) < 1:
            raise ValueError('max_steps must be positive')
        self.max_steps = int(max_steps)

    def reset(self, context=None):
        self.task = str(_get(context, 'task', 'transfer 10 cookies into target box'))
        self.is_2x10 = '20 cookies' in self.task.lower() or '2x10' in self.task.lower()
        column = re.search('source column\\s+(\\d+)', self.task, re.I)
        self.source_column = int(column.group(1)) if column else 1
        first = re.search('grasp\\s+(\\d+)\\s+cookies?\\s+first', self.task, re.I)
        n = int(first.group(1)) if first else 0 if 'skip grasp 1' in self.task.lower() else 5
        self.batch_sizes = [n, 10 - n, n, 10 - n] if self.is_2x10 else [5, 5]
        columns = np.unique(self.source_layout[:, 0])
        indices = np.flatnonzero(np.isclose(self.source_layout[:, 0], columns[self.source_column - 1]))
        indices = indices[np.argsort(self.source_layout[indices, 1])]
        self.batch_ledger = []
        start = 0
        for batch, count in enumerate(self.batch_sizes):
            self.batch_ledger.append({'batch_index': batch, 'planned_count': count, 'planned_indices': indices[start:start + count].astype(int).tolist(), 'taken_indices': [], 'held_indices': [], 'placed_indices': [], 'extracted_candidate_indices': [], 'recovery_indices': [], 'state': 'skipped' if count == 0 else 'planned', 'attempts': 0, 'grasp_verified': None, 'placement_verified': None, 'identity_verified': None, 'observed_object_ids': [], 'held_batch_association_interval': [0, count], 'identity_basis': 'source-layout labels associated with the visually selected group'})
            start += count
        self.batch_index = 0
        self.stage = 'SELECT'
        self.phase_steps = self.total_steps = 0
        self.max_steps = 3200 if self.is_2x10 else 500
        self.opening = self.actual_opening = 1.0
        self.finished = self.failed = False
        self.failure_reason = None
        self.batch_reports = []
        self.transition_history = []
        self.stage_reports = []
        self._stage_batch = 0
        self.recovery_events = []
        self.completed_indices = []
        self.current_report = None
        self._report = None
        self._obs = self._scene = None
        self._metric = None
        self._segment_reports = []
        self._last_progress = 0
        self._stable = self._jam = self._low_force = 0
        self._last_close_opening = None
        self._last_close_min_force = 0.0
        self._last_safe_pose = None
        self._source_path = []
        self._carry_path = []
        self._carry_source_high_index = None
        self._carry_source_high_pose = None
        self._safe_source_high = None
        self._target_clear_pose = None
        self._target_path = []
        self._target_box_reference = None
        self._place_box_frozen = False
        self._placement_goal = None
        self._v9_preplace_observed = set()
        self._v9_preplace_measurement = None
        self._v9_target_measurement = None
        self._source_recovery_count = 0
        self._observe_count = 0
        self._observe_signature = None
        self._observe_progress = []
        self._push_attempts = {}
        self._push_before = {}
        self._repairing = False
        self._repair_attempts = 0
        self._held_pose_confidence = 0.0
        self._held_width = None
        self._last_slip_measurement = None
        self._slip_exceedances = 0
        self._held_geometry_generation = None
        self._request_held_refresh = False
        self._held_center_local = None
        self._held_cookie_local = CANONICAL.T.copy()
        self._group_offset = PAD_OFFSET - CANONICAL.T @ np.array([0.0, 0.0, 0.01])
        self._release_goal = 1.0
        self._release_checked = False
        self._alignment_corrections = 0
        self._alignment_observations = 0
        self._near_coarse_done = False
        self._last_alignment_measurement = None
        self._pending_event_ids = []
        self._source_memory = {}
        self._alignment_attempt_views = 0
        self._alignment_geometry = None
        self._alignment_arrival_step = None
        self._alignment_arrival_measurement = None
        self._alignment_view_stable = 0
        self._motion_verify_samples = 0
        self._motion_valid_samples = 0
        self._motion_passes = []
        self._placement_evidence_ids = set()
        self._handled_count_ids = set()
        self._repair_unit = None
        self._inventory = {}
        self._rule_reason = 'initial planned source selection'
        self._actual_held_count = None
        self._b2_reset()
        self._skip_zero_batches()

    @property
    def phase_name(self):
        return self.stage

    @property
    def count(self):
        return self.batch_sizes[self.batch_index] if self.batch_index < len(self.batch_sizes) else 0

    @property
    def remaining_steps(self):
        return max(0, self.max_steps - self.total_steps)

    @property
    def execution_count(self):
        """Physical repair/grasp quantity; planned batch size stays immutable."""
        unit = self._repair_unit or {}
        interval = unit.get('count_interval')
        if unit.get('location') in ('held', 'target') and interval and (interval[0] == interval[1]):
            return int(interval[0])
        return self.count

    @property
    def public_evidence(self):
        held = (self._scene or {}).get('held') or {}
        return dict(measurement_id=held.get('count_measurement_id', held.get('measurement_id')), measurement_time=held.get('measurement_time'), measurement_step=held.get('measurement_step'), geometry_version=((self._scene or {}).get('group') or {}).get('geometry_version'), object_id=list(self.ledger['observed_object_ids']) or None, camera=held.get('camera'), planned_object_hypotheses=list(self.ledger['planned_indices']), rule_reason=self._rule_reason, object_inventory=copy.deepcopy(self._inventory), repair_unit=copy.deepcopy(self._repair_unit), planned_count=self.count, actual_held_count=self._actual_held_count)

    @property
    def ledger(self):
        return self.batch_ledger[min(self.batch_index, len(self.batch_ledger) - 1)]

    def perception_request(self):
        entry = self.ledger
        target_column = self.batch_index // 2 if self.is_2x10 else self.batch_index
        pose = None if self._obs is None else np.asarray(self._obs['observation.eef_pose'])[:7]
        request = {'task': self.task, 'variant': self.variant, 'batch_index': self.batch_index, 'verify_batch_id': self.batch_index, 'source_column': self.source_column, 'count': self.count, 'source_start_index': sum(self.batch_sizes[:self.batch_index]), 'stage': ('PLACE' if getattr(self, '_b2_resume_stage', None) == 'PLACE' else 'HELD_OBSERVE') if self.stage == 'SLIP_RECHECK' else 'TARGET_OBSERVE' if self.stage == 'COMPLETION_CANDIDATE' else 'OBJECT_LOCALIZE' if self.stage == 'HELD_COUNT_RECONCILE' else 'HELD_OBSERVE' if self.stage == 'SLIP_VIEW' else self.stage, 'target_column': min(target_column, 1), 'expected_target_count': sum(self.batch_sizes[:self.batch_index + 1]), 'planned_indices': list(entry['planned_indices']), 'requested_indices': list(entry['planned_indices']), 'taken_indices': [i for e in self.batch_ledger for i in e['taken_indices']], 'placed_indices': list(self.completed_indices), 'completed_indices': list(self.completed_indices), 'remaining_steps': self.remaining_steps, 'preinsert_position': getattr(self, '_preinsert', None), 'preinsert_rotation': getattr(self, '_pick_rotation', None), 'expected_group_center': getattr(self, '_selected_center', None), 'expected_group_rotation': getattr(self, '_selected_rotation', None), 'batch_expected_local_center': getattr(self, '_expected_local_center', None), 'placed_bounds_by_batch': [{'batch_index': e['batch_index'], 'target_column': e['batch_index'] // 2 if self.is_2x10 else e['batch_index'], 'bounds': e.get('target_local_bounds'), 'verified': e['state'] == 'committed'} for e in self.batch_ledger if e.get('target_local_bounds') is not None], 'held_center_tool': self._held_center_local, 'held_rotation_tool': self._held_cookie_local, 'held_pose_confidence': self._held_pose_confidence, 'held_reobserve': self._request_held_refresh or self.stage == 'PREPLACE_OBSERVE', 'held_reference_epoch': self._b2_reference_epoch, 'target_box_reference': self._target_box_reference, 'release_opening_goal': self._release_goal, 'actual_opening': self.actual_opening, 'grasp_reference_eef': getattr(self, '_grasp_reference_eef', None), 'observation_pass': self._observe_count + self._alignment_observations, 'alignment_geometry': self._alignment_geometry, 'source_retry_memory': copy.deepcopy(self._source_memory.get(self.batch_index, {})), 'repair_unit': copy.deepcopy(self._repair_unit), 'force_refit': self.stage in ('LOCAL_ALIGN', 'XY_COARSE', 'ALIGN_OBSERVE', 'ALIGN_RETURN', 'VERIFY', 'TARGET_OBSERVE', 'SOURCE_OBSERVE', 'PREPLACE_OBSERVE') or (self.stage == 'ALIGN_APPROACH' and pose is not None and (np.linalg.norm(pose[:3] - self._preinsert) < 0.012)), 'straighten_attempts': self._push_attempts.get(self.batch_index, 0), 'eef_for_request': pose}
        if self.stage == 'COMPLETION_CANDIDATE':
            last = len(self.batch_sizes) - 1
            request.update(batch_index=last, verify_batch_id=last, count=self.batch_sizes[last], source_start_index=sum(self.batch_sizes[:last]))
        return request

    def _skip_zero_batches(self):
        while self.batch_index < len(self.batch_sizes) and (not self.batch_sizes[self.batch_index]):
            self.batch_reports.append({'batch_index': self.batch_index, 'planned_count': 0, 'skipped': True, 'outcome': 'skipped'})
            self.batch_index += 1
        if self.batch_index >= len(self.batch_sizes):
            verified = all((not e['planned_count'] or e['state'] == 'committed' for e in self.batch_ledger))
            self.finished = True
            self._advance('DONE' if verified else 'UNVERIFIED')

    def _advance(self, stage):
        self.transition_history.append({'step': self.total_steps, 'from': self.stage, 'to': stage})
        if self.phase_steps:
            self.stage_reports.append({'stage': self.stage, 'batch_index': self._stage_batch, 'start_step': self.total_steps - self.phase_steps, 'end_step': self.total_steps, 'reached_segments': list(self._segment_reports), **(self._metric or {'first_error': None, 'last_error': None, 'best_error': None})})
        self.stage, self.phase_steps = (stage, 0)
        self._stage_batch = self.batch_index
        self._metric = None
        self._segment_reports = []
        self._last_progress = self.total_steps
        self._stable = self._jam = self._low_force = 0
        self._last_close_opening = None
        self._last_close_min_force = 0.0

    def _progress(self, error):
        error = float(error)
        if self._metric is None:
            self._metric = {'first_error': error, 'last_error': error, 'best_error': error}
        else:
            previous = self._metric['best_error']
            self._metric['last_error'] = error
            if error < previous:
                self._metric['best_error'] = error
            if previous - error > 0.0003:
                self._last_progress = self.total_steps

    def _waypoint_reached(self):
        if self._metric is not None:
            self._segment_reports.append({'end_step': self.total_steps, **self._metric})
        self._metric = None
        self._last_progress = self.total_steps

    def _event(self, kind, reason):
        event = {'id': len(self.recovery_events), 'type': kind, 'batch_index': self.batch_index, 'reason': reason, 'start_step': self.total_steps, 'success': None, 'recovered': None, 'state': 'pending'}
        self.recovery_events.append(event)
        self._pending_event_ids.append(event['id'])
        return event

    def _resolve_events(self, criterion, domain=None):
        source_types = {'return_source', 'insertion_retreat', 'source_straighten', 'source_reobserve'}
        for index in list(self._pending_event_ids):
            event = self.recovery_events[index]
            correct_domain = domain is None or (event['type'] in source_types if domain == 'source' else event['type'] not in source_types)
            if event['batch_index'] == self.batch_index and correct_domain:
                event.update(success=True, recovered=True, state='recovered', end_step=self.total_steps, success_criterion=criterion)
                self._pending_event_ids.remove(index)

    def _finish_unknown(self, reason):
        self.failure_reason = reason
        self.finished = True
        if self.current_report is not None:
            self.current_report.update(outcome='unverified', reason=reason)
            self.batch_reports.append(copy.deepcopy(self.current_report))
            self.current_report = self._report = None
        for index in self._pending_event_ids:
            self.recovery_events[index].update(success=False, recovered=False, state='unresolved', end_step=self.total_steps, end_reason=reason)
        self._pending_event_ids.clear()
        self._advance('UNVERIFIED')

    def _hold(self):
        action = np.zeros(14, dtype=float)
        action[6], action[13] = (2 * float(np.clip(self.opening, 0, 1)) - 1, 1.0)
        return action

    def _touch(self, obs):
        return np.maximum(np.asarray(obs.get('observation.force', np.zeros(18)), dtype=float)[12:14], 0.0)

    def _servo(self, obs, target, rotation, speed=0.004, tolerance=0.0025):
        pose = np.asarray(obs['observation.eef_pose'], dtype=float)[:7]
        target = np.asarray(target, dtype=float)
        error = target - pose[:3]
        distance = float(np.linalg.norm(error))
        effective = max(0.0012, speed * distance / 0.02) if distance < 0.02 and speed > 0.002 else speed
        delta = error * min(0.35, effective / max(distance, 1e-09))
        desired = np.empty(4)
        mujoco.mju_mat2Quat(desired, np.asarray(rotation, dtype=float).ravel())
        orientation_error = np.empty(3)
        mujoco.mju_subQuat(orientation_error, desired, pose[3:7])
        angle = float(np.linalg.norm(orientation_error))
        action = self._hold()
        action[:3] = np.clip(delta / self.translation_scale, -1, 1)
        action[3:6] = orientation_error * min(0.35, min(0.05, self.rotation_scale) / max(angle, 1e-09)) / self.rotation_scale
        velocity = np.asarray(obs.get('observation.velocity', np.zeros(16)), dtype=float)
        reached = distance < tolerance and angle < 0.07 and (np.linalg.norm(velocity[:7]) < 0.4)
        self._progress(distance + 0.025 * angle)
        if reached and max(self._touch(obs)) < 0.15:
            self._last_safe_pose = (pose[:3].copy(), _mat(pose[3:7]))
        return (action, bool(reached))

    def _motion_checkpoint(self, obs, target, rotation, tolerance=0.0025):
        """A geometric transit point does not require joint velocity to vanish."""
        pose = np.asarray(obs['observation.eef_pose'], float)[:7]
        desired = np.empty(4)
        mujoco.mju_mat2Quat(desired, np.asarray(rotation, float).ravel())
        error = np.empty(3)
        mujoco.mju_subQuat(error, desired, pose[3:7])
        return bool(np.linalg.norm(np.asarray(target) - pose[:3]) < tolerance and np.linalg.norm(error) < 0.07)

    def _motion_recovery_target(self, waypoints):
        index = 0
        if len(waypoints) < 3:
            return index
        safe_z = getattr(self, '_preinsert', np.array([0.0, 0.0, np.inf]))[2] + 0.006
        previous = np.asarray(waypoints[0][0], float)
        if previous[2] < safe_z:
            return index
        first_direction = np.asarray(waypoints[1][0], float) - previous
        direction_length = float(np.linalg.norm(first_direction))
        if direction_length < 1e-08:
            return index
        first_direction /= direction_length
        traveled = 0.0
        for candidate in range(1, len(waypoints)):
            point, rotation = waypoints[candidate]
            point = np.asarray(point, float)
            segment = point - previous
            length = float(np.linalg.norm(segment))
            if point[2] < safe_z or traveled + length > 0.012:
                break
            angle = float(np.arccos(np.clip((np.trace(np.asarray(waypoints[0][1]).T @ np.asarray(rotation)) - 1.0) / 2.0, -1.0, 1.0)))
            if angle > 0.035 or (length > 1e-08 and float(segment @ first_direction) / length < np.cos(np.deg2rad(10.0))):
                break
            index = candidate
            traveled += length
            previous = point
        return index

    def _motion_retreat_sample(self, obs, scene):
        state, placement = self._placement_state(scene)
        release, opened, free = self._release_state(obs, scene)
        self._motion_verify_samples += 1
        admissible = state == 'valid' and bool(placement.get('stationary', False)) and opened and free and (release != 'invalid')
        self._motion_current_admissible = admissible
        identifier = placement.get('measurement_id')
        if identifier is not None and identifier not in self._placement_evidence_ids:
            self._placement_evidence_ids.add(identifier)
            self._motion_valid_samples = self._motion_valid_samples + 1 if admissible else 0
        if self.current_report is not None:
            self.current_report.update(placement_state=state, placement_reason=placement.get('reason'), target_count=placement.get('count', scene.get('target_count')), release_state=release, motion_parallel_verify_samples=self._motion_verify_samples)
        return (placement, opened, free)

    def _source_observation_target(self, scene):
        """Observe the current planned source frontier, not the last free pose.

        The last free pose may be over the target after release. Public source
        CAD plus its existing RGB box anchor supplies the current source view;
        a previously reached source high pose is the fallback.
        """
        source = scene.get('source_box') or {}
        frontier = scene.get('source_frontier') or {}
        if frontier.get('observation_scope') == 'full_column_identity' and source.get('center') is not None:
            center = np.asarray(source['center'], float)
            columns = np.unique(self.source_layout[:, 0])
            nominal = np.asarray(_get(self.geometry, 'source_bin_center_m', [0.175, 0.315]), float)
            group_center = np.r_[center[:2], float(_get(self.geometry, 'cookie_model_z_m', 0.7685))]
            group_center[0] += columns[self.source_column - 1] - nominal[0]
            pitch = (0.0, -15.0, -35.0, -35.0)[self.source_column - 1]
            rotation = _rotation('y', np.deg2rad(pitch)) @ CANONICAL
            span = 2 * float(_get(self.geometry, 'source_bin_half_size_m', [0.1186, 0.1010833])[1]) + 0.03
            viewing_distance = span / (2 * np.tan(np.deg2rad(70.0) / 2))
            position = group_center + [0.0, 0.0, self.half_size[2] + viewing_distance - 0.095] - rotation @ PAD_OFFSET
            self._source_observe_basis = dict(kind='measured_column_frontier', scope='full_column_identity', measurement_id=frontier.get('measurement_id'), view_position=position.tolist(), reason='CAD full-column field of view; precise initial row identity has not been observed')
            return (position, rotation)
        if frontier.get('frontier_center') is not None and frontier.get('confidence', 0.0) >= 0.45 and (int(frontier.get('source_column', 0)) == self.source_column):
            center = np.asarray(frontier['frontier_center'], float).copy()
            axis = _rotation('z', float(source.get('yaw', 0.0)))[:, 1]
            if frontier.get('state') == 'noncontiguous' and (frontier.get('repair_unit') or {}).get('center') is not None:
                center = np.asarray(frontier['repair_unit']['center'], float)
            else:
                center += axis * max(0, self.count - 1) * 0.008833333333333 / 2
            pitch = (0.0, -15.0, -35.0, -35.0)[self.source_column - 1]
            rotation = _rotation('y', np.deg2rad(pitch)) @ CANONICAL
            clearance = max(0.04, float(_get(self.geometry, 'source_bin_wall_height_m', 0.026)) + 0.014)
            position = center + [0.0, 0.0, 0.01 + clearance] - rotation @ PAD_OFFSET
            position[0] += 0.004 if self._source_recovery_count % 2 == 0 else -0.004
            self._source_observe_basis = dict(kind='measured_column_frontier', measurement_id=frontier.get('measurement_id'), frontier_center=frontier['frontier_center'], view_position=position.tolist(), confidence=frontier['confidence'], reason='overview RGB candidate; wrist precision required before insertion')
            return (position, rotation)
        center = source.get('center')
        columns = np.unique(self.source_layout[:, 0])
        indices = np.flatnonzero(np.isclose(self.source_layout[:, 0], columns[self.source_column - 1]))
        indices = indices[np.argsort(self.source_layout[indices, 1])]
        start = sum(self.batch_sizes[:self.batch_index])
        indices = indices[start:start + self.count]
        sign = 1.0 if self._source_recovery_count % 2 == 0 else -1.0
        if center is not None and float(source.get('confidence', 0.0) or 0.0) >= 0.5 and len(indices):
            delta = np.asarray(center, float)[:2] - np.asarray(_get(self.geometry, 'source_bin_center_m', [0.0, 0.0]), float)
            group_center = np.r_[self.source_layout[indices].mean(0) + delta, float(_get(self.geometry, 'cookie_model_z_m', 0.7685))]
            pitch = (0.0, -15.0, -35.0, -35.0)[self.source_column - 1]
            rotation = _rotation('y', np.deg2rad(pitch)) @ CANONICAL
            clearance = max(0.04, float(_get(self.geometry, 'source_bin_wall_height_m', 0.026)) + 0.014)
            position = group_center + [0.0, 0.0, 0.01 + clearance] - rotation @ PAD_OFFSET
            position[0] += sign * 0.004
            self._source_observe_basis = {'kind': 'public_source_box_and_planned_frontier', 'source_column': self.source_column, 'source_start_index': start, 'planned_indices': indices.astype(int).tolist(), 'source_box_confidence': float(source['confidence']), 'group_center': group_center.tolist(), 'view_position': position.tolist()}
            return (position, rotation)
        if self._safe_source_high is not None:
            position, rotation = self._safe_source_high
            position = position.copy()
            position[0] += sign * 0.004
            self._source_observe_basis = {'kind': 'previously_reached_source_high', 'view_position': position.tolist()}
            return (position, rotation.copy())
        return None

    def _group(self, scene):
        group = scene.get('group') or {}
        return group if group.get('center') is not None else None

    def _apply_group_geometry(self, group):
        self._selected_center = np.asarray(group['center'], dtype=float).copy()
        self._selected_rotation = np.asarray(group.get('rotation', np.eye(3)), dtype=float).copy()
        trusted_tilt = bool(group.get('tilt_verified', False)) and float(group.get('confidence', 0.0) or 0.0) >= 0.6
        if not trusted_tilt:
            yaw = float(np.arctan2(self._selected_rotation[1, 0], self._selected_rotation[0, 0]))
            self._selected_rotation = _rotation('z', yaw)
        self._selected_tilt_verified = trusted_tilt
        self._source_geometry_version = copy.deepcopy(group.get('geometry_version'))
        up = self._selected_rotation[:, 2]
        fall = np.arctan2(up[1], max(up[2], 0.1)) if trusted_tilt and float(group.get('tilt_deg', 0.0)) > 5 else 0.0
        pitch = (0.0, -15.0, -35.0, -35.0)[self.source_column - 1]
        self._pick_rotation = _rotation('y', np.deg2rad(pitch)) @ _rotation('x', -np.clip(fall, -0.6, 0.6)) @ CANONICAL
        if self._source_geometry_version is not None:
            self._source_geometry_version['required_eef_rotation'] = self._pick_rotation.copy()
        self._pick = self._selected_center + [0, 0, 0.01] - self._pick_rotation @ PAD_OFFSET
        self._clearance = max(0.04, float(_get(self.geometry, 'source_bin_wall_height_m', 0.026)) + 0.014)
        self._high = self._pick + [0.0, 0.0, self._clearance]
        self._preinsert = self._pick + [0.0, 0.0, 0.024]
        self._trial = self._pick + [0.0, 0.0, 0.012]
        self._held_cookie_local = self._pick_rotation.T @ self._selected_rotation
        self._group_offset = PAD_OFFSET - self._pick_rotation.T @ np.array([0.0, 0.0, 0.01])

    def _select(self, scene):
        frontier = scene.get('source_frontier') or {}
        if frontier.get('requires_wrist_view') and (frontier.get('frontier_center') is not None or frontier.get('observation_scope') == 'full_column_identity'):
            view = self._source_observation_target(scene)
            if view is not None:
                self._source_observe_pose = view
                self._source_recovery_count += 1
                self._rule_reason = 'whole-column overview finds remaining objects; acquire wrist geometry at actual frontier'
                self._event('source_reobserve', self._rule_reason)['observation_pose_basis'] = copy.deepcopy(self._source_observe_basis)
                self._advance('SOURCE_OBSERVE')
                return True
        group = self._group(scene)
        target = scene.get('target_box') or {}
        if group is None or target.get('center') is None or float(group.get('confidence', 0.0)) < 0.15:
            return False
        self.group = group
        self._apply_group_geometry(group)
        width = float(group.get('width_m', self.count * 2 * self.half_size[1] + max(0, self.count - 1) * 0.0025))
        self._initial_opening = opening_from_gap(width + 0.0015)
        self.opening = self._initial_opening
        self._min_opening = opening_from_gap(max(GAP_ZERO_M, self.count * 2 * self.half_size[1] - 0.0025))
        self._target_force = 0.45 if self.count == 1 else 1.3
        self._stable_force = 0.35 if self.count == 1 else 1.0
        self._held_width = None
        self._last_slip_measurement = None
        self._slip_exceedances = 0
        self._held_geometry_generation = None
        self._held_center_local = None
        self._held_pose_confidence = 0.0
        self._repairing = False
        self._repair_unit = None
        self._actual_held_count = None
        self._placement_evidence_ids = set()
        self._place_box_frozen = False
        self._placement_goal = None
        self._target_box_reference = copy.deepcopy(target)
        self._alignment_corrections = 0
        memory = self._source_memory.setdefault(self.batch_index, {'observation_count': 0, 'observations': [], 'failures': []})
        self._alignment_observations = memory['observation_count']
        self._alignment_attempt_views = 0
        self._alignment_geometry = None
        self._alignment_arrival_step = None
        self._alignment_view_stable = 0
        self._near_coarse_done = False
        self._last_alignment_measurement = None
        self._source_path = []
        self._carry_path = []
        self._carry_source_high_index = None
        self._carry_source_high_pose = None
        self._safe_source_high = None
        self._observe_count = 0
        self.ledger['attempts'] += 1
        self.ledger['state'] = 'source_approach'
        self._repair_unit = copy.deepcopy(group.get('source_unit'))
        self.current_report = self._report = {'batch_index': self.batch_index, 'planned_count': self.count, 'indices': list(self.ledger['planned_indices']), 'attempt': self.ledger['attempts'], 'variant': self.variant, 'count_verified': None, 'placement_verified': None, 'alignment_verified': None, 'tilt_deg': float(group.get('tilt_deg', 0.0)), 'group_confidence': float(group.get('confidence', 0.0)), 'position_error_m': float(group.get('position_error_m', 0.0))}
        tilt = float(group.get('tilt_deg', 0.0))
        pushes = self._push_attempts.get(self.batch_index, 0)
        before = self._push_before.get(self.batch_index)
        improved = before is None or tilt < before - 3.0
        if sum(self.batch_sizes[:self.batch_index]) > 0 and group.get('tilt_verified', False) and (tilt > 10.0) and (pushes < 2) and improved and (self.remaining_steps > 150):
            centers = np.asarray(scene.get('cookies', {}).get('centers', []), dtype=float)
            ids = group.get('indices', [])
            first = centers[int(ids[0])].copy() if len(ids) and len(centers) > int(ids[0]) else self._selected_center - self._selected_rotation[:, 1] * max(0, self.count - 1) * 0.0088333333 / 2
            floor = float(_get(self.geometry, 'source_floor_z_m', 0.753)) + 0.003
            self._push_start = np.array([first[0], first[1] - self._selected_rotation[1, 2] * self.half_size[2] - 0.012, floor + 0.004])
            self._push_end = self._push_start + [0, 0.016, 0]
            self._push_high = self._push_start + [0, 0, 0.05]
            self._push_rotation = self._pick_rotation if self.source_column == 4 else CANONICAL.copy()
            self._push_attempts[self.batch_index] = pushes + 1
            self._push_before[self.batch_index] = tilt
            self._event('source_straighten', 'visually verified group lean')
            self.opening = 0.1
            self._advance('PUSH_APPROACH')
        else:
            self._advance('APPROACH')
        return True

    def _place_pose(self, scene, clearance=0.0):
        candidate = scene['target_box']
        if self._target_box_reference is None:
            self._target_box_reference = copy.deepcopy(candidate)
        if not self._place_box_frozen:
            current_confidence = float(candidate.get('confidence', 0.0) or 0.0)
            visible = candidate.get('visible', True) is not False and (not candidate.get('occluded', False))
            rim_visible = float(candidate.get('visible_rim_fraction', 1.0) or 0.0) >= 0.65 or bool(candidate.get('anchor_confirmed', False))
            if visible and rim_visible and (current_confidence >= 0.65):
                self._target_box_reference = copy.deepcopy(candidate)
            if self.current_report is not None:
                self.current_report['target_box_reference'] = copy.deepcopy(self._target_box_reference)
                self.current_report['target_box_reference_step'] = self.total_steps
        box = self._target_box_reference
        box_rotation = _rotation('z', float(box.get('yaw', 0.0)))
        column = self.batch_index // 2 if self.is_2x10 else self.batch_index
        x = np.unique(self.slots[:, 0])[column]
        ys = self.slots[np.isclose(self.slots[:, 0], x), 1]
        if self.is_2x10:
            n = self.batch_sizes[0]
            rows = np.arange(n) if self.batch_index % 2 == 0 else np.arange(n, 10)
            y = float(np.mean(ys[rows])) + ((-0.004 if self.batch_index % 2 == 0 else 0.004) if n else 0.0)
        else:
            y = float(np.mean(ys))
        floor_top = float(_get(self.geometry, 'target_floor_z_m', 0.003)) + 0.003
        self._expected_local_center = np.array([x, y, floor_top + self.half_size[2] + 0.002])
        center = np.asarray(box['center'], dtype=float) + box_rotation @ (self._expected_local_center + [0, 0, clearance])
        tool_rotation = box_rotation @ self._held_cookie_local.T
        offset = self._held_center_local if self._held_center_local is not None else self._group_offset
        return (center - tool_rotation @ offset, tool_rotation)

    def _snapshot_pose(self, obs):
        pose = np.asarray(obs['observation.eef_pose'], dtype=float)[:7]
        return (pose[:3].copy(), _mat(pose[3:7]))

    def _remember_source_path(self, obs):
        if self.stage not in ('APPROACH', 'ALIGN_APPROACH', 'LOCAL_ALIGN', 'XY_COARSE', 'ALIGN_OBSERVE', 'ALIGN_RETURN', 'DESCEND') or max(self._touch(obs)) > 0.4:
            return
        pose = self._snapshot_pose(obs)
        if not self._source_path or np.linalg.norm(pose[0] - self._source_path[-1][0]) >= 0.003:
            self._source_path.append(pose)
            self._source_path = self._source_path[-36:]

    def _start_source_recovery(self, reason, carrying=False):
        if carrying and (not self._b2_return_allowed()):
            self._b2_begin(reason)
            return
        obs = self._obs
        event = self._event('return_source' if carrying else 'insertion_retreat', reason)
        if self.current_report is not None:
            self.current_report.update(outcome='recovery_needed', reason=reason)
            self.batch_reports.append(copy.deepcopy(self.current_report))
            self.current_report = self._report = None
        self.ledger['state'] = 'recovery_needed'
        self.ledger['recovery_indices'] = list(self.ledger['planned_indices'])
        self._source_recovery_count += 1
        self.opening = self.opening if carrying else self.actual_opening
        self._carrying_recovery = carrying
        if carrying:
            start = self._carry_source_high_index or 0
            self._recovery_waypoints = list(reversed(self._carry_path[start:]))
            if self._carry_source_high_pose is not None:
                if not self._recovery_waypoints or np.linalg.norm(self._recovery_waypoints[-1][0] - self._carry_source_high_pose[0]) > 0.002:
                    self._recovery_waypoints.append(self._carry_source_high_pose)
            elif not self._recovery_waypoints and self._target_path:
                self._recovery_waypoints = list(reversed(self._target_path))
                if self._safe_source_high is not None:
                    self._recovery_waypoints.append(self._safe_source_high)
            event['path_basis'] = 'reversed_reached_carry_path_to_source_high'
            if not self._recovery_waypoints:
                self._finish_unknown('no reached carrying path available for controlled return')
                return
        else:
            self._recovery_waypoints = list(reversed(self._source_path))
            if self._safe_source_high is not None:
                self._recovery_waypoints.append(self._safe_source_high)
            if not self._recovery_waypoints and self._last_safe_pose is not None:
                self._recovery_waypoints = [self._last_safe_pose]
        if not self._recovery_waypoints:
            self._finish_unknown('no previously reached free pose available for retreat')
        else:
            event['waypoint_count'] = len(self._recovery_waypoints)
            self._advance('RECOVER_PATH')

    def _retry(self, reason, obs=None, carrying=False):
        if obs is not None:
            self._obs = obs
        self._start_source_recovery(reason, carrying)

    def _held_update(self, scene):
        held = scene.get('held') or {}
        if self.variant != 'a' and float(held.get('pose_confidence', 0.0) or 0.0) >= 0.6:
            center = held.get('center_tool')
            rotation = held.get('rotation_tool')
            if center is not None and rotation is not None:
                center = np.asarray(center, dtype=float)
                rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
                if np.all(np.isfinite(center)) and np.linalg.norm(center) < 0.12 and np.all(np.isfinite(rotation)):
                    self._held_center_local = center.copy()
                    self._held_cookie_local = rotation.copy()
                    self._held_pose_confidence = float(held['pose_confidence'])
                    if self.current_report is not None:
                        self.current_report['post_grasp_pose_measured'] = True
                        self.current_report['held_center_tool'] = center.tolist()
                        self.current_report['held_rotation_tool'] = rotation.tolist()
        width = held.get('width_m')
        if width is not None and float(held.get('pose_confidence', 0.0) or 0.0) >= 0.5:
            self._held_width = float(width)
        count = held.get('count', scene.get('held_count'))
        confidence = float(held.get('count_confidence', scene.get('held_count_confidence', 0.0)) or 0.0)
        if count is not None and confidence >= 0.6:
            self._actual_held_count = int(count)
            if self.current_report is not None:
                self.current_report.update(observed_held_count=int(count), count_verified=int(count) == self.execution_count, held_count_measurement_id=held.get('count_measurement_id'))
            return int(count) == self.execution_count
        return None

    def _begin_count_reconcile(self, scene):
        held = scene.get('held') or {}
        identifier = held.get('count_measurement_id')
        if identifier is not None and identifier in self._handled_count_ids:
            return False
        self._count_resume_stage = self.stage
        self._count_observations = []
        self._count_hold_pose = self._snapshot_pose(self._obs)
        self._request_held_refresh = True
        self._rule_reason = 'independent held quantity disagrees; reconcile current region before selecting a return'
        self._advance('HELD_COUNT_RECONCILE')
        return True

    def _reconcile_count(self, scene):
        held = scene.get('held') or {}
        identifier = held.get('count_measurement_id')
        fresh = identifier is not None and identifier not in self._handled_count_ids
        if fresh:
            self._handled_count_ids.add(identifier)
            if held.get('count') is not None and float(held.get('count_confidence', 0.0) or 0.0) >= 0.6:
                self._count_observations.append(int(held['count']))
        consistent = len(self._count_observations) >= 2 and self._count_observations[-1] == self._count_observations[-2]
        if consistent:
            actual = self._count_observations[-1]
            if actual > 0:
                self._repair_unit = dict(unit_id=f'held:{self.batch_index}:{identifier}', location='held', count_interval=[actual, actual], object_hypotheses=list(self.ledger['planned_indices']), association='candidate identities; quantity measured independently', envelope=dict(width_m=self._held_width, center_tool=self._held_center_local, rotation_tool=self._held_cookie_local), gaps=None, protected_neighbors=[i for e in self.batch_ledger for i in e['placed_indices']], full_path_status='existing forward carry path; source return not established', measurement_id=identifier)
                self.ledger['actual_held_count_interval'] = [actual, actual]
                self.ledger['partial'] = actual != self.count
                self._rule_reason = 'measured physical count retained in independent RepairUnit; resume forward path with actual envelope'
                self._placement_goal = None
                self._request_held_refresh = False
                self._advance(self._count_resume_stage)
                return self._hold()
            if max(self._touch(self._obs)) < 0.12:
                location = scene.get('object_location') or {}
                self.ledger.update(actual_held_count_interval=[0, 0], state='hand_empty_location_pending')
                self._rule_reason = 'two fresh zero counts and clear fingers; inspect target instead of returning empty to source'
                self._request_held_refresh = False
                if location.get('state') == 'target' and location.get('confidence', 0.0) >= 0.6:
                    self._commit(scene.get('placement') or {}, False)
                else:
                    self._start_target_observation(self._rule_reason)
                return self._hold()
        if self.phase_steps >= 6:
            self.ledger['actual_held_count_interval'] = [0, self.count]
            self._rule_reason = 'held quantity still uncertain after fresh current-region samples; resume slowly with unknown inventory'
            self._request_held_refresh = False
            self._advance(self._count_resume_stage)
        return self._hold()

    def _confirm_observed_grasp(self, valid_count, obs):
        reference = getattr(self, '_grasp_reference_eef', None)
        rise = 0.0 if reference is None else float(np.asarray(obs['observation.eef_pose'])[2] - reference[2])
        gap_error = abs(gap_from_opening(self.actual_opening) - self.execution_count * 2 * self.half_size[1])
        touch = self._touch(obs)
        stable = min(touch) >= self._stable_force and max(touch) / max(min(touch), 0.1) < 3.0
        if valid_count is True and rise >= 0.009 and (gap_error <= 0.0025) and stable and (self.execution_count == self.count):
            association = (self._scene or {}).get('object_association') or {}
            ids = association.get('object_ids')
            identified = ids is not None and len(ids) == self.execution_count and (association.get('confidence', 0.0) >= 0.6) and (association.get('measurement_id') is not None) and (association.get('source_gap_verified') is True)
            actual_ids = [int(i) for i in ids] if identified else []
            interval = [self.count, self.count] if identified and set(actual_ids) == set(self.ledger['planned_indices']) else [0, self.count]
            self.ledger.update(taken_indices=actual_ids, held_indices=actual_ids, observed_object_ids=actual_ids, grasp_verified=True, identity_verified=True if identified else None, actual_held_count_interval=[self.execution_count, self.execution_count], held_batch_association_interval=interval)
            if self._repair_unit is not None:
                self._repair_unit.update(location='held', count_interval=[self.execution_count, self.execution_count], envelope=dict(frame='tool', width_m=self._held_width, center_tool=self._held_center_local, rotation_tool=self._held_cookie_local))
            self.current_report.update(grasp_verified=True, quantity_verified=True, identity_verified=True if identified else None, observed_object_ids=actual_ids, planned_object_hypotheses=list(self.ledger['planned_indices']), observed_trial_rise_m=rise)
            if not self._repairing:
                self._resolve_events('trial_lift_reached_with_independent_rgb_count_width_and_bilateral_contact', 'source')
            return True
        return False

    def _near_group_update(self, obs, scene):
        group = self._group(scene)
        if self._near_coarse_done or not group or float(group.get('confidence', 0.0)) < 0.6:
            return False
        measured = np.asarray(obs['observation.eef_pose'], dtype=float)[:3]
        if max(self._touch(obs)) > 0.4 or measured[2] - self._pick[2] < 0.018:
            return False
        newer = np.asarray(group['center'], dtype=float)
        delta = newer - self._selected_center
        delta[2] = 0.0
        if 0.0003 < np.linalg.norm(delta) <= 0.025:
            self._pick += delta
            self._trial += delta
            self._preinsert += delta
            self._high += delta
            self._selected_center += delta
            self._near_coarse_done = True
            self._last_progress = self.total_steps
            self.current_report['near_group_update_m'] = delta.tolist()
            self.current_report['near_group_confidence'] = float(group['confidence'])
            return True
        return False

    def _capture_alignment(self, obs, local):
        version = local.get('geometry_version') or {}
        old_version = (self._alignment_geometry or {}).get('geometry_version') or {}
        if (version.get('g2_active') or old_version.get('g2_active')) and version and old_version and (version.get('group_key') != old_version.get('group_key') or version.get('calibration_key') != old_version.get('calibration_key')):
            self._alignment_geometry = None
        if local.get('state') not in ('correct', 'aligned') or float(local.get('confidence', 0.0) or 0.0) < 0.6:
            return False
        measurement = local.get('measurement_id', local.get('measurement_step'))
        if self._alignment_geometry is not None and measurement is not None and (measurement == self._alignment_geometry.get('measurement_id', self._alignment_geometry.get('measurement_step'))):
            return True
        geometry = copy.deepcopy(local)
        world = geometry.get('desired_pad_mid_world')
        if world is None:
            position, rotation = self._snapshot_pose(obs)
            correction = np.asarray(local.get('correction_world', [0.0, 0.0, 0.0]), float)
            world = position + rotation @ PAD_OFFSET + correction
        world = np.asarray(world, float)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            return False
        geometry['desired_pad_mid_world'] = world.copy()
        geometry['controller_capture_step'] = self.total_steps
        geometry['measurement_eef'] = np.asarray(obs['observation.eef_pose'])[:7].copy()
        self._alignment_geometry = geometry
        return True

    def _alignment_now(self, obs, scene):
        local = scene.get('local_align') or {}
        self._capture_alignment(obs, local)
        geometry = self._alignment_geometry
        if geometry is None or self.total_steps - geometry['controller_capture_step'] > 30:
            return local
        local = copy.deepcopy(geometry)
        position, rotation = self._snapshot_pose(obs)
        error = np.asarray(local['desired_pad_mid_world'], float) - position - rotation @ PAD_OFFSET
        axis = np.asarray(local.get('observable_axis_world', self._selected_rotation[:, 1]), float).copy()
        axis[2] = 0.0
        axis /= max(np.linalg.norm(axis), 1e-09)
        broad = np.asarray(local.get('broad_axis_world', self._selected_rotation[:, 0]), float).copy()
        broad[2] = 0.0
        broad -= axis * float(broad @ axis)
        broad /= max(np.linalg.norm(broad), 1e-09)
        relative, broad_error = (float(error @ axis), float(error @ broad))
        correction = axis * relative + broad * broad_error
        opening_error = float(local.get('opening_error_m', 0.0) or 0.0)
        if local.get('gap_inner_width_m') is not None:
            opening_error = abs(gap_from_opening(self.actual_opening) - float(local['gap_inner_width_m']))
        gap = float(local.get('gap_width_m', 0.006) or 0.006)
        uncertainty = float(local.get('uncertainty_m', 0.001) or 0.001)
        aligned = abs(relative) + uncertainty < (gap - 0.001) / 2 and opening_error < 0.0008 and (abs(broad_error) < 0.003)
        local.update(correction_world=correction, relative_error_m=abs(relative), relative_signed_error_m=relative, broad_error_m=broad_error, observable_axis_world=axis, broad_axis_world=broad, valid=aligned, state='aligned' if aligned else 'correct', opening_error_m=opening_error)
        return local

    def _local_align(self, obs, scene):
        supplied = scene.get('local_align') or {}
        if self._alignment_geometry is None and (not (supplied.get('state') in ('correct', 'aligned') and float(supplied.get('confidence', 0.0) or 0.0) >= 0.6)):
            self._near_group_update(obs, scene)
        local = self._alignment_now(obs, scene)
        confidence = float(local.get('confidence', 0.0) or 0.0)
        state = local.get('state', 'unknown')
        if state in ('correct', 'aligned') and confidence >= 0.6:
            correction = np.asarray(local.get('correction_world', [0, 0, 0]), dtype=float)
            if correction.shape != (3,) or not np.all(np.isfinite(correction)):
                correction = np.zeros(3)
            measured = np.asarray(obs['observation.eef_pose'], dtype=float)[:3]
            axis = np.asarray(local.get('observable_axis_world', self._selected_rotation[:, 1]), dtype=float)
            axis /= max(np.linalg.norm(axis), 1e-09)
            broad = np.asarray(local.get('broad_axis_world', self._selected_rotation[:, 0]), dtype=float)
            broad /= max(np.linalg.norm(broad), 1e-09)
            broad_error = float(local.get('broad_error_m', correction @ broad) or 0.0)
            measurement = local.get('measurement_id', local.get('measurement_step'))
            fresh = measurement is None or measurement != self._last_alignment_measurement
            if abs(broad_error) >= 0.003 and max(self._touch(obs)) < 0.4 and fresh:
                step = correction.copy()
                step[2] = 0.0
                step *= min(1.0, 0.003 / max(np.linalg.norm(step), 1e-09))
                desired = measured + step
                delta = desired - self._preinsert
                delta[2] = 0.0
                for name in ('_pick', '_preinsert', '_trial', '_high', '_selected_center'):
                    getattr(self, name)[:] += delta
                self._xy_coarse_target = self._preinsert.copy()
                self._xy_coarse_target[2] = max(self._preinsert[2], measured[2])
                self._last_alignment_measurement = measurement
                self._alignment_corrections += 1
                self._near_coarse_done = False
                self.current_report.update(xy_coarse_error_m=abs(broad_error), xy_coarse_correction_m=step.tolist(), xy_coarse_measurement=measurement, alignment_verified=False)
                self._advance('XY_COARSE')
                return self._hold()
            desired = measured + correction
            step = axis * float((desired - self._preinsert) @ axis)
            step *= min(1.0, 0.001 / max(np.linalg.norm(step), 1e-09))
            if fresh and np.linalg.norm(step) > 0.0001:
                for name in ('_pick', '_preinsert', '_trial', '_high', '_selected_center'):
                    getattr(self, name)[:] += step
                self._alignment_corrections += 1
            if fresh:
                self._last_alignment_measurement = measurement
            opening = local.get('recommended_opening')
            if opening is not None:
                self.opening = float(np.clip(opening, 0, 1))
            action, reached = self._servo(obs, self._preinsert, self._pick_rotation, 0.001, 0.0007)
            self.current_report.update(alignment_corrections=self._alignment_corrections, local_alignment_reason=local.get('reason'), local_relative_error_m=local.get('relative_error_m'), local_broad_error_m=broad_error, alignment_geometry_frame=measurement)
            self._near_coarse_done = reached and abs(broad_error) < 0.003
            if reached and bool(local.get('valid', False)):
                self.current_report['alignment_verified'] = True
                self._advance('DESCEND')
            elif self.phase_steps >= 14 and self.total_steps - self._last_progress >= 12:
                self.current_report['local_alignment_stall_reason'] = 'relative correction made no further observed pose/alignment progress'
                self._request_alignment_observation(obs, scene)
            return action
        action, reached = self._servo(obs, self._preinsert, self._pick_rotation, 0.0015, 0.001)
        stalled = self.phase_steps >= 14 and self.total_steps - self._last_progress >= 12
        if not reached and (not stalled):
            return action
        if self.phase_steps >= 5:
            if stalled:
                self.current_report['near_observation_pose_residual_m'] = float(np.linalg.norm(self._preinsert - np.asarray(obs['observation.eef_pose'])[:3]))
            self._request_alignment_observation(obs, scene)
        return action

    def _request_alignment_observation(self, obs, scene):
        group = self._group(scene)
        if self._alignment_attempt_views < 2:
            self._alignment_attempt_views += 1
            memory = self._source_memory.setdefault(self.batch_index, {'observation_count': 0, 'observations': [], 'failures': []})
            view_index = memory['observation_count']
            memory['observation_count'] += 1
            self._alignment_observations = memory['observation_count']
            offsets = [(0.004, 0.0, 0.0, 0.06), (-0.004, 0.0, 0.0, -0.06), (0.0, -0.006, 0.004, 0.08), (0.0, 0.006, 0.004, -0.08), (0.008, -0.004, 0.006, 0.1), (-0.008, 0.004, 0.006, -0.1)]
            dx, dy, dz, angle = offsets[view_index % len(offsets)]
            base, measured_rotation = self._snapshot_pose(obs)
            self._alignment_view = (base + [dx, dy, dz], measured_rotation @ _rotation('z', angle))
            self._alignment_arrival_step = None
            self._alignment_arrival_measurement = None
            self._alignment_view_stable = 0
            event = self._event('source_reobserve', 'local fingertip-to-gap estimate unknown or not progressing')
            event['view_index'] = view_index
            self._alignment_event_id = event['id']
            self._advance('ALIGN_OBSERVE')
        elif group and float(group.get('confidence', 0.0)) >= 0.45:
            self.current_report.update(alignment_verified=False, alignment_mode='unknown_force_guided_probe')
            self._advance('DESCEND')
        else:
            self._start_source_recovery('local group remains unobserved after distinct near views')

    def _release_state(self, obs, scene):
        opened = abs(self.actual_opening - self._release_goal) < 0.012
        free = max(self._touch(obs)) < 0.12
        held = scene.get('held') or {}
        following = held.get('following')
        released = held.get('released')
        pose_confidence = max(float(held.get('pose_confidence', 0.0) or 0.0), float(held.get('motion_confidence', 0.0) or 0.0))
        if following is True and pose_confidence >= 0.6:
            return ('invalid', opened, free)
        if opened and free and (released is True or following is False):
            return ('valid', opened, free)
        return ('unknown', opened, free)

    def _placement_state(self, scene):
        placement = scene.get('placement') or {}
        confidence = float(placement.get('confidence', 0.0) or 0.0)
        state = str(placement.get('state', 'unknown'))
        if confidence < 0.6 or state not in ('valid', 'invalid'):
            return ('unknown', placement)
        if state == 'valid':
            count = placement.get('count')
            count_confidence = float(placement.get('count_confidence', 0.0) or 0.0)
            if count is None or count_confidence < 0.6:
                return ('unknown', placement)
            expected = self.count if placement.get('count_scope') == 'batch' else sum(self.batch_sizes[:self.batch_index + 1])
            if int(count) != expected:
                return ('invalid', {**placement, 'reason': 'reliable placement count differs from the required batch/target count'})
        return (state, placement)

    def _commit(self, placement, verified=True):
        entry = self.ledger
        if self.execution_count != self.count:
            verified = False
            entry['partial'] = True
            entry['repair_unit'] = copy.deepcopy(self._repair_unit)
        entry['state'] = 'committed' if verified else 'placed_unverified'
        entry['placement_verified'] = True if verified else None
        entry['held_indices'] = []
        if verified:
            entry['taken_indices'] = list(entry['observed_object_ids'])
            entry['placed_indices'] = list(entry['observed_object_ids'])
            entry['recovery_indices'] = []
            self.completed_indices.extend((i for i in entry['observed_object_ids'] if i not in self.completed_indices))
            bounds = placement.get('target_local_bounds')
            if bounds is not None:
                entry['target_local_bounds'] = np.asarray(bounds, dtype=float).tolist()
            self._resolve_events('batch_placement_visually_valid_and_released')
        self.current_report.update(outcome='committed' if verified else 'placed_unverified', placement_verified=True if verified else None, release_state=self._release_state(self._obs, self._scene)[0])
        self.batch_reports.append(copy.deepcopy(self.current_report))
        self.current_report = self._report = None
        self.batch_index += 1
        self._repairing = False
        self._repair_attempts = 0
        self._observe_count = 0
        self._observe_signature = None
        self._source_recovery_count = 0
        self._skip_zero_batches()
        if not self.finished:
            self._advance('SELECT')

    def _start_target_observation(self, reason):
        needs_exit = self._target_clear_pose is not None and np.asarray(self._obs['observation.eef_pose'])[2] < self._target_clear_pose[0][2] - 0.003 and (self.stage in ('PLACE', 'OPEN', 'RELEASE_LIFT', 'REPAIR_DESCEND', 'REPAIR_CLOSE', 'REPAIR_LIFT'))
        if needs_exit:
            self.opening = self.opening if min(self._touch(self._obs)) >= 0.1 else self.actual_opening
            self._pending_target_observation = reason
            self._target_exit_path = list(reversed(self._target_path)) or [self._target_clear_pose]
            if np.linalg.norm(self._target_exit_path[-1][0] - self._target_clear_pose[0]) > 0.002:
                self._target_exit_path.append(self._target_clear_pose)
            self._event('target_contact_retreat', reason)
            self._advance('TARGET_RETREAT')
            return
        pose = self._snapshot_pose(self._obs)
        if not hasattr(self, '_target_observe_base') or self._observe_count == 0:
            self._target_observe_base = (pose[0].copy(), pose[1].copy())
        offsets = [(0.008, 0.0, 0.04), (-0.008, 0.0, -0.04), (0.0, 0.008, 0.08), (0.0, -0.008, -0.08), (0.012, 0.006, 0.1), (-0.012, -0.006, -0.1), (0.008, -0.008, 0.14), (-0.008, 0.008, -0.14)]
        limit = min(len(offsets), 2 + self.remaining_steps // 120)
        if self._observe_count >= limit:
            self._finish_unknown('placement unknown after distinct reached views without sufficient audit evidence')
            return
        dx, dy, angle = offsets[self._observe_count]
        self._observe_count += 1
        base, rotation = self._target_observe_base
        self._target_view = (base + [dx, dy, 0.006], rotation @ _rotation('z', angle))
        event = self._event('target_reobserve', reason)
        self._observe_event_id = event['id']
        self._advance('TARGET_OBSERVE')

    def _start_target_repair(self, placement):
        if placement.get('count_scope') != 'batch':
            self._start_target_observation('column occupancy has no isolated RepairUnit; acquire individual object association')
            return
        center = placement.get('center')
        rotation = placement.get('rotation')
        if not placement.get('repairable', False) or center is None or rotation is None:
            self._start_target_observation('confirmed invalid placement has no reliable regrasp pose')
            return
        if self.remaining_steps < 45:
            self._finish_unknown('insufficient remaining control budget for observed target repair')
            return
        actual = placement.get('count')
        if actual is None or float(placement.get('count_confidence', 0.0) or 0.0) < 0.6:
            self._start_target_observation('repair quantity remains uncertain')
            return
        self._repairing = True
        self._placement_evidence_ids = set()
        self._repair_unit = dict(unit_id=f'target:{self.batch_index}:{placement.get('measurement_id')}', location='target', count_interval=[int(actual), int(actual)], object_hypotheses=[], envelope={'center': center, 'rotation': rotation, 'width_m': placement.get('width_m')}, gaps=placement.get('gaps'), protected_neighbors=[i for e in self.batch_ledger for i in e['placed_indices']], full_path_status='existing observed target approach', measurement_id=placement.get('measurement_id'))
        self._place_box_frozen = False
        self._placement_goal = None
        target = self._scene.get('target_box') or {}
        if target.get('center') is not None and float(target.get('confidence', 0.0) or 0.0) >= 0.65 and (not target.get('occluded', False)):
            self._target_box_reference = copy.deepcopy(target)
        self._repair_attempts += 1
        self.ledger['state'] = 'target_repair'
        self.ledger['recovery_indices'] = list(self.ledger['planned_indices'])
        self._event('target_regrasp_reseat', str(placement.get('reason', 'invalid placement')))
        obj_rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        self._repair_rotation = obj_rotation @ CANONICAL
        self._repair_pick = np.asarray(center, dtype=float) + obj_rotation[:, 2] * 0.01 - self._repair_rotation @ PAD_OFFSET
        self._repair_high = self._repair_pick + [0, 0, 0.04]
        width = float(placement.get('width_m', int(actual) * 2 * self.half_size[1]))
        self.opening = opening_from_gap(width + 0.0015)
        self._min_opening = opening_from_gap(max(GAP_ZERO_M, int(actual) * 2 * self.half_size[1] - 0.0025))
        self._held_cookie_local = self._repair_rotation.T @ obj_rotation
        self._group_offset = PAD_OFFSET - self._repair_rotation.T @ (obj_rotation[:, 2] * 0.01)
        self._held_center_local = None
        self._held_pose_confidence = 0.0
        self._held_width = None
        self._advance('REPAIR_APPROACH')

    def act(self, obs, scene):
        self.total_steps += 1
        self.phase_steps += 1
        self._obs, self._scene = (obs, scene)
        self._inventory = copy.deepcopy(scene.get('object_inventory') or {})
        self.actual_opening = float(np.asarray(obs['observation.state'], dtype=float)[7])
        if self.finished:
            return self._hold()
        target_geometry = scene.get('target_geometry') or {}
        target_measurement = target_geometry.get('measurement_id')
        if target_measurement != self._v9_target_measurement:
            self._v9_target_measurement = target_measurement
            if target_geometry.get('occupancy_epoch_changed'):
                for entry in self.batch_ledger:
                    entry.pop('target_local_bounds', None)
                if self.stage in ('TRANSPORT', 'PREPLACE_OBSERVE'):
                    self._place_box_frozen = False
                    self._placement_goal = None
                if self.current_report is not None:
                    self.current_report['target_geometry_version'] = target_geometry.get('geometry_version')
        self._remember_source_path(obs)
        touch = self._touch(obs)
        pose, actual_rotation = self._snapshot_pose(obs)
        if self.stage in ('TRIAL_LIFT', 'LIFT', 'TRANSPORT', 'PLACE', 'PREPLACE_OBSERVE'):
            current = (pose.copy(), actual_rotation.copy())
            if not self._carry_path or np.linalg.norm(pose - self._carry_path[-1][0]) >= 0.003:
                self._carry_path.append(current)
        if self.stage in ('PLACE', 'REPAIR_DESCEND') and max(touch) < 0.4:
            current = (pose.copy(), actual_rotation.copy())
            if not self._target_path or np.linalg.norm(pose - self._target_path[-1][0]) >= 0.003:
                self._target_path.append(current)
                self._target_path = self._target_path[-24:]
        moving = self.stage not in ('SELECT', 'CLOSE', 'VERIFY', 'LOCAL_ALIGN', 'OPEN', 'REPAIR_CLOSE', 'HELD_COUNT_RECONCILE', 'SLIP_RECHECK', 'SLIP_VIEW', 'COMPLETION_CANDIDATE')
        if moving and self._metric is not None and (self.total_steps - self._last_progress > 35):
            if self.stage == 'PREPLACE_OBSERVE':
                self._event('target_preplacement_view', 'bounded target view did not reach; continue with measured uncertainty')
                self._advance('TRANSPORT')
                return self._hold()
            elif self.stage.startswith('RECOVER'):
                self._finish_unknown('previously reached recovery path made no observed progress')
            elif self.stage == 'TARGET_RETREAT':
                self._finish_unknown('previously reached target retreat path made no observed progress')
            elif self.stage.startswith('REPAIR') or self.stage in ('RETRACT', 'RELEASE_LIFT', 'TARGET_OBSERVE'):
                self._start_target_observation('target motion stalled in public pose feedback')
            else:
                self._start_source_recovery(self.stage.lower() + ' stopped making pose progress', carrying=self.stage in ('TRIAL_LIFT', 'LIFT', 'TRANSPORT', 'PLACE'))
            return self._hold()
        if self.stage == 'SLIP_RECHECK':
            return self._b2_step(obs, scene)
        if self.stage == 'SLIP_VIEW':
            return self._b2_view_step(obs, scene)
        if self.stage == 'COMPLETION_CANDIDATE':
            return self._hold()
        if self.stage == 'HELD_COUNT_RECONCILE':
            self._held_update(scene)
            return self._reconcile_count(scene)
        if self.stage == 'SELECT':
            if not self._select(scene) and self.phase_steps > 12:
                source_view = self._source_observation_target(scene)
                if self.remaining_steps > 35 and source_view is not None:
                    self._source_observe_pose = source_view
                    self._source_recovery_count += 1
                    event = self._event('source_reobserve', 'source group unavailable')
                    event['observation_pose_basis'] = copy.deepcopy(self._source_observe_basis)
                    self._advance('SOURCE_OBSERVE')
                else:
                    self._finish_unknown('source group unavailable with no useful reached observation pose')
            return self._hold()
        if self.stage == 'SOURCE_OBSERVE':
            destination, rotation = self._source_observe_pose
            error = destination - pose
            distance = float(np.linalg.norm(error))
            high_free_transit = max(touch) < 0.12 and pose[2] >= destination[2] - 0.003 and (getattr(self, '_source_observe_basis', {}).get('kind') in ('public_source_box_and_planned_frontier', 'previously_reached_source_high', 'measured_column_frontier'))
            speed = 0.003
            if high_free_transit and distance > 0.02:
                speed = 0.015 if np.linalg.norm(error[:2]) >= abs(error[2]) else 0.01
            action, reached = self._servo(obs, destination, rotation, speed)
            if reached:
                self._advance('SELECT')
            return action
        if self.stage == 'APPROACH':
            action, reached = self._servo(obs, self._high, self._pick_rotation, 0.018)
            checkpoint = max(touch) < 0.4 and pose[2] >= self._preinsert[2] + 0.01 and self._motion_checkpoint(obs, self._high, self._pick_rotation, 0.004)
            if reached or checkpoint:
                self._safe_source_high = (pose.copy(), actual_rotation.copy())
                self._source_path = [self._safe_source_high]
                self._advance('ALIGN_APPROACH' if self.variant != 'a' else 'DESCEND')
            return action
        if self.stage == 'ALIGN_APPROACH':
            if np.linalg.norm(pose - self._preinsert) < 0.012:
                self._near_group_update(obs, scene)
            free_distance = float(np.linalg.norm(pose - self._preinsert))
            speed = 0.0045 if max(touch) < 0.4 and free_distance > 0.006 else 0.0025
            action, reached = self._servo(obs, self._preinsert, self._pick_rotation, speed)
            if reached:
                self._advance('LOCAL_ALIGN')
            return action
        if self.stage == 'LOCAL_ALIGN':
            return self._local_align(obs, scene)
        if self.stage == 'XY_COARSE':
            action, reached = self._servo(obs, self._xy_coarse_target, self._pick_rotation, 0.0025, 0.0012)
            self._xy_stable = getattr(self, '_xy_stable', 0) + 1 if reached else 0
            if self._xy_stable >= 2:
                self._xy_stable = 0
                self._advance('LOCAL_ALIGN')
            return action
        if self.stage == 'ALIGN_OBSERVE':
            action, reached = self._servo(obs, *self._alignment_view, 0.0015, 0.0025)
            if not reached:
                self._alignment_view_stable = 0
                return action
            self._alignment_view_stable += 1
            local = scene.get('local_align') or {}
            measurement = local.get('measurement_id', local.get('measurement_step'))
            if self._alignment_arrival_step is None:
                self._alignment_arrival_step = self.total_steps
                self._alignment_arrival_measurement = measurement
                return action
            new_frame = measurement is not None and measurement != self._alignment_arrival_measurement
            usable = self._alignment_view_stable >= 2 and new_frame and self._capture_alignment(obs, local)
            if usable or self._alignment_view_stable >= 4:
                memory = self._source_memory[self.batch_index]
                result = {'step': self.total_steps, 'frame_id': measurement, 'result': 'new_geometry' if usable else 'still_unobservable', 'position': pose.tolist(), 'confidence': float(local.get('confidence', 0.0) or 0.0)}
                memory['observations'].append(result)
                memory['observations'] = memory['observations'][-12:]
                event = self.recovery_events[self._alignment_event_id]
                event.update(observation_result=result, sample_step=self.total_steps)
                self._advance('ALIGN_RETURN')
            return action
        if self.stage == 'ALIGN_RETURN':
            action, reached = self._servo(obs, self._preinsert, self._pick_rotation, 0.0015, 0.0025)
            if reached:
                self._advance('LOCAL_ALIGN')
            return action
        if self.stage == 'DESCEND':
            low_force_stall = max(touch) > 8.0 and self.total_steps - self._last_progress > 3
            self._jam = self._jam + 1 if max(touch) > 20.0 or low_force_stall else 0
            if self._jam >= 3:
                self._start_source_recovery('insertion force rose while downward pose progress stopped')
                return self._hold()
            remaining = np.linalg.norm(pose - self._pick)
            slower = sum(self.batch_sizes[:self.batch_index]) > 0 or self._source_recovery_count > 0
            speed = (0.0025 if remaining > 0.015 else 0.0012) if slower else 0.0035 if remaining > 0.006 else 0.0012
            if self.variant != 'a' and self.current_report.get('alignment_verified') is not True:
                speed = min(speed, 0.0012)
            action, reached = self._servo(obs, self._pick, self._pick_rotation, speed, 0.0013)
            if reached:
                self._advance('CLOSE')
            return action
        if self.stage in ('CLOSE', 'REPAIR_CLOSE'):
            repair = self.stage == 'REPAIR_CLOSE'
            if self._last_close_opening is None or self._last_close_opening - self.actual_opening > 0.0005 or float(min(touch)) - self._last_close_min_force > 0.1:
                self._last_progress = self.total_steps
            self._last_close_opening = self.actual_opening
            self._last_close_min_force = float(min(touch))
            if min(touch) < self._target_force:
                self.opening = max(self._min_opening, self.opening - (0.004 if max(touch) > 0.1 else 0.015))
            gap = gap_from_opening(self.actual_opening)
            expected = self.execution_count * 2 * self.half_size[1]
            width_ok = abs(gap - expected) <= 0.0025
            force_ok = min(touch) >= self._stable_force and max(touch) / max(min(touch), 0.1) < 3.0
            self._stable = self._stable + 1 if width_ok and force_ok else 0
            target, rotation = (self._repair_pick, self._repair_rotation) if repair else (self._pick, self._pick_rotation)
            action, _ = self._servo(obs, target, rotation, 0.0008)
            if self._stable >= 3:
                self._held_width = gap
                self._grasp_reference_eef = np.asarray(obs['observation.eef_pose'], dtype=float)[:7].copy()
                if not repair:
                    self._carry_path = [(pose.copy(), actual_rotation.copy())]
                    self._carry_source_high_index = None
                    self._carry_source_high_pose = None
                self.ledger['state'] = 'grasp_candidate'
                self.ledger['extracted_candidate_indices'] = list(self.ledger['planned_indices'])
                self.current_report.update(grip_touch_n=touch.tolist(), measured_gripper_opening=self.actual_opening, measured_inner_gap_m=gap, compressed_width_residual_m=gap - expected, width_consistent=True, force_consistent=True)
                if repair:
                    self._repair_unit['location'] = 'held'
                    self._repair_trial = pose + [0, 0, 0.012]
                    self._advance('REPAIR_LIFT')
                else:
                    self._advance('TRIAL_LIFT')
            elif self.total_steps - self._last_progress > 22:
                if repair:
                    self._start_target_observation('target regrasp did not obtain consistent width and bilateral force')
                else:
                    self._start_source_recovery('grip width or bilateral contact did not stabilize')
            return action
        if self.stage in ('TRIAL_LIFT', 'LIFT', 'REPAIR_LIFT'):
            trial_checkpoint = self.stage == 'TRIAL_LIFT'
            target = self._repair_trial if self.stage == 'REPAIR_LIFT' else self._high
            rotation = self._repair_rotation if self.stage == 'REPAIR_LIFT' else self._pick_rotation
            action, reached = self._servo(obs, target, rotation, 0.0024 if self.stage == 'REPAIR_LIFT' else 0.0045)
            if trial_checkpoint:
                checkpoint = self._trial.copy()
                checkpoint[2] = pose[2]
                reached = pose[2] >= self._trial[2] - 0.0013 and self._motion_checkpoint(obs, checkpoint, rotation, 0.003)
            elif self.stage == 'LIFT':
                reached = reached or self._motion_checkpoint(obs, self._high, rotation)
            self._low_force = self._low_force + 1 if min(touch) < 0.1 else 0
            if self._low_force >= 5:
                if self._repairing:
                    self._start_target_observation('target group lost bilateral contact during trial lift')
                else:
                    self._start_source_recovery('grip force lost during lift', carrying=True)
                return action
            valid_count = self._held_update(scene)
            if valid_count is False:
                if self._repairing:
                    self._start_target_observation('target repair held-count evidence disagrees')
                else:
                    self._begin_count_reconcile(scene)
            elif reached:
                self._confirm_observed_grasp(valid_count, obs)
                if self.stage == 'TRIAL_LIFT':
                    self._advance('LIFT')
                else:
                    if self.stage == 'LIFT':
                        self._b2_departure_width = self._held_width or gap_from_opening(self.actual_opening)
                        self._carry_source_high_pose = (pose.copy(), actual_rotation.copy())
                        self._carry_path.append(self._carry_source_high_pose)
                        self._carry_source_high_index = len(self._carry_path) - 1
                    self.ledger['state'] = 'holding' if valid_count else 'holding_unverified'
                    self._advance('TRANSPORT')
            return action
        if self.stage == 'PREPLACE_OBSERVE':
            target = self._v9_preplace_waypoints[0] if self._v9_preplace_waypoints else self._v9_preplace_view[0]
            action, reached = self._servo(obs, target, self._v9_preplace_view[1], 0.008)
            if reached and self._v9_preplace_waypoints:
                self._v9_preplace_waypoints.pop(0)
                self._waypoint_reached()
            if reached and (not self._v9_preplace_waypoints) and (target_measurement != self._v9_preplace_measurement):
                candidate = scene['target_box']
                if float(candidate.get('confidence', 0.0) or 0.0) >= 0.65 and (not candidate.get('occluded', False)):
                    self._target_box_reference = copy.deepcopy(candidate)
                self._place_box_frozen = False
                self._placement_goal = None
                self._advance('TRANSPORT')
            return action
        if self.stage in ('TRANSPORT', 'PLACE'):
            valid_count = self._held_update(scene)
            if valid_count is False and self._begin_count_reconcile(scene):
                return self._hold()
            held = scene.get('held') or {}
            generation = held.get('geometry_generation')
            if generation != self._held_geometry_generation:
                self._held_geometry_generation = generation
                self._slip_exceedances = 0
                self._last_slip_measurement = None
            slip = float(held.get('slip_m', scene.get('slip_m', 0.0)) or 0.0)
            slip_confident = self.variant != 'a' and float(held.get('pose_confidence', 0.0) or 0.0) >= 0.6
            measurement = held.get('measurement_id', held.get('measurement_step', obs.get('time')))
            fresh = measurement is None or measurement != self._last_slip_measurement
            if fresh:
                self._slip_exceedances = self._slip_exceedances + 1 if slip_confident and slip > 0.004 else 0
                self._last_slip_measurement = measurement
            self._request_held_refresh = bool(slip_confident and slip > 0.0025)
            if self._slip_exceedances >= 2:
                if self._repairing:
                    self._start_target_observation('measured target group slipped during replacement')
                else:
                    self._b2_begin('measured object-to-tool translation changed by over 4 mm')
                return self._hold()
            if self.stage == 'PLACE' and self._placement_goal is not None:
                target, rotation = self._placement_goal
            else:
                target, rotation = self._place_pose(scene, self._clearance if self.stage == 'TRANSPORT' else 0.0)
            speed = 0.015 if self.stage == 'TRANSPORT' else 0.0045 if np.linalg.norm(pose - target) > 0.012 else 0.0012
            interval = self.ledger.get('actual_held_count_interval')
            if interval is not None and interval[0] != interval[1]:
                speed = min(speed, 0.003)
            if slip_confident and slip > 0.0025:
                speed = min(speed, 0.003)
            action, reached = self._servo(obs, target, rotation, speed)
            if reached:
                self._put_position, self._put_rotation = (target.copy(), rotation.copy())
                if self.stage == 'TRANSPORT':
                    view = target_geometry.get('preplacement_observation') or {}
                    if self.batch_index not in self._v9_preplace_observed and view.get('needed') and view.get('valid'):
                        self._v9_preplace_observed.add(self.batch_index)
                        self._v9_preplace_measurement = target_measurement
                        self._v9_preplace_view = (np.asarray(view['eef_position'], float), np.asarray(view['eef_rotation'], float))
                        self._v9_preplace_waypoints = [np.asarray(p, float) for p in view.get('waypoints', [view['eef_position']])]
                        self._event('target_preplacement_view', str(view.get('reason', 'visible target contour conflicts with anchor')))
                        if self.current_report is not None:
                            self.current_report['preplacement_view'] = copy.deepcopy(view)
                        self._advance('PREPLACE_OBSERVE')
                        return self._hold()
                    self._place_box_frozen = True
                    place_position, place_rotation = self._place_pose(scene, 0.0)
                    self._placement_goal = (place_position.copy(), place_rotation.copy())
                    if self.current_report is not None:
                        self.current_report['target_box_frozen_step'] = self.total_steps
                        self.current_report['placement_goal_position'] = place_position.tolist()
                        self.current_report['placement_goal_rotation'] = place_rotation.tolist()
                    self._target_clear_pose = (pose.copy(), actual_rotation.copy())
                    self._target_path = [self._target_clear_pose]
                    self._advance('PLACE')
                else:
                    self.ledger['state'] = 'placed_pending_release'
                    actual_width = self._held_width if self._held_width is not None else gap_from_opening(self.actual_opening)
                    self._release_goal = opening_from_gap(actual_width + 0.006)
                    self.current_report.update(release_actual_width_m=actual_width, release_gap_goal_m=gap_from_opening(self._release_goal), release_opening_goal=self._release_goal, post_grasp_pose_measured=self._held_pose_confidence >= 0.6)
                    self._advance('OPEN')
            return action
        if self.stage == 'OPEN':
            self.opening = min(self._release_goal, max(self.opening, self.actual_opening) + 0.015)
            action, _ = self._servo(obs, self._put_position, self._put_rotation, 0.0008)
            release, opened, free = self._release_state(obs, scene)
            self.current_report.update(release_state=release, actual_release_opening=self.actual_opening, actual_opening_reached=opened, bilateral_contact_cleared=free)
            if opened:
                self._stable = self._stable + 1
                if self._stable >= 3:
                    self._retract = pose + [0, 0, 0.045]
                    if self._target_clear_pose is not None:
                        self._retract[2] = max(pose[2], self._target_clear_pose[0][2])
                    self._motion_verify_samples = self._motion_valid_samples = 0
                    self._advance('RETRACT')
            else:
                self._stable = 0
                if self.phase_steps > 25 and max(touch) > 1.0:
                    self.opening = self.actual_opening
                    self._release_lift = pose + [0, 0, 0.006]
                    self._event('release_disengage', 'opening blocked by continuing fingertip contact')
                    self._advance('RELEASE_LIFT')
                    return self._hold()
            return action
        if self.stage == 'RELEASE_LIFT':
            action, reached = self._servo(obs, self._release_lift, self._put_rotation, 0.001)
            if reached:
                self._advance('OPEN')
            return action
        if self.stage == 'RETRACT':
            action, reached = self._servo(obs, self._retract, self._put_rotation, 0.006)
            placement, opened, free = self._motion_retreat_sample(obs, scene)
            clearance_reached = reached or self._motion_checkpoint(obs, self._retract, self._put_rotation)
            if clearance_reached:
                self.ledger['state'] = 'placed_pending_audit'
                self._target_observe_base = (pose.copy(), actual_rotation.copy())
                if self.variant != 'c' and self._motion_verify_samples >= 6 and opened and free:
                    self._commit(placement, self._motion_current_admissible)
                else:
                    self._advance('VERIFY')
            return action
        if self.stage == 'VERIFY':
            state, placement = self._placement_state(scene)
            release, opened, free = self._release_state(obs, scene)
            signature = (state, placement.get('count'), round(float(placement.get('confidence', 0.0) or 0.0), 2))
            if self._observe_signature is not None and signature != self._observe_signature:
                self._observe_progress.append({'step': self.total_steps, 'before': self._observe_signature, 'after': signature})
            self._observe_signature = signature
            self.current_report.update(placement_state=state, placement_reason=placement.get('reason'), target_count=placement.get('count', scene.get('target_count')), release_state=release, observation_passes=self._observe_count, observation_progress=list(self._observe_progress))
            admissible = state == 'valid' and bool(placement.get('stationary', False)) and opened and free and (release != 'invalid')
            identifier = placement.get('measurement_id')
            if identifier is not None and identifier not in self._placement_evidence_ids:
                self._placement_evidence_ids.add(identifier)
                self._stable = self._stable + 1 if admissible else 0
            if admissible:
                self._commit(placement, True)
            elif self.phase_steps >= 6:
                if self.variant != 'c' and opened and free:
                    self._commit(placement, False)
                elif state == 'invalid' or release == 'invalid':
                    self._start_target_repair(placement)
                else:
                    self._start_target_observation('placement audit lacks independent pose/support evidence')
            return self._hold()
        if self.stage == 'TARGET_OBSERVE':
            action, reached = self._servo(obs, *self._target_view, 0.003)
            if reached:
                self._advance('VERIFY')
            return action
        if self.stage == 'TARGET_RETREAT':
            if not self._target_exit_path:
                self._start_target_observation(self._pending_target_observation)
                return self._hold()
            action, reached = self._servo(obs, *self._target_exit_path[0], 0.0025)
            if reached:
                self._target_exit_path.pop(0)
                self._waypoint_reached()
            return action
        if self.stage == 'REPAIR_APPROACH':
            action, reached = self._servo(obs, self._repair_high, self._repair_rotation, 0.006)
            if reached:
                self._target_clear_pose = (pose.copy(), actual_rotation.copy())
                self._target_path = [self._target_clear_pose]
                self._advance('REPAIR_DESCEND')
            return action
        if self.stage == 'REPAIR_DESCEND':
            action, reached = self._servo(obs, self._repair_pick, self._repair_rotation, 0.0012)
            self._jam = self._jam + 1 if max(touch) > 12.0 else 0
            if self._jam >= 3:
                self.opening = self.actual_opening
                self._start_target_observation('target regrasp insertion obstructed')
                return self._hold()
            elif reached:
                self._advance('REPAIR_CLOSE')
            return action
        if self.stage == 'RECOVER_PATH':
            if not self._recovery_waypoints:
                if self._carrying_recovery:
                    self._advance('RECOVER_LOWER')
                else:
                    self.opening = max(self.actual_opening, self._initial_opening)
                    self._advance('RECOVER_CLEAR')
                return self._hold()
            lookahead = self._motion_recovery_target(self._recovery_waypoints)
            point, rotation = self._recovery_waypoints[lookahead]
            action, reached = self._servo(obs, point, rotation, 0.0035)
            if lookahead:
                reached = reached or self._motion_checkpoint(obs, point, rotation)
            if reached:
                del self._recovery_waypoints[:lookahead + 1]
                self._waypoint_reached()
            return action
        if self.stage == 'RECOVER_LOWER':
            action, reached = self._servo(obs, self._pick, self._pick_rotation, 0.0012)
            if reached:
                self._release_goal = opening_from_gap((self._held_width or gap_from_opening(self.actual_opening)) + 0.006)
                self.opening = self._release_goal
                self._advance('RECOVER_SOURCE_OPEN')
            return action
        if self.stage == 'RECOVER_SOURCE_OPEN':
            if abs(self.actual_opening - self._release_goal) < 0.012 and max(touch) < 0.12:
                self._advance('RECOVER_SOURCE_UP')
            elif self.phase_steps > 25:
                self._finish_unknown('source return release still lacks actual opening/contact clearance')
            return self._hold()
        if self.stage == 'RECOVER_SOURCE_UP':
            target, rotation = self._safe_source_high if self._safe_source_high is not None else (self._high, self._pick_rotation)
            action, reached = self._servo(obs, target, rotation, 0.0035)
            if reached:
                self.ledger.update(state='reobserve_source', held_indices=[], taken_indices=[], extracted_candidate_indices=[])
                self._advance('SELECT')
            return action
        if self.stage == 'RECOVER_CLEAR':
            if abs(self.actual_opening - self.opening) < 0.015:
                self._advance('SELECT')
            return self._hold()
        if self.stage == 'PUSH_APPROACH':
            action, reached = self._servo(obs, self._push_high, self._push_rotation, 0.008)
            if reached:
                self._advance('PUSH_DESCEND')
            return action
        if self.stage in ('PUSH_DESCEND', 'PUSH_SWEEP'):
            target = self._push_start if self.stage == 'PUSH_DESCEND' else self._push_end
            action, reached = self._servo(obs, target, self._push_rotation, 0.0018 if self.stage == 'PUSH_DESCEND' else 0.0012)
            self._jam = self._jam + 1 if max(touch) > 12.0 else 0
            if self._jam >= 3:
                self.opening = self.actual_opening
                self._advance('PUSH_RETRACT')
            elif reached:
                self._advance('PUSH_SWEEP' if self.stage == 'PUSH_DESCEND' else 'PUSH_RETRACT')
            return action
        if self.stage == 'PUSH_RETRACT':
            action, reached = self._servo(obs, self._push_high, self._push_rotation, 0.0035)
            if reached:
                self._advance('SELECT')
            return action
        return self._hold()
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.source1.d_supervisor import DSupervisor

class RuleControllerD(DSupervisor, RuleController):
    pass
RuleController = RuleControllerD
