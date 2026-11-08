"""Target-only post-plan observation and five-object reseating experiment.

No score/info/episode environment is accepted. Visual matched never ends an episode.
"""
from __future__ import annotations
import copy
import numpy as np
import mujoco
from v10_leaves_17e0363838a9.two5_source1.legacy.rule_controller import CANONICAL, PAD_OFFSET, gap_from_opening, opening_from_gap, _mat
from v10_leaves_17e0363838a9.two5_source1.target_vision import VisionEstimator, _rot_z, _project, _sample_map
from v10_leaves_17e0363838a9.two5_source1.slot_audit import SlotAuditor
from v10_leaves_17e0363838a9.two5_source1.path_feasibility import PathFeasibility
from v10_leaves_17e0363838a9.two5_source1.repair_costs import COSTS
from v10_leaves_17e0363838a9.two5_source1.public_ik import PublicIK

def repair_candidate(report, box, ledger, source_column, half):
    """Only a complete isolated five-band group; never convert column capacity to count."""
    if not box or not box.get('anchor_confirmed') or float(box.get('confidence', 0.0)) < 0.6 or box.get('occluded'):
        return (None, 'fresh trusted target pose unavailable')
    columns = report.get('columns', [])
    if report.get('geometry_version') != box.get('geometry_version'):
        return (None, 'object and box geometry versions differ')
    if len(columns) != 2 or any((c.get('state') != 'measured' or c.get('count') != 5 or (not c.get('cross_view_consistent')) for c in columns)):
        return (None, 'two target columns need five separately observed objects each')
    options = []
    for c in columns:
        if c.get('visually_slot_matched'):
            continue
        if not c.get('cameras'):
            continue
        errors = np.asarray(c['internal_residual_xy_m'], float)
        if np.max(abs(errors[:, 1])) > 0.0015 or np.max(abs(errors[:, 0])) > 0.004:
            continue
        views = [v for v in c['per_camera'] if v.get('state') == 'measured' and v.get('count') == 5]
        if any((min((o['top_band_width_m'] for o in v['objects'])) < 1.8 * half[0] for v in views)):
            continue
        shift = np.asarray(c['common_offset_xy_m'], float)
        if np.linalg.norm(shift) < 0.0007:
            continue
        options.append((float(np.linalg.norm(shift)), c))
    if not options:
        return (None, 'spacing/tilt, incomplete views or no resolved common offset: obtain another view')
    _, c = max(options, key=lambda x: x[0])
    column = int(c['column'])
    points = np.asarray(c['centers_local'], float)
    z = float(report.get('object_center_z_local_m', 0.0))
    if points.shape == (5, 2):
        points = np.column_stack((points, np.full(5, z)))
    lo = points.min(0) - half
    hi = points.max(0) + half
    center = (lo + hi) / 2
    size = (hi - lo) / 2
    entry = ledger[column]
    if entry['planned_count'] != 5:
        return (None, 'original batch association is not five objects')
    mid = report['measurement_id']
    version = report.get('geometry_version')
    ids = [f'target:{column}:band:{i}:{mid}' for i in range(5)]
    other = columns[1 - column]
    protected = [dict(object_id=f'target:{1 - column}:band:{i}:{mid}', center_local=list(p[:2]) + [z], rotation_local=np.eye(3).tolist(), half_size=half.tolist(), box_key='target_box', geometry_version=version) for i, p in enumerate(other['centers_local'])]
    unit = dict(unit_id=f'terminal:{column}:{mid}', scope='batch', object_scope='batch', batch_index=column, source_column=source_column, count_interval=[5, 5], object_ids=ids, frame='box', original_planned_indices=list(entry['planned_indices']), native_ids_confirmed=False, association_basis='five RGB instances in the original batch target column; native IDs are hypotheses', center_local=center.tolist(), half_size=size.tolist(), rotation_local=np.eye(3).tolist(), geometry_version=version, measurement_id=mid, common_offset_xy_m=c['common_offset_xy_m'], uncertainty_calibrated=False, protected_objects=protected)
    return (unit, 'five whole RGB top bands, common offset and independent original batch association')

class TerminalController:

    def __init__(self, parent, context, *, vision=None, checker=None, auditor=None):
        self.parent = parent
        self.base = parent.controller
        self.config = parent.vision.config
        self.geometry = self.config.cookie_transfer
        self.half = np.asarray(self.geometry.cookie_half_size_m, float)
        self.slots = np.asarray(self.geometry.target_slots_local_m, float)
        self.total_steps = int(parent.step)
        self.max_steps = int(self.base.max_steps)
        self.stage = 'COMPLETE_CANDIDATE'
        self.command = float(self.base.opening)
        self.stage_steps = 0
        self.observe_steps = 0
        self.attempted = set()
        self.unit = None
        self.rule_reason = 'parent two-batch plan ended; official termination remains external'
        self.events = []
        self.last_report = None
        self.box = None
        self.last_scene = {}
        self.path = []
        self.index = 0
        self.stable = 0
        self.costs = COSTS['medians']
        self.target = vision if vision is not None else VisionEstimator(self.config)
        for key in ('camera_adjustments', 'wrist_mount', 'wrist_calibration', 'cookie_colors', 'calibration_generation'):
            setattr(self.target, key, copy.deepcopy(getattr(parent.vision, key)))
        self.target.v6_target_anchor = None
        self.target.v9_initial_target_prior = copy.deepcopy((parent.vision.boxes or {}).get('target_box'))
        self.auditor = auditor if auditor is not None else SlotAuditor(self.config, self.target._v6_color_mask)
        self.checker = checker if checker is not None else PathFeasibility(self.config)
        self.last_evidence = {}
        self._grasp_reference = None
        self.fresh_held_ids = set()
        self.held = None
        self.rollback = None
        self.ik = PublicIK(self.target.robot_model, self.config) if hasattr(self.target, 'robot_model') else None
        self.origin_eef = None
        self.view_assessment = {}
        self._preaudit_handoff = False
        self._parent_target_anchor = None
        self.preaudit_evidence = {}
        self._preaudit_unit = None
        self._preaudit_contact_epoch = 0
        self._preaudit_touching = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def phase_name(self):
        return self.stage

    @property
    def finished(self):
        return False

    @property
    def public_evidence(self):
        return dict(measurement_id=(self.last_report or {}).get('measurement_id'), geometry_version=(self.last_report or {}).get('geometry_version'), rule_reason=self.rule_reason, terminal_phase=self.stage, slot_measurement=self.last_report, repair_unit=self.unit, held=self.held, path_assessment=self.last_evidence, remaining_steps=self.max_steps - self.total_steps, target_box=self.box, target_geometry=self.target.v9_target_geometry, view_assessment=self.view_assessment, raw_slot_matching_is_official_success=False, repair_events=self.events, preaudit=self.preaudit_evidence)

    def prepare_from_parent(self, obs):
        """Observe alongside the parent's unchanged final retract/verify action."""
        self.total_steps = int(self.parent.step)
        self.observe_steps += 1
        self._observe_parent_anchor(obs)
        release, opened, free = self.base._release_state(obs, self.base._scene or {})
        if release == 'valid' and opened and free:
            unit, _ = repair_candidate(self.last_report or {}, self.box, self.base.batch_ledger, self.base.source_column, self.half)
            if unit is not None:
                self._preaudit_unit = dict(unit=copy.deepcopy(unit), report=copy.deepcopy(self.last_report), box=copy.deepcopy(self.box), contact_epoch=self._preaudit_contact_epoch, cached_at_policy_step=self.total_steps)

    def _observe_parent_anchor(self, obs):
        for key in ('camera_adjustments', 'wrist_mount', 'wrist_calibration', 'cookie_colors', 'calibration_generation'):
            setattr(self.target, key, copy.deepcopy(getattr(self.parent.vision, key)))
        touching = max(self.base._touch(obs)) >= 0.12
        if touching and (not self._preaudit_touching):
            self._preaudit_contact_epoch += 1
            self._preaudit_unit = None
        self._preaudit_touching = touching
        current = copy.deepcopy((self.base._scene or {}).get('target_box') or (self.parent.vision.boxes or {}).get('target_box') or {})
        signature = self.target._v9_target_calibration_signature()
        same_calibration = str(current.get('geometry_version', '')).startswith(signature + ':target:')
        if self._parent_target_anchor is not None and (not same_calibration or current.get('geometry_version') != self._parent_target_anchor.get('geometry_version')):
            self._parent_target_anchor = None
            self._preaudit_unit = None
        if same_calibration and current.get('anchor_confirmed') and current.get('visible') and (not current.get('occluded')) and (float(current.get('confidence', 0.0) or 0.0) >= 0.6):
            self._parent_target_anchor = copy.deepcopy(current)
        self.box = copy.deepcopy(self._parent_target_anchor or current)
        if not same_calibration:
            self.box['anchor_confirmed'] = False
        self.target.v6_target_anchor = copy.deepcopy(self._parent_target_anchor)
        self.target.v9_target_epoch = int(getattr(self.parent.vision, 'v9_target_epoch', 0))
        self.target.v9_target_geometry = copy.deepcopy(getattr(self.parent.vision, 'v9_target_geometry', {}))
        cameras = self.target._robot_cameras(obs)
        cameras['left_wrist'] = self.target._v6_wrist_camera(cameras['left_wrist'], obs)
        cameras = {n: self.target._adjust_camera(c, self.target.camera_adjustments.get(n, np.zeros(7))) for n, c in cameras.items()}
        self.cameras = cameras
        version = self.box.get('geometry_version')
        views = self.box.get('pose_diagnostics') or {}
        accepted = {n: c for n, c in cameras.items() if n in ('front', 'left_wrist') and views.get(n, {}).get('valid')}
        left = views.get('left_wrist', {})
        if self._parent_target_anchor is not None and left.get('valid') and (left.get('complete') or left.get('metric_observable')):
            accepted = {'left_wrist': cameras['left_wrist']}
        self.last_report = self.auditor.measure(obs, accepted, self.box, version)
        self.last_report['object_center_z_local_m'] = float(self.auditor.floor + self.half[2])
        self.last_report['measurement_selection'] = 'fresh RGB slots in the original confirmed parent target/calibration frame'
        self.last_scene = dict(target_box=self.box, geometry_version=version)
        self.preaudit_evidence = dict(parent_phase=self.base.stage, anchor_basis='latest real confirmed parent target pose; original measurement identity retained', anchor_measurement_id=(self._parent_target_anchor or {}).get('measurement_id'), anchor_measurement_step=(self._parent_target_anchor or {}).get('measurement_step'), current_parent_box_visible=current.get('visible'), slot_measurement_id=self.last_report.get('measurement_id'), slot_measurement_time=self.last_report.get('measurement_time'), parent_motion_unchanged=True)

    def _cached_preaudit_unit(self, obs):
        cached = self._preaudit_unit
        if cached is None:
            return (None, None)
        if cached['contact_epoch'] != self._preaudit_contact_epoch or max(self.base._touch(obs)) >= 0.12 or cached['box'].get('geometry_version') != (self.box or {}).get('geometry_version'):
            self._preaudit_unit = None
            return (None, 'cached unit invalidated by contact or target geometry epoch')
        oldbox = cached['box']
        newbox = self.box
        oldr = _rot_z(float(oldbox.get('yaw', 0.0)))
        newr = _rot_z(float(newbox.get('yaw', 0.0)))
        oldcols = cached['report']['columns']
        newcols = (self.last_report or {}).get('columns', [])
        comparisons = []
        for old in oldcols:
            current = next((x for x in newcols if x.get('column') == old['column']), {})
            if current.get('state') != 'measured':
                comparisons.append(dict(column=old['column'], state='occluded_or_unknown_preserve_world_measurement'))
                continue
            if current.get('count') != old.get('count'):
                self._preaudit_unit = None
                return (None, 'new observed row count contradicts cached unit')
            points = np.asarray(current['centers_local'], float)
            world = np.column_stack((points, np.full(len(points), self.auditor.floor + self.half[2]))) @ newr.T + newbox['center']
            oldframe = (world - np.asarray(oldbox['center'])) @ oldr
            differences = oldframe[:, :2] - np.asarray(old['centers_local'], float)

            def metric_sigma(box):
                return np.asarray(((box.get('pose_diagnostics') or {}).get('left_wrist') or {}).get('position_std_m', [box.get('position_error_m', 0.004)] * 2), float)
            allowance = metric_sigma(oldbox) + metric_sigma(newbox)
            allowance[0] += float(oldbox.get('position_error_m', 0.0)) + float(newbox.get('position_error_m', 0.0))
            allowance[1] += float(old.get('pixel_sigma_y_m', 0.0)) + float(current.get('pixel_sigma_y_m', 0.0)) + max(float(oldbox.get('position_error_m', 0.0)), float(newbox.get('position_error_m', 0.0)))
            comparisons.append(dict(column=old['column'], max_change_xy_m=np.max(abs(differences), axis=0).tolist(), reported_interval_overlap_allowance_m=allowance.tolist()))
            if np.any(abs(differences) > allowance):
                self._preaudit_unit = None
                return (None, 'fresh object positions contradict retained complete-unit measurement')
        self.preaudit_evidence.update(retained_unit_measurement_id=cached['unit']['measurement_id'], retained_unit_measurement_time=cached['report']['measurement_time'], retained_unit_age_seconds=float(obs['time']) - float(cached['report']['measurement_time']), retained_unit_contact_epoch=cached['contact_epoch'], current_image_comparison=comparisons)
        self.box = copy.deepcopy(oldbox)
        self.last_scene = dict(target_box=self.box, geometry_version=self.box.get('geometry_version'))
        return (copy.deepcopy(cached['unit']), 'retained complete measured unit; fresh RGB shows no resolved state contradiction')

    def _hold(self):
        action = np.zeros(14)
        action[6] = 2 * self.command - 1
        action[13] = 1.0
        return action

    def _go(self, stage, reason):
        self.events.append(dict(step=self.total_steps, from_stage=self.stage, to_stage=stage, reason=reason))
        self.stage = stage
        self.stage_steps = 0
        self.rule_reason = reason
        self.stable = 0

    def _servo(self, obs, position, rotation, speed=0.004):
        pose = np.asarray(obs['observation.eef_pose'], float)[:7]
        error = np.asarray(position) - pose[:3]
        distance = np.linalg.norm(error)
        delta = error * min(0.35, speed / max(distance, 1e-09))
        desired = np.empty(4)
        mujoco.mju_mat2Quat(desired, np.asarray(rotation).ravel())
        angular = np.empty(3)
        mujoco.mju_subQuat(angular, desired, pose[3:7])
        angle = np.linalg.norm(angular)
        action = self._hold()
        action[:3] = delta / self.base.translation_scale
        action[3:6] = angular * min(0.35, 0.04 / max(angle, 1e-09)) / self.base.rotation_scale
        reached = bool(distance < 0.0015 and angle < 0.025 and (np.linalg.norm(np.asarray(obs['observation.velocity'])[:7]) < 0.4))
        return (np.clip(action, -1, 1), reached)

    def _observe(self, obs):
        if self._preaudit_handoff:
            self._preaudit_handoff = False
            self._observe_parent_anchor(obs)
            return
        self.target.step += 1
        cameras = self.target._robot_cameras(obs)
        cameras['left_wrist'] = self.target._v6_wrist_camera(cameras['left_wrist'], obs)
        cameras = {n: self.target._adjust_camera(c, self.target.camera_adjustments.get(n, np.zeros(7))) for n, c in cameras.items()}
        self.box = self.target._v9_fit_target(obs, cameras)
        self.target.boxes = {'target_box': self.box}
        self.cameras = cameras
        version = self.box.get('geometry_version', self.target.v9_target_geometry.get('geometry_version'))
        self.box['geometry_version'] = version
        views = self.target.v9_target_geometry.get('per_camera', {})
        accepted = {n: c for n, c in cameras.items() if n in ('front', 'left_wrist') and views.get(n, {}).get('valid')}
        left = views.get('left_wrist', {})
        if self.box.get('anchor_confirmed') and left.get('valid') and (left.get('complete') or left.get('metric_observable')):
            accepted = {'left_wrist': cameras['left_wrist']}
        self.last_report = self.auditor.measure(obs, accepted, self.box, version)
        self.last_report['measurement_selection'] = 'current metric wrist when independently observable; overview retained in target_geometry'
        self.last_report['object_center_z_local_m'] = float(self.auditor.floor + self.half[2])
        self.last_scene = dict(target_box=self.box, geometry_version=version)

    def _free_view(self, obs):
        pose = np.asarray(obs['observation.eef_pose'], float)[:7]
        rotation = _mat(pose[3:7])
        best = None
        if self.origin_eef is None:
            self.origin_eef = pose[:3].copy()
        if not self.box or self.box.get('center') is None:
            return None
        camera = self.cameras['left_wrist']
        image = obs['observation.images.left_wrist']
        mask, _ = self.target._v6_gripper_silhouette(camera, image.shape)
        points = self.target._box_points('target') @ _rot_z(self.box.get('yaw', 0.0)).T + np.asarray(self.box['center'])
        deltas = [np.array([dx, dy, dz]) for dz in (0.02, 0.035, 0.05, 0.07) for dx, dy in ((0.0, 0.0), (-0.018, 0.0), (0.018, 0.0), (0.0, -0.018), (0.0, 0.018))]
        h, w = image.shape[:2]
        f = h / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
        look_at = np.asarray(self.box['center']) + [0, 0, float(self.geometry.target_bin_wall_height_m) / 2]
        for depth in (0.14, 0.17, 0.2, 0.24, 0.28, 0.3):
            for iy in (0.2, 0.32, 0.44):
                for ix in (0.3, 0.5, 0.7):
                    ray = np.array([(ix * w - (w - 1) / 2) * depth / f, ((h - 1) / 2 - iy * h) * depth / f, -depth])
                    delta = look_at - camera['rotation'] @ ray - camera['pos']
                    if delta[2] >= 0.012:
                        deltas.append(delta)
        candidates = []
        attempts = []
        maximum_coverage = 0.0
        for delta in deltas:
            if delta[2] > 0.1 or np.linalg.norm(delta[:2]) > 0.09 or pose[2] + delta[2] > self.origin_eef[2] + 0.1:
                continue
            view = {**camera, 'pos': camera['pos'] + delta}
            uv, z = _project(points, view, image.shape)
            occ, inside = _sample_map((mask > 0).astype(float), uv, default=1.0)
            visible = inside & (z > 0.01) & (occ < 0.1)
            score = float(np.mean(visible))
            maximum_coverage = max(maximum_coverage, score)
            edges = visible[:96].reshape(4, 24).mean(1)
            two_edges = any((edges[j] >= 0.8 and edges[(j + 1) % 4] >= 0.8 for j in range(4)))
            if score < 0.65 and (not two_edges):
                continue
            candidates.append((float(np.linalg.norm(delta)), score, delta))
        for length, score, delta in sorted(candidates, key=lambda x: (x[0], -x[1])):
            dz = float(delta[2])
            waypoints = [dict(name='terminal_view_rise', position=pose[:3] + [0, 0, dz], rotation=rotation, opening=float(np.asarray(obs['observation.state'])[7]), carrying=False, box_key='target_box'), dict(name='terminal_view_translate', position=pose[:3] + delta, rotation=rotation, opening=float(np.asarray(obs['observation.state'])[7]), carrying=False, box_key='target_box')]
            protected = self._protected_all()
            check = self.checker.check_path(obs, waypoints, None, {**self.last_scene, 'path_scope': 'empty_hand_observation'}, protected_objects=protected)
            ik = self.ik.check_ik_path(obs, waypoints) if check['state'] == 'feasible' and self.ik is not None else {'state': 'unknown'}
            attempts.append(dict(delta=delta.tolist(), rim_coverage=score, path=check, ik=ik))
            if check['state'] == 'feasible' and ik['state'] == 'reachable':
                best = (length, waypoints, check)
                break
        self.view_assessment = dict(candidates=len(candidates), maximum_predicted_rim_coverage=maximum_coverage, assessed=attempts, selected=best is not None, reason='complete bounded RGB view and full D/F path checks')
        if best:
            return (best[1], dict(path=best[2], ik=attempts[-1]['ik'], view=self.view_assessment))
        return None

    def _protected_all(self):
        objects = []
        for c in (self.last_report or {}).get('columns', []):
            if c.get('state') != 'measured' or c.get('count') != 5:
                inner = np.asarray(self.geometry.target_bin_half_size_m) - self.geometry.bin_wall_thickness_m
                return [dict(object_id='unsegmented_target_exclusion_region', box_key='target_box', bounds_local=[[-float(inner[0]), float(inner[0])], [-float(inner[1]), float(inner[1])], [float(self.auditor.floor), float(self.geometry.target_bin_wall_height_m) + 0.03]], geometry_version=self.box.get('geometry_version'), evidence_source='fixed CAD target-region exclusion for view motion; quantity and identities unknown')]
            for i, p in enumerate(c['centers_local']):
                objects.append(dict(object_id=f'view:{c['column']}:{i}', center_local=list(p[:2]) + [float(self.auditor.floor + self.half[2])], half_size=self.half.tolist(), rotation_local=np.eye(3).tolist(), box_key='target_box', geometry_version=self.box.get('geometry_version')))
        return objects if len(objects) == 10 else None

    def _plan(self, obs, unit, pinch_height_m=0.01):
        br = _rot_z(self.box.get('yaw', 0.0))
        r = br @ CANONICAL
        center = np.asarray(self.box['center']) + br @ np.asarray(unit['center_local'])
        pick = center + [0, 0, pinch_height_m] - r @ PAD_OFFSET
        high = pick + [0, 0, 0.065]
        pre = pick + [0, 0, 0.024]
        width = 2 * float(unit['half_size'][1])
        opened = opening_from_gap(width + 0.003)
        closed = opening_from_gap(width)
        release = opening_from_gap(width + 0.006)
        xs = np.unique(self.slots[:, 0])
        ys = self.slots[np.isclose(self.slots[:, 0], xs[unit['batch_index']]), 1]
        local = np.array([xs[unit['batch_index']], float(np.mean(ys)), float(self.auditor.floor + self.half[2] + 0.002)])
        relative = r.T @ (center - pick)
        desired_center = np.asarray(self.box['center']) + br @ local
        place = desired_center - r @ relative

        def wp(name, p, opening, carrying=False, reference=False, speed=0.004):
            return dict(name=name, position=np.asarray(p), rotation=r, opening=opening, carrying=carrying, reference=reference, box_key='target_box', speed=speed)
        path = [wp('approach_high', high, opened), wp('preinsert', pre, opened), wp('descend', pick, opened, speed=0.0015), wp('close', pick, closed, True), wp('trial_lift', pick + [0, 0, 0.045], closed, True, speed=0.003), wp('lift', high, closed, True), wp('place_high', place + [0, 0, 0.065], closed, True), wp('place_pre', place + [0, 0, 0.024], closed, True), wp('place', place, closed, True, True, speed=0.0015), wp('release', place, release, False), wp('withdraw', place + [0, 0, 0.045], release, False)]
        envelope = dict(center_tool=relative, rotation_tool=r.T @ br, half_size=unit['half_size'], width_m=width, measurement_id=unit['measurement_id'])
        check = self.checker.check_path(obs, path, envelope, self.last_scene, repair_unit=unit, protected_objects=unit['protected_objects'])
        ik = self.ik.check_ik_path(obs, path) if self.ik is not None else {'state': 'unknown'}
        rollback = [wp('return_original', pick, closed, True, speed=0.002), wp('rollback_release', pick, opened, False), wp('rollback_withdraw', high, opened, False)]
        rbcheck = self.checker.check_path(obs, [path[0], path[1], path[2], path[3], path[4], *rollback], envelope, self.last_scene, repair_unit=unit, protected_objects=unit['protected_objects'])
        rbik = self.ik.check_ik_path(obs, [path[0], path[1], path[2], path[3], path[4], *rollback]) if self.ik is not None else {'state': 'unknown'}
        pose = np.asarray(obs['observation.eef_pose'])[:3]
        moves = 0
        for w in path:
            if w['name'] not in ('close', 'release'):
                moves += int(np.ceil(np.linalg.norm(w['position'] - pose) / w['speed'])) + 4
            pose = w['position']
        cost = int(moves + self.costs.get('CLOSE', 16) + self.costs.get('OPEN', 7) + self.costs.get('VERIFY', 6) + 8)
        self.last_evidence = dict(forward=check, rollback=rbcheck, ik=ik, rollback_ik=rbik, estimated_steps=cost, remaining_steps=self.max_steps - self.total_steps, cost_basis='public parent_e measured CLOSE/OPEN/VERIFY medians + geometric moves at existing servo speeds + observation reserve')
        if check['state'] != 'feasible' or rbcheck['state'] != 'feasible' or ik['state'] != 'reachable' or (rbik['state'] != 'reachable'):
            conflicts = (check.get('conflicts') or []) + (rbcheck.get('conflicts') or [])
            if pinch_height_m == 0.01 and conflicts and all(('wall' in c['obstacle'] and c['piece'].startswith('L_finger_') for c in conflicts)):
                shapes = self.checker.gripper_vertices_tool(obs, opened)
                pieces = {c['piece'] for c in conflicts}
                bottom = min((float((np.asarray(shapes[name]) @ r.T + pick)[:, 2].min()) for name in pieces))
                wall_top = float(self.box['center'][2] + self.geometry.target_bin_wall_height_m)
                rise = wall_top + self.checker.collision_epsilon_m - bottom
                cookie_top = float(center[2] + unit['half_size'][2])
                if rise > 0 and cookie_top > wall_top + self.checker.collision_epsilon_m:
                    accepted = self._plan(obs, unit, pinch_height_m + rise)
                    self.last_evidence['grasp_height_recovery'] = dict(delta_m=rise, pinch_height_m=pinch_height_m + rise, blocking_pieces=sorted(pieces), nominal_conflicts=copy.deepcopy(conflicts), wall_top_world_z=wall_top, cookie_top_world_z=cookie_top, remaining_vertical_contact_overlap_m=cookie_top - wall_top - self.checker.collision_epsilon_m, basis='measured target group and fixed wall/gripper CAD; full forward/rollback/IK/cost recomputed')
                    return accepted
            return False
        if self.max_steps - self.total_steps < cost + 10:
            return False
        self.path = path
        self.rollback = rollback
        self.envelope = envelope
        self.unit = unit
        self.desired_place_center = desired_center
        self.group = dict(center=center, rotation=br, width_m=width, confidence=0.7)
        self.force_command_floor = opening_from_gap(5 * 2 * self.half[1] - 0.0025)
        self.command = opened
        self.index = 0
        self._grasp_reference = None
        self._held_reset(unit)
        return True

    def _held_reset(self, unit):
        self.target.selected_group_key = ('terminal_repair', unit['unit_id'])
        self.target.selected_group = copy.deepcopy(self.group)
        for key in ('v6_held_previous', 'v6_held_reference', 'v6_release_track', 'v6_held_cache', 'held_anchor'):
            setattr(self.target, key, None)
        self.target.v6_held_generation += 1
        self.target.v6_held_method = None
        self.target.v6_use_bevel_held = False
        self.fresh_held_ids = set()
        self.held = None

    def _held_measure(self, obs):
        self.target.step += 1
        cameras = self.target._robot_cameras(obs)
        cameras['left_wrist'] = self.target._v6_wrist_camera(cameras['left_wrist'], obs)
        cameras = {n: self.target._adjust_camera(c, self.target.camera_adjustments.get(n, np.zeros(7))) for n, c in cameras.items()}
        request = dict(stage='REPAIR_LIFT', count=5, batch_index=self.unit['batch_index'], grasp_reference_eef=self._grasp_reference, held_reobserve=True, grasp_body_reference=dict(frame='world', location='target', count=5, center_world=np.asarray(self.group['center']).tolist(), rotation_world=np.asarray(self.group['rotation']).tolist(), measurement_id=self.unit['measurement_id'], basis='complete pregrasp RGB target group; initializer only'))
        count, conf = self.target._count_visible_held(obs, cameras, self.group, request)
        self.held = self.target._v6_held_observation(obs, cameras, self.group, request, count, conf)
        self.held.update(measurement_id=f'terminal_held:{self.unit['unit_id']}:{obs['time']}', count=count, count_confidence=conf, original_target_rotation=self.group['rotation'], repair_unit_id=self.unit['unit_id'])
        if count == 5 and conf >= 0.6 and (self.held.get('pose_confidence', 0.0) >= 0.6):
            self.fresh_held_ids.add(self.held['measurement_id'])
        return len(self.fresh_held_ids) >= 2

    def act(self, obs):
        self.total_steps += 1
        self.stage_steps += 1
        if self.stage in ('COMPLETE_CANDIDATE', 'TERMINAL_OBSERVE'):
            self.observe_steps += 1
            if self.stage_steps == 1 or self.stage_steps % 5 == 0:
                self._observe(obs)
            if self.observe_steps < 8:
                return self._hold()
            unit, reason = self._cached_preaudit_unit(obs)
            if unit is None:
                unit, reason = repair_candidate(self.last_report or {}, self.box, self.base.batch_ledger, self.base.source_column, self.half)
            self.rule_reason = reason
            if unit and unit['batch_index'] not in self.attempted and self._plan(obs, unit):
                self._preaudit_unit = None
                self.attempted.add(unit['batch_index'])
                self._go('TERMINAL_REPAIR', 'complete forward/rollback paths feasible within remaining measured budget')
            elif self.observe_steps % 10 == 0 and max(self.base._touch(obs)) < 0.12:
                view = self._free_view(obs)
                if view:
                    self.path, self.last_evidence = view
                    self.index = 0
                    self._go('TERMINAL_VIEW', 'move to an actually less-occluded target viewpoint')
            return self._hold()
        if self.stage == 'TERMINAL_VIEW':
            w = self.path[self.index]
            action, reached = self._servo(obs, w['position'], w['rotation'])
            if reached:
                self.index += 1
                if self.index == len(self.path):
                    self.observe_steps = 0
                    self._go('TERMINAL_OBSERVE', 'reached fresh target view')
            return action
        w = self.path[self.index]
        name = w['name']
        touch = self.base._touch(obs)
        if name == 'close':
            if min(touch) < 1.3:
                self.command = max(self.force_command_floor, self.command - (0.004 if max(touch) > 0.1 else 0.015))
            measured_gap = gap_from_opening(float(np.asarray(obs['observation.state'])[7]))
            width_ok = abs(measured_gap - 5 * 2 * self.half[1]) <= 0.0025
            force_ok = min(touch) >= 1.0 and max(touch) / max(min(touch), 0.1) < 3.0
            self.stable = self.stable + 1 if width_ok and force_ok else 0
            action, _ = self._servo(obs, w['position'], w['rotation'], 0.0008)
            if self.stable >= 3:
                self._grasp_reference = np.asarray(obs['observation.eef_pose'])[:7].copy()
                self.index += 1
                self.stage_steps = 0
            elif self.stage_steps > int(self.costs.get('CLOSE', 16)) * 2:
                self.path = [dict(self.path[2], name='rollback_release', opening=self.path[0]['opening'], carrying=False), dict(self.path[0], name='rollback_withdraw', carrying=False)]
                self.index = 0
                self.rule_reason = 'dual touch not established; release at original supported location'
            return action
        if name in ('release', 'rollback_release'):
            self.command = min(w['opening'], self.command + 0.025)
            action = self._hold()
            self.stable = self.stable + 1 if float(np.asarray(obs['observation.state'])[7]) >= w['opening'] - 0.04 and max(touch) < 0.12 else 0
            if self.stable >= 5:
                self.index += 1
                self.stage_steps = 0
            return action
        action, reached = self._servo(obs, w['position'], w['rotation'], w.get('speed', 0.004))
        if reached and name == 'trial_lift':
            if not self._held_measure(obs):
                if self.stage_steps > 30:
                    self.path = self.rollback
                    self.index = 0
                    self.stage_steps = 0
                    self.rule_reason = 'actual five-object count/pose unresolved; rollback without crediting a repaired batch'
                return self._hold()
            self.envelope = {k: self.held[k] for k in ('center_tool', 'rotation_tool', 'width_m', 'bounds_tool')}
            self.envelope['half_size'] = [float(self.half[0]), float(self.held['width_m']) / 2, float(self.half[2])]
            target_rotation = _rot_z(self.box.get('yaw', 0.0)) @ np.asarray(self.held['rotation_tool']).T
            place_node = next((x for x in self.path if x['name'] == 'place'))
            old_place_position = place_node['position'].copy()
            corrected = self.desired_place_center - target_rotation @ np.asarray(self.held['center_tool'])
            for node in self.path[self.index + 1:]:
                if node['name'].startswith('place') or node['name'] in ('release', 'withdraw'):
                    dz = float(node['position'][2] - old_place_position[2])
                    node['position'] = corrected + [0, 0, dz]
                    node['rotation'] = target_rotation
            assessment = self.checker.check_path(obs, self.path[self.index + 1:], self.envelope, self.last_scene, repair_unit=self.unit, protected_objects=self.unit['protected_objects'])
            if assessment['state'] != 'feasible':
                self.path = self.rollback
                self.index = 0
                self.rule_reason = 'measured held geometry changes remaining feasibility; return original group'
            else:
                self.index += 1
            self.stage_steps = 0
            return self._hold()
        if reached:
            self.index += 1
            self.stage_steps = 0
            if self.index >= len(self.path):
                self.observe_steps = 0
                self._go('COMPLETE_CANDIDATE', 'reseat or rollback finished; continue public observation and native hold5')
        return action
