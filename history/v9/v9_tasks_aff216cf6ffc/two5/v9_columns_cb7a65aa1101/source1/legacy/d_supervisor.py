"""Geometric supervisor around the unchanged ABC source/state controller."""
from __future__ import annotations
import copy
import numpy as np
from v9_tasks_aff216cf6ffc.two5.v9_columns_cb7a65aa1101.source1.legacy.path_feasibility import PathFeasibility, _rz

def _opening(gap):
    return float(np.clip((float(gap) - 0.0053616) / 0.085, 0.0, 1.0))

class DSupervisor:

    def reset(self, context=None):
        super().reset(context)
        if not hasattr(self, '_d_checker'):
            self._d_checker = PathFeasibility(self.config)
        self._d_report = {}
        self._d_view_pending = False
        self._d_view_active = False
        self._d_observed = set()
        self._d_envelope = None
        self._d_protected = None
        self._d_release_goal = None
        self._d_pressure = []
        self._d_batch = None

    @property
    def public_evidence(self):
        base = super().public_evidence
        return {**base, 'D_path': self._d_report, 'D_view_pending': self._d_view_pending}

    def perception_request(self):
        r = super().perception_request()
        if self._d_view_pending or self._d_view_active:
            r.update(force_refit=True, force_target_observation=True, held_reobserve=True)
        return r

    def _d_objects(self, scene):
        entries = scene.get('target_occupancy') or {}
        objects = []
        known = set()
        for key, entry in entries.items():
            if float(entry.get('confidence', 0.0) or 0.0) < 0.55 or entry.get('occluded'):
                continue
            known.add(int(key))
            bounds = entry.get('bounds')
            if bounds is None:
                continue
            bounds = np.asarray(bounds, float)
            if bounds.shape != (3, 2):
                continue
            box = scene['target_box']
            rotation = _rz(float(entry.get('box_yaw', box.get('yaw', 0.0))))
            center = np.asarray(entry.get('box_center', box['center'])) + rotation @ bounds.mean(1)
            objects.append(dict(object_id=f'prior_target_column:{key}:{entry.get('measurement_id')}', frame='world', center_world=center, rotation_world=rotation, half_size=np.diff(bounds, axis=1).ravel() / 2))
        return (objects, len(known) >= len(np.unique(self.slots[:, 0])))

    def _d_held(self, scene):
        h = scene.get('held') or {}
        if h.get('center_tool') is not None and h.get('rotation_tool') is not None and (float(h.get('pose_confidence', 0.0) or 0.0) >= 0.6):
            self._d_envelope = {k: copy.deepcopy(h.get(k)) for k in ('center_tool', 'rotation_tool', 'width_m', 'bounds_tool', 'measurement_id')}
        elif self._d_envelope is None and self._held_center_local is not None:
            self._d_envelope = dict(center_tool=self._held_center_local, rotation_tool=self._held_cookie_local, width_m=self._held_width)
        return self._d_envelope

    def _d_macro(self, scene, position, rotation, extra):
        env = self._d_held(scene)
        if env is None or env.get('width_m') is None:
            return (None, None)
        close = float(np.asarray(self._obs['observation.state'])[7])
        release = _opening(float(env['width_m']) + extra)
        high = np.asarray(position) + [0, 0, max(0.04, float(getattr(self, '_clearance', 0.04)))]
        nodes = [dict(name='target_high', position=high, rotation=rotation, opening=close, carrying=True), dict(name='descend_place', position=position, rotation=rotation, opening=close, carrying=True, reference=True), dict(name='release_open', position=position, rotation=rotation, opening=release, carrying=False), dict(name='withdraw_high', position=high, rotation=rotation, opening=release, carrying=False)]
        if self._d_protected is None or self.stage == 'TRANSPORT':
            protected, known = self._d_objects(scene)
        else:
            protected, known = self._d_protected
        report = self._d_checker.check_path(self._obs, nodes, env, scene, protected_objects=protected)
        if not scene['target_box'].get('anchor_confirmed', False) and report['state'] != 'blocked':
            report.update(state='unknown', status='unknown', reason='target anchor unconfirmed: complete independent rim view required before descent', next_action='observe_target')
        elif not known and report['state'] == 'feasible':
            report.update(state='unknown', status='unknown', reason='protected target-object coverage remains partial', next_action='observe_target')
        return (report, (release, protected, known))

    def _d_brief(self, report, **extra):
        if report is None:
            return dict(state='unknown', reason='actual held envelope missing', **extra)
        return dict(state=report['state'], reason=report.get('reason'), conflicts=report.get('conflicts', [])[:6], feasible_y_intervals=report.get('feasible_y_intervals'), coverage=report.get('coverage'), geometry_version=report.get('geometry_version'), **extra)

    def _place_pose(self, scene, clearance=0.0):
        position, rotation = super()._place_pose(scene, 0.0)
        nominal = position.copy()
        chosen = None
        last = None
        box = scene['target_box']
        br = np.asarray(box.get('rotation', _rz(float(box.get('yaw', 0.0)))), float)
        for extra in (0.006, 0.004, 0.002, 0.001):
            report, data = self._d_macro(scene, position, rotation, extra)
            last = report
            if report is None:
                break
            if report['state'] != 'blocked':
                chosen = (position.copy(), report, data, extra)
                break
            old_y = report.get('requested_reference_y')
            candidates = []
            for interval in report.get('feasible_y_intervals') or []:
                lo, hi = interval
                if lo is None or hi is None or lo > hi:
                    continue
                candidates.append((abs((lo + hi) / 2 - old_y), (lo + hi) / 2))
            for _, y in sorted(candidates):
                adjusted = nominal + br[:, 1] * (y - old_y)
                check, newdata = self._d_macro(scene, adjusted, rotation, extra)
                if check is not None and check['state'] != 'blocked':
                    chosen = (adjusted, check, newdata, extra)
                    break
            if chosen:
                break
        if chosen:
            position, report, data, extra = chosen
            self._d_release_goal = data[0]
            self._d_protected = (data[1], data[2])
            shift = float(np.linalg.norm(position - nominal))
            self._d_report = self._d_brief(report, release_extra_m=extra, target_adjustment_m=shift, rule='explicit held/gripper feasible interval and full descend/open/withdraw sweep')
            if shift > 0.0001:
                ref = self._target_box_reference or box
                rr = _rz(float(ref.get('yaw', 0.0)))
                self._expected_local_center = np.asarray(self._expected_local_center) + rr.T @ (position - nominal)
        else:
            self._d_release_goal = None
            self._d_report = self._d_brief(last, rule='no feasible held/gripper target channel; obtain an effective view')
        self._d_transport_low = (position.copy(), rotation.copy())
        self._d_transport_high = (position + np.array([0.0, 0.0, max(0.04, float(getattr(self, '_clearance', 0.04)))]), rotation.copy())
        return (position + np.array([0, 0, clearance]), rotation)

    def _d_begin_view(self, reason):
        self._d_view_pending = True
        self._rule_reason = reason
        if self.stage in ('PLACE', 'OPEN', 'RETRACT', 'RELEASE_LIFT', 'REPAIR_DESCEND'):
            self._pending_target_observation = reason
            self._target_exit_path = list(reversed(self._target_path))
            if self._target_clear_pose is not None:
                if not self._target_exit_path or np.linalg.norm(self._target_exit_path[-1][0] - self._target_clear_pose[0]) > 0.002:
                    self._target_exit_path.append(self._target_clear_pose)
            if self._target_exit_path:
                self._advance('TARGET_RETREAT')
            else:
                self._advance('TRANSPORT')
        elif self.stage != 'PREPLACE_OBSERVE':
            self._advance('TRANSPORT')
        return self._hold()

    def _start_target_observation(self, reason):
        if self._d_view_pending and self.stage == 'TARGET_RETREAT':
            self._advance('TRANSPORT')
            return
        return super()._start_target_observation(reason)

    def _d_pressure_alarm(self, obs):
        relevant = self.stage == 'PLACE' or (self.stage == 'SLIP_RECHECK' and getattr(self, '_b2_resume_stage', None) == 'PLACE')
        if not relevant:
            self._d_pressure = []
            return False
        goal = self._placement_goal
        if goal is None:
            return False
        pos = np.asarray(obs['observation.eef_pose'])[:3]
        touch = self._touch(obs)
        now = float(obs.get('time', 0.0))
        self._d_pressure.append((now, pos.copy(), float(max(touch)), float(min(touch))))
        self._d_pressure = self._d_pressure[-10:]
        if len(self._d_pressure) < 8:
            return False
        old = self._d_pressure[0]
        high = pos[2] - np.asarray(goal[0])[2] > 0.004
        stalled = np.linalg.norm(pos - old[1]) < 0.0005 and now - old[0] >= 0.3
        asymmetric = max(touch) > 12.0 and max(touch) > 3 * max(min(touch), 0.1)
        return bool(high and stalled and asymmetric)

    def act(self, obs, scene):
        self._obs, self._scene = (obs, scene)
        if self._d_batch != self.batch_index:
            self._d_batch = self.batch_index
            self._d_protected = None
            self._d_envelope = None
            self._d_release_goal = None
            self._d_report = {}
            self._d_pressure = []
        if self._d_pressure_alarm(obs):
            self.total_steps += 1
            self.phase_steps += 1
            return self._d_begin_view('persistent high-position obstruction across held-fit rebases: exit and relocate')
        if self._d_view_pending and self.stage == 'TRANSPORT':
            view = (scene.get('target_geometry') or {}).get('preplacement_observation') or {}
            if view.get('valid'):
                self._v9_preplace_view = (np.asarray(view['eef_position']), np.asarray(view['eef_rotation']))
                self._v9_preplace_waypoints = [np.asarray(p) for p in view.get('waypoints', [view['eef_position']])]
                self._v9_preplace_measurement = (scene.get('target_geometry') or {}).get('measurement_id')
                self._d_view_active = True
                self._d_view_pending = False
                self._d_observed.add(self.batch_index)
                self._advance('PREPLACE_OBSERVE')
        before = self.stage
        if before == 'OPEN' and self._d_release_goal is not None:
            self._release_goal = self._d_release_goal
        action = super().act(obs, scene)
        if before == 'PREPLACE_OBSERVE' and self.stage == 'TRANSPORT':
            self._d_view_active = False
            self._d_view_pending = False
            self._d_protected = None
            self._d_observed.add(self.batch_index)
        if before == 'TRANSPORT' and self.stage == 'TRANSPORT' and scene['target_box'].get('anchor_confirmed', False):
            protected, _ = self._d_objects(scene)
            planned = getattr(self, '_d_transport_high', None)
            if protected and planned is not None:
                position, actual_rotation = self._snapshot_pose(obs)
                shortfall = float(planned[0][2] - position[2])
                if shortfall > 0.0025:
                    raised = position.copy()
                    raised[2] = planned[0][2]
                    env = self._d_held(scene)
                    opening = float(np.asarray(obs['observation.state'])[7])
                    width = (env or {}).get('width_m') or self._held_width or self.execution_count * 2 * self.half_size[1]
                    released = self._d_release_goal if self._d_release_goal is not None else _opening(float(width) + 0.006)
                    low, rotation = self._d_transport_low
                    nodes = [dict(name='actual_xy_raise', position=raised, rotation=actual_rotation, opening=opening, carrying=True), dict(name='target_high', position=planned[0], rotation=rotation, opening=opening, carrying=True), dict(name='descend_place', position=low, rotation=rotation, opening=opening, carrying=True, reference=True), dict(name='release_open', position=low, rotation=rotation, opening=released, carrying=False), dict(name='withdraw_high', position=planned[0], rotation=rotation, opening=released, carrying=False)]
                    checked = self._d_checker.check_path(obs, nodes, env, scene, protected_objects=protected)
                    self._d_report['transport_height_correction'] = dict(active=True, measurement_time=float(obs.get('time', 0.0)), shortfall_m=shortfall, actual_xy=position[:2].tolist(), planned_high_z=float(raised[2]), full_macro_state=checked['state'], current_lift_conflicts=checked.get('segments', [{}])[0].get('conflicts', []), coverage=checked.get('coverage'), held_measurement_id=(env or {}).get('measurement_id'), reason='actual high corridor not reached; retain clamp and raise at current XY/R')
                    self._rule_reason = 'occupied target: recover actual transport height before continuing lateral motion'
                    return self._servo(obs, raised, actual_rotation, 0.004)[0]
        if before == 'TRANSPORT' and self.stage in ('TRANSPORT', 'PLACE'):
            state = self._d_report.get('state')
            at_high = self.stage == 'PLACE' or (self._target_clear_pose is not None and np.linalg.norm(np.asarray(obs['observation.eef_pose'])[:3] - self._target_clear_pose[0]) < 0.004)
            unconfirmed = not (scene.get('target_box') or {}).get('anchor_confirmed', False)
            if at_high and (state == 'blocked' or (state == 'unknown' and (unconfirmed or self.batch_index not in self._d_observed))):
                if unconfirmed or self.batch_index not in self._d_observed:
                    return self._d_begin_view(self._d_report.get('reason', 'target channel unobserved'))
                if state == 'blocked':
                    if hasattr(self, '_b2_return_allowed') and self._b2_return_allowed():
                        self._start_source_recovery('target gripper channel remains infeasible after a new view', carrying=True)
                    else:
                        self._rule_reason = 'target channel infeasible; remain at reached high pose for associated-object replanning'
                    return self._hold()
        if before in ('OPEN', 'RETRACT') and self._d_report.get('geometry_version') is not None:
            version = (scene.get('target_box') or {}).get('geometry_version')
            if version is not None and version != self._d_report['geometry_version']:
                return self._d_begin_view('target frame changed during release/withdrawal; reacquire gripper channel')
        if self.stage == 'OPEN' and self._d_release_goal is not None:
            self._release_goal = self._d_release_goal
        return action

    def _start_source_recovery(self, reason, carrying=False):
        if carrying and self._carry_path and (self._carry_source_high_pose is not None):
            scene = self._scene or {}
            env = self._d_held(scene)
            waypoints = [dict(name='reached_carry_return', position=p, rotation=r, opening=float(self.actual_opening), carrying=True, box_key='source_box') for p, r in reversed(self._carry_path[self._carry_source_high_index or 0:])]
            width = (env or {}).get('width_m') or self._held_width
            if width is not None:
                release = _opening(float(width) + 0.006)
                high = self._safe_source_high or self._carry_source_high_pose
                waypoints.extend([dict(name='return_source_descend', position=self._pick, rotation=self._pick_rotation, opening=float(self.actual_opening), carrying=True, box_key='source_box', reference=True), dict(name='return_source_open', position=self._pick, rotation=self._pick_rotation, opening=release, carrying=False, box_key='source_box'), dict(name='return_source_withdraw', position=high[0], rotation=high[1], opening=release, carrying=False, box_key='source_box')])
            report = self._d_checker.check_path(self._obs, waypoints, env, scene, protected_objects=None)
            source = scene.get('source_box') or {}
            verified = bool(source.get('wall_pose_verified', False))
            cert = bool(hasattr(self, '_b2_return_allowed') and self._b2_return_allowed())
            self._d_report = self._d_brief(report, domain='source_return', source_walls_verified=verified, reached_path_certificate=cert, reason_override='unverified source walls remain unknown; reached reverse path uses current payload and observed vacancy')
            reliable_conflicts = [c for c in report.get('conflicts', []) if verified or ':wall:' not in c.get('obstacle', '')]
            if not verified and (not reliable_conflicts):
                self._d_report['state'] = 'unknown'
            if reliable_conflicts or not cert:
                self._d_begin_view('source-return clearance or vacancy needs a new effective observation')
                return
        return super()._start_source_recovery(reason, carrying)
