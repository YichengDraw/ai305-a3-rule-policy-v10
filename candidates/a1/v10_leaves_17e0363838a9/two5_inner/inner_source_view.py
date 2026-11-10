"""Finite source-camera views from public kinematics and observed source geometry."""
from __future__ import annotations
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from v10_leaves_17e0363838a9.two5_inner.observation_path import PathFeasibility, PublicPathIK

def _project(points, camera, shape):
    h, w = shape[:2]
    f = h / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
    p = (np.asarray(points) - camera['pos']) @ camera['rotation']
    depth = -p[:, 2]
    d = np.maximum(depth, 1e-05)
    return (np.column_stack([(w - 1) / 2 + f * p[:, 0] / d, (h - 1) / 2 - f * p[:, 1] / d]), depth)

def source_evidence(scene, count):
    """Describe measured information, independently of the measurement ID."""
    front = scene.get('source_frontier') or {}
    local = scene.get('local_align') or {}
    if not front:
        group = scene.get('group') or {}
        available = group.get('center') is not None and float(group.get('confidence', 0.0) or 0.0) >= 0.6
        return dict(resolved=bool(available), measurement_id=group.get('measurement_id'), measurement_step=group.get('measurement_step'), geometry_valid=bool(available), selected_count=int(local.get('visible_peak_count', 0) or 0), object_count=int(local.get('visible_peak_count', 0) or 0), planned_count=int(count), count_verified=False, front_clear=False, rear_clear=False, identity_anchor=False, row_ranges=[], gap_measured=local.get('gap_width_m') is not None, gap_insertable=local.get('gap_width_m') is not None and float(local['gap_width_m']) > 0.001, local_state=local.get('state'), source_state='parent_source_fit_available' if available else 'unobserved', source_group_confidence=float(group.get('confidence', 0.0) or 0.0), scope='source-group geometry reacquisition; native object identities remain hypotheses')
    raw = front.get('coarse_current_observation') or front
    objects = raw.get('objects') or []
    selected = int(front.get('selected_count', 0) or 0)
    solved = bool(front.get('geometry_valid') and selected == int(count) and (front.get('selection_state') == 'contiguous_measured_unit') and (not front.get('requires_wrist_view')))
    ranges = []
    for camera, data in (raw.get('per_camera') or {}).items():
        positions = np.asarray(data.get('row_positions_local_m', []), float)
        ranges.append((camera, len(positions), bool(data.get('front_clear_band')), bool(data.get('rear_clear_band')), bool(data.get('all_twenty_resolved')), None if not len(positions) else [float(positions.min()), float(positions.max())]))
    gap = local.get('gap_width_m')
    return dict(resolved=solved, measurement_id=raw.get('measurement_id'), measurement_step=raw.get('measurement_step'), geometry_valid=bool(front.get('geometry_valid')), selected_count=selected, object_count=len(objects), front_clear=bool(raw.get('front_clear_band')), rear_clear=bool(raw.get('rear_clear_band')), identity_anchor=bool(front.get('identity_anchor_valid')), row_ranges=ranges, gap_measured=gap is not None, gap_insertable=gap is not None and float(gap) > 0.001, local_state=local.get('state'), source_state=front.get('state'))

def observation_outcome(before, after):
    if after.get('resolved'):
        return 'resolved'
    gained = any((bool(after.get(k)) and (not before.get(k)) for k in ('front_clear', 'rear_clear', 'identity_anchor', 'gap_measured')))
    gained = gained or int(after.get('object_count', 0)) > int(before.get('object_count', 0))
    old = {r[0]: r for r in before.get('row_ranges', [])}
    for r in after.get('row_ranges', []):
        a = old.get(r[0])
        if a is None and r[1]:
            gained = True
        elif a is not None and r[1] > a[1]:
            gained = True
        elif a is not None and r[5] is not None and (a[5] is not None):
            if r[5][0] < a[5][0] - 0.0015 or r[5][1] > a[5][1] + 0.0015:
                gained = True
    return 'informative_but_incomplete' if gained else 'uninformative'

class SourceViewPlanner:

    def __init__(self, vision):
        self.vision = vision
        self.checker = PathFeasibility(vision.config)
        self.ik = PublicPathIK(vision.robot_model, vision.config)
        self.reset()

    def reset(self):
        self.active = None

    def _regions(self, scene, request):
        v = self.vision
        box = scene['source_box']
        front = scene.get('source_frontier') or {}
        c = np.asarray(box['center'], float)
        yaw = float(box.get('yaw', 0.0))
        co, si = (np.cos(yaw), np.sin(yaw))
        rot = np.array([[co, -si, 0.0], [si, co, 0.0], [0.0, 0.0, 1.0]])
        col = int(request.get('source_column', 1))
        count = int(request.get('count', 0))
        pitch = 0.008833333333333
        x = float(v.columns[col - 1] - v.scene.source_bin_center_m[0])
        group = scene.get('group') or {}
        if group.get('center') is not None and group.get('broad_reference_measurement_id') is not None:
            x = float((np.asarray(group['center'], float) - c) @ rot[:, 0])
        top = float(v.scene.cookie_model_z_m + v.cookie_half[2])
        half = np.asarray(v.scene.source_bin_half_size_m, float)
        identity = front.get('observation_scope') == 'full_column_identity' or not front.get('identity_anchor_valid', False)
        if front.get('frontier_center') is not None:
            first = float((np.asarray(front['frontier_center']) - np.r_[c[:2], 0.0]) @ rot[:, 1])
        else:
            rows = v.layout[np.isclose(v.layout[:, 0], v.columns[col - 1])]
            first = float(rows[:, 1].min() - v.scene.source_bin_center_m[1])
            first += int(request.get('source_start_index', 0)) * pitch
        rear = first + max(count - 1, 0) * pitch
        boxes = []

        def add(name, xlo, xhi, ylo, yhi, expected):
            corners = np.array([[xx, yy, zz] for zz in (top - 0.005, top + 0.005) for xx in (xlo, xhi) for yy in (ylo, yhi)]) @ rot.T + np.r_[c[:2], 0.0]
            sample = np.array([[xx, yy, top] for yy in np.linspace(ylo, yhi, 24) for xx in np.linspace(xlo, xhi, 7)]) @ rot.T + np.r_[c[:2], 0.0]
            boxes.append(dict(name=name, corners=corners, samples=sample, expected_evidence=expected, column=col, first_row_prior=first, world_axis=rot[:, 1]))
        if identity:
            add('column_identity', -half[0] - 0.006, half[0] + 0.006, -half[1] - 0.015, half[1] + 0.015, ['source_box_walls', 'column_context', 'front_empty_band', 'ordered_row_prefix'])
        add('front_and_batch_end', x - v.cookie_half[0] - 0.006, x + v.cookie_half[0] + 0.006, first - 0.016, rear + pitch + 0.012, ['front_empty_band', 'first_object', 'batch_end', 'protected_rear_neighbour'])
        add('front_prefix', x - v.cookie_half[0] - 0.006, x + v.cookie_half[0] + 0.006, first - 0.018, first + min(count + 1, 8) * pitch + 0.012, ['front_empty_band', 'overlapping_ordered_prefix'])
        add('rear_gap', x - v.cookie_half[0] - 0.006, x + v.cookie_half[0] + 0.006, rear - 0.02, rear + pitch + 0.014, ['batch_end', 'protected_rear_neighbour', 'insertion_gap'])
        return boxes

    def _protected(self, scene):
        objects = []
        for key, entry in (scene.get('target_occupancy') or {}).items():
            bounds = entry.get('bounds')
            if bounds is None or entry.get('occluded'):
                continue
            b = np.asarray(bounds, float)
            if b.shape != (3, 2):
                continue
            yaw = float(entry.get('box_yaw', (scene.get('target_box') or {}).get('yaw', 0.0)))
            rr = Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix()
            center = np.asarray(entry.get('box_center', scene['target_box']['center'])) + rr @ b.mean(1)
            objects.append(dict(object_id='observed_target_column:' + str(key), frame='world', center_world=center, rotation_world=rr, half_size=np.diff(b, axis=1).ravel() / 2))
        front = scene.get('source_frontier') or {}
        if front.get('contact_epoch') == getattr(self.vision, 'v9_source_contact_epoch', None):
            for item in front.get('objects') or []:
                objects.append(dict(object_id=item.get('spatial_id', 'source_object'), frame='world', center_world=item['center'], rotation_world=np.eye(3), half_size=self.vision.cookie_half))
        box = scene.get('source_box') or {}
        if box.get('center') is not None:
            sourcehalf = np.asarray(self.vision.scene.source_bin_half_size_m, float)
            center = np.asarray(box['center'], float).copy()
            center[2] = float(self.vision.scene.cookie_model_z_m)
            yaw = float(box.get('yaw', 0.0))
            rr = Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix()
            objects.append(dict(object_id='source_observation_CAD_occupied_volume_prior', frame='world', center_world=center, rotation_world=rr, half_size=[sourcehalf[0] + 0.025, sourcehalf[1] + 0.025, float(self.vision.cookie_half[2])], basis='possible cookie region for high observation motion; occupancy unknown'))
        return objects

    def plan(self, observation, cameras, scene, request):
        v = self.vision
        stage = str(request.get('stage', '')).upper()
        front = scene.get('source_frontier') or {}
        needed = bool(stage == 'SOURCE_OBSERVE' or (stage == 'SELECT' and (request.get('source_view_required') or front.get('requires_wrist_view') or float((scene.get('group') or {}).get('confidence', 0.0) or 0.0) < 0.15)))
        key = (int(request.get('source_column', 1)), request.get('batch_index'), request.get('source_start_index'), int(request.get('count', 0)), getattr(v, 'v9_source_contact_epoch', None), int(v.calibration_generation), tuple(np.round(np.asarray(v.wrist_mount, float), 8)))
        context = repr(key)
        result = dict(needed=needed, valid=False, context_key=context, object_scope='source_observation_plan', measurement_id='source_view_plan:' + str(v.step), measurement_step=v.step, measurement_time=float(observation.get('time', 0.0)), camera='left_wrist', evidence=source_evidence(scene, request.get('count', 0)))
        if not needed:
            return dict(result, reason='current source geometry is available')
        if stage == 'SOURCE_OBSERVE' and request.get('source_view_active') and (self.active is not None):
            return {**self.active, 'evidence': result['evidence']}
        held = scene.get('held') or {}
        touch = np.asarray(observation.get('observation.force', np.zeros(18)))[12:14]
        if held.get('currently_held') is True or max(touch) > 0.15:
            return dict(result, reason='source observation needs current object-location/empty-hand evidence', outcome='partial')
        image = np.asarray(observation['observation.images.left_wrist'])
        camera = cameras['left_wrist']
        position, rotation = v._eef_rotation(observation)
        h, w = image.shape[:2]
        mask, _ = v._v6_gripper_silhouette(camera, image.shape)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        done = {x.get('view_id') for x in request.get('source_view_history', []) if x.get('context_key') == context and x.get('outcome') in ('informative_but_incomplete', 'uninformative', 'unreachable')}
        failures = [x for x in request.get('source_view_history', []) if x.get('context_key') == context]
        reassociate = len(failures) >= 3 and all((x.get('outcome') in ('uninformative', 'unreachable') for x in failures[-3:]))
        result['escalation'] = 'new_observation_direction_and_column_association_review' if reassociate else None
        result['development_escalation_default'] = 3
        canonical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        pitch = (0.0, -15.0, -35.0, -35.0)[key[0] - 1]
        base = Rotation.from_rotvec([0.0, np.deg2rad(pitch), 0.0]).as_matrix() @ canonical
        rotations = [rotation, base] + [Rotation.from_rotvec([0.0, 0.0, a]).as_matrix() @ base for a in (-0.18, 0.18)]
        if reassociate:
            rotations = [Rotation.from_rotvec([0.0, 0.0, a]).as_matrix() @ base for a in (-0.35, 0.35, -0.5, 0.5)] + rotations
        f = h / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
        lower = np.array([(8 - (w - 1) / 2) / f, ((h - 1) / 2 - (h - 9)) / f])
        upper = np.array([(w - 9 - (w - 1) / 2) / f, ((h - 1) / 2 - 8) / f])
        candidates = []
        for region_index, region in enumerate(self._regions(scene, request)):
            corners = region['corners']
            look = corners.mean(0)
            for ri, rr in enumerate(rotations):
                change = rr @ rotation.T
                camrot = change @ camera['rotation']
                campos = position + change @ (camera['pos'] - position)
                relative = (corners - look) @ camrot
                for ray_index, (ix, iy) in enumerate(((0.5, 0.22), (0.32, 0.3), (0.68, 0.3), (0.5, 0.4))):
                    rayxy = np.array([(ix * w - (w - 1) / 2) / f, ((h - 1) / 2 - iy * h) / f])
                    required = max(0.07, float(relative[:, 2].max() + 0.015))
                    for axis in range(2):
                        q = relative[:, axis]
                        z = relative[:, 2]
                        required = max(required, float(np.max((q + upper[axis] * z) / (upper[axis] - rayxy[axis]))), float(np.max(-(q + lower[axis] * z) / (rayxy[axis] - lower[axis]))))
                    for scale_index, scale in enumerate((1.05, 1.2)):
                        view_id = '{}:{}:{}:{}'.format(region['name'], ri, ray_index, scale_index)
                        if view_id in done:
                            continue
                        depth = required * scale
                        desiredcam = look - camrot @ np.r_[rayxy * depth, -depth]
                        target = position + desiredcam - campos
                        if not np.all(np.isfinite(target)) or not 0.79 < target[2] < 1.1:
                            continue
                        view = dict(camera, pos=desiredcam, rotation=camrot)
                        uv, z = _project(corners, view, image.shape)
                        if not np.all((z > 0) & (uv[:, 0] > 8) & (uv[:, 0] < w - 9) & (uv[:, 1] > 8) & (uv[:, 1] < h - 9)):
                            continue
                        sample, sd = _project(region['samples'], view, image.shape)
                        ixy = np.rint(sample).astype(int)
                        inside = (sd > 0) & (ixy[:, 0] >= 0) & (ixy[:, 0] < w) & (ixy[:, 1] >= 0) & (ixy[:, 1] < h)
                        visible = inside.copy()
                        visible[inside] &= mask[ixy[inside, 1], ixy[inside, 0]] == 0
                        coverage = float(visible.mean())
                        checkpitch = np.array([look, look + region['world_axis'] * 0.008833333333333])
                        puv, _ = _project(checkpitch, view, image.shape)
                        pixels = float(np.linalg.norm(puv[1] - puv[0]))
                        if coverage < 0.85 or pixels < 4.0:
                            continue
                        cost = float(np.linalg.norm(target - position)) + 0.05 * float(Rotation.from_matrix(rotation.T @ rr).magnitude())
                        candidates.append((region_index, cost, -coverage, view_id, target, rr, view, region, pixels))
        protected = self._protected(scene)
        checked = []
        opening = float(np.asarray(observation['observation.state'])[7])
        pathscene = {**scene, 'path_scope': 'empty_hand_observation'}
        for _, cost, negcoverage, view_id, target, rr, view, region, pixels in sorted(candidates, key=lambda a: a[:3]):
            high = max(float(position[2]), float(target[2]))
            rise = position.copy()
            rise[2] = high
            above = target.copy()
            above[2] = high
            nodes = [dict(name='source_view_rise', position=rise, rotation=rotation, opening=opening, carrying=False, box_key='source_box'), dict(name='source_view_translate', position=above, rotation=rr, opening=opening, carrying=False, box_key='source_box')]
            if abs(target[2] - high) > 0.0001:
                nodes.append(dict(name='source_view_settle', position=target, rotation=rr, opening=opening, carrying=False, box_key='source_box'))
            sweep = self.checker.check_path(observation, nodes, None, pathscene, protected_objects=protected)
            receipt = dict(view_id=view_id, path_state=sweep['state'], reason=sweep.get('reason'), conflicts=sweep.get('conflicts', [])[:2])
            if sweep['state'] != 'feasible':
                checked.append(receipt)
                continue
            ik = self.ik.check_ik_path(observation, nodes, position_tolerance_m=0.0025, rotation_tolerance_rad=0.07)
            receipt['ik'] = ik
            if ik['state'] != 'reachable':
                checked.append(receipt)
                continue
            actual = self.ik.last_fk_path[-1]
            acamera = dict(camera, pos=actual['position'] + actual['rotation'] @ rotation.T @ (camera['pos'] - position), rotation=actual['rotation'] @ rotation.T @ camera['rotation'])
            uv, z = _project(region['corners'], acamera, image.shape)
            if not np.all((z > 0) & (uv[:, 0] > 8) & (uv[:, 0] < w - 9) & (uv[:, 1] > 8) & (uv[:, 1] < h - 9)):
                checked.append(dict(receipt, reason='actual FK view clips required source region'))
                continue
            actualnodes = [dict(**node, opening=opening, carrying=False, box_key='source_box') for node in self.ik.last_fk_path]
            actualsweep = self.checker.check_path(observation, actualnodes, None, pathscene, protected_objects=protected)
            if actualsweep['state'] != 'feasible':
                checked.append(dict(receipt, actual_fk_path=actualsweep['state']))
                continue
            result.update(valid=True, issue=region['name'], view_id=view_id, expected_evidence=region['expected_evidence'], eef_position=target.tolist(), eef_rotation=rr.tolist(), waypoints=[dict(name=node['name'], position=np.asarray(node['position']).tolist(), rotation=np.asarray(node['rotation']).tolist()) for node in nodes], predicted_unoccluded_fraction=-negcoverage, pixels_per_pitch=pixels, actual_fk_complete=True, path_state=sweep['state'], actual_fk_path_state=actualsweep['state'], ik=ik, candidate_checks=checked, reason='source evidence region fits calibrated camera FOV and reachable full empty-hand path')
            self.active = dict(result)
            return result
        return dict(result, reason='finite source views lack a reachable clear evidence region', outcome='unreachable', candidate_count=len(candidates), candidate_checks=checked)
