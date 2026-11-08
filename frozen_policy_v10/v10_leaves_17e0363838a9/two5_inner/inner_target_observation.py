"""Tested finite FOV solver; explicit parent release or measured loaded scope."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from v10_leaves_17e0363838a9.two5_inner.target_vision import _rot_z, _project, _sample_map

def plan_target_observation(self, observation, cameras, held, *, empty_hand_confirmed=False):
    carrying = bool(held.get('center_tool') is not None and float(held.get('pose_confidence', 0.0) or 0.0) >= 0.6)
    if not carrying and (not empty_hand_confirmed):
        return dict(needed=True, valid=False, reason='current loaded envelope or parent confirmed release needed')
    scope = 'loaded_target_observation' if carrying else 'empty_hand_observation'
    packet = self.v9_target_geometry
    views = packet.get('per_camera', {})
    left = views.get('left_wrist', {})
    cached = getattr(self, '_d_view_plan_cache', None)
    if self.v6_request_stage == 'PREPLACE_OBSERVE' and cached is not None and (cached.get('target_geometry_version') == packet.get('geometry_version')):
        return dict(cached)
    prior_error = left.get('prior_forward_visible_median_px')
    needed = bool(getattr(self, 'd_force_target_observation', False) or (not self.boxes['target_box'].get('anchor_confirmed', False) and prior_error is not None and (prior_error > 4.0)))
    result = {'needed': needed, 'valid': False, 'observation_scope': scope, 'carrying': carrying, 'reason': 'visible calibrated wrist contour disagrees with provisional target anchor' if needed else 'no contradictory visible target contour', 'measurement_id': f'target_view_plan:{packet.get('measurement_id')}:{observation.get('time')}', 'target_measurement_id': packet.get('measurement_id'), 'measurement_step': self.step, 'measurement_time': float(observation.get('time', 0.0)), 'camera': 'left_wrist', 'held_measurement_id': held.get('measurement_id'), 'prior_forward_visible_median_px': prior_error}
    if not needed:
        return result
    camera = cameras['left_wrist']
    image = np.asarray(observation['observation.images.left_wrist'])
    robot, _ = self._v6_gripper_silhouette(camera, image.shape)
    occluded = cv2.dilate(robot, np.ones((5, 5), np.uint8))
    position, rotation = self._eef_rotation(observation)
    bounds = held.get('bounds_tool')
    if bounds is not None and float(held.get('pose_confidence', 0.0)) >= 0.5:
        bounds = np.asarray(bounds, float)
        if bounds.shape == (3, 2):
            tool = np.array([[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]])
            uv, z = _project(tool @ rotation.T + position, camera, image.shape)
            if np.all(z > 0):
                cv2.fillConvexPoly(occluded, np.rint(cv2.convexHull(uv.astype(np.float32))).astype(np.int32), 255)
    box = self.boxes['target_box']
    br = _rot_z(float(box['yaw']))
    center = np.asarray(box['center'])
    half = np.asarray(self.scene.target_bin_half_size_m) + self.scene.bin_wall_thickness_m
    corners = np.array([[x, y, z] for z in (0.0, self.scene.target_bin_wall_height_m) for x in (-half[0], half[0]) for y in (-half[1], half[1])]) @ br.T + center
    rim = self._box_points('target') @ br.T + center
    if getattr(self, 'd_force_target_observation', False) and (not box.get('anchor_confirmed', False)):
        nominal = np.asarray(self.scene.target_bin_world_position_m, float).copy()
        nominal[2] = center[2]
        local = np.array([[x, y, z] for z in (0.0, self.scene.target_bin_wall_height_m) for x in (-half[0], half[0]) for y in (-half[1], half[1])])
        priors = [local @ _rot_z(a).T + nominal + [dx, dy, 0.0] for a in (-0.08, 0.0, 0.08) for dx in (-0.02, 0.02) for dy in (-0.02, 0.02)]
        corners = np.vstack([corners, *priors])
    candidates = []
    h, w = image.shape[:2]

    def consider(delta, view_rotation=rotation):
        change = view_rotation @ rotation.T
        view = {**camera, 'pos': position + change @ (camera['pos'] - position) + delta, 'rotation': change @ camera['rotation']}
        uv, z = _project(corners, view, image.shape)
        complete = bool(np.all(z > 0) & np.all(uv[:, 0] > 8) & np.all(uv[:, 0] < w - 9) & np.all(uv[:, 1] > 8) & np.all(uv[:, 1] < h - 9))
        ruv, rz = _project(rim, view, image.shape)
        occ, inside = _sample_map(occluded.astype(float) / 255.0, ruv, default=1.0)
        visible = inside & (rz > 0) & (occ < 0.1)
        coverage = float(np.mean(visible))
        edges = visible[:96].reshape(4, 24).mean(1)
        two_edges = any((edges[j] >= 0.8 and edges[(j + 1) % 4] >= 0.8 for j in range(4)))
        if complete and (coverage >= 0.65 or two_edges):
            candidates.append((float(np.linalg.norm(delta)), coverage, delta, view_rotation))
    for dz in (0.02, 0.035, 0.05, 0.07):
        for dx, dy in ((0.0, 0.0), (0.012, 0.0), (-0.012, 0.0), (0.0, 0.012), (0.0, -0.012)):
            consider(np.array([dx, dy, dz]))
        if candidates:
            break
    if not candidates:
        f = h / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
        look_at = corners.mean(0)
        lower = np.array([(8 - (w - 1) / 2) / f, ((h - 1) / 2 - (h - 9)) / f])
        upper = np.array([(w - 9 - (w - 1) / 2) / f, ((h - 1) / 2 - 8) / f])
        view_rotations = [rotation] + [Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix() @ rotation for yaw in (-0.12, 0.12, -0.24, 0.24)] + [Rotation.from_rotvec([0.0, pitch, 0.0]).as_matrix() @ rotation for pitch in (-0.12, 0.12)]
        for view_rotation in view_rotations:
            change = view_rotation @ rotation.T
            view_camera = dict(camera, pos=position + change @ (camera['pos'] - position), rotation=change @ camera['rotation'])
            relative = (corners - look_at) @ view_camera['rotation']
            for image_y in (0.24, 0.32, 0.4):
                for image_x in (0.5, 0.32, 0.68):
                    ray_xy = np.array([(image_x * w - (w - 1) / 2) / f, ((h - 1) / 2 - image_y * h) / f])
                    if np.any(ray_xy <= lower) or np.any(ray_xy >= upper):
                        continue
                    required = max(0.1, float(relative[:, 2].max() + 0.015))
                    for axis in range(2):
                        q = relative[:, axis]
                        z = relative[:, 2]
                        required = max(required, float(np.max((q + upper[axis] * z) / (upper[axis] - ray_xy[axis]))), float(np.max(-(q + lower[axis] * z) / (ray_xy[axis] - lower[axis]))))
                    for scale in (1.03, 1.18, 1.38):
                        depth = required * scale
                        ray = np.r_[ray_xy * depth, -depth]
                        delta = look_at - view_camera['rotation'] @ ray - view_camera['pos']
                        if delta[2] < 0.012 or position[2] + delta[2] > 1.1 or np.linalg.norm(delta[:2]) > 0.25:
                            continue
                        consider(delta, view_rotation)
    if candidates:
        from v10_leaves_17e0363838a9.two5_inner.observation_path import PathFeasibility, PublicPathIK
        if not hasattr(self, '_d_view_checker'):
            self._d_view_checker = PathFeasibility(self.config)
        if not hasattr(self, '_d_view_ik'):
            self._d_view_ik = PublicPathIK(self.robot_model, self.config)
        protected = []
        for key, entry in (self.v7_occupancy or {}).items():
            b = entry.get('bounds')
            if b is None or entry.get('occluded') or float(entry.get('confidence', 0.0) or 0.0) < 0.55:
                continue
            b = np.asarray(b, float)
            if b.shape != (3, 2):
                continue
            rr = _rot_z(float(entry.get('box_yaw', box.get('yaw', 0.0))))
            protected.append(dict(object_id=f'target_column:{key}', frame='world', center_world=np.asarray(entry.get('box_center', box['center'])) + rr @ b.mean(1), rotation_world=rr, half_size=np.diff(b, axis=1).ravel() / 2))
        checked = []
        for _, coverage, delta, view_rotation in sorted(candidates, key=lambda x: (np.linalg.norm(Rotation.from_matrix(rotation.T @ x[3]).as_rotvec()) > 0.001, x[0], -x[1])):
            points = [position + np.array([0.0, 0.0, delta[2]]), position + delta] if np.linalg.norm(delta[:2]) > 0.001 else [position + delta]
            waypoints = [dict(name='loaded_target_view_rise' if j == 0 else 'loaded_target_view_translate', position=p, rotation=view_rotation, opening=float(np.asarray(observation['observation.state'])[7]), carrying=carrying, box_key='target_box') for j, p in enumerate(points)]
            envelope = held if held.get('center_tool') is not None else None
            sweep = self._d_view_checker.check_path(observation, waypoints, envelope, {**self.boxes, 'path_scope': scope}, protected_objects=protected)
            if sweep['state'] != 'feasible':
                checked.append(dict(state=sweep['state'], reason=sweep.get('reason'), conflicts=sweep.get('conflicts', [])[:2]))
                continue
            ik = self._d_view_ik.check_ik_path(observation, waypoints, position_tolerance_m=0.0025, rotation_tolerance_rad=0.07)
            checked.append(dict(state=ik['state'], path_state=sweep['state'], translation=delta.tolist(), ik=ik))
            if ik['state'] != 'reachable':
                continue
            actual = self._d_view_ik.last_fk_path[-1]
            actual_camera = dict(camera, pos=actual['position'] + actual['rotation'] @ rotation.T @ (camera['pos'] - position), rotation=actual['rotation'] @ rotation.T @ camera['rotation'])
            auv, az = _project(corners, actual_camera, image.shape)
            actual_complete = bool(np.all(az > 0) & np.all(auv[:, 0] > 8) & np.all(auv[:, 0] < w - 9) & np.all(auv[:, 1] > 8) & np.all(auv[:, 1] < h - 9))
            aruv, arz = _project(rim, actual_camera, image.shape)
            aocc, ainside = _sample_map(occluded.astype(float) / 255.0, aruv, default=1.0)
            avisible = ainside & (arz > 0) & (aocc < 0.1)
            aedges = avisible[:96].reshape(4, 24).mean(1)
            actual_coverage = float(np.mean(avisible))
            if not actual_complete or not (actual_coverage >= 0.65 or any((aedges[j] >= 0.8 and aedges[(j + 1) % 4] >= 0.8 for j in range(4)))):
                checked[-1]['actual_fk_view'] = 'box initialization domain or visible rim lacks image margin'
                continue
            actual_nodes = [dict(**point, opening=float(np.asarray(observation['observation.state'])[7]), carrying=carrying, box_key='target_box') for point in self._d_view_ik.last_fk_path]
            actual_sweep = self._d_view_checker.check_path(observation, actual_nodes, envelope, {**self.boxes, 'path_scope': scope}, protected_objects=protected)
            if actual_sweep['state'] != 'feasible':
                checked[-1]['actual_fk_path'] = actual_sweep['state']
                continue
            result.update(valid=True, eef_position=(position + delta).tolist(), eef_rotation=view_rotation.tolist(), translation=delta.tolist(), predicted_unoccluded_rim_fraction=coverage, actual_fk_unoccluded_rim_fraction=actual_coverage, actual_fk_complete=True, observation_rotation_change_rad=float(np.linalg.norm(Rotation.from_matrix(rotation.T @ view_rotation).as_rotvec())), target_geometry_version=packet.get('geometry_version'), waypoints=[p.tolist() for p in points], path_state=sweep['state'], ik=ik, geometric_basis='fixed camera-to-EEF mount; CAD initialization-domain FOV depth; finite yaw/pitch; actual held/gripper sweep and public IK')
            result['reason'] = 'finite camera-depth observation passes predicted/FK FOV and loaded sweep checks'
            self._d_view_plan_cache = dict(result)
            return result
        result.update(reason='complete-view candidates lack a verified loaded sweep or reachable public IK', candidate_checks=checked)
    else:
        result['reason'] = 'no finite camera-depth candidate predicts a complete unoccluded box view'
    return result

class InnerMetricTarget:
    """Separate target measurements; source/held parent estimator is untouched."""

    def __init__(self, parent_vision):
        from v10_leaves_17e0363838a9.two5_inner.target_vision import VisionEstimator
        self.parent = parent_vision
        self.vision = VisionEstimator(parent_vision.config)
        self.signature = None
        self.last_geometry = {}
        self.last_view = None

    def update(self, observation, scene, stage):
        import copy
        v, p = (self.vision, self.parent)
        for key in ('camera_adjustments', 'wrist_mount', 'wrist_calibration', 'cookie_colors', 'calibration_generation'):
            setattr(v, key, copy.deepcopy(getattr(p, key)))
        signature = v._v9_target_calibration_signature()
        prior = copy.deepcopy(scene['target_box'])
        if self.signature != signature:
            v.v6_target_anchor = prior
            v.v6_target_anchor['anchor_confirmed'] = False
            v.v9_target_pending = None
            self.last_view = None
            self.signature = signature
        v.step = int(p.step)
        v.v6_request_stage = stage
        cameras = v._robot_cameras(observation)
        cameras['left_wrist'] = v._v6_wrist_camera(cameras['left_wrist'], observation)
        cameras = {n: v._adjust_camera(c, v.camera_adjustments.get(n, np.zeros(7))) for n, c in cameras.items()}
        box = v._v9_fit_target(observation, cameras)
        v.boxes = dict(source_box=scene['source_box'], target_box=box)
        v.v7_occupancy = copy.deepcopy(scene.get('target_occupancy') or {})
        result = dict(scene)
        result['target_box'] = box
        result['inner_target_geometry'] = copy.deepcopy(v.v9_target_geometry)
        if box.get('anchor_confirmed'):
            p.v6_target_anchor = copy.deepcopy(box)
            if p.boxes is not None:
                p.boxes['target_box'] = copy.deepcopy(box)
        if stage == 'INNER_TARGET_VIEW':
            if box.get('anchor_confirmed'):
                view = dict(needed=False, valid=False, reason='fresh calibrated metric target is already confirmed')
            elif self.last_view is not None:
                view = self.last_view
            else:
                v.d_force_target_observation = True
                view = plan_target_observation(v, observation, cameras, scene.get('held') or {})
                if view.get('valid'):
                    self.last_view = copy.deepcopy(view)
            result['inner_target_view'] = view
        else:
            self.last_view = None
        self.last_geometry = copy.deepcopy(v.v9_target_geometry)
        return result
