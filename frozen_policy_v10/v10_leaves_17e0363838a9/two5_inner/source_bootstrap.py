"""First source observation from public joints, fixed CAD and a coarse box prior."""
from __future__ import annotations
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from v10_leaves_17e0363838a9.two5_inner.observation_path import PathFeasibility, PublicPathIK

class InitialSourceView:

    def __init__(self, vision):
        self.vision = vision
        self.geometry = vision.config.cookie_transfer
        self.checker = PathFeasibility(vision.config)
        self.ik = PublicPathIK(vision.robot_model, vision.config)

    def plan(self, observation, scene, request, image_shape=None):
        vision, geometry = (self.vision, self.geometry)
        shape = image_shape or observation['observation.images.left_wrist'].shape
        cameras = vision._robot_cameras(observation)
        cameras['left_wrist'] = vision._v6_wrist_camera(cameras['left_wrist'], observation)
        camera = vision._adjust_camera(cameras['left_wrist'], vision.camera_adjustments.get('left_wrist', np.zeros(7)))
        position, rotation = vision._eef_rotation(observation)
        column = int(request['source_column'])
        nominal_source = dict(center=np.r_[geometry.source_bin_center_m, 0.0], yaw=0.0)
        half = np.asarray(geometry.source_bin_half_size_m) + geometry.bin_wall_thickness_m
        z0 = float(geometry.source_wall_base_z_m)
        z1 = z0 + float(geometry.source_bin_wall_height_m)
        layout = np.asarray(geometry.cookie_source_positions_m, float)
        xs = np.unique(layout[:, 0])
        rows = layout[np.isclose(layout[:, 0], xs[column - 1])]
        rows = rows[np.argsort(rows[:, 1])][:int(request['count']) + 1]
        pitch = float(np.min(np.diff(xs)))
        lo = np.array([xs[column - 1] - pitch, nominal_source['center'][1] - half[1], z0])
        hi = np.array([xs[column - 1] + pitch, rows[-1, 1] + float(geometry.cookie_half_size_m[1]), z1])
        front_roi = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        corners = np.vstack([front_roi + [dx, dy, 0.0] for dx in (-0.02, 0.02) for dy in (-0.02, 0.02)])
        look_at = corners.mean(0)
        rim = front_roi
        robot, _ = vision._v6_gripper_silhouette(camera, shape)
        height, width = shape[:2]
        focal = height / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
        lower = np.array([(8 - (width - 1) / 2) / focal, ((height - 1) / 2 - (height - 9)) / focal])
        upper = np.array([(width - 9 - (width - 1) / 2) / focal, ((height - 1) / 2 - 8) / focal])
        from v10_leaves_17e0363838a9.two5_inner.perception import _project, _sample_map
        from v10_leaves_17e0363838a9.two5_inner.rule_controller import CANONICAL, _rotation
        source_rotation = _rotation('y', np.deg2rad((0.0, -15.0, -35.0, -35.0)[column - 1])) @ CANONICAL
        rotations = [source_rotation, rotation] + [Rotation.from_rotvec([0.0, 0.0, a]).as_matrix() @ rotation for a in (-0.12, 0.12, -0.24, 0.24)]
        rotations += [Rotation.from_rotvec([0.0, a, 0.0]).as_matrix() @ rotation for a in (-0.12, 0.12)]
        candidates = []
        for view_rotation in rotations:
            change = view_rotation @ rotation.T
            view = dict(camera, pos=position + change @ (camera['pos'] - position), rotation=change @ camera['rotation'])
            relative = (corners - look_at) @ view['rotation']
            for image_y in (0.24, 0.32, 0.4):
                for image_x in (0.5, 0.32, 0.68):
                    ray_xy = np.array([(image_x * width - (width - 1) / 2) / focal, ((height - 1) / 2 - image_y * height) / focal])
                    if np.any(ray_xy <= lower) or np.any(ray_xy >= upper):
                        continue
                    required = max(0.1, float(relative[:, 2].max() + 0.015))
                    for axis in range(2):
                        q, z = (relative[:, axis], relative[:, 2])
                        required = max(required, float(np.max((q + upper[axis] * z) / (upper[axis] - ray_xy[axis]))), float(np.max(-(q + lower[axis] * z) / (ray_xy[axis] - lower[axis]))))
                    for scale in (1.03, 1.18, 1.38):
                        depth = required * scale
                        delta = look_at - view['rotation'] @ np.r_[ray_xy * depth, -depth] - view['pos']
                        view_camera = dict(view, pos=view['pos'] + delta)
                        uv, z = _project(corners, view_camera, shape)
                        complete = bool(np.all(z > 0) and np.all(uv[:, 0] > 8) and np.all(uv[:, 0] < width - 9) and np.all(uv[:, 1] > 8) and np.all(uv[:, 1] < height - 9))
                        projected, rz = _project(rim, view_camera, shape)
                        occluded, inside = _sample_map((robot > 0).astype(float), projected, default=1.0)
                        visible = float(np.mean(inside & (rz > 0) & (occluded < 0.1)))
                        if complete and visible >= 0.65:
                            candidates.append((float(np.linalg.norm(delta)), delta, view_rotation, visible))
        target = scene.get('target_box') or dict(center=np.asarray(geometry.target_bin_world_position_m), yaw=0.0)
        public_scene = dict(source_box=nominal_source, target_box=target, path_scope='empty_hand_observation')
        protected = [dict(object_id='initial_source_occupied_domain', frame='box', box_key='source_box', bounds_local=[[-float(half[0] + 0.02), float(half[0] + 0.02)], [-float(half[1] + 0.02), float(half[1] + 0.02)], [float(geometry.source_floor_z_m), z1 + 2 * float(geometry.cookie_half_size_m[2])]], evidence_source='fixed initial occupied-region prior; no object identity asserted')]
        report = dict(needed=True, valid=False, issue='initial_source_identity', source_column=column, context_key=f'initial_source:{column}:{vision.calibration_generation}', measurement_id=(scene.get('group') or {}).get('measurement_id'), measurement_step=vision.step, expected_evidence=['source_front_wall', 'source_column_identity', 'first_component_and_next_boundary'], coverage_scope='initial_front_component; no full20-row requirement', prior_is_grasp_measurement=False, candidate_count=len(candidates), checked=[])
        opening = float(np.asarray(observation['observation.state'])[7])
        for index, (_, delta, view_rotation, visible) in enumerate(sorted(candidates, key=lambda item: item[0])):
            goal = position + delta
            high = max(float(position[2]), float(goal[2]))
            points = [position + [0, 0, high - position[2]], np.r_[goal[:2], high], goal]
            nodes = [dict(name=f'bootstrap_{j}', position=p, rotation=view_rotation, opening=opening, carrying=False, box_key='source_box') for j, p in enumerate(points)]
            checks = [self.checker.check_path(observation, nodes, None, public_scene, protected_objects=protected), self.checker.check_path(observation, [dict(n, box_key='target_box') for n in nodes], None, public_scene, protected_objects=[])]
            entry = dict(view_id=index, eef_position=goal.tolist(), eef_rotation=view_rotation.tolist(), path_states=[x['state'] for x in checks])
            report['checked'].append(entry)
            if any((x['state'] != 'feasible' for x in checks)):
                continue
            ik = self.ik.check_ik_path(observation, nodes, position_tolerance_m=0.0025, rotation_tolerance_rad=0.07)
            entry['ik'] = ik
            if ik['state'] != 'reachable':
                continue
            actual = self.ik.last_fk_path[-1]
            actual_camera = dict(camera, pos=actual['position'] + actual['rotation'] @ rotation.T @ (camera['pos'] - position), rotation=actual['rotation'] @ rotation.T @ camera['rotation'])
            uv, z = _project(corners, actual_camera, shape)
            actual_complete = bool(np.all(z > 0) and np.all(uv[:, 0] > 8) and np.all(uv[:, 0] < width - 9) and np.all(uv[:, 1] > 8) and np.all(uv[:, 1] < height - 9))
            if not actual_complete:
                entry['actual_fk_view'] = 'initial source domain clipped'
                continue
            actual_nodes = [dict(point, opening=opening, carrying=False, box_key='source_box') for point in self.ik.last_fk_path]
            actual_checks = [self.checker.check_path(observation, actual_nodes, None, public_scene, protected_objects=protected), self.checker.check_path(observation, [dict(n, box_key='target_box') for n in actual_nodes], None, public_scene, protected_objects=[])]
            entry['actual_fk_path_states'] = [item['state'] for item in actual_checks]
            if any((item['state'] != 'feasible' for item in actual_checks)):
                continue
            report.update(valid=True, view_id=index, waypoints=copy.deepcopy(nodes), eef_position=goal, eef_rotation=view_rotation, path_state='feasible', path_checks=checks, ik=ik, actual_fk_path_checks=actual_checks, actual_fk_view=True, predicted_visible_rim_fraction=visible, reason='first high source overview passes source/target CAD, public IK and actual-FK FOV')
            return report
        report['reason'] = 'no first source view passes the supplied initial-domain path and public IK checks'
        return report
