"""Public RGB/CAD source frame and current neighbouring-object hypotheses.

The initial twenty-row identity frame is older evidence. Each occupancy result
is measured again from the current RGB; a layout position alone is never marked
as a currently verified object. No episode environment or seed is accepted.
"""
from __future__ import annotations
import copy
import cv2
import numpy as np
from scipy.signal import find_peaks

def calibrated_source_cameras(vision, observation):
    cameras = vision._robot_cameras(observation)
    cameras['left_wrist'] = vision._v6_wrist_camera(cameras['left_wrist'], observation)
    return {name: vision._adjust_camera(camera, vision.camera_adjustments.get(name, np.zeros(7))) for name, camera in cameras.items()}

def _project(points, camera, shape):
    h, w = shape[:2]
    f = h / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
    p = (np.asarray(points) - camera['pos']) @ camera['rotation']
    d = np.maximum(-p[:, 2], 1e-05)
    return (np.column_stack([(w - 1) / 2 + f * p[:, 0] / d, (h - 1) / 2 - f * p[:, 1] / d]), -p[:, 2])

class SourceGeometry:

    def __init__(self, config):
        self.scene = getattr(config, 'cookie_transfer', config)
        self.layout = np.asarray(self.scene.cookie_source_positions_m, float)
        self.columns = np.unique(self.layout[:, 0])
        self.half = np.asarray(self.scene.cookie_half_size_m, float)
        self.reset()

    def reset(self):
        self.broad_anchors = {}

    def estimate(self, observation, scene, cameras, *, measurement_step, color_mask):
        frame = scene.get('source_row_frame') or (scene.get('object_inventory') or {}).get('source_row_frame')
        if frame is None:
            return dict(state='unknown', reason='initial full-column row frame not yet observed', source_box=scene.get('source_box'), protected_objects=[], objects=[], coverage={'state': 'unknown', 'reason': 'no measured row identity frame'})
        rows = np.asarray(frame.get('row_centres_world', []), float)
        column = int(frame['source_column'])
        if rows.shape != (20, 3) or frame.get('created_contact_epoch') != 0:
            return dict(state='unknown', reason='initial undisturbed twenty-row RGB identity frame unavailable', source_box=scene.get('source_box'), protected_objects=[], objects=[], coverage={'state': 'unknown'})
        axis = np.asarray(frame['row_axis_world'], float)
        broad = np.asarray(frame['broad_axis_world'], float)
        rotation = np.column_stack([broad, axis, np.cross(broad, axis)])
        nominal = self.layout[np.isclose(self.layout[:, 0], self.columns[column - 1])]
        nominal = nominal[np.argsort(nominal[:, 1])]
        base = np.r_[np.asarray(self.scene.source_bin_center_m, float), 0.0]
        measured_along = rows @ axis
        nominal_along = np.column_stack([nominal, np.full(20, self.scene.cookie_model_z_m)]) @ axis
        row_translation = float(np.median(measured_along - nominal_along))
        group = scene.get('group') or {}
        edge = group.get('broad_measurement') or (scene.get('local_align') or {}).get('broad_measurement') or {}
        key = (column, frame.get('calibration_generation'))
        if key not in self.broad_anchors and edge.get('observable') and edge.get('outer_edge_observed') and (float(edge.get('confidence', 0.0) or 0.0) >= 0.65) and (edge.get('center_world') is not None):
            self.broad_anchors[key] = copy.deepcopy(edge)
        edge = self.broad_anchors.get(key)
        original = copy.deepcopy(scene.get('source_box') or {})
        box_center = np.asarray(original.get('center', base), float).copy()
        box_center += axis * (float(base @ axis) + row_translation - float(box_center @ axis))
        if edge is not None:
            nominal_broad = float(np.mean(nominal[:, 0]))
            box_broad = float(np.asarray(edge['center_world']) @ broad) - nominal_broad + float(base @ broad)
            box_center += broad * (box_broad - float(box_center @ broad))
        geometry_version = {'source_row_measurement_id': frame['measurement_id'], 'broad_measurement_id': None if edge is None else edge.get('measurement_id'), 'calibration_generation': frame.get('calibration_generation')}
        box = {**original, 'center': box_center.tolist(), 'rotation': rotation.tolist(), 'geometry_version': geometry_version, 'measurement_id': 'source_frame:' + str(frame['measurement_id']) + ':' + str(None if edge is None else edge.get('measurement_id')), 'measurement_step': frame['measurement_step'], 'row_axis_verified': True, 'broad_axis_verified': edge is not None, 'wall_pose_state': 'full_initial_RGB_rows_and_visible_outer_edge' if edge is not None else 'row_axis_only; broad_wall_prior_uncertain', 'row_axis_correction_m': float(row_translation), 'position_uncertainty_m': {'row': frame.get('row_uncertainty_m', 0.0015), 'broad': frame.get('broad_uncertainty_m', 0.006) if edge is None else edge.get('uncertainty_m', 0.002)}, 'basis': 'initial full twenty RGB rows + fixed source layout; outer edge supplies broad translation when observed'}
        row_values = rows @ axis
        lo, hi = (float(row_values.min() - 0.014), float(row_values.max() + 0.014))
        yy = np.linspace(lo, hi, 900)
        dy = float(yy[1] - yy[0])
        xx = np.linspace(-self.half[0] * 0.45, self.half[0] * 0.45, 20)
        midpoint = np.asarray(box['center'], float)
        base_broad = float(midpoint @ broad)
        top = float(self.scene.cookie_model_z_m + self.half[2])
        objects = []
        per_camera = {}
        selected = (group.get('source_unit') or {}).get('object_hypotheses', [])
        selected_centers = [np.asarray(o['center'], float) for o in selected if o.get('center') is not None]
        for col in range(1, len(self.columns) + 1):
            bx = base_broad + float(self.columns[col - 1] - self.scene.source_bin_center_m[0])
            points = np.asarray([broad * (bx + x) + axis * y + [0, 0, top] for y in yy for x in xx])
            views = []
            for name, camera in cameras.items():
                if 'observation.images.' + name not in observation:
                    continue
                image = np.asarray(observation['observation.images.' + name])
                uv, depth = _project(points, camera, image.shape)
                valid = ((depth > 0) & (uv[:, 0] > 2) & (uv[:, 0] < image.shape[1] - 3) & (uv[:, 1] > 2) & (uv[:, 1] < image.shape[0] - 3)).reshape(len(yy), len(xx))
                patch = cv2.remap(image, uv[:, 0].astype(np.float32).reshape(len(yy), len(xx)), uv[:, 1].astype(np.float32).reshape(len(yy), len(xx)), cv2.INTER_LINEAR)
                hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                colored = (color_mask(patch) > 0) & valid
                coverage = valid.mean(1)
                fraction = colored.sum(1) / np.maximum(valid.sum(1), 1)
                visible = (coverage > 0.95) & (np.mean(hsv[:, :, 2] < 28, axis=1) < 0.2)
                signal = np.mean(hsv[:, :, 2].astype(float) * np.where(colored, 1.0, 0.12), axis=1)
                signal = cv2.GaussianBlur(signal.astype(np.float32)[:, None], (1, 5), 0).ravel()
                peaks, _ = find_peaks(signal, prominence=max(5.0, float(np.std(signal[visible])) * 0.2 if np.any(visible) else 5.0), distance=max(4, int(0.0047 / dy)))
                peaks = np.asarray([i for i in peaks if visible[i] and fraction[i] > 0.35], int)
                test = np.asarray([broad * bx + axis * y + [0, 0, top] for y in (float(row_values[0]), float(row_values[0] + 0.0088333333))])
                pix, _ = _project(test, camera, image.shape)
                pitch_px = float(np.linalg.norm(pix[1] - pix[0]))
                positions = yy[peaks]
                views.append((name, positions, pitch_px, fraction, visible))
                per_camera[f'{col}:{name}'] = dict(row_positions_world_axis_m=positions.tolist(), pixels_per_pitch=pitch_px, visible_fraction=float(np.mean(visible)))
            for row, y in enumerate(row_values):
                high = []
                support = []
                for name, positions, pixels, fraction, visible in views:
                    j = int(np.argmin(abs(yy - y)))
                    center_support = bool(visible[j] and fraction[j] > 0.35)
                    if center_support:
                        support.append(name)
                    if pixels >= 4 and len(positions):
                        closest = float(positions[np.argmin(abs(positions - y))])
                        if abs(closest - y) < 0.0025:
                            high.append((pixels, name, closest))
                best = max(high, default=None)
                measured_y = float(y if best is None else best[2])
                center = broad * bx + axis * measured_y + [0, 0, self.scene.cookie_model_z_m]
                verified = bool(best is not None and len(support) >= 2 and (edge is not None))
                own = col == column and any((abs(float((center - c) @ axis)) < 0.004 for c in selected_centers))
                objects.append(dict(object_id=f'source_spatial_hypothesis:{col}:{row}', object_scope='individual_objects', identity_state='unconfirmed_layout_row_hypothesis', source_column=col, row_hypothesis=row, frame='world', box_key='source_box', center_world=center.tolist(), rotation_world=rotation.tolist(), half_size=self.half.tolist(), geometry_version=geometry_version, measurement_id=f'source_neighbour:{col}:{row}:{measurement_step}', measurement_step=int(measurement_step), state='verified_current_RGB_occupancy' if verified else 'single_view_RGB_supported' if best is not None else 'unobserved_layout_hypothesis', verified=verified, cameras=support, precise_camera=None if best is None else best[1], part_of_selected_source_unit=own, uncertainty_m={'row': 0.0015 if best is not None else 0.004, 'broad': box['position_uncertainty_m']['broad']}))
        protected = [o for o in objects if o['verified'] and (not o['part_of_selected_source_unit'])]
        return dict(state='measured' if edge is not None else 'partial', source_box=box, objects=objects, protected_objects=protected, measurement_id=f'source_geometry:{column}:{measurement_step}', measurement_step=int(measurement_step), geometry_version=geometry_version, current_contact_epoch=frame.get('current_contact_epoch'), identity_measurement_id=frame['measurement_id'], coverage=dict(state='current_RGB_partial', verified_count=sum((o['verified'] for o in objects)), protected_count=len(protected), unknown_hypotheses=[o['object_id'] for o in objects if not o['verified']], selected_source_column=column, note='unobserved layout hypotheses are not current empty space'), per_camera=per_camera)

def source_planning_scene(scene, evidence):
    """Return a planning copy. The original perception box is preserved."""
    return {**scene, 'source_box': evidence.get('source_box', scene.get('source_box')), 'source_geometry': evidence, 'source_neighbours': evidence.get('protected_objects', [])}

def source_path_coverage(checker, observation, waypoints, evidence):
    """Identify unobserved source hypotheses inside this gripper path region.

    The exact swept collision result remains the D checker's responsibility.
    This query only prevents a supplied partial RGB list from implying empty
    source space. It uses the same actual-opening CAD and <=2deg rotations.
    """
    from scipy.spatial.transform import Rotation, Slerp
    if evidence.get('state') == 'unknown' or not (evidence.get('source_box') or {}).get('broad_axis_verified'):
        return dict(state='unknown', reason='source broad-axis wall geometry not yet verified', unknown_hypotheses=[])
    unknown = [o for o in evidence.get('objects', []) if not o.get('verified') and (not o.get('part_of_selected_source_unit'))]
    relevant = set()
    height = float(checker.scene.cookie_model_z_m + checker.scene.cookie_half_size_m[2])
    nodes = [p for p in waypoints if p.get('box_key') == 'source_box']
    for start, end in zip(nodes[:-1], nodes[1:]):
        ra = np.asarray(start['rotation'], float)
        rb = np.asarray(end['rotation'], float)
        angle = float(Rotation.from_matrix(ra.T @ rb).magnitude())
        parts = max(1, int(np.ceil(angle / np.deg2rad(2.0))))
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([ra, rb]))
        all_points = []
        for t in np.linspace(0.0, 1.0, parts + 1):
            rot = slerp(t).as_matrix()
            pos = (1 - t) * np.asarray(start['position']) + t * np.asarray(end['position'])
            opening = (1 - t) * start['opening'] + t * end['opening']
            for vertices in checker.gripper_vertices_tool(observation, opening).values():
                points = vertices @ rot.T + pos
                if points[:, 2].min() <= height:
                    all_points.append(points)
        if not all_points:
            continue
        points = np.concatenate(all_points)
        lo = points.min(0)
        hi = points.max(0)
        for obj in unknown:
            center = np.asarray(obj['center_world'])
            half = np.asarray(obj['half_size'])
            rotation = np.asarray(obj['rotation_world'])
            extent = np.abs(rotation) @ half
            if np.all(center + extent >= lo) and np.all(center - extent <= hi):
                relevant.add(obj['object_id'])
    return dict(state='covered' if not relevant else 'unknown', unknown_hypotheses=sorted(relevant), reason='current RGB verifies source hypotheses in the supplied low gripper path' if not relevant else 'current RGB has not verified source hypotheses in the low gripper path')
