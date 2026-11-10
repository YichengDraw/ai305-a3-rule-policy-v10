"""Bounded source1 contact breakaway before the first trial-lift action.

Geometry comes from the retained measured source pose, current public pinch
width and finger contact. The core is an attachment hypothesis; additional side
attachments and hidden floor pixels remain unknown. No episode truth is read.
"""
from __future__ import annotations
import itertools
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from v10_leaves_17e0363838a9.two5_source1.legacy.rule_controller import PAD_OFFSET, gap_from_opening
from v10_leaves_17e0363838a9.two5_source1.legacy.path_feasibility import PathFeasibility, PublicPathIK, _poly, _z_slice
CORNERS = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
EXPERIMENT_MAX_M = 0.001

class SourcePretrialPlanner:

    def __init__(self, vision):
        self.vision = vision
        self.checker = PathFeasibility(vision.config)
        self.ik = PublicPathIK(self.checker.model, vision.config, self.checker)

    def plan(self, observation, scene, request, retained):
        out = dict(valid=False, scope='low source contact primitive before trial lift', pose_state='attachment_hypothesis', reason='pretrial source breakaway unavailable')
        if int(request.get('source_column', 0)) != 1 or int(request.get('count', 0)) != 5:
            return out
        if retained.get('center') is None or float(retained.get('confidence', 0.0) or 0.0) < 0.6:
            return {**out, 'reason': 'retained measured source geometry unavailable'}
        broad_id = retained.get('broad_reference_measurement_id')
        if broad_id is None:
            return {**out, 'reason': 'no retained observed outer-column edge reference'}
        touch = np.maximum(np.asarray(observation['observation.force'])[12:14], 0.0)
        held = scene.get('held') or {}
        width = gap_from_opening(float(np.asarray(observation['observation.state'])[7]))
        half = np.asarray(self.vision.cookie_half, float).copy()
        if min(touch) < 0.1 or held.get('count') != 5 or float(held.get('count_confidence', 0.0) or 0.0) < 0.6 or (abs(width - 10 * half[1]) > 0.0025):
            return {**out, 'reason': 'current row count, actual width and bilateral touch do not support the clamped core'}
        box = scene.get('source_box') or {}
        if box.get('center') is None or float(box.get('confidence', 0.0) or 0.0) < 0.6:
            return {**out, 'reason': 'public source frame unavailable'}
        p = np.asarray(observation['observation.eef_pose'], float)[:7]
        r = Rotation.from_quat(np.r_[p[4:7], p[3]]).as_matrix()
        body_r = np.asarray(retained['rotation'], float)
        center = np.asarray(retained['center'], float).copy()
        row = body_r[:, 1] / np.linalg.norm(body_r[:, 1])
        pad = p[:3] + r @ PAD_OFFSET
        center += row * float(np.dot(pad - center, row))
        half[1] = max(width, 10 * half[1]) / 2
        attachment = dict(center_tool=r.T @ (center - p[:3]), rotation_tool=r.T @ body_r, half_size=half, width_m=width, pose_state='attachment_hypothesis', measurement_id=retained['measurement_id'], source_geometry_version=copy.deepcopy(retained.get('geometry_version')), basis='retained measured broad/vertical source pose, current pad-midpoint row coordinate and actual compressed pinch width', observed_core_row_count=5, total_side_attachment_count='unknown', original_uncompressed_source_width_m=retained.get('width_m'))
        b = self.checker._box({'source_box': box}, 'source_box')
        if b is None:
            return out
        outward = -b['rotation'][:, 0]
        shapes = self.checker.gripper_vertices_tool(observation, float(np.asarray(observation['observation.state'])[7]))
        shapes = {**shapes, 'clamped_core_hypothesis': CORNERS * half @ attachment['rotation_tool'].T + attachment['center_tool']}
        low = []
        for vertices in shapes.values():
            local = (vertices @ r.T + p[:3] - b['center']) @ b['rotation']
            clipped = _z_slice(_poly(local), b['z0'], b['z1'])
            if len(clipped):
                low.append(clipped)
        if not low:
            return {**out, 'reason': 'no low source contact geometry'}
        clearance = float(np.vstack(low)[:, 0].min() + b['inner'][0])
        requested = min(EXPERIMENT_MAX_M, clearance - self.checker.collision_epsilon_m)
        out.update(available_outer_wall_clearance_m=clearance, requested_m=max(0.0, requested), maximum_experimental_displacement_m=EXPERIMENT_MAX_M, outer_edge_reference_measurement_id=broad_id, source_box=box, attachment=attachment, outer_space_state='retained observed outside edge; floor visibility unknown', additional_side_contacts='unknown; outward motion separates the intended core from inner-column layout hypotheses')
        public_centers = np.asarray((scene.get('cookies') or {}).get('centers', []), float)
        offsets = []
        if public_centers.shape == (len(self.vision.layout), 3):
            for column in range(1, len(self.vision.columns)):
                indices = np.flatnonzero(np.isclose(self.vision.layout[:, 0], self.vision.columns[column]))
                offsets.append(float(np.min((public_centers[indices] - center) @ b['rotation'][:, 0])))
        out['inner_column_layout_support_offsets_m'] = offsets
        out['inner_column_geometry_state'] = 'layout hypotheses; not current observed objects'
        if len(offsets) != len(self.vision.columns) - 1 or min(offsets) <= 0:
            return {**out, 'reason': 'current public source layout does not establish an outward separating direction'}
        if requested <= 0:
            return {**out, 'reason': 'physical low gripper/core CAD has no outward source-wall interval'}
        target = p[:3] + requested * outward

        def node(name, position):
            return dict(name=name, position=np.asarray(position), rotation=r, opening=float(np.asarray(observation['observation.state'])[7]), carrying=True, box_key='source_box')
        trial = target.copy()
        trial[2] = float(np.asarray(request['pretrial_trial_target'])[2])
        high = target.copy()
        high[2] = float(np.asarray(request['pretrial_high_target'])[2])
        nodes = [node('source_low_outward_breakaway', target), node('original_trial_after_breakaway', trial), node('source_high_after_breakaway', high)]
        protected = []
        local = request.get('alignment_geometry') or {}
        if local.get('rear_gap_center') is not None and local.get('gap_width_m') is not None and (float(local.get('confidence', 0.0) or 0.0) >= 0.6):
            rr = np.asarray(local['rotation'], float)
            rear = np.asarray(local['rear_gap_center'], float) - rr[:, 2] * self.vision.cookie_half[2] + rr[:, 1] * (float(local['gap_width_m']) + 2 * self.vision.cookie_half[1]) / 2
            protected.append(dict(object_id='measured_source_rear_neighbour', frame='world', center_world=rear, rotation_world=rr, half_size=np.asarray(self.vision.cookie_half), measurement_id=local.get('measurement_id'), geometry_basis='retained actual rear-gap measurement; native identity remains a hypothesis'))
        for key, entry in (scene.get('target_occupancy') or {}).items():
            if entry.get('bounds') is None or entry.get('occluded'):
                continue
            bounds = np.asarray(entry['bounds'])
            yaw = float(entry.get('box_yaw', (scene.get('target_box') or {}).get('yaw', 0.0)))
            rr = Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix()
            protected.append(dict(object_id='observed_target_column:' + str(key), frame='world', center_world=np.asarray(entry.get('box_center', scene['target_box']['center'])) + rr @ bounds.mean(1), rotation_world=rr, half_size=np.diff(bounds, axis=1).ravel() / 2))
        geometry = self.checker.check_path(observation, nodes, attachment, {'source_box': box, 'path_scope': 'loaded_source_pretrial'}, protected_objects=protected)
        geometry['basis'] = 'physical gripper CAD and explicitly retained source attachment hypothesis; no measured held pose'
        if geometry.get('full_path_status') == 'observation_path_checked':
            geometry['full_path_status'] = 'source_contact_prefix_checked'
        ik = self.ik.check_ik_path(observation, nodes, position_tolerance_m=0.0008, rotation_tolerance_rad=0.015)
        out.update(path=geometry, ik=ik, waypoints=nodes, protected_objects=protected, outward_world=outward, target=target, rotation=r, geometry_coverage='full physical hand/core source prefix, source walls/floor and observed rear/target objects; inner-column objects are contact hypotheses')
        if geometry['state'] != 'feasible' or ik['state'] != 'reachable':
            return {**out, 'reason': 'low breakaway/trial/high prefix lacks the existing CAD/IK clearance'}
        return {**out, 'valid': True, 'reason': 'bounded observed source1 breakaway prefix before lifting; actual separation still to be executed'}
