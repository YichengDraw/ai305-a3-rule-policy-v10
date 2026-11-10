"""Nominal CAD swept-volume checks using public robot and visual scene inputs.

This module never receives an episode environment, object truth or contact graph.
It reports geometric feasibility for the supplied objects and their association.
Official completion/success and force-based execution remain outside this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp
from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
from a3_dual_arm_sim.sim.model import build_model

def _rz(a):
    c, s = (np.cos(a), np.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def _corners(half):
    h = np.asarray(half, float)
    return np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])

def _unique_axes(axes):
    a = np.asarray(axes, float).reshape(-1, 3)
    length = np.linalg.norm(a, axis=1)
    a = a[length > 1e-09] / length[length > 1e-09, None]
    if not len(a):
        return a
    j = np.argmax(abs(a), axis=1)
    a *= np.where(a[np.arange(len(a)), j] < 0, -1.0, 1.0)[:, None]
    return np.unique(np.round(a, 7), axis=0)

@dataclass
class Poly:
    vertices: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    pairs: np.ndarray

def _poly(vertices):
    v = np.unique(np.round(np.asarray(vertices, float), 10), axis=0)
    hull = ConvexHull(v)
    pairs = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]], hull.simplices[:, [1, 2]], hull.simplices[:, [0, 2]]]), axis=1), axis=0)
    return Poly(v, _unique_axes(hull.equations[:, :3]), _unique_axes(v[pairs[:, 1]] - v[pairs[:, 0]]), pairs)

def _separation(a, b):
    broad = np.maximum(b.vertices.min(0) - a.vertices.max(0), a.vertices.min(0) - b.vertices.max(0))
    if np.max(broad) > 0:
        return float(np.max(broad))
    axes = _unique_axes(np.vstack((a.normals, b.normals, np.cross(a.edges[:, None, :], b.edges[None, :, :]).reshape(-1, 3))))
    ap = a.vertices @ axes.T
    bp = b.vertices @ axes.T
    gap = np.maximum(bp.min(0) - ap.max(0), ap.min(0) - bp.max(0))
    return float(np.max(gap))

def _z_slice(poly, low, high):
    v = poly.vertices
    inside = (v[:, 2] >= low) & (v[:, 2] <= high)
    points = [v[inside]]
    a = v[poly.pairs[:, 0]]
    b = v[poly.pairs[:, 1]]
    for level in (low, high):
        crossed = (a[:, 2] - level) * (b[:, 2] - level) < 0
        aa = a[crossed]
        bb = b[crossed]
        if len(aa):
            points.append(aa + (bb - aa) * ((level - aa[:, 2]) / (bb[:, 2] - aa[:, 2]))[:, None])
    return np.concatenate(points) if any((len(x) for x in points)) else np.empty((0, 3))

def _subtract(intervals, forbidden):
    out = []
    lo, hi = forbidden
    for a, b in intervals:
        if hi <= a or lo >= b:
            out.append([a, b])
            continue
        if a < lo:
            out.append([a, min(lo, b)])
        if hi < b:
            out.append([max(hi, a), b])
    return out

def tool_envelope(envelope, grasp_position, grasp_rotation, box=None):
    """Convert an isolated measured object envelope at its reached grasp pose."""
    e = dict(envelope)
    frame = e.get('frame', 'tool')
    if frame == 'tool' or e.get('center_tool') is not None:
        return e
    r = np.asarray(e.get('rotation_world', e.get('rotation', np.eye(3))), float)
    c = np.asarray(e.get('center_world', e.get('center', e.get('center_local'))), float)
    if frame in ('box', 'box_local'):
        if box is None:
            raise ValueError('box frame requires its visually measured pose')
        br = np.asarray(box.get('rotation', _rz(float(box.get('yaw', 0.0)))), float)
        c = np.asarray(box['center']) + br @ c
        r = br @ np.asarray(e.get('rotation_local', r), float)
    gr = np.asarray(grasp_rotation, float)
    return {**e, 'frame': 'tool', 'center_tool': gr.T @ (c - np.asarray(grasp_position)), 'rotation_tool': gr.T @ r, 'half_size': e.get('half_size')}

class PathFeasibility:

    def __init__(self, config):
        self.config = config
        self.scene = getattr(config, 'cookie_transfer', config)
        bundle = build_model(config, scene='cookie_transfer')
        self.model = bundle.model
        self.data = mujoco.MjData(self.model)
        self.arm_addresses = [int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in (*LEFT_JOINTS, *RIGHT_JOINTS)]
        self.finger_addresses = [[int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_finger_{f}_joint')]) for f in ('inner', 'outer')] for side in ('L', 'R')]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'L_eef')
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'L_flange')
        self.geoms = []
        for gid in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[gid])
            inside = False
            while body > 0:
                if body == root:
                    inside = True
                    break
                body = int(self.model.body_parentid[body])
            if inside and self.model.geom_contype[gid] and self.model.geom_conaffinity[gid]:
                self.geoms.append(gid)
        self.closed = None
        self.opened = None
        self.collision_epsilon_m = 0.00015

    def _gripper_cache(self, observation):
        if self.closed is not None:
            return
        state = np.asarray(observation['observation.state'], float)
        self.data.qpos[self.arm_addresses] = np.r_[state[:7], state[8:15]]
        saved = []
        for opening in (0.0, 1.0):
            self.data.qpos[self.finger_addresses[0]] = (1 - opening) * 0.0425
            self.data.qpos[self.finger_addresses[1]] = (1 - float(state[15])) * 0.0425
            mujoco.mj_kinematics(self.model, self.data)
            origin = self.data.site_xpos[self.site].copy()
            rotation = self.data.site_xmat[self.site].reshape(3, 3).copy()
            pieces = {}
            for gid in self.geoms:
                kind = int(self.model.geom_type[gid])
                size = self.model.geom_size[gid]
                if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                    vertices = _corners(size)
                elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    angles = (np.arange(64) + 0.5) * 2 * np.pi / 64
                    radius = size[0] / np.cos(np.pi / 64)
                    vertices = np.array([[radius * np.cos(a), radius * np.sin(a), z] for z in (-size[1], size[1]) for a in angles])
                else:
                    raise ValueError(f'unsupported physical gripper geom type {kind}')
                world = vertices @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
                pieces[name] = (world - origin) @ rotation
            saved.append(pieces)
        self.closed, self.opened = saved

    def gripper_vertices_tool(self, observation, opening):
        self._gripper_cache(observation)
        o = float(opening)
        if not 0 <= o <= 1:
            raise ValueError('opening is a physical normalized aperture in [0,1]')
        return {name: (1 - o) * v + o * self.opened[name] for name, v in self.closed.items()}

    def _held_vertices(self, envelope):
        if not envelope:
            return None
        e = envelope.get('envelope', envelope)
        if e.get('center_tool') is not None:
            half = e.get('half_size')
            if half is None:
                width = e.get('width_m')
                if width is None:
                    return None
                half = np.asarray(self.scene.cookie_half_size_m, float).copy()
                half[1] = float(width) / 2
            return _corners(half) @ np.asarray(e.get('rotation_tool', np.eye(3)), float).T + np.asarray(e['center_tool'], float)
        if e.get('bounds_tool') is not None:
            b = np.asarray(e['bounds_tool'], float)
            if b.shape != (3, 2):
                return None
            return _corners((b[:, 1] - b[:, 0]) / 2) + b.mean(1)
        return None

    def _box(self, scene, key):
        box = scene.get(key)
        if not box or box.get('center') is None:
            return None
        kind = 'source' if key == 'source_box' else 'target'
        half = np.asarray(getattr(self.scene, kind + '_bin_half_size_m'), float)
        height = float(getattr(self.scene, kind + '_bin_wall_height_m'))
        floor = float(getattr(self.scene, kind + '_floor_z_m'))
        z0 = float(self.scene.source_wall_base_z_m) if kind == 'source' else 0.0
        r = np.asarray(box.get('rotation', _rz(float(box.get('yaw', 0.0)))), float)
        center = np.asarray(box['center'], float)
        t = float(self.scene.bin_wall_thickness_m)
        obstacles = []
        for axis in (0, 1):
            other = 1 - axis
            for sign in (-1, 1):
                c = np.zeros(3)
                c[axis] = sign * half[axis]
                c[2] = z0 + height / 2
                h = np.array([half[0] + t, half[1] + t, height / 2])
                h[axis] = t
                obstacles.append((f'{key}:wall:{axis}:{sign}', _poly(_corners(h) + c)))
        obstacles.append((f'{key}:floor', _poly(_corners([half[0], half[1], t / 2]) + [0, 0, floor])))
        return dict(key=key, center=center, rotation=r, half=half, inner=half - t, z0=z0, z1=z0 + height, obstacles=obstacles, geometry_version=box.get('geometry_version', scene.get('geometry_version')), measurement_id=box.get('measurement_id'))

    @staticmethod
    def _object_poly(obj, box):
        if obj.get('bounds_local') is not None or obj.get('bounds') is not None:
            bounds = np.asarray(obj.get('bounds_local', obj.get('bounds')), float)
            if bounds.shape != (3, 2):
                return None
            vertices = _corners(np.diff(bounds, axis=1).ravel() / 2) + bounds.mean(1)
        else:
            half = obj.get('half_size')
            center = obj.get('center_local', obj.get('center_world', obj.get('center')))
            if half is None or center is None:
                return None
            rotation = np.asarray(obj.get('rotation_local', obj.get('rotation_world', obj.get('rotation', np.eye(3)))), float)
            vertices = _corners(half) @ rotation.T + np.asarray(center, float)
        if obj.get('frame', 'box') == 'world':
            vertices = (vertices - box['center']) @ box['rotation']
        return _poly(vertices)

    def _scope_reason(self, unit):
        if unit is None:
            return None
        scope = unit.get('object_scope', unit.get('scope'))
        if scope not in ('batch', 'isolated_batch', 'individual_objects'):
            return 'repair unit is not an isolated associated object group'
        if unit.get('source_column') is None or unit.get('batch_index') is None:
            return 'repair unit lacks original source and batch association'
        count = unit.get('count_interval')
        if count is None or len(count) != 2 or count[0] != count[1]:
            return 'repair quantity is not independently resolved'
        if len(self.scene.target_slots_local_m) == 10 and int(count[0]) != 5:
            return 'two-by-five reseat requires its original five-object batch'
        ids = unit.get('object_ids', unit.get('object_hypotheses'))
        if ids is not None and len(ids) != int(count[0]):
            return 'repair object association and physical count have different scope'
        return None

    def check_path(self, observation, waypoints, actual_envelope, public_scene, *, repair_unit=None, protected_objects=None, opening_schedule=None):
        """Check the complete named path. No coordinate is silently clipped.

        Every waypoint has position, rotation (3x3), physical opening [0,1],
        carrying bool and optional box_key. Include the reached closed grasp
        pose as the first carrying waypoint. After carrying becomes false,
        that object remains a protected obstacle at its last placement pose.
        """
        scope_reason = self._scope_reason(repair_unit)
        if scope_reason:
            return {'state': 'unknown', 'status': 'unknown', 'reason': scope_reason, 'next_action': 'observe_and_associate_objects', 'segments': [], 'conflicts': []}
        if not waypoints:
            return {'state': 'unknown', 'status': 'unknown', 'reason': 'no complete path supplied', 'next_action': 'supply_named_path', 'segments': [], 'conflicts': []}
        public_pose = np.asarray(observation['observation.eef_pose'], float)[:7]
        rotation = Rotation.from_quat(np.r_[public_pose[4:], public_pose[3]]).as_matrix()
        state = np.asarray(observation['observation.state'], float)
        initial = dict(name='current', position=public_pose[:3], rotation=rotation, opening=float(state[7]), carrying=bool(waypoints[0].get('carrying', True)), box_key=waypoints[0].get('box_key', 'target_box'))
        nodes = [initial]
        for i, w in enumerate(waypoints):
            n = dict(w)
            n['position'] = np.asarray(n['position'], float)
            n['rotation'] = np.asarray(n['rotation'], float).reshape(3, 3)
            n['opening'] = float(opening_schedule[i] if opening_schedule is not None else n.get('opening', nodes[-1]['opening']))
            n['carrying'] = bool(n.get('carrying', nodes[-1]['carrying']))
            n['box_key'] = n.get('box_key', nodes[-1]['box_key'])
            nodes.append(n)
        held = self._held_vertices(actual_envelope)
        unknown = []
        if held is None and any((n['carrying'] for n in nodes)):
            unknown.append('actual tool-frame held envelope unavailable')
        if protected_objects is None:
            protected_objects = public_scene.get('objects')
            if protected_objects is None:
                protected_objects = []
                unknown.append('protected-object coverage not supplied')
        boxes = {key: self._box(public_scene, key) for key in set((n['box_key'] for n in nodes))}
        if any((box is None for box in boxes.values())):
            return {'state': 'unknown', 'status': 'unknown', 'reason': 'visual box pose missing for a path segment', 'next_action': 'observe_box', 'segments': [], 'conflicts': []}
        conflicts = []
        contacts = []
        segments = []
        intervals = [[-np.inf, np.inf]]
        reference_y = None
        released = None
        references = [i for i, n in enumerate(nodes[1:]) if n.get('reference')]
        reference_index = references[-1] if references else None
        attachments = [i for i, (a, b) in enumerate(zip(nodes, nodes[1:])) if not a['carrying'] and b['carrying']]
        attachment_index = attachments[0] if attachments else None
        fixed_unit_world = None
        if attachment_index is not None and held is not None:
            grasp = nodes[attachment_index + 1]
            fixed_unit_world = held @ grasp['rotation'].T + grasp['position']
        self._gripper_cache(observation)
        for index, (start, end) in enumerate(zip(nodes, nodes[1:])):
            key = end['box_key']
            box = boxes[key]
            angle = float(Rotation.from_matrix(start['rotation'].T @ end['rotation']).magnitude())
            subdivisions = max(1, int(np.ceil(angle / np.deg2rad(2.0))))
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix([start['rotation'], end['rotation']]))
            objects = []
            for j, obj in enumerate(protected_objects):
                if obj.get('box_key', key) != key:
                    continue
                poly = self._object_poly(obj, box)
                if poly is None:
                    unknown.append(f'object {obj.get('object_id', j)} envelope unknown')
                    continue
                if obj.get('geometry_version') and box['geometry_version'] and (obj['geometry_version'] != box['geometry_version']):
                    unknown.append(f'object {obj.get('object_id', j)} geometry version stale')
                    continue
                objects.append((str(obj.get('object_id', j)), poly))
            if start['carrying'] and (not end['carrying']) and (held is not None):
                released = _poly((held @ start['rotation'].T + start['position'] - box['center']) @ box['rotation'])
            if released is not None:
                objects.append(('released_unit', released))
            if fixed_unit_world is not None and index <= attachment_index:
                objects.append(('repair_unit_before_grasp', _poly((fixed_unit_world - box['center']) @ box['rotation'])))
            seg = dict(index=index, name=end.get('name', str(index)), box_key=key, subdivisions=subdivisions, minimum_separation_m=None, conflicts=[])
            for sub in range(subdivisions):
                times = np.array([sub / subdivisions, (sub + 1) / subdivisions])
                rr = slerp(times).as_matrix()
                pp = np.array([(1 - a) * start['position'] + a * end['position'] for a in times])
                openings = [(1 - a) * start['opening'] + a * end['opening'] for a in times]
                local_vertices = []
                for pos, rot, opening in zip(pp, rr, openings):
                    shapes = self.gripper_vertices_tool(observation, opening)
                    if end['carrying'] and held is not None:
                        shapes = {**shapes, 'held_unit': held}
                    local_vertices.append({name: (v @ rot.T + pos - box['center']) @ box['rotation'] for name, v in shapes.items()})
                for name in local_vertices[0]:
                    sweep = _poly(np.vstack([local_vertices[0][name], local_vertices[1][name]]))
                    clipped = _z_slice(sweep, box['z0'], box['z1'])
                    adjustable = bool(end.get('adjustable_y', reference_index is not None and index >= reference_index))
                    if len(clipped) and adjustable:
                        lo = -box['inner'][1] - float(clipped[:, 1].min())
                        hi = box['inner'][1] - float(clipped[:, 1].max())
                        intervals = [[max(a, lo), min(b, hi)] for a, b in intervals if max(a, lo) <= min(b, hi)]
                        for oid, ob in objects:
                            if oid in ('released_unit', 'repair_unit_before_grasp'):
                                continue
                            if np.ptp(clipped[:, 0]) == 0:
                                continue
                            if clipped[:, 0].max() <= ob.vertices[:, 0].min() or clipped[:, 0].min() >= ob.vertices[:, 0].max():
                                continue
                            if clipped[:, 2].max() <= ob.vertices[:, 2].min() or clipped[:, 2].min() >= ob.vertices[:, 2].max():
                                continue
                            intervals = _subtract(intervals, [float(ob.vertices[:, 1].min() - clipped[:, 1].max()), float(ob.vertices[:, 1].max() - clipped[:, 1].min())])
                    for obstacle, poly in box['obstacles'] + objects:
                        if obstacle == 'repair_unit_before_grasp' and name == 'held_unit' and (index == attachment_index):
                            continue
                        gap = _separation(sweep, poly)
                        if seg['minimum_separation_m'] is None or gap < seg['minimum_separation_m']:
                            seg['minimum_separation_m'] = gap
                        if gap < -self.collision_epsilon_m:
                            event = dict(piece=name, obstacle=obstacle, subsegment=sub, separation_m=gap)
                            separating = False
                            if obstacle == 'released_unit' and name.startswith('L_finger_') and start['carrying'] and (not end['carrying']) and (end['opening'] >= start['opening']) and (angle <= np.deg2rad(3.0)):
                                before = _separation(_poly(local_vertices[0][name]), poly)
                                after = _separation(_poly(local_vertices[1][name]), poly)
                                separating = after >= -self.collision_epsilon_m and after > before
                            if separating:
                                contacts.append({'segment': index, 'name': seg['name'], **event, 'kind': 'expected_initial_pad_contact', 'reason': 'fixed-pose opening separates the pad from the released envelope'})
                            elif obstacle == 'repair_unit_before_grasp' and name.startswith('L_finger_') and (index == attachment_index) and (end['opening'] <= start['opening']) and (np.linalg.norm(end['position'] - start['position']) <= 0.001) and (angle <= np.deg2rad(3.0)):
                                contacts.append({'segment': index, 'name': seg['name'], **event, 'kind': 'intended_grasp_pad_contact', 'reason': 'stationary closing at the isolated associated unit'})
                            else:
                                seg['conflicts'].append(event)
                    if end.get('reference') and name == 'held_unit':
                        reference_y = float(np.mean(local_vertices[1][name][:, 1]))
            if seg['conflicts']:
                conflicts.extend(({'segment': index, 'name': seg['name'], **x} for x in seg['conflicts']))
            segments.append(seg)
        finite_intervals = [[None if not np.isfinite(a) else float(a), None if not np.isfinite(b) else float(b)] for a, b in intervals]
        feasible_y = None if reference_y is None else [[None if a is None else a + reference_y, None if b is None else b + reference_y] for a, b in finite_intervals]
        released_once = False
        coverage = dict(descend=False, release_open=False, withdraw_after_release=False)
        for start, end in zip(nodes, nodes[1:]):
            dz = float(end['position'][2] - start['position'][2])
            coverage['descend'] |= dz < -0.003
            if start['carrying'] and (not end['carrying']) and (end['opening'] > start['opening']):
                coverage['release_open'] = True
                released_once = True
            if released_once and (not end['carrying']) and (dz > 0.003):
                coverage['withdraw_after_release'] = True
        full_path = all(coverage.values())
        observation_only = bool(public_scene.get('path_scope') == 'empty_hand_observation' and repair_unit is None and (actual_envelope is None) and (not any((n['carrying'] for n in nodes))))
        if not full_path and (not observation_only):
            unknown.append('supplied path omits a descend, release-opening or post-release withdrawal segment')
        result_state = 'blocked' if conflicts else 'unknown' if unknown else 'feasible'
        return dict(state=result_state, status=result_state, reason='nominal physical CAD sweep intersects named geometry' if conflicts else '; '.join(dict.fromkeys(unknown)) if unknown else 'complete supplied gripper/held sweep clears supplied geometry', next_action='change_pose_or_rearrange' if conflicts else 'observe_missing_geometry' if unknown else 'execute_and_verify', segments=segments, conflicts=conflicts, contact_events=contacts, coverage=coverage, full_path_status='checked' if full_path else 'empty_hand_observation_checked' if observation_only else 'incomplete', translation_y_intervals=finite_intervals, feasible_y_intervals=feasible_y, requested_reference_y=reference_y, geometry_version=public_scene.get('geometry_version', next(iter(boxes.values()))['geometry_version']), box_geometry_versions={k: b['geometry_version'] for k, b in boxes.items()}, basis='physical gripper CAD and measured envelope; visual-only mesh does not create physical collision', uncertainty_calibrated=False, rotation_sweep_resolution_deg=2.0, cylinder_radial_error_bound_m=6e-05, ik_state='not_checked', release_assumption='unit stays at the last carrying pose after the release waypoint; execution must verify')
