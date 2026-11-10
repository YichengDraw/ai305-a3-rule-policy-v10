"""Visibility of sampled world points using public joints and fixed robot CAD.

This projects opaque gripper triangles and an explicitly estimated payload.
It never reads episode object transforms or renders segmentation/depth images.
"""
from __future__ import annotations
import itertools
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], int)

def _camera_points(world, camera):
    return (np.asarray(world, float) - np.asarray(camera['pos'])) @ np.asarray(camera['rotation'])

def _project_camera(points, camera, shape):
    h, w = shape[:2]
    depth = -np.asarray(points)[:, 2]
    focal = h / 2 / np.tan(np.deg2rad(float(camera['fovy'])) / 2)
    safe = np.maximum(depth, 1e-05)
    return (np.column_stack([(w - 1) / 2 + focal * points[:, 0] / safe, (h - 1) / 2 - focal * points[:, 1] / safe]), depth)

def _clip_near(triangle, near=0.0001):
    result = []
    previous = triangle[-1]
    previous_inside = -previous[2] >= near
    for current in triangle:
        inside = -current[2] >= near
        if inside != previous_inside:
            t = (-near - previous[2]) / (current[2] - previous[2])
            result.append(previous + t * (current - previous))
        if inside:
            result.append(current)
        previous, previous_inside = (current, inside)
    return result

def triangle_occlusion(sample_uv, sample_depth, meshes, camera, shape):
    """Perspective-correct triangle depth at the requested image samples."""
    occluded = np.zeros(len(sample_uv), bool)
    if not len(sample_uv):
        return occluded
    sample_uv = np.asarray(sample_uv, float)
    sample_depth = np.asarray(sample_depth, float)
    min_uv = sample_uv.min(0)
    max_uv = sample_uv.max(0)
    for world, faces in meshes:
        vertices = _camera_points(world, camera)
        for face in np.asarray(faces, int):
            polygon = _clip_near(vertices[face])
            for k in range(1, len(polygon) - 1):
                tri = np.array([polygon[0], polygon[k], polygon[k + 1]])
                uv, depth = _project_camera(tri, camera, shape)
                low, high = (uv.min(0), uv.max(0))
                if np.any(high < min_uv) or np.any(low > max_uv):
                    continue
                ids = np.flatnonzero(~occluded & (sample_uv[:, 0] >= low[0]) & (sample_uv[:, 0] <= high[0]) & (sample_uv[:, 1] >= low[1]) & (sample_uv[:, 1] <= high[1]))
                if not len(ids):
                    continue
                a, b, c = uv
                denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(denominator) < 1e-12:
                    continue
                q = sample_uv[ids]
                u = ((b[1] - c[1]) * (q[:, 0] - c[0]) + (c[0] - b[0]) * (q[:, 1] - c[1])) / denominator
                v = ((c[1] - a[1]) * (q[:, 0] - c[0]) + (a[0] - c[0]) * (q[:, 1] - c[1])) / denominator
                w = 1 - u - v
                inside = (u >= -1e-09) & (v >= -1e-09) & (w >= -1e-09)
                inverse = u / depth[0] + v / depth[1] + w / depth[2]
                hit = inside & (inverse > 0)
                triangle_depth = np.full(len(ids), np.inf)
                triangle_depth[hit] = 1 / inverse[hit]
                occluded[ids[hit & (triangle_depth < sample_depth[ids] - 1e-06)]] = True
    return occluded

def public_gripper_meshes(vision, observation):
    """Use the independent robot model, including actual finite jaw travel."""
    data, model = (vision.robot_data, vision.robot_model)
    state = np.asarray(observation['observation.state'], float)
    addresses = [list(x) for x in vision.finger_addresses]
    saved = [data.qpos[x].copy() for x in addresses]
    meshes = []
    try:
        for side, index in enumerate((7, 15)):
            data.qpos[addresses[side]] = (1 - float(state[index])) * 0.0425
        mujoco.mj_kinematics(model, data)
        for geom, vertices, faces, _ in vision._v6_gripper_meshes:
            world = np.asarray(vertices) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            meshes.append((world.copy(), np.asarray(faces, int)))
    finally:
        for indices, old in zip(addresses, saved):
            data.qpos[indices] = old
        mujoco.mj_kinematics(model, data)
    return meshes

def payload_mesh(vision, observation, held, request):
    held = held or {}
    request = request or {}
    force = np.asarray(observation.get('observation.force', np.zeros(18)), float)
    stage = str(request.get('stage', '')).upper()
    carrying = held.get('currently_held') is True or (max(force[12:14]) >= 0.12 and stage in ('TRIAL_LIFT', 'LIFT', 'TRANSPORT', 'PLACE', 'PREPLACE_OBSERVE', 'HELD_OBSERVE', 'HELD_REOBSERVE', 'HELD_RETREAT', 'RECOVER_PATH', 'RECOVER_LOWER'))
    if not carrying:
        return (None, True, 'hand_not_reported_carrying')
    confidence = float(held.get('pose_confidence', 0.0) or 0.0)
    if confidence >= 0.6:
        center = held.get('center_tool')
        rotation = held.get('rotation_tool')
        basis = 'current_public_RGB_held_pose'
    elif float(request.get('held_pose_confidence', 0.0) or 0.0) >= 0.6:
        center = request.get('held_center_tool')
        rotation = request.get('held_rotation_tool')
        basis = 'retained_measured_tool_relative_payload_pose'
    else:
        return (None, False, 'payload_projection_unknown')
    if center is None or rotation is None:
        return (None, False, 'payload_projection_unknown')
    width = held.get('width_m')
    if width is None:
        width = 0.0053616 + 0.085 * float(np.asarray(observation['observation.state'])[7])
    half = np.asarray(vision.cookie_half, float).copy()
    half[1] = max(0.0, float(width)) / 2
    vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * half
    vertices = vertices @ np.asarray(rotation, float).T + np.asarray(center, float)
    pose = np.asarray(observation['observation.eef_pose'], float)
    tool = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
    return ((vertices @ tool.T + pose[:3], BOX_FACES), True, basis)

def sample_visibility(vision, observation, camera, world_points, shape, held=None, request=None):
    camera_points = _camera_points(world_points, camera)
    uv, depth = _project_camera(camera_points, camera, shape)
    h, w = shape[:2]
    inside = (depth > 0.0001) & (uv[:, 0] > 1) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1) & (uv[:, 1] < h - 2)
    meshes = public_gripper_meshes(vision, observation)
    payload, known, basis = payload_mesh(vision, observation, held, request)
    if payload is not None:
        meshes.append(payload)
    occluded = np.zeros(len(uv), bool)
    valid = np.flatnonzero(inside)
    if len(valid):
        occluded[valid] = triangle_occlusion(uv[valid], depth[valid], meshes, camera, shape)
    if not known:
        occluded[inside] = True
    return dict(uv=uv, depth=depth, in_frame=inside, occluded=occluded, visible=inside & ~occluded, payload_projection_known=known, payload_basis=basis, projection_basis='fixed gripper CAD + public joints/EEF + explicit RGB payload estimate')
