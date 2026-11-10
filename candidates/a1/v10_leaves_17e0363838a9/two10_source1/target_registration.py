"""Register target-local sampling to a camera's own current RGB box contour."""
from __future__ import annotations
import numpy as np
from v10_leaves_17e0363838a9.two10_source1.target_occupancy import measure_visible_released

def _rotation(yaw):
    c, s = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def registered_occupancy(vision, observation, cameras, request, held):
    box = vision.boxes['target_box']
    evidence = dict(state='parent_projection', measurement_step=int(vision.step), measurement_id=f'target_local_registration:{vision.step}', scope='target-local RGB sampling only; world anchor and original camera calibration retained')
    vision.v9_target_geometry['occupancy_registration'] = evidence
    if not box.get('anchor_confirmed') or float(box.get('confidence', 0.0)) < 0.6:
        evidence['reason'] = 'confirmed world target anchor required'
        return None
    if vision.v9_target_geometry.get('pose_state') == 'suspected_moved':
        evidence['reason'] = 'pending world target motion must be resolved first'
        return None
    camera = cameras['front']
    center = np.asarray(box['center'], float)
    half = np.asarray(vision.scene.target_bin_half_size_m) + vision.scene.bin_wall_thickness_m
    corners = np.array([[x, y, z] for z in (0.0, vision.scene.target_bin_wall_height_m) for x in (-half[0], half[0]) for y in (-half[1], half[1])])
    previous = (vision.v9_target_geometry.get('per_camera') or {}).get('front', {})
    seed = np.r_[np.asarray(previous.get('center', center))[:2], previous.get('yaw', box['yaw'])]
    fit = vision._v9_target_view('front', observation['observation.images.front'], camera, seed, float(center[2]), corners)
    evidence['camera_measurement'] = fit
    if not (fit.get('valid') and fit.get('metric_observable') and fit.get('complete')):
        evidence['reason'] = 'current independent complete metric box contour required'
        return None
    translation = float(np.linalg.norm(np.asarray(fit['center']) - center))
    yaw = float(fit['yaw']) - float(box['yaw'])
    evidence.update(camera_world_anchor_disagreement_m=translation, yaw_disagreement_rad=yaw)
    if translation <= 0.008 and abs(yaw) <= 0.06:
        evidence['reason'] = 'camera and world target fit agree at existing geometric scales'
        return None
    change = _rotation(float(fit['yaw'])) @ _rotation(float(box['yaw'])).T
    offset = np.asarray(fit['center']) - change @ center
    projection = dict(camera, pos=change.T @ (np.asarray(camera['pos']) - offset), rotation=change.T @ np.asarray(camera['rotation']))
    target_cameras = dict(cameras, front=projection)
    result = measure_visible_released(vision, observation, target_cameras, request, held)
    evidence.update(state='camera_relative_target_projection', reason='own RGB box contour anchors local pixel coordinates', registration_translation_m=offset, registration_yaw_rad=yaw, world_anchor_unchanged=True, source_and_held_projection_unchanged=True, coverage=vision.v10_target_visibility)
    return result
