"""Released target measurements while carrying, with explicit RGB visibility.

Only full column coverage (possibly from overlapping cameras) replaces a prior
column envelope in this first isolated version. Partial observations and their
unknown intervals are exported for the later corridor-certificate stage.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.signal import find_peaks
from v10_leaves_17e0363838a9.two10_source1.public_visibility import sample_visibility

def _runs(mask, coordinates):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    return [[float(coordinates[g[0]]), float(coordinates[g[-1]])] for g in groups]

def _rotation(angle):
    c, s = (np.cos(angle), np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def measure_visible_released(vision, observation, cameras, request, held):
    box = vision.boxes['target_box']
    diagnostics = dict(measurement_step=int(vision.step), measurement_id=f'target_visibility:{vision.step}', target_geometry_version=box.get('geometry_version'), columns={}, ground_truth_used=False, partial_policy='retain prior envelope until sampled column coverage is complete')
    vision.v10_target_visibility = diagnostics
    if float(box.get('confidence', 0.0) or 0.0) < 0.55:
        diagnostics['reason'] = 'target pose confidence below existing occupancy threshold'
        return vision.v7_occupancy
    scene = vision.scene
    rotation = _rotation(float(box.get('yaw', 0.0)))
    center = np.asarray(box['center'], float)
    columns = np.unique(np.asarray(scene.target_slots_local_m)[:, 0])
    half_y = float(scene.target_bin_half_size_m[1]) - float(scene.bin_wall_thickness_m)
    floor = float(scene.target_floor_z_m) + float(scene.bin_wall_thickness_m) / 2
    top = floor + 2 * float(vision.cookie_half[2])
    ys = np.linspace(-half_y + 0.001, half_y - 0.001, 300)
    dy = float(ys[1] - ys[0])
    now = int(vision.step)
    for column, x in enumerate(columns):
        measurements = []
        for name in ('front', 'left_wrist'):
            if name not in cameras or 'observation.images.' + name not in observation:
                continue
            image = np.asarray(observation['observation.images.' + name])
            xs = np.linspace(x - vision.cookie_half[0] * 0.55, x + vision.cookie_half[0] * 0.55, 18)
            local = np.array([[xx, y, top] for y in ys for xx in xs])
            world = local @ rotation.T + center
            visibility = sample_visibility(vision, observation, cameras[name], world, image.shape, held, request)
            uv = visibility['uv']
            patch = cv2.remap(image, uv[:, 0].astype(np.float32).reshape(len(ys), len(xs)), uv[:, 1].astype(np.float32).reshape(len(ys), len(xs)), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            colors = vision._v6_color_mask(patch) > 0
            clear = visibility['visible'].reshape(len(ys), len(xs))
            signal = (colors & clear).sum(1) / np.maximum(clear.sum(1), 1)
            support = cv2.medianBlur((signal > 0.4).astype(np.uint8)[:, None], 5).ravel().astype(bool)
            row_clear = (clear.mean(1) >= 0.96) & ((np.mean(hsv[:, :, 2] < 28, axis=1) <= 0.12) | support)
            if not visibility['payload_projection_known']:
                row_clear[:] = False
            ids = np.flatnonzero(support & row_clear)
            endpoints = None if not len(ids) else [float(ys[ids[0]] - dy), float(ys[ids[-1]] + dy)]
            value = np.mean(hsv[:, :, 2].astype(float), axis=1) * np.clip(signal * 2, 0, 1)
            value = cv2.GaussianBlur(value.astype(np.float32)[:, None], (1, 5), 0).ravel()
            peaks, _ = find_peaks(value, prominence=max(5.0, float(np.std(value)) * 0.22), distance=max(5, int(0.0047 / dy)))
            peaks = np.asarray([p for p in peaks if signal[p] > 0.4 and row_clear[p]], int)
            measurements.append(dict(camera=name, clear=row_clear, support=support, endpoints=endpoints, count=int(len(peaks)), pixel_occluded_fraction=float(np.mean(visibility['occluded'])), covered_rows=int(row_clear.sum()), payload_basis=visibility['payload_basis']))
        if not measurements:
            diagnostics['columns'][str(column)] = {'state': 'unknown', 'reason': 'no sampled camera'}
            continue
        measurements.sort(key=lambda m: 0 if m['camera'] == 'left_wrist' and vision.v6_wrist_target_complete else 1)
        covered = np.zeros(len(ys), bool)
        support = np.zeros(len(ys), bool)
        for item in measurements:
            newly_seen = item['clear'] & ~covered
            support[newly_seen] = item['support'][newly_seen]
            covered |= item['clear']
        public_views = [{k: v for k, v in item.items() if k not in ('clear', 'support')} for item in measurements]
        entry_diagnostic = dict(state='measured' if covered.all() else 'partially_observed', cameras=public_views, covered_row_fraction=float(covered.mean()), unknown_y_intervals=_runs(~covered, ys), visible_occupied_y_intervals=_runs(support & covered, ys), visible_free_y_intervals=_runs(~support & covered, ys), coverage_complete=bool(covered.all()))
        diagnostics['columns'][str(column)] = entry_diagnostic
        fully_measured = [m for m in measurements if m['clear'].all() and m['endpoints'] is not None]
        if len(fully_measured) > 1:
            difference = float(np.max(np.abs(np.array(fully_measured[0]['endpoints']) - fully_measured[1]['endpoints'])))
            entry_diagnostic['independent_endpoint_disagreement_m'] = difference
            if difference > 0.008:
                entry_diagnostic.update(state='conflicting_views', reason='existing 8mm cross-view boundary criterion')
                continue
        if not covered.all():
            continue
        ids = np.flatnonzero(support)
        previous = vision.v7_occupancy.get(str(column))
        empty_cameras = [m for m in measurements if m['clear'].all() and m['endpoints'] is None]
        if not len(ids) and previous is not None and (previous.get('bounds') is not None) and (len(empty_cameras) < 2):
            entry_diagnostic.update(state='empty_requires_independent_confirmation')
            continue
        bounds = None if not len(ids) else np.array([[x - vision.cookie_half[0], x + vision.cookie_half[0]], [float(ys[ids[0]] - dy), float(ys[ids[-1]] + dy)], [floor, top]])
        preferred = measurements[0]
        full_candidates = [m for m in measurements if m['clear'].all()]
        count = full_candidates[0]['count'] if full_candidates else None
        if len(full_candidates) > 1 and any((abs(m['count'] - count) > 1 for m in full_candidates)):
            count = None
        entry = dict(camera=preferred['camera'], cameras=[m['camera'] for m in measurements], bounds=None if bounds is None else bounds.tolist(), count=count, confidence=0.65 if bounds is None else 0.68, uncertainty_m=0.002 if preferred['camera'] == 'left_wrist' else 0.0035, uncertainty_calibrated=False, scope='column', column=column, measurement_step=now, measurement_id=f'occupancy:{column}:{now}', excludes_held=True, occluded=False, coverage_complete=True, evidence_source='released RGB top bands; public CAD/payload occlusion; complete row coverage', box_center=center.tolist(), box_yaw=float(box.get('yaw', 0.0)), target_geometry_version=box.get('geometry_version'), fresh_visible_while_carrying=True, per_camera=public_views)
        vision.v7_occupancy[str(column)] = entry
    return vision.v7_occupancy
