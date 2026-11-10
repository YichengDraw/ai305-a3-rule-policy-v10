"""Released target measurements while carrying, with explicit RGB visibility.

W1a candidate (parent: frozen V10 two10_source1). The per-camera sampling
below is unchanged. Row fusion and the entry-write decision moved into
fusion_conflict.decide_column_entry so overlapping-view occupied/free
disagreements are camera-order invariant and, beyond the existing 8mm
cross-view scale, inadmissible for overwriting the stored column entry in
either direction (neither creating nor clearing occupancy). Only unconflicted
complete-coverage measurements update `v7_occupancy`; the entry layout is
unchanged for all consumers, and conflicts are reported per column in
`v10_target_visibility` diagnostics with their ROI and per-view ids.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.signal import find_peaks
from v10_leaves_17e0363838a9.two10_source1.public_visibility import sample_visibility
from v10_leaves_17e0363838a9.two10_source1.fusion_conflict import decide_column_entry

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
            measurements.append(dict(camera=name, clear=row_clear, support=support, endpoints=endpoints, count=int(len(peaks)), pixel_occluded_fraction=float(np.mean(visibility['occluded'])), covered_rows=int(row_clear.sum()), payload_basis=visibility['payload_basis'], measurement_id=f'occupancy_view:{column}:{name}:{now}'))
        if not measurements:
            diagnostics['columns'][str(column)] = {'state': 'unknown', 'reason': 'no sampled camera'}
            continue
        geometry = dict(column=column, x=float(x), cookie_half_x=float(vision.cookie_half[0]), floor=floor, top=top, now=now, wrist_target_complete=bool(vision.v6_wrist_target_complete), box_center=center.tolist(), box_yaw=float(box.get('yaw', 0.0)), target_geometry_version=box.get('geometry_version'))
        previous = vision.v7_occupancy.get(str(column))
        entry, entry_diagnostic = decide_column_entry(measurements, ys, geometry, previous)
        diagnostics['columns'][str(column)] = entry_diagnostic
        if entry is not None:
            vision.v7_occupancy[str(column)] = entry
    return vision.v7_occupancy
