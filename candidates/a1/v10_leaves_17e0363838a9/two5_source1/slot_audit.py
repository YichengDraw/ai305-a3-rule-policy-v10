"""Per-cookie top-band measurements for the two-by-five target.

Inputs are RGB, estimated calibrated cameras/target pose and fixed CAD. This
observer reports visual slot evidence; it never supplies the official score.
Pixel-derived uncertainty here is a diagnostic value awaiting offline calibration.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

def rotation_z(yaw):
    c, s = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def project(points, camera, shape):
    h, w = shape[:2]
    p = (np.asarray(points) - np.asarray(camera['pos'])) @ np.asarray(camera['rotation'])
    z = -p[:, 2]
    f = h / 2 / np.tan(np.deg2rad(camera['fovy']) / 2)
    uv = np.column_stack(((w - 1) / 2 + f * p[:, 0] / np.maximum(z, 1e-06), (h - 1) / 2 - f * p[:, 1] / np.maximum(z, 1e-06)))
    return (uv, z)

class SlotAuditor:

    def __init__(self, config, color_mask):
        self.geometry = getattr(config, 'cookie_transfer', config)
        self.slots = np.asarray(self.geometry.target_slots_local_m, float)
        self.columns = np.unique(self.slots[:, 0])
        self.half = np.asarray(self.geometry.cookie_half_size_m, float)
        self.bin_half = np.asarray(self.geometry.target_bin_half_size_m, float)
        self.wall = float(self.geometry.bin_wall_thickness_m)
        self.floor = float(self.geometry.target_floor_z_m) + self.wall / 2
        self.tolerance = np.asarray(self.geometry.target_slot_tolerance_m, float)
        self.color_mask = color_mask

    def _column_view(self, observation, camera, name, box, column):
        image = np.asarray(observation['observation.images.' + name], np.uint8)
        R = rotation_z(float(box.get('yaw', 0.0)))
        center = np.asarray(box['center'], float)
        inner = self.bin_half - self.wall
        mid = float(np.mean(self.columns))
        xlow = -inner[0] if column == 0 else mid
        xhigh = mid if column == 0 else inner[0]
        xs = np.linspace(xlow + 0.0003, xhigh - 0.0003, 96)
        ys = np.linspace(-inner[1] + 0.0005, inner[1] - 0.0005, 421)
        top = self.floor + 2 * self.half[2]
        local = np.array([[x, y, top] for y in ys for x in xs])
        world = local @ R.T + center
        uv, depth = project(world, camera, image.shape)
        valid = (depth > 0) & (uv[:, 0] > 1) & (uv[:, 0] < image.shape[1] - 2) & (uv[:, 1] > 1) & (uv[:, 1] < image.shape[0] - 2)
        if valid.mean() < 0.95:
            return dict(camera=name, state='unknown', reason='target column top plane clipped', visible_fraction=float(valid.mean()))
        patch = cv2.remap(image, uv[:, 0].astype(np.float32).reshape(len(ys), len(xs)), uv[:, 1].astype(np.float32).reshape(len(ys), len(xs)), cv2.INTER_LINEAR)
        colors = self.color_mask(patch) > 0
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        middle = abs(xs - self.columns[column]) < self.half[0] * 0.55
        color_rows = colors[:, middle].mean(1)
        value = np.mean(hsv[:, middle, 2].astype(float), axis=1)
        black = float(np.mean(hsv[:, :, 2] < 28))
        if black > 0.18:
            return dict(camera=name, state='unknown', reason='gripper or other dark occlusion', black_fraction=black)
        signal = cv2.GaussianBlur((value * np.minimum(color_rows * 2, 1)).astype(np.float32)[:, None], (1, 5), 0).ravel()
        dy = float(ys[1] - ys[0])
        distance = max(3, int(0.0045 / dy))
        peaks, _ = find_peaks(signal, distance=distance, prominence=max(3.0, float(np.std(signal)) * 0.16))
        peaks = np.asarray([i for i in peaks if color_rows[i] > 0.35], int)
        observed = []
        for i in peaks:
            offset = 0.0
            if 0 < i < len(signal) - 1:
                den = float(signal[i - 1] - 2 * signal[i] + signal[i + 1])
                if abs(den) > 1e-06:
                    offset = float(np.clip(0.5 * (signal[i - 1] - signal[i + 1]) / den, -0.5, 0.5))
            y = float(ys[i] + offset * dy)
            band = colors[max(0, i - 3):min(len(ys), i + 4)].mean(0)
            available = xs[band > 0.5]
            if len(available) < 3:
                continue
            lo, hi = np.quantile(available, [0.02, 0.98])
            width = float(hi - lo)
            if width < self.half[0] * 1.35:
                continue
            x = float((lo + hi) / 2)
            point = np.array([[x, y, top], [x, y + 0.001, top]]) @ R.T + center
            puv, _ = project(point, camera, image.shape)
            pixels_per_m = float(np.linalg.norm(puv[1] - puv[0]) / 0.001)
            observed.append(dict(center_local=[x, y, self.floor + self.half[2]], top_band_width_m=width, pixel_sigma_y_m=0.5 / max(pixels_per_m, 1.0), color_fraction=float(color_rows[i])))
        return dict(camera=name, state='measured' if observed else 'unknown', reason='independently detected RGB top bands' if observed else 'no whole-width top band', count=len(observed), objects=observed, visible_fraction=float(valid.mean()), black_fraction=black)

    def measure(self, observation, cameras, box, geometry_version=None):
        now = float(observation.get('time', 0.0))
        identifier = f'slots:{geometry_version}:{now:.6f}'
        result = dict(measurement_id=identifier, measurement_time=now, geometry_version=geometry_version, evidence_source='public RGB individual top bands and fixed CAD', uncertainty_calibrated=False, columns=[])
        if not box or box.get('center') is None:
            return dict(result, state='unknown', reason='target pose unavailable')
        for column, x in enumerate(self.columns):
            per_camera = [self._column_view(observation, camera, name, box, column) for name, camera in cameras.items() if 'observation.images.' + name in observation]
            expected = self.slots[np.isclose(self.slots[:, 0], x)]
            expected = expected[np.argsort(expected[:, 1])]
            usable = [v for v in per_camera if v.get('state') == 'measured' and v['count'] == len(expected)]
            item = dict(column=column, per_camera=per_camera, state='unknown', reason='individual row count not established')
            if usable:
                positions = np.asarray([[obj['center_local'][:2] for obj in view['objects']] for view in usable])
                centers = np.median(positions, axis=0)
                spread = np.max(abs(positions - centers), axis=(0, 1))
                sigma = max((obj['pixel_sigma_y_m'] for view in usable for obj in view['objects']))
                row = np.argsort(centers[:, 1])
                col = np.arange(len(expected))
                errors = centers[row] - expected[col]
                shift = np.median(errors, axis=0)
                internal = errors - shift
                max_error = np.max(abs(errors), axis=0)
                matched = bool(np.all(abs(errors) <= self.tolerance))
                systematic = bool(np.max(abs(internal[:, 1])) < 0.0015 and np.max(abs(internal[:, 0])) < 0.004)
                classification = 'matched' if matched else 'common_offset' if systematic else 'spacing_or_tilt'
                item.update(state='measured', reason='independently observed row bands matched one-to-one to configured slots', cameras=[v['camera'] for v in usable], count=len(centers), centers_local=centers.tolist(), assignment=col.tolist(), errors_xy_m=errors.tolist(), common_offset_xy_m=shift.tolist(), internal_residual_xy_m=internal.tolist(), max_abs_error_xy_m=max_error.tolist(), camera_spread_xy_m=spread.tolist(), pixel_sigma_y_m=float(sigma), classification=classification, visually_slot_matched=matched, cross_view_consistent=bool(np.max(spread) < 0.0025))
            result['columns'].append(item)
        result['state'] = 'measured' if all((x['state'] == 'measured' for x in result['columns'])) else 'unknown'
        result['visually_slot_matched'] = bool(result['state'] == 'measured' and all((x['visually_slot_matched'] for x in result['columns'])))
        return result
