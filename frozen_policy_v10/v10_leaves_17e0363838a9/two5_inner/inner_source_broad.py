"""Public source1 outer-edge translation query for inner source columns.

Prepared helper only. It does not modify a VisionEstimator, row identities,
inventory, camera calibration, or controller state.
"""
import copy
import numpy as np
from v10_leaves_17e0363838a9.two5_inner.public_visibility import sample_visibility

def observe_inner_broad(vision, observation, cameras, group, request, outer_edge_observer):
    column = int(request.get('source_column', 1))
    packet = dict(observable=False, source_column=column, own_inner_boundary_state='unobserved', basis='source1 observed outside edge plus fixed relative source-column CAD', source1_edge=None, corrected_group_center=None)
    if column not in (2, 3):
        return {**packet, 'reason': 'inner-only prepared adapter; parent source1/4 unchanged'}
    source = vision.boxes.get('source_box') or {}
    if source.get('center') is None:
        return {**packet, 'reason': 'public source-box coordinate frame unavailable'}
    yaw = float(source.get('yaw', 0.0))
    c, s = (np.cos(yaw), np.sin(yaw))
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    column_positions = np.asarray(vision.columns, float)
    offset = float(column_positions[column - 1] - column_positions[0])
    center = np.asarray(group['center'], float)
    shadow = dict(center=center - offset * rotation[:, 0], rotation=rotation)
    query = copy.deepcopy(request)
    query['source_column'] = 1
    measured = outer_edge_observer(observation, cameras, shadow, query)
    packet.update(source1_shadow_center=shadow['center'].tolist(), relative_column_offset_m=offset, source1_edge=measured, input_group_center=center.tolist(), row_center_unchanged=True, reason=measured.get('reason'))
    if measured.get('observable') and measured.get('center_world') is not None:
        count = int(request.get('count', 0))
        row_offsets = np.arange(count) * 0.008833333333333 - (count - 1) * 0.008833333333333 / 2
        offsets = np.r_[np.arange(-12, -2), np.arange(2, 18)] * 0.0002
        boundary = np.asarray(measured['center_world']) - vision.cookie_half[0] * rotation[:, 0]
        boundary = boundary + vision.cookie_half[2] * rotation[:, 2]
        points = np.array([boundary + rotation @ np.array([x, y + d, 0.0]) for y in row_offsets for x in offsets for d in (-0.0005, 0.0, 0.0005)])
        image = np.asarray(observation['observation.images.left_wrist'])
        view = sample_visibility(vision, observation, cameras['left_wrist'], points, image.shape, held=None, request=request)
        visible = np.asarray(view['visible']).reshape(count, len(offsets), 3)
        uv = np.asarray(view['uv'])
        import cv2
        patch = cv2.remap(image, uv[:, 0].astype(np.float32).reshape(count, len(offsets) * 3), uv[:, 1].astype(np.float32).reshape(count, len(offsets) * 3), cv2.INTER_LINEAR)
        colour = (vision._v6_color_mask(patch) > 0).reshape(count, len(offsets), 3)
        supported = []
        for j in range(count):
            if float(visible[j].mean()) >= 0.95 and float(colour[j, :10].mean()) <= 0.2 and (float(colour[j, 10:].mean()) >= 0.72):
                supported.append(j)
        packet.update(source1_visibility=dict(basis='public robot CAD and same RGB outside/inside brackets; no episode truth', supported_rows=supported, row_visible_fraction=visible.mean((1, 2)).tolist(), payload_projection_known=view['payload_projection_known'], payload_basis=view['payload_basis']))
        if not view['payload_projection_known'] or len(supported) < min(count, 2):
            return {**packet, 'reason': 'source1 outside edge lacks clear public CAD/RGB brackets; retain unknown and request an earlier clear anchor view'}
        shift = float((np.asarray(measured['center_world']) - shadow['center']) @ rotation[:, 0])
        packet.update(observable=True, confidence=measured['confidence'], uncertainty_m=measured['uncertainty_m'], inferred_broad_shift_m=shift, corrected_group_center=(center + shift * rotation[:, 0]).tolist(), measurement_id=measured['measurement_id'], measurement_step=measured['measurement_step'], reason='current source1 outside colour transition supports shared broad translation; inner border remains unobserved')
    return packet
