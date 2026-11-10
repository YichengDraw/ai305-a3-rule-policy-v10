"""Source-only RGB adapter with one shared source geometry packet.

The parent keeps its held/count/target algorithms. Its source-fit entry receives
exactly the F source packet before those algorithms consume group/held_anchor.
"""
import copy, types
import numpy as np
from v10_leaves_17e0363838a9.two5_outer.f_parent_perception import VisionEstimator as ParentVision
from v10_leaves_17e0363838a9.two5_outer.f_source_perception import VisionEstimator as SourceVision
from v10_leaves_17e0363838a9.two5_outer.source_geometry import SourceGeometry, calibrated_source_cameras, source_planning_scene

class VisionEstimator:

    def __init__(self, config):
        self.parent = ParentVision(config)
        self.source = SourceVision(config)
        self.geometry = SourceGeometry(config)
        self.last_source = {}
        self.source_fit = None
        original = self.parent._fit_selected_group
        adapter = self

        def source_fit(parent, observation, cameras, centers, indices, source_error, allow_tilt=True):
            packet = adapter.source_fit
            if packet is not None and list(map(int, indices)) == packet['indices']:
                return copy.deepcopy(packet['fit'])
            return original(observation, cameras, centers, indices, source_error, allow_tilt=allow_tilt)
        self.parent._fit_selected_group = types.MethodType(source_fit, self.parent)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def reset(self, context=None):
        self.parent.reset(context)
        self.source.reset(context)
        self.geometry.reset()
        self.last_source = {}
        self.source_fit = None

    def estimate(self, observation, request=None):
        request = request or {}
        stage = str(request.get('stage', '')).upper()
        source_stages = ('SELECT', 'APPROACH', 'LOCAL_ALIGN', 'XY_COARSE', 'ALIGN_APPROACH', 'ALIGN_OBSERVE', 'ALIGN_RETURN', 'SOURCE_OBSERVE', 'OBSERVE_SOURCE', 'DESCEND', 'CLOSE', 'PUSH_APPROACH', 'PUSH_DESCEND', 'PUSH_SWEEP', 'PUSH_RETRACT')
        current = None
        if stage in source_stages:
            current = self.source.estimate(observation, request)
            self.last_source = current
            group = current['group']
            indices = list(map(int, group['indices']))
            fit = copy.deepcopy(group)
            fit['centers'] = np.asarray(current['cookies']['centers'], float)[indices].copy()
            self.source_fit = dict(indices=indices, fit=fit, source_geometry_version=copy.deepcopy(group.get('geometry_version')))
        result = self.parent.estimate(observation, request)
        if current is not None:
            for key in ('source_box', 'cookies', 'group', 'local_align', 'source_presence', 'source_frontier', 'source_row_frame', 'source_tilt_deg', 'position_error_m'):
                if key in current:
                    result[key] = current[key]
            inventory = dict(result.get('object_inventory') or {})
            for key in ('source_column_frontier', 'source_row_frame'):
                if key in (current.get('object_inventory') or {}):
                    inventory[key] = current['object_inventory'][key]
            result['object_inventory'] = inventory
            result['source_camera_calibration'] = current.get('camera_calibration')
            result['source_wrist_calibration'] = current.get('wrist_calibration')
            if stage == 'SELECT' and int(request.get('source_column', 1)) == 4:
                evidence = self.geometry.estimate(observation, result, calibrated_source_cameras(self.source, observation), measurement_step=self.source.step, color_mask=self.source._v6_color_mask)
                result = source_planning_scene(result, evidence)
        else:
            for key in ('source_frontier', 'source_row_frame'):
                if key in self.last_source:
                    result[key] = self.last_source[key]
        if self.source_fit is not None:
            selected = self.parent.selected_group or {}
            fit = self.source_fit['fit']
            consistent = np.allclose(np.asarray(selected.get('center', [np.nan] * 3)), np.asarray(fit['center']), rtol=0, atol=1e-12) and np.allclose(np.asarray(selected.get('rotation', np.eye(3))), np.asarray(fit['rotation']), rtol=0, atol=1e-12)
            result['source_sync'] = dict(state='shared_source_packet' if consistent else 'parent_source_packet_changed', geometry_version=self.source_fit['source_geometry_version'], source_center=np.asarray(fit['center']).tolist(), parent_selected_center=None if not selected else np.asarray(selected['center']).tolist(), source_rotation=np.asarray(fit['rotation']).tolist(), parent_selected_rotation=None if not selected else np.asarray(selected['rotation']).tolist(), held_anchor=None if self.parent.held_anchor is None else [np.asarray(x).tolist() for x in self.parent.held_anchor], object_association='spatial source hypotheses; count does not confirm planned identity')
        return result
