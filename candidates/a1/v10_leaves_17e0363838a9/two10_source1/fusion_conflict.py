"""Order-invariant row-level fusion for released-target column occupancy.

W1a candidate module (parent: frozen V10 two10_source1). Pure numpy logic; no
image, calibration or physics imports so the fusion contracts run anywhere.

Semantics (first isolated version, W1.1 of the execution plan):

- Per camera, a sampled row carries `occupied evidence` when the camera sees
  it clearly (visible, not dark) and with colour support, `free evidence`
  when it sees it clearly without support; otherwise no evidence.
- Merged row state is order invariant: occupied if ANY clear camera reports
  support on the row, free when at least one clear camera saw the row and no
  clear camera reports support, unknown otherwise.
- A row is `conflicting` when different clear cameras hold occupied and free
  evidence for the same row. The total Y extent of conflicting rows is
  compared against the existing 8mm cross-view boundary criterion already
  used by the parent for endpoint disagreement. Beyond that scale the whole
  column measurement is inadmissible: it may neither create nor clear the
  stored occupancy entry (the 1110 failure mode: a stale-retained held-payload
  projection leaked foreground pixels into one view and was fused into a
  confident obstacle). Small boundary disagreements stay admissible.
- W1b write protocol on top of the W1a fusion layer, from the replayed
  21006 trace: (A) conflicted support rows may not CREATE a new occupied
  obstacle where the prior entry is empty/absent — the 440 failure wrote a
  0.68-confidence obstacle from five 1-2mm support slivers that the wrist
  camera contradicted (30 rows, 7.3mm, below the 8mm scale); refreshing an
  existing occupied entry keeps the sub-tolerance conservative behaviour.
  (B) A prior positive is refuted when every one of at least two cameras
  independently reports its bounds Y interval clear and free (>=95% covered);
  the parent's whole-column double-full-empty requirement alone left the
  false 440 obstacle uncleared for 2755 steps while both cameras repeatedly
  saw the interval free under 95% column coverage.
- Admissible entries keep the parent's exact field layout; consumers of
  `v7_occupancy` are unchanged. Conflicts and their ROIs are reported in the
  per-column diagnostics only.
"""
from __future__ import annotations
import numpy as np

CROSS_VIEW_TOLERANCE_M = 0.008


def _runs(mask, coordinates):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    return [[float(coordinates[g[0]]), float(coordinates[g[-1]])] for g in groups]


def _interval_extent(intervals):
    return float(sum(hi - lo for lo, hi in intervals))


def fuse_row_states(measurements, ys, tolerance_m=CROSS_VIEW_TOLERANCE_M):
    """Fuse per-camera row evidence without camera-order dependence.

    Each measurement dict needs `camera` (str), `clear` (bool[len(ys)]) and
    `support` (bool[len(ys)]); support outside clear rows is ignored.
    """
    n = len(ys)
    clear_any = np.zeros(n, bool)
    occupied = np.zeros(n, bool)
    conflict = np.zeros(n, bool)
    for item in measurements:
        clear = np.asarray(item['clear'], bool)
        support = np.asarray(item['support'], bool) & clear
        clear_any |= clear
        occupied |= support
    for i, a in enumerate(measurements):
        clear_a = np.asarray(a['clear'], bool)
        occ_a = np.asarray(a['support'], bool) & clear_a
        for b in measurements[i + 1:]:
            clear_b = np.asarray(b['clear'], bool)
            occ_b = np.asarray(b['support'], bool) & clear_b
            shared = clear_a & clear_b
            conflict |= shared & (occ_a ^ occ_b)
    conflict_intervals = _runs(conflict, ys)
    conflict_extent_m = _interval_extent(conflict_intervals)
    free = clear_any & ~occupied
    admissible = len(measurements) < 2 or conflict_extent_m <= float(tolerance_m)
    return dict(covered=clear_any, support=occupied, free=free, conflict=conflict, conflict_rows=int(conflict.sum()), conflict_intervals_y_m=conflict_intervals, conflict_extent_m=conflict_extent_m, admissible=bool(admissible), tolerance_m=float(tolerance_m), cameras=[m['camera'] for m in measurements], row_count=n)


def roi_independently_refuted(measurements, ys, previous, min_coverage=0.95):
    """W1b rule B: a prior positive is refuted when every one of at least two
    cameras independently reports the prior bounds Y interval as clear and
    free, covering at least `min_coverage` of its rows, with no support rows
    inside it. Whole-column double-full-empty is no longer the only way to
    clear a stale obstacle; per-ROI independent free evidence is enough."""
    bounds = (previous or {}).get('bounds')
    if not bounds:
        return False
    lo, hi = (float(bounds[1][0]), float(bounds[1][1]))
    inside = (np.asarray(ys, float) >= lo) & (np.asarray(ys, float) <= hi)
    total = int(inside.sum())
    if total <= 0 or len(measurements) < 2:
        return False
    qualifying = 0
    for m in measurements:
        clear = np.asarray(m['clear'], bool)
        occupied = np.asarray(m['support'], bool) & clear
        if occupied[inside].any():
            continue
        free_rows = int((clear & ~occupied & inside).sum())
        if free_rows >= min_coverage * total:
            qualifying += 1
    return qualifying >= 2


def decide_column_entry(measurements, ys, geometry, previous):
    """Apply the parent's write rules on top of order-invariant fusion.

    `geometry` carries column, x, cookie_half_x, floor, top, now and
    wrist_target_complete. `previous` is the stored column entry or None.
    Returns (entry_or_None, diagnostic). entry_or_None is None whenever the
    parent would not have written (plus the new inadmissible-conflict case).
    The entry layout is identical to the parent's.
    """
    fusion = fuse_row_states(measurements, ys)
    covered = fusion['covered']
    support = fusion['support']
    column = int(geometry['column'])
    now = int(geometry['now'])
    order = sorted(measurements, key=lambda m: (0 if m['camera'] == 'left_wrist' and geometry.get('wrist_target_complete') else 1, str(m['camera'])))
    public_views = [{k: v for k, v in item.items() if k not in ('clear', 'support')} for item in order]
    diagnostic = dict(state='measured' if covered.all() else 'partially_observed', cameras=public_views, covered_row_fraction=float(covered.mean()), unknown_y_intervals=_runs(~covered, ys), visible_occupied_y_intervals=_runs(support & covered, ys), visible_free_y_intervals=_runs(fusion['free'] & covered, ys), coverage_complete=bool(covered.all()), row_fusion=dict(conflict_rows=fusion['conflict_rows'], conflict_extent_m=fusion['conflict_extent_m'], conflict_intervals_y_m=fusion['conflict_intervals_y_m'], admissible=fusion['admissible'], cameras=fusion['cameras'], order_invariant=True, merged_rule='occupied if any clear camera reports support; conflict rows resolved conservatively as occupied'))
    if not measurements:
        return (None, dict(state='unknown', reason='no sampled camera'))
    if len(measurements) > 1 and not fusion['admissible']:
        diagnostic.update(state='conflicting_views', reason=f"overlapping-view occupied/free disagreement spans {fusion['conflict_extent_m']:.4f}m of rows, above the existing {fusion['tolerance_m']:.4f}m cross-view scale", conflict_roi=fusion['conflict_intervals_y_m'], conflict_cameras=fusion['cameras'], per_camera_measurement_ids=[m.get('measurement_id') for m in measurements], needs_observation='separate held_foreground from released_in_column for the conflict ROI before this column state changes', retained_prior=None if previous is None else previous.get('measurement_id'), inadmissible_for_entry_update=True)
        return (None, diagnostic)
    if not covered.all():
        return (None, diagnostic)
    fully_measured = [m for m in measurements if np.asarray(m['clear'], bool).all() and m['endpoints'] is not None]
    if len(fully_measured) > 1:
        difference = float(np.max(np.abs(np.array(fully_measured[0]['endpoints']) - fully_measured[1]['endpoints'])))
        diagnostic['independent_endpoint_disagreement_m'] = difference
        if difference > CROSS_VIEW_TOLERANCE_M:
            diagnostic.update(state='conflicting_views', reason='existing 8mm cross-view boundary criterion')
            return (None, diagnostic)
    ids = np.flatnonzero(support)
    empty_cameras = [m for m in measurements if np.asarray(m['clear'], bool).all() and m['endpoints'] is None]
    refuted_prior = False
    creating_occupied = bool(len(ids)) and (previous is None or previous.get('bounds') is None)
    if creating_occupied and fusion['conflict_rows'] > 0:
        diagnostic.update(state='creation_requires_conflict_free_support', reason=f"conflicted sliver support ({fusion['conflict_rows']} rows, {fusion['conflict_extent_m']:.4f}m) may not create a new occupied obstacle while another view contradicts the support rows", conflict_roi=fusion['conflict_intervals_y_m'], conflict_cameras=fusion['cameras'], per_camera_measurement_ids=[m.get('measurement_id') for m in measurements], needs_observation='re-observe after separating held_foreground from released_in_column; no obstacle is created from contradictory support', retained_prior=None if previous is None else previous.get('measurement_id'))
        return (None, diagnostic)
    if not len(ids) and previous is not None and (previous.get('bounds') is not None):
        if roi_independently_refuted(measurements, ys, previous):
            refuted_prior = True
            diagnostic.update(state='prior_refuted_by_independent_roi_free', reason='every measured camera independently reports the prior obstacle interval clear and free; whole-column double-full-empty no longer required', refuted_prior=previous.get('measurement_id'), per_camera_measurement_ids=[m.get('measurement_id') for m in measurements])
        elif len(empty_cameras) < 2:
            diagnostic.update(state='empty_requires_independent_confirmation')
            return (None, diagnostic)
    floor = float(geometry['floor'])
    top = float(geometry['top'])
    x = float(geometry['x'])
    cookie_half_x = float(geometry['cookie_half_x'])
    dy = float(ys[1] - ys[0])
    bounds = None if not len(ids) else np.array([[x - cookie_half_x, x + cookie_half_x], [float(ys[ids[0]] - dy), float(ys[ids[-1]] + dy)], [floor, top]])
    preferred = order[0]
    full_candidates = [m for m in order if np.asarray(m['clear'], bool).all()]
    count = full_candidates[0]['count'] if full_candidates else None
    if len(full_candidates) > 1 and any((abs(m['count'] - count) > 1 for m in full_candidates)):
        count = None
    entry = dict(camera=preferred['camera'], cameras=[m['camera'] for m in order], bounds=None if bounds is None else bounds.tolist(), count=count, confidence=0.65 if bounds is None else 0.68, uncertainty_m=0.002 if preferred['camera'] == 'left_wrist' else 0.0035, uncertainty_calibrated=False, scope='column', column=column, measurement_step=now, measurement_id=f'occupancy:{column}:{now}', excludes_held=True, occluded=False, coverage_complete=True, evidence_source='released RGB top bands; public CAD/payload occlusion; complete row coverage', box_center=list(geometry['box_center']), box_yaw=float(geometry['box_yaw']), target_geometry_version=geometry.get('target_geometry_version'), fresh_visible_while_carrying=True, per_camera=public_views)
    diagnostic['state'] = 'prior_refuted_by_independent_roi_free' if refuted_prior else 'measured'
    return (entry, diagnostic)
