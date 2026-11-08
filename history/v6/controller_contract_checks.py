"""Pure-data V6 controller checks; no environment, rendering, or policy scoring."""
from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.paths import resource_root
from rule_controller import CANONICAL, RuleController, gap_from_opening, opening_from_gap


def public_held_samples():
    """RGB/public trace replay: dev_b_smoke_r3 advanced12001, fits 96/108.

    These are the estimator's tool-relative observations, without object GT.
    The second observation was recorded during PLACE after contact.
    """
    common = {"count": 5, "count_confidence": .7, "width_m": .03166666666666666,
              "geometry_generation": 0}
    return ({**common, "center_tool": [5.7463164398648796e-5, -.0024296767237872276, -.0030796827249272634],
             "rotation_tool": [[.00042521251923131816, -.9999599331018566, -.008941553849514627],
                               [.02941812320196695, .008950193148209273, -.9995271222282485],
                               [.9995671029108485, .00016196771292068579, .029420750243197135]],
             "pose_confidence": .8198585943, "slip_m": .0004216399, "measurement_id": 96},
            {**common, "center_tool": [.0002182866832456869, -.005746123340033424, -.0023064858113273174],
             "rotation_tool": [[-.004916994215969715, -.9999604826290887, -.007406507144369346],
                               [.3402594252064013, .005291617944228598, -.9403167138457877],
                               [.9403187474075417, -.007143665706889263, .34021996019366446]],
             "pose_confidence": .8715132873, "slip_m": .0033107481333944874, "measurement_id": 108})


def rotation_distance(a, b):
    return float(np.arccos(np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.) / 2., -1., 1.)))


def setup(variant="c"):
    config = load_config(resource_root() / "configs/cookie_batch.yaml")
    c = RuleController(config)
    c.reset(SimpleNamespace(task="transfer 10 cookies into target box", seed=12000))
    c.set_variant(variant)
    c.set_budget(500)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, CANONICAL.ravel())
    state = np.zeros(16)
    state[7] = state[15] = 1.
    obs = {"observation.eef_pose": np.r_[.1, .25, .82, q, .1, -.25, .92, q],
           "observation.state": state, "observation.velocity": np.zeros(16),
           "observation.force": np.zeros(18), "time": 0.}
    layout = np.asarray(config.cookie_transfer.cookie_source_positions_m)
    center = np.r_[layout[:5].mean(axis=0), config.cookie_transfer.cookie_model_z_m]
    scene = {"group": {"indices": list(range(5)), "center": center, "rotation": np.eye(3),
                       "confidence": .8, "width_m": .0416666667, "tilt_deg": 0.,
                       "position_error_m": .001},
             "target_box": {"center": config.cookie_transfer.target_bin_world_position_m, "yaw": 0., "confidence": .8},
             "held": {}, "placement": {"state": "unknown", "confidence": 0.}}
    c._obs, c._scene = obs, scene
    assert c._select(scene)
    return c, obs, scene


def run():
    checks = []
    for n in (1, 2, 5, 9, 10):
        width = n * (.019 / 3)
        u = opening_from_gap(width + .006)
        assert abs(gap_from_opening(u) - width - .006) < 1e-12
    checks.append("opening inverse includes the 5.3616 mm zero gap")

    c, obs, scene = setup()
    c._put_position = obs["observation.eef_pose"][:3].copy()
    c._put_rotation = CANONICAL.copy()
    c._release_goal = opening_from_gap(5 * .019 / 3 + .006)
    c.opening = c._release_goal
    obs["observation.state"][7] = .309
    c._advance("OPEN")
    for _ in range(6):
        c.act(obs, scene)
    assert c.stage == "OPEN" and c.current_report["actual_opening_reached"] is False
    obs["observation.state"][7] = c._release_goal
    for _ in range(3):
        c.act(obs, scene)
    assert c.stage == "RETRACT"
    assert c.current_report["release_state"] == "unknown"
    checks.append("commanded release alone cannot satisfy actual opening or RGB release")

    c, obs, scene = setup()
    obs["observation.state"][7] = .31
    c.actual_opening = .31
    c._source_path = [(np.array([.1, .25, .81]), CANONICAL.copy()),
                      (np.array([.1, .25, .79]), CANONICAL.copy())]
    c._safe_source_high = c._source_path[0]
    c.opening = .40
    c._start_source_recovery("test insertion obstruction")
    assert abs(c.opening - .31) < 1e-12 and c.stage == "RECOVER_PATH"
    assert np.allclose(c._recovery_waypoints[0][0], [.1, .25, .79])
    assert c.recovery_events[-1]["success"] is None
    checks.append("jam retreat preserves measured opening and reverses reached points")

    c, obs, scene = setup()
    obs["observation.eef_pose"][:3] = [.067, .1, .772]
    c.actual_opening = .31
    c._last_safe_pose = (np.array([.1, .25, .764]), CANONICAL.copy())
    c._carry_path = [(np.array(p), CANONICAL.copy()) for p in (
        [.1, .25, .805], [.085, .17, .805], [.067, .1, .805], [.067, .1, .790], [.067, .1, .772])]
    c._carry_source_high_index = 0
    c._carry_source_high_pose = c._carry_path[0]
    c._start_source_recovery("test real placement slip", carrying=True)
    assert c.stage == "RECOVER_PATH"
    points = [point for point, _ in c._recovery_waypoints]
    assert np.allclose(points[0], [.067, .1, .772])
    assert np.allclose(points[-1], [.1, .25, .805])
    assert all(point[2] >= .8 for point in points if point[1] > .13)
    assert not any(np.allclose(point, [.1, .25, .764]) for point in points)
    checks.append("carrying return reverses PLACE to transit height before crossing boxes")

    c, obs, scene = setup("b")
    obs["observation.eef_pose"][:3] = c._preinsert
    c._advance("LOCAL_ALIGN")
    scene["local_align"] = {"state": "correct", "valid": False, "confidence": .8,
                            "correction_world": [0., .001, 0.], "relative_error_m": .001}
    c._local_align(obs, scene)
    corrected = c._preinsert.copy()
    c._local_align(obs, scene)
    assert np.allclose(c._preinsert, corrected)
    assert c.stage == "LOCAL_ALIGN" and c.current_report["alignment_verified"] is not True
    obs["observation.eef_pose"][:3] = corrected
    scene["local_align"].update(state="aligned", valid=True, correction_world=[0., 0., 0.])
    c._local_align(obs, scene)
    assert c.stage == "DESCEND" and c.current_report["alignment_verified"] is True
    checks.append("local correction is anchored to measured pose and valid is explicit")

    c, obs, scene = setup("b")
    original_group = np.asarray(scene["group"]["center"]).copy()
    for name in ("_pick", "_preinsert", "_high", "_trial", "_selected_center"):
        getattr(c, name)[0] += .019
    obs["observation.eef_pose"][:3] = c._preinsert
    old_preinsert = c._preinsert.copy()
    assert c._near_group_update(obs, scene)
    assert abs(c._preinsert[0] - old_preinsert[0] + .019) < 1e-12
    coarse_x = c._preinsert[0]
    c._advance("LOCAL_ALIGN")
    scene["local_align"] = {"state": "correct", "valid": False, "confidence": .8,
                            "correction_world": [0., .001, 0.], "observable_axis_world": [0., 1., 0.],
                            "measurement_id": 10}
    action = c._local_align(obs, scene)
    assert abs(c._preinsert[0] - coarse_x) < 1e-12 and action[0] < 0.
    assert np.allclose(scene["group"]["center"], original_group)
    checks.append("19 mm reliable near-view relocation survives Y fine servo without mutating perception")

    for visual_state in ("unknown", "correct"):
        c, obs, scene = setup("b")
        obs["observation.eef_pose"][:3] = c._preinsert - [.0011667, 0., 0.]
        scene["group"]["confidence"] = .18
        scene["local_align"] = {"state": visual_state, "valid": False,
                                "confidence": .8 if visual_state == "correct" else 0.,
                                "correction_world": [0., 0., 0.], "measurement_id": 1}
        c._advance("LOCAL_ALIGN")
        for _ in range(25):
            c.act(obs, scene)
            if c.stage != "LOCAL_ALIGN":
                break
        assert c.stage == "ALIGN_OBSERVE" and not c.finished
        assert c.current_report["alignment_verified"] is not True
    checks.append("stationary 1.167 mm preinsert residual triggers a new view without declaring alignment")

    c, obs, scene = setup("b")
    measured_center = np.array([0., -.004, .001])
    scene["held"] = {"count": 5, "count_confidence": .8, "pose_confidence": .8,
                     "center_tool": measured_center, "rotation_tool": CANONICAL.T,
                     "width_m": 5 * .019 / 3}
    assert c._held_update(scene) is True
    assert np.allclose(c._held_center_local, measured_center)
    assert not c.ledger["taken_indices"] and not c.ledger["placed_indices"]
    obs["observation.force"][12:14] = 1.3
    c.actual_opening = opening_from_gap(5 * .019 / 3)
    c._grasp_reference_eef = obs["observation.eef_pose"][:7].copy()
    c._grasp_reference_eef[2] -= .012
    assert c._confirm_observed_grasp(True, obs)
    assert c.ledger["taken_indices"] == list(range(5)) and not c.ledger["placed_indices"]
    position, rotation = c._place_pose(scene)
    object_center = position + rotation @ measured_center
    expected = np.asarray(scene["target_box"]["center"]) + c._expected_local_center
    assert np.allclose(object_center, expected)
    checks.append("post-grasp tool-relative pose determines placement; taken and placed differ")

    c, obs, scene = setup("c")
    c._event("insertion_retreat", "test source recovery")
    c._event("target_regrasp_reseat", "test target repair")
    c._event("target_reobserve", "test target observation")
    obs["observation.force"][12:14] = 1.3
    c.actual_opening = opening_from_gap(5 * .019 / 3)
    c._grasp_reference_eef = obs["observation.eef_pose"][:7].copy()
    c._grasp_reference_eef[2] -= .012
    assert c._confirm_observed_grasp(True, obs)
    assert c.recovery_events[0]["success"] is True
    assert all(event["success"] is None for event in c.recovery_events[1:])
    c._commit({"state": "valid", "count": 5}, True)
    assert all(event["success"] is True for event in c.recovery_events)
    checks.append("observed trial grasp resolves source events; target events await verified placement")

    c, obs, scene = setup("b")
    c._advance("TRANSPORT")
    first, _ = c._place_pose(scene)
    scene["target_box"] = {**scene["target_box"], "center": np.asarray(scene["target_box"]["center"]) + [.007, 0., 0.], "confidence": .85, "visible_rim_fraction": .9}
    corrected, _ = c._place_pose(scene)
    assert abs(corrected[0] - first[0] - .007) < 1e-12
    c._place_box_frozen = True
    c._advance("PLACE")
    scene["target_box"] = {**scene["target_box"], "center": np.asarray(scene["target_box"]["center"]) + [.007, 0., 0.], "confidence": .3}
    second, _ = c._place_pose(scene)
    assert np.allclose(corrected, second)
    checks.append("clear transit view updates box pose; occluded PLACE refit cannot move the goal")

    c, obs, scene = setup("b")
    before, after = public_held_samples()
    scene["held"] = before
    obs["observation.force"][12:14] = 1.3
    obs["observation.state"][7] = opening_from_gap(before["width_m"])
    c._held_update(scene)
    c._advance("TRANSPORT")
    high_position, high_rotation = c._place_pose(scene, c._clearance)
    obs["observation.eef_pose"][:3] = high_position
    mujoco.mju_mat2Quat(obs["observation.eef_pose"][3:7], high_rotation.ravel())
    assert c._placement_goal is None and c._place_box_frozen is False
    c.act(obs, scene)
    assert c.stage == "PLACE" and c._place_box_frozen is True
    locked_position, locked_rotation = (value.copy() for value in c._placement_goal)
    assert np.allclose(c.current_report["placement_goal_position"], locked_position)
    assert np.allclose(c.current_report["placement_goal_rotation"], locked_rotation)
    observed_goals = []
    servo = c._servo

    def record_goal(observation, target, rotation, *args, **kwargs):
        observed_goals.append((np.asarray(target).copy(), np.asarray(rotation).copy()))
        return servo(observation, target, rotation, *args, **kwargs)

    c._servo = record_goal
    scene["held"] = after
    c.act(obs, scene)
    assert c.stage == "PLACE" and len(observed_goals) == 1
    assert np.array_equal(observed_goals[0][0], locked_position)
    assert np.array_equal(observed_goals[0][1], locked_rotation)
    assert np.array_equal(c._placement_goal[0], locked_position)
    assert np.array_equal(c._placement_goal[1], locked_rotation)
    assert np.allclose(c._held_center_local, after["center_tool"])
    assert np.allclose(c._held_cookie_local, after["rotation_tool"])
    # The public fit still updates monitoring. Recomputing a live PLACE goal
    # from that same fit would move and rotate the commanded placement.
    dynamic_position, dynamic_rotation = c._place_pose(scene)
    public_replay = {"source": "dev_b_smoke_r3/2x5_advanced_12001_n5/policy_trace/seed_12001.jsonl",
                     "held_measurements": [96, 108],
                     "dynamic_goal_translation_mm": float(np.linalg.norm(dynamic_position - locked_position) * 1000.),
                     "dynamic_goal_rotation_deg": float(np.rad2deg(rotation_distance(locked_rotation, dynamic_rotation))),
                     "cached_goal_translation_mm": float(np.linalg.norm(observed_goals[0][0] - locked_position) * 1000.),
                     "cached_goal_rotation_deg": float(np.rad2deg(rotation_distance(locked_rotation, observed_goals[0][1])))}
    assert public_replay["dynamic_goal_translation_mm"] > 1.
    assert public_replay["dynamic_goal_rotation_deg"] > 10.
    assert public_replay["cached_goal_translation_mm"] == 0.
    assert public_replay["cached_goal_rotation_deg"] < 1e-4
    checks.append("reached TRANSPORT locks full PLACE goal while fresh held geometry continues monitoring")

    for restart in ("select", "repair", "reset"):
        c, obs, scene = setup("c")
        c._placement_goal = (np.array([.1, .1, .77]), CANONICAL.copy())
        c._place_box_frozen = True
        if restart == "select":
            assert c._select(scene)
        elif restart == "repair":
            c._start_target_repair({"repairable": True, "center": [.067, .1, .7715],
                                    "rotation": np.eye(3), "reason": "contract repair"})
            assert c.stage == "REPAIR_APPROACH"
        else:
            c.reset(SimpleNamespace(task="transfer 10 cookies into target box", seed=12000))
        assert c._placement_goal is None and c._place_box_frozen is False
    checks.append("new selection, target repair and episode reset clear the complete placement goal")

    c, obs, scene = setup("b")
    scene["held"] = {"count": 5, "count_confidence": .8, "pose_confidence": .8,
                     "center_tool": [0., -.004, 0.], "rotation_tool": CANONICAL.T,
                     "width_m": 5 * .019 / 3, "slip_m": .005, "measurement_id": 1,
                     "geometry_generation": 0}
    c._carry_path = [(np.array([.1, .25, .82]), CANONICAL.copy())]
    c._carry_source_high_index = 0
    c._carry_source_high_pose = c._carry_path[0]
    c._advance("TRANSPORT")
    c.act(obs, scene)
    assert c._slip_exceedances == 1 and c.stage == "TRANSPORT"
    c.act(obs, scene)
    assert c._slip_exceedances == 1 and c.stage == "TRANSPORT"
    scene["held"]["measurement_id"] = 2
    scene["held"]["geometry_generation"] = 1
    c.act(obs, scene)
    assert c._held_geometry_generation == 1 and c._slip_exceedances == 1 and c.stage == "TRANSPORT"
    c.act(obs, scene)
    assert c._slip_exceedances == 1 and c.stage == "TRANSPORT"
    scene["held"]["measurement_id"] = 3
    c.act(obs, scene)
    assert c.stage == "RECOVER_PATH"
    checks.append("slip requires two fresh observations in the same geometry generation")

    for verdict in ("unknown", "invalid", "valid"):
        c, obs, scene = setup("c")
        c._release_goal = .4
        c.opening = .4
        obs["observation.state"][7] = .4
        scene["held"] = {"released": True, "following": False, "pose_confidence": .8}
        scene["placement"] = {"state": verdict, "confidence": .8, "count": 5,
                              "count_confidence": .8, "count_scope": "batch",
                              "center": [.067, .1, .7715], "rotation": np.eye(3),
                              "repairable": True, "reason": "test audit"}
        c._advance("VERIFY")
        for _ in range(6):
            if c.stage != "VERIFY":
                break
            c.act(obs, scene)
        if verdict == "unknown":
            assert c.stage == "TARGET_OBSERVE" and c.batch_index == 0
            assert not c.ledger["placed_indices"]
        elif verdict == "invalid":
            assert c.stage == "REPAIR_APPROACH" and c.batch_index == 0
            assert not c.ledger["placed_indices"] and c.recovery_events[-1]["success"] is None
        else:
            assert c.batch_index == 1 and c.batch_ledger[0]["state"] == "committed"
            assert c.completed_indices == list(range(5))
    checks.append("placement valid commits; invalid repairs; unknown reobserves")

    c, obs, scene = setup("a")
    c._release_goal = .4
    obs["observation.state"][7] = .4
    c._advance("VERIFY")
    for _ in range(6):
        c.act(obs, scene)
    assert c.batch_index == 1 and c.batch_ledger[0]["state"] == "placed_unverified"
    assert not c.completed_indices
    checks.append("a/b ablation may continue execution without claiming verified commitment")

    for stage in ("RECOVER_PATH", "TARGET_RETREAT"):
        c, obs, scene = setup("c")
        obs["observation.eef_pose"][:3] = [.1, .25, .764]
        obs["observation.state"][7] = .31
        c.opening = c.actual_opening = .31
        points = [(np.array([.1, .25, .7641 + index * .003]), CANONICAL.copy())
                  for index in range(16)]
        if stage == "RECOVER_PATH":
            c._recovery_waypoints = points
            c._carrying_recovery = False
            expected_stage = "RECOVER_CLEAR"
        else:
            c._target_exit_path = points
            c._target_clear_pose = points[-1]
            c._pending_target_observation = "test reached path"
            expected_stage = "TARGET_OBSERVE"
        c._advance(stage)
        for steps in range(250):
            action = c.act(obs, scene)
            assert not c.finished, (stage, steps, c.failure_reason)
            if c.stage == expected_stage:
                break
            assert c.stage == stage
            # Delayed but healthy public pose feedback. The first waypoint's
            # error is 0.1 mm; later goals begin several millimetres away.
            obs["observation.eef_pose"][:3] += action[:3] * c.translation_scale * .5
        assert c.stage == expected_stage and steps > 35
        assert len(c.stage_reports[-1]["reached_segments"]) == 16
    checks.append("multi-waypoint source/target retreat refreshes progress despite a near-zero first goal")
    print("PASS: " + str(len(checks)) + " groups; " + "; ".join(checks))
    print("PUBLIC_PLACE_REPLAY " + json.dumps(public_replay, sort_keys=True))


if __name__ == "__main__":
    run()
