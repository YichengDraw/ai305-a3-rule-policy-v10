"""Unchanged public FK/IK methods extracted from candidate_f FixedRobotPitchSearch."""
import numpy as np
import mujoco
from v9_tasks_aff216cf6ffc.two5.v9_columns_cb7a65aa1101.source1.legacy.rule_controller import _mat

class PublicIK:

    def __init__(self, model, config, path_checker=None):
        from a3_dual_arm_sim.sim.actions import CartesianDeltaAdapter
        from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
        self.model = model
        self.config = config
        self.path_checker = path_checker
        self.adapter = CartesianDeltaAdapter(model, config)
        self.qids = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in (*LEFT_JOINTS, *RIGHT_JOINTS)]
        self.fingers = [[int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{side}_finger_{finger}_joint')]) for finger in ('inner', 'outer')] for side in ('L', 'R')]
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'L_eef')
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'L_2f85_base_collision')

    def public_kinematics(self, observation):
        state = np.asarray(observation['observation.state'], float)
        work = self.adapter._work
        work.qpos[self.qids] = np.r_[state[:7], state[8:15]]
        for i, opening in enumerate((state[7], state[15])):
            work.qpos[self.fingers[i]] = (1 - np.clip(opening, 0, 1)) * 0.0425
        mujoco.mj_kinematics(self.model, work)
        return (work, state[:7].copy())

    def check_ik_path(self, observation, waypoints):
        from scipy.spatial.transform import Rotation, Slerp
        work, q = self.public_kinematics(observation)
        max_position = max_angle = max_joint_delta = 0.0
        samples = 0
        if not waypoints:
            return dict(state='unknown', reason='no waypoints supplied')
        public = np.asarray(observation['observation.eef_pose'], float)[:7]
        nodes = [dict(name='public_current', position=public[:3], rotation=_mat(public[3:7]))]
        nodes.extend(({**point, 'name': point.get('name', f'waypoint_{i}'), 'position': np.asarray(point['position'], float), 'rotation': np.asarray(point['rotation'], float).reshape(3, 3)} for i, point in enumerate(waypoints)))
        for start, end in zip(nodes[:-1], nodes[1:]):
            length = float(np.linalg.norm(end['position'] - start['position']))
            angle = float(np.linalg.norm(Rotation.from_matrix(start['rotation'].T @ end['rotation']).as_rotvec()))
            steps = max(1, int(np.ceil(length / 0.012)), int(np.ceil(angle / 0.08)))
            slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.array([start['rotation'], end['rotation']])))
            for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
                position = (1 - t) * start['position'] + t * end['position']
                rotation = slerp(t).as_matrix()
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, rotation.ravel())
                previous = q.copy()
                q = self.adapter.solve(position, quat, q)
                work.qpos[list(self.adapter._qpos_ids[0])] = q
                mujoco.mj_kinematics(self.model, work)
                actual = np.empty(4)
                error = np.empty(3)
                mujoco.mju_mat2Quat(actual, work.site_xmat[self.site])
                mujoco.mju_subQuat(error, quat, actual)
                position_error = float(np.linalg.norm(work.site_xpos[self.site] - position))
                angle_error = float(np.linalg.norm(error))
                samples += 1
                max_position = max(max_position, position_error)
                max_angle = max(max_angle, angle_error)
                max_joint_delta = max(max_joint_delta, float(np.max(abs(q - previous))))
                if not np.all(np.isfinite(q)) or position_error > 0.0008 or angle_error > 0.015:
                    return dict(state='unreachable', segment=[start['name'], end['name']], position_error_m=position_error, rotation_error_rad=angle_error, samples=samples, criterion='official expert .8mm/.015rad FK residual')
        return dict(state='reachable', max_position_error_m=max_position, max_rotation_error_rad=max_angle, max_sample_joint_delta_rad=max_joint_delta, samples=samples, criterion='sampled entire path; fixed kinematics only')
