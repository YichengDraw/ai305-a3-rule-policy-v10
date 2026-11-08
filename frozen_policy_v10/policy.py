"""Select from the public task and source column; official scoring is external."""
import re
from v10_leaves_17e0363838a9.two5_source1.policy import make_policy_b as make_two5_source1
from v10_leaves_17e0363838a9.two5_inner.policy import make_policy_b as make_two5_inner
from v10_leaves_17e0363838a9.two5_outer.policy import make_policy_b as make_two5_outer
from v10_leaves_17e0363838a9.two10_source1.policy import make_policy_b as make_two10_source1
from v10_leaves_17e0363838a9.two10_inner.policy import make_policy_b as make_two10_inner
from v10_leaves_17e0363838a9.two10_outer.policy import make_policy_b as make_two10_outer

class VisualForcePolicy:
    action_mode = 'cartesian_delta'
    name = 'v10_visual_force_rules'

    def __init__(self):
        self.makers = {'two5_source1': make_two5_source1, 'two5_inner': make_two5_inner, 'two5_outer': make_two5_outer, 'two10_source1': make_two10_source1, 'two10_inner': make_two10_inner, 'two10_outer': make_two10_outer}
        self.parents = {}
        self.active = None

    def reset(self, context):
        task = 'two10' if '20 cookies' in str(context.task).lower() else 'two5'
        match = re.search(r'source column\s+(\d+)', str(context.task), re.I)
        self.selected_source_column = int(match.group(1)) if match else 1
        branch = 'source1' if self.selected_source_column == 1 else 'inner' if self.selected_source_column in (2, 3) else 'outer'
        self.selected_parent = task + '_' + branch
        if self.selected_parent not in self.parents:
            self.parents[self.selected_parent] = self.makers[self.selected_parent]()
        self.active = self.parents[self.selected_parent]
        self.active.reset(context)

    def act(self, observation, task=''):
        if self.active is None:
            raise RuntimeError('reset before act')
        return self.active.act(observation, task)

    def __getattr__(self, name):
        active = self.__dict__.get('active')
        if active is None:
            raise AttributeError(name)
        return getattr(active, name)

    @property
    def finished(self):
        return False if self.active is None else self.active.finished

    @property
    def phase_name(self):
        return 'UNINITIALIZED' if self.active is None else self.active.phase_name

    def close(self):
        for parent in self.parents.values():
            parent.close()

def make_policy_b():
    return VisualForcePolicy()

def make_policy():
    return VisualForcePolicy()

RUNTIME_DEPENDENCIES = ['policy.py', 'v10_leaves_17e0363838a9/__init__.py', 'v10_leaves_17e0363838a9/two10_inner/__init__.py', 'v10_leaves_17e0363838a9/two10_inner/perception.py', 'v10_leaves_17e0363838a9/two10_inner/placement_audit.py', 'v10_leaves_17e0363838a9/two10_inner/policy.py', 'v10_leaves_17e0363838a9/two10_inner/rule_controller.py', 'v10_leaves_17e0363838a9/two10_outer/__init__.py', 'v10_leaves_17e0363838a9/two10_outer/b2_supervisor.py', 'v10_leaves_17e0363838a9/two10_outer/f_hybrid_policy.py', 'v10_leaves_17e0363838a9/two10_outer/f_parent_perception.py', 'v10_leaves_17e0363838a9/two10_outer/f_parent_policy.py', 'v10_leaves_17e0363838a9/two10_outer/f_parent_rule_controller.py', 'v10_leaves_17e0363838a9/two10_outer/f_source_controller.py', 'v10_leaves_17e0363838a9/two10_outer/f_source_perception.py', 'v10_leaves_17e0363838a9/two10_outer/path_feasibility.py', 'v10_leaves_17e0363838a9/two10_outer/perception.py', 'v10_leaves_17e0363838a9/two10_outer/placement_audit.py', 'v10_leaves_17e0363838a9/two10_outer/policy.py', 'v10_leaves_17e0363838a9/two10_outer/rule_controller.py', 'v10_leaves_17e0363838a9/two10_outer/source_geometry.py', 'v10_leaves_17e0363838a9/two10_source1/__init__.py', 'v10_leaves_17e0363838a9/two10_source1/b2_supervisor.py', 'v10_leaves_17e0363838a9/two10_source1/d_supervisor.py', 'v10_leaves_17e0363838a9/two10_source1/path_feasibility.py', 'v10_leaves_17e0363838a9/two10_source1/perception.py', 'v10_leaves_17e0363838a9/two10_source1/placement_audit.py', 'v10_leaves_17e0363838a9/two10_source1/policy.py', 'v10_leaves_17e0363838a9/two10_source1/public_visibility.py', 'v10_leaves_17e0363838a9/two10_source1/rule_controller.py', 'v10_leaves_17e0363838a9/two10_source1/source_view.py', 'v10_leaves_17e0363838a9/two10_source1/target_occupancy.py', 'v10_leaves_17e0363838a9/two10_source1/target_registration.py', 'v10_leaves_17e0363838a9/two5_inner/__init__.py', 'v10_leaves_17e0363838a9/two5_inner/inner_source_broad.py', 'v10_leaves_17e0363838a9/two5_inner/inner_source_view.py', 'v10_leaves_17e0363838a9/two5_inner/inner_target_observation.py', 'v10_leaves_17e0363838a9/two5_inner/observation_path.py', 'v10_leaves_17e0363838a9/two5_inner/perception.py', 'v10_leaves_17e0363838a9/two5_inner/placement_audit.py', 'v10_leaves_17e0363838a9/two5_inner/policy.py', 'v10_leaves_17e0363838a9/two5_inner/public_visibility.py', 'v10_leaves_17e0363838a9/two5_inner/rule_controller.py', 'v10_leaves_17e0363838a9/two5_inner/source_bootstrap.py', 'v10_leaves_17e0363838a9/two5_inner/target_vision.py', 'v10_leaves_17e0363838a9/two5_outer/__init__.py', 'v10_leaves_17e0363838a9/two5_outer/b2_supervisor.py', 'v10_leaves_17e0363838a9/two5_outer/f_hybrid_policy.py', 'v10_leaves_17e0363838a9/two5_outer/f_parent_perception.py', 'v10_leaves_17e0363838a9/two5_outer/f_parent_policy.py', 'v10_leaves_17e0363838a9/two5_outer/f_parent_rule_controller.py', 'v10_leaves_17e0363838a9/two5_outer/f_source_controller.py', 'v10_leaves_17e0363838a9/two5_outer/f_source_perception.py', 'v10_leaves_17e0363838a9/two5_outer/path_feasibility.py', 'v10_leaves_17e0363838a9/two5_outer/perception.py', 'v10_leaves_17e0363838a9/two5_outer/placement_audit.py', 'v10_leaves_17e0363838a9/two5_outer/policy.py', 'v10_leaves_17e0363838a9/two5_outer/rule_controller.py', 'v10_leaves_17e0363838a9/two5_outer/source_geometry.py', 'v10_leaves_17e0363838a9/two5_source1/__init__.py', 'v10_leaves_17e0363838a9/two5_source1/e_source_view.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/__init__.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/b2_supervisor.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/d_supervisor.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/path_feasibility.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/perception.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/placement_audit.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/policy.py', 'v10_leaves_17e0363838a9/two5_source1/legacy/rule_controller.py', 'v10_leaves_17e0363838a9/two5_source1/path_feasibility.py', 'v10_leaves_17e0363838a9/two5_source1/policy.py', 'v10_leaves_17e0363838a9/two5_source1/public_ik.py', 'v10_leaves_17e0363838a9/two5_source1/repair_costs.py', 'v10_leaves_17e0363838a9/two5_source1/slot_audit.py', 'v10_leaves_17e0363838a9/two5_source1/source_pretrial.py', 'v10_leaves_17e0363838a9/two5_source1/target_local_projection.py', 'v10_leaves_17e0363838a9/two5_source1/target_vision.py', 'v10_leaves_17e0363838a9/two5_source1/terminal_controller.py']
