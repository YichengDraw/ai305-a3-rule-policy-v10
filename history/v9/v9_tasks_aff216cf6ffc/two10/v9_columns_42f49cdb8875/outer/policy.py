"""Route by the public source-column instruction; no seed/profile/count lookup."""
import re
RUNTIME_DEPENDENCIES = ('v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/b2_supervisor.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_hybrid_policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_parent_perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_parent_policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_parent_rule_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_source_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_source_perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/path_feasibility.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/placement_audit.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/rule_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/source_geometry.py')
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.outer.f_parent_policy import VisualForcePolicy as ParentPolicy
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.outer.f_hybrid_policy import VisualForcePolicy as HybridPolicy

class VisualForcePolicy:
    action_mode = 'cartesian_delta'

    def __init__(self, variant='b'):
        self.variant = variant
        self.active = None
        self.selected_parent = None

    def __getattr__(self, name):
        if self.active is None:
            raise AttributeError(name)
        return getattr(self.active, name)

    def reset(self, context):
        self.close()
        match = re.search('source column\\s+(\\d+)', str(context.task), re.I)
        column = int(match.group(1)) if match else 1
        self.active = (ParentPolicy if column == 1 else HybridPolicy)(self.variant)
        self.selected_parent = 'original_parent_source1' if column == 1 else 'parent_act_held_target_with_F_source'
        self.active.reset(context)

    def act(self, observation, task=''):
        return self.active.act(observation, task)

    @property
    def finished(self):
        return bool(self.active is not None and self.active.finished)

    @property
    def phase_name(self):
        return 'UNINITIALIZED' if self.active is None else self.active.phase_name

    def close(self):
        if self.active is not None:
            self.active.close()

def make_policy():
    return VisualForcePolicy('b')

def make_policy_a():
    return VisualForcePolicy('a')

def make_policy_b():
    return VisualForcePolicy('b')

def make_policy_c():
    return VisualForcePolicy('c')
