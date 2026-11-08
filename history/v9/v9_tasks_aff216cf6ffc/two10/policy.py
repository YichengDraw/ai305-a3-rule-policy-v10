"""Select a complete policy from the public requested source column."""
import re
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.source1.policy import make_policy_b as make_source1
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.inner.policy import make_policy_b as make_inner
from v9_tasks_aff216cf6ffc.two10.v9_columns_42f49cdb8875.outer.policy import make_policy_b as make_outer

class ColumnRouter:
    action_mode = 'cartesian_delta'
    name = 'v9_public_source_column_policy'

    def __init__(self):
        self.parents = {'source1': make_source1(), 'inner': make_inner(), 'outer': make_outer()}
        self.active = None

    def reset(self, context):
        match = re.search('source column\\s+(\\d+)', str(context.task), re.I)
        self.selected_source_column = int(match.group(1)) if match else 1
        self.selected_parent = 'source1' if self.selected_source_column == 1 else 'inner' if self.selected_source_column in (2, 3) else 'outer'
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
        for policy in self.parents.values():
            policy.close()

def make_policy_b():
    return ColumnRouter()

def make_policy():
    return ColumnRouter()
RUNTIME_DEPENDENCIES = ['v9_tasks_aff216cf6ffc/two10/policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/__init__.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/inner/__init__.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/inner/perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/inner/placement_audit.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/inner/policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/inner/rule_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/__init__.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/b2_supervisor.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_hybrid_policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_parent_perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_parent_policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_parent_rule_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_source_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/f_source_perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/path_feasibility.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/placement_audit.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/rule_controller.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/outer/source_geometry.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/__init__.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/b2_supervisor.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/d_supervisor.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/path_feasibility.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/perception.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/placement_audit.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/policy.py', 'v9_tasks_aff216cf6ffc/two10/v9_columns_42f49cdb8875/source1/rule_controller.py']
