"""Task-only router: fixed V7 for 10-cookie tasks and the selected candidate for 20-cookie tasks."""
from task_route_parents_ac88e32538bd.candidate20.policy import make_policy_b as make_2x10
from task_route_parents_ac88e32538bd.v7.policy import make_policy_b as make_v7

RUNTIME_DEPENDENCIES = ['policy.py', 'task_route_parents_ac88e32538bd/__init__.py', 'task_route_parents_ac88e32538bd/candidate20/__init__.py', 'task_route_parents_ac88e32538bd/candidate20/perception.py', 'task_route_parents_ac88e32538bd/candidate20/placement_audit.py', 'task_route_parents_ac88e32538bd/candidate20/policy.py', 'task_route_parents_ac88e32538bd/candidate20/rule_controller.py', 'task_route_parents_ac88e32538bd/v7/__init__.py', 'task_route_parents_ac88e32538bd/v7/perception.py', 'task_route_parents_ac88e32538bd/v7/placement_audit.py', 'task_route_parents_ac88e32538bd/v7/policy.py', 'task_route_parents_ac88e32538bd/v7/rule_controller.py']


class TaskOnlyPolicy:
    action_mode="cartesian_delta"
    name="task_only_v7_2x5_G3_SO2_2x10"
    def __init__(self):
        self._parents={"2x5":make_v7(),"2x10":make_2x10()}
        self._active=None
        self.selected_parent=None
    def reset(self,context):
        is20="20 cookies" in context.task.lower()
        self.selected_parent="G3_SO2" if is20 else "v7"
        self._active=self._parents["2x10" if is20 else "2x5"]
        self._active.reset(context)
    def act(self,observation,task=""):
        if self._active is None:raise RuntimeError("reset before act")
        return self._active.act(observation,task)
    def __getattr__(self,name):
        active=self.__dict__.get("_active")
        if active is None:raise AttributeError(name)
        return getattr(active,name)
    @property
    def finished(self):return False if self._active is None else self._active.finished
    @property
    def phase_name(self):return "UNINITIALIZED" if self._active is None else self._active.phase_name
    def close(self):
        for policy in self._parents.values():policy.close()


def make_policy():return TaskOnlyPolicy()
def make_policy_b():return TaskOnlyPolicy()
