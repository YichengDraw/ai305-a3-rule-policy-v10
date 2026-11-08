"""Original V8 2x5 until its plan ends; then an independent RGB terminal supervisor."""
from __future__ import annotations
from v9_tasks_aff216cf6ffc.two5.v9_columns_cb7a65aa1101.source1.legacy.policy import make_policy_b as make_legacy
from v9_tasks_aff216cf6ffc.two5.v9_columns_cb7a65aa1101.source1.terminal_controller import TerminalController

class TerminalPolicy:
    action_mode = 'cartesian_delta'
    name = 'v9_2x5_selected_with_terminal3'

    def __init__(self):
        self.parent = make_legacy()
        self.terminal = None
        self.context = None

    def reset(self, context):
        self.context = context
        self.terminal = None
        return self.parent.reset(context)

    def _plan_done(self):
        c = self.parent.controller
        return bool(c is not None and c.finished and (c.batch_index >= len(c.batch_sizes)))

    def act(self, observation, task=''):
        if self.terminal is None and (not self._plan_done()):
            return self.parent.act(observation, task)
        if self.terminal is None:
            self.terminal = TerminalController(self.parent, self.context)
        return self.terminal.act(observation)

    @property
    def controller(self):
        return self.terminal if self.terminal is not None else self.parent.controller

    @property
    def vision(self):
        return self.parent.vision

    @property
    def finished(self):
        return False if self.terminal is not None or self._plan_done() else self.parent.finished

    @property
    def phase_name(self):
        if self.terminal is not None:
            return self.terminal.phase_name
        return 'COMPLETE_CANDIDATE' if self._plan_done() else self.parent.phase_name

    def close(self):
        return self.parent.close()

def make_policy():
    return TerminalPolicy()

def make_policy_b():
    return TerminalPolicy()
