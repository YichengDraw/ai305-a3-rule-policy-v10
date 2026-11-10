"""Original V8 2x5 until its plan ends; then an independent RGB terminal supervisor."""
from __future__ import annotations
from v10_leaves_17e0363838a9.two5_source1.legacy.policy import make_policy_b as make_legacy
from v10_leaves_17e0363838a9.two5_source1.terminal_controller import TerminalController

class TerminalPolicy:
    action_mode = 'cartesian_delta'
    name = 'v9_2x5_selected_with_terminal3'

    def __init__(self):
        self.parent = make_legacy()
        self.terminal = None
        self.context = None
        self.preaudit = None

    def reset(self, context):
        self.context = context
        self.terminal = None
        self.preaudit = None
        return self.parent.reset(context)

    def _plan_done(self):
        c = self.parent.controller
        return bool(c is not None and c.finished and (c.batch_index >= len(c.batch_sizes)))

    def act(self, observation, task=''):
        if self.terminal is None and (not self._plan_done()):
            action = self.parent.act(observation, task)
            c = self.parent.controller
            if c.batch_index == len(c.batch_sizes) - 1 and c.stage in ('RETRACT', 'VERIFY'):
                if self.preaudit is None:
                    self.preaudit = TerminalController(self.parent, self.context)
                self.preaudit.prepare_from_parent(observation)
            return action
        if self.terminal is None:
            self.terminal = self.preaudit if self.preaudit is not None else TerminalController(self.parent, self.context)
            if self.preaudit is not None:
                self.terminal.total_steps = int(self.parent.step)
                self.terminal.stage_steps = 0
                self.terminal._preaudit_handoff = True
        return self.terminal.act(observation)

    @property
    def public_evidence(self):
        return {'terminal_preaudit': self.preaudit.public_evidence if self.preaudit is not None else None}

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
