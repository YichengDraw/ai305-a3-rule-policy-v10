"""Keep the selected parent's act/target/held; delegate only F source methods."""
import copy
from v10_leaves_17e0363838a9.two5_outer.f_parent_rule_controller import RuleController as ParentController
from v10_leaves_17e0363838a9.two5_outer.f_source_controller import RuleController as SourceController, FixedRobotPitchSearch

class RuleController(ParentController):

    def reset(self, context=None):
        super().reset(context)
        self._f_planner = None
        self._f_selected_pitch = None
        self._f_pinch_height = 0.01
        self._f_search_key = None
        self._f_pitch_results = None

    def prepare_fixed_robot_search(self, model, path_checker=None):
        self._f_planner = FixedRobotPitchSearch(model, self.config, path_checker)

    def _source_observation_target(self, scene):
        return SourceController._source_observation_target(self, scene)

    def _select(self, scene):
        return SourceController._select(self, scene)

    def _apply_group_geometry(self, group):
        return SourceController._apply_group_geometry(self, group)

    def _confirm_observed_grasp(self, valid_count, obs):
        return SourceController._confirm_observed_grasp(self, valid_count, obs)

    @property
    def public_evidence(self):
        result = ParentController.public_evidence.fget(self)
        result['finite_pitch_search'] = copy.deepcopy(self._f_pitch_results)
        return result
