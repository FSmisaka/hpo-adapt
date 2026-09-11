"""Stage 5 tests: Revised Setup Agent (execute the 7 intervention types)."""

import pytest

from hpo_adapt.agents.revised_setup import (
    REVISED_SETUP_PROMPT,
    HeuristicRevisedSetupAgent,
    InvalidActionError,
    LLMRevisedSetupAgent,
)
from hpo_adapt.schemas import (
    ActionType,
    BoundaryConcentrationEntry,
    DiagnosticState,
    GoodRegionEntry,
    MetricStatus,
    RLAction,
)
from hpo_adapt.validator import validate_setup
from tests.test_hpo import make_setup, make_task


def diag_with_good_region(lr_mean=0.006, lr_lower=0.004, lr_upper=0.008, width=0.3):
    state = DiagnosticState()
    state.search_space.good_region = {
        "learning_rate": GoodRegionEntry(
            mean=lr_mean, lower=lr_lower, upper=lr_upper, width_ratio=width, status=MetricStatus.VALID
        )
    }
    state.search_space.boundary_concentration = {
        "learning_rate": BoundaryConcentrationEntry(
            all=0.4, good=0.9, upper=0.85, lower=0.05, status=MetricStatus.VALID
        )
    }
    return state


class TestExpand:
    def test_expand_upper_log(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.EXPAND_SEARCH_SPACE,
            target="learning_rate",
            parameters={"direction": "upper"},
        )
        result = HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)
        new_lr = result.new_setup.hyperparameter("learning_rate")
        old_lr = setup.hyperparameter("learning_rate")
        assert new_lr.high == pytest.approx(old_lr.high * 10.0)
        assert new_lr.low == old_lr.low
        assert validate_setup(result.new_setup, task) == []
        assert result.new_setup.setup_id == "setup_001"
        assert result.new_setup.phase == 1
        assert len(result.changes) == 1 and result.changes[0].hyperparameter == "learning_rate"

    def test_expand_lower_log(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.EXPAND_SEARCH_SPACE,
            target="learning_rate",
            parameters={"direction": "lower"},
        )
        result = HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)
        new_lr = result.new_setup.hyperparameter("learning_rate")
        old_lr = setup.hyperparameter("learning_rate")
        assert new_lr.low == pytest.approx(old_lr.low / 10.0)

    def test_expand_linear_param(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.EXPAND_SEARCH_SPACE,
            target="batch_size",
            parameters={"direction": "upper"},
        )
        result = HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)
        new_bs = result.new_setup.hyperparameter("batch_size")
        old_bs = setup.hyperparameter("batch_size")
        # metadata high = 512 caps the expansion
        assert new_bs.high == pytest.approx(min(old_bs.high + 0.5 * (old_bs.high - old_bs.low), 512))

    def test_expand_categorical_invalid(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.EXPAND_SEARCH_SPACE, target="optimizer_type", parameters={}
        )
        with pytest.raises(InvalidActionError):
            HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)


class TestShiftShrink:
    def test_shift_moves_center_preserving_width(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.SHIFT_SEARCH_REGION,
            target="learning_rate",
            parameters={"target_region": [0.004, 0.008]},
        )
        result = HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)
        old_lr, new_lr = setup.hyperparameter("learning_rate"), result.new_setup.hyperparameter("learning_rate")
        assert new_lr is not None and old_lr is not None
        old_log_span = old_lr.high / old_lr.low
        new_log_span = new_lr.high / new_lr.low
        # width preserved in log space
        assert new_log_span == pytest.approx(old_log_span, rel=1e-6)
        # new center (geometric mean) moved toward sqrt(0.004*0.008)
        new_center = (new_lr.low * new_lr.high) ** 0.5
        assert new_center == pytest.approx((0.004 * 0.008) ** 0.5, rel=1e-3)
        assert validate_setup(result.new_setup, task) == []

    def test_shrink_around_good_region(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.SHRINK_SEARCH_SPACE,
            target="learning_rate",
            parameters={"target_region": [0.004, 0.008]},
        )
        result = HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)
        new_lr = result.new_setup.hyperparameter("learning_rate")
        assert new_lr is not None
        # 25% margin of the good region on each side (log space)
        assert new_lr.low >= 0.004 * 0.8
        assert new_lr.high <= 0.008 * 1.2
        # strictly narrower than before
        import math

        old_lr = setup.hyperparameter("learning_rate")
        assert math.log(new_lr.high) - math.log(new_lr.low) < math.log(old_lr.high) - math.log(old_lr.low)

    def test_shift_uses_diagnostic_good_region_when_params_missing(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.SHIFT_SEARCH_REGION, target="learning_rate")
        result = HeuristicRevisedSetupAgent().revise(task, setup, diag_with_good_region(), action)
        assert result.changes, "should use diagnostic good region as target"


class TestFreeze:
    def test_freeze_uses_default_value(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.FREEZE_HYPERPARAMETER, target="batch_size")
        result = HeuristicRevisedSetupAgent().revise(task, setup, DiagnosticState(), action)
        assert result.new_setup.hyperparameter("batch_size") is None
        assert result.new_setup.frozen["batch_size"] == 64
        assert validate_setup(result.new_setup, task) == []

    def test_freeze_explicit_value(self):
        task, setup = make_task(), make_setup()
        action = RLAction(
            action=ActionType.FREEZE_HYPERPARAMETER,
            target="optimizer_type",
            parameters={"value": "sgd"},
        )
        result = HeuristicRevisedSetupAgent().revise(task, setup, DiagnosticState(), action)
        assert result.new_setup.frozen["optimizer_type"] == "sgd"

    def test_freeze_unknown_target_invalid(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.FREEZE_HYPERPARAMETER, target="nonexistent")
        with pytest.raises(InvalidActionError):
            HeuristicRevisedSetupAgent().revise(task, setup, DiagnosticState(), action)


class TestBudgetAndExploration:
    def test_increase_budget(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.INCREASE_BUDGET, parameters={"additional_trials": 10})
        result = HeuristicRevisedSetupAgent().revise(
            task, setup, DiagnosticState(), action, remaining_global_trials=25
        )
        assert result.new_setup.budget.max_trials == setup.budget.max_trials + 10
        assert validate_setup(result.new_setup, task) == []

    def test_increase_budget_limited_by_remaining(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.INCREASE_BUDGET, parameters={"additional_trials": 50})
        result = HeuristicRevisedSetupAgent().revise(
            task, setup, DiagnosticState(), action, remaining_global_trials=5
        )
        assert result.new_setup.budget.max_trials == setup.budget.max_trials + 5

    def test_increase_budget_exhausted_invalid(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.INCREASE_BUDGET)
        with pytest.raises(InvalidActionError):
            HeuristicRevisedSetupAgent().revise(
                task, setup, DiagnosticState(), action, remaining_global_trials=0
            )

    def test_increase_exploration_tpe(self):
        task, setup = make_task(), make_setup()
        setup.optimizer.config = {"n_startup_trials": 10}
        action = RLAction(action=ActionType.INCREASE_EXPLORATION, target="optimizer")
        result = HeuristicRevisedSetupAgent().revise(task, setup, DiagnosticState(), action)
        assert result.new_setup.optimizer.config["n_startup_trials"] == 20

    def test_stop_marks_terminated_and_preserves_setup(self):
        task, setup = make_task(), make_setup()
        action = RLAction(action=ActionType.STOP)
        result = HeuristicRevisedSetupAgent().revise(task, setup, DiagnosticState(), action)
        assert result.new_setup.terminated is True
        assert result.new_setup.search_space == setup.search_space


class TestLLMBackend:
    def test_parses_valid_llm_revision(self):
        task, setup = make_task(), make_setup()
        new_setup_json = setup.model_dump()
        new_setup_json.update({"setup_id": "setup_001", "phase": 1})
        reply_json = {
            "base_setup_id": "setup_000",
            "action": "EXPAND_SEARCH_SPACE",
            "changes": [
                {
                    "hyperparameter": "learning_rate",
                    "old": {"low": 1e-5, "high": 1e-2, "log": True},
                    "new": {"low": 1e-5, "high": 1e-1, "log": True},
                    "reason": "Good trials concentrate near the upper boundary.",
                }
            ],
            "new_setup": new_setup_json,
            "confidence": 0.87,
            "assumptions": [],
        }
        import json

        class FakeClient:
            def complete(self, system: str, user: str) -> str:
                assert "{{CURRENT_SETUP}}" not in system
                assert "setup_000" in system
                return f"```json\n{json.dumps(reply_json)}\n```"

        action = RLAction(
            action=ActionType.EXPAND_SEARCH_SPACE,
            target="learning_rate",
            parameters={"direction": "upper"},
        )
        result = LLMRevisedSetupAgent(FakeClient()).revise(task, setup, DiagnosticState(), action)
        assert result.new_setup.phase == 1
        assert result.confidence == 0.87
        assert REVISED_SETUP_PROMPT.startswith("You are the Revised HPO Setup Agent.")
