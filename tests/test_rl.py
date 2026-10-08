"""Stage 7 tests: RL Controller — tabular Q-learning, reward, state keys."""

import math

import pytest

from hpo_adapt.rl import (
    ALL_ACTIONS,
    QController,
    QLearningConfig,
    RewardConfig,
    build_state,
    compute_reward,
    state_key,
)
from hpo_adapt.rules import RuleEngine
from hpo_adapt.schemas import (
    ActionType,
    BoundaryConcentrationEntry,
    ConfigurationDiversity,
    DiagnosticState,
    GoodRegionEntry,
    ImportanceEntry,
    Metric,
    MetricStatus,
    OptimizationStatus,
    Setup,
)
from hpo_adapt.schemas.diagnostic import BoolMetric, BudgetStatus
from tests.test_hpo import make_setup


def truncation_state() -> DiagnosticState:
    """Classic search-space-truncation diagnostic pattern."""
    state = DiagnosticState(optimization=OptimizationStatus(stagnation=BoolMetric.valid(True)))
    state.optimization.best_objective = Metric.valid(0.5)
    state.search_space.importance = {
        "learning_rate": ImportanceEntry(
            global_importance=0.78, local_importance=0.7, rank=1, status=MetricStatus.VALID
        ),
        "dropout": ImportanceEntry(
            global_importance=0.03, local_importance=0.05, rank=2, status=MetricStatus.VALID
        ),
    }
    state.search_space.boundary_concentration = {
        "learning_rate": BoundaryConcentrationEntry(
            all=0.43, good=0.91, upper=0.86, lower=0.05, status=MetricStatus.VALID
        )
    }
    state.search_space.good_region = {
        "learning_rate": GoodRegionEntry(
            mean=0.008, lower=0.006, upper=0.010, width_ratio=0.20, status=MetricStatus.VALID
        )
    }
    state.search_space.configuration_diversity = ConfigurationDiversity(
        recent=0.2, global_=0.24, ratio=0.83, status=MetricStatus.VALID
    )
    state.budget = BudgetStatus(
        total_trials=100,
        completed_trials=50,
        budget_consumed=Metric.valid(0.5),
        budget_remaining=Metric.valid(0.5),
    )
    return state


class TestReward:
    def test_reward_formula(self):
        r = compute_reward(0.5, 0.55, 20, 100, RewardConfig(lambda_cost=0.1))
        expected = (0.55 - 0.5) / (0.5 + 1e-8) - 0.1 * (20 / 100)
        assert r == pytest.approx(expected, abs=1e-9)

    def test_reward_clip(self):
        r = compute_reward(1.0, 100.0, 50, 100)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_invalid_action_penalty(self):
        r = compute_reward(0.5, 0.5, 0, 100, invalid=True)
        assert r == pytest.approx(-0.2, abs=1e-9)

    def test_negative_reward_clipped(self):
        r = compute_reward(1.0, 0.0, 100, 100)
        assert r == pytest.approx(-1.0, abs=1e-9)


class TestStateKeys:
    def test_state_key_structure(self):
        engine = RuleEngine()
        structured = build_state(truncation_state(), make_setup(), engine)
        key = state_key(structured)
        assert len(key) == 6 + 2 * 3  # 6 global + top-2 candidates x 3 levels
        assert structured["global"]["stagnation"] is True
        assert structured["candidates"][0][0] == "high"  # importance level

    def test_insufficient_metrics_map_to_none(self):
        engine = RuleEngine()
        structured = build_state(DiagnosticState(), make_setup(), engine)
        assert all(v == "None" for v in list(state_key(structured))[:6])


class TestQController:
    def test_select_action_returns_schema_valid_action(self):
        controller = QController(rule_engine=RuleEngine(), config=QLearningConfig(seed=0))
        state = truncation_state()
        action = controller.select_action(state, make_setup())
        assert action.action in {a for a in ALL_ACTIONS}
        assert action.confidence is not None and 0.0 <= action.confidence <= 1.0
        assert action.q_value is not None

    def test_explore_then_exploit_with_rule_tiebreak(self):
        controller = QController(
            rule_engine=RuleEngine(),
            config=QLearningConfig(epsilon_start=0.0, seed=0),  # pure exploitation
        )
        state = truncation_state()
        setup = make_setup()
        engine_rec = RuleEngine().recommend(state, setup)
        action = controller.select_action(state, setup, rule_recommendation=engine_rec)
        # all Q = 0 -> tie broken by rule recommendation
        assert action.action == ActionType.EXPAND_SEARCH_SPACE
        assert action.target == "learning_rate"
        assert action.parameters["direction"] == "upper"

    def test_q_update_formula(self):
        controller = QController(config=QLearningConfig(learning_rate=0.1, discount_factor=0.9))
        s1 = tuple(["True"] * 6) + (("high", "high", "narrow"),)
        s2 = tuple(["False"] * 6) + (("low", "low", "wide"),)
        controller._q_row(s2)[ActionType.STOP.value] = 0.8
        new_q = controller.update(s1, ActionType.EXPAND_SEARCH_SPACE, 0.5, s2)
        # 0 + 0.1 * (0.5 + 0.9 * 0.8 - 0)
        assert new_q == pytest.approx(0.1 * (0.5 + 0.9 * 0.8), abs=1e-9)

    def test_terminal_update(self):
        controller = QController(config=QLearningConfig(learning_rate=0.5))
        s1 = tuple(["True"] * 6)
        new_q = controller.update(s1, ActionType.STOP, -0.2, None, terminal=True)
        assert new_q == pytest.approx(0.5 * (-0.2), abs=1e-9)

    def test_epsilon_decay(self):
        controller = QController(config=QLearningConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995))
        state = truncation_state()
        for _ in range(10):
            controller.select_action(state, make_setup())
        assert controller.epsilon == pytest.approx(0.995**10, abs=1e-12)

    def test_no_candidates_restricts_hyper_actions(self):
        controller = QController(config=QLearningConfig(epsilon_start=0.0, seed=0))
        action = controller.select_action(DiagnosticState(), make_setup())
        assert action.action not in {
            ActionType.EXPAND_SEARCH_SPACE,
            ActionType.SHIFT_SEARCH_REGION,
            ActionType.SHRINK_SEARCH_SPACE,
            ActionType.FREEZE_HYPERPARAMETER,
        }

    def test_stop_gated_by_stagnation_evidence(self):
        # STOP must not be selectable (not even under full exploration) when
        # stagnation is False: unguided premature termination is not allowed.
        state = truncation_state()
        state.optimization.stagnation = BoolMetric.valid(False)
        controller = QController(config=QLearningConfig(seed=0, epsilon_start=1.0))
        for _ in range(50):
            action = controller.select_action(state, make_setup())
            assert action.action != ActionType.STOP

    def test_stop_gated_by_budget_consumed(self):
        # Stagnation alone is not enough: most of the budget must be consumed.
        state = truncation_state()  # stagnation=True, budget_consumed=0.5 -> allowed
        controller = QController(config=QLearningConfig(seed=0, epsilon_start=0.0))
        assert ActionType.STOP in controller._available_actions(state)
        state.budget.budget_consumed = Metric.valid(0.1)
        state.budget.budget_remaining = Metric.valid(0.9)
        assert ActionType.STOP not in controller._available_actions(state)

    def test_stop_available_under_stagnation_and_spent_budget(self):
        state = truncation_state()
        controller = QController(config=QLearningConfig(seed=0, epsilon_start=0.0))
        actions = controller._available_actions(state)
        assert ActionType.STOP in actions
        assert len(actions) == len(ALL_ACTIONS)

    def test_deterministic_with_seed(self):
        c1 = QController(rule_engine=RuleEngine(), config=QLearningConfig(seed=42, epsilon_start=1.0))
        c2 = QController(rule_engine=RuleEngine(), config=QLearningConfig(seed=42, epsilon_start=1.0))
        state = truncation_state()
        setup = make_setup()
        seq1 = [c1.select_action(state, setup).action for _ in range(20)]
        seq2 = [c2.select_action(state, setup).action for _ in range(20)]
        assert seq1 == seq2

    def test_persistence_roundtrip(self, tmp_path):
        controller = QController(config=QLearningConfig(seed=7))
        state = truncation_state()
        key = controller.observe(state, make_setup())
        controller.update(key, ActionType.EXPAND_SEARCH_SPACE, 0.3, None, terminal=True)
        path = tmp_path / "q.json"
        controller.save(path)
        other = QController(config=QLearningConfig(seed=99))
        other.load(path)
        assert other.q_table == controller.q_table
        assert other.epsilon == controller.epsilon


class TestLearningLoop:
    def test_q_values_converge_toward_profitable_action(self):
        """Sanity: repeatedly rewarding EXPAND makes it the greedy choice."""
        controller = QController(
            rule_engine=RuleEngine(),
            config=QLearningConfig(epsilon_start=0.0, learning_rate=0.5),
        )
        state = truncation_state()
        setup = make_setup()
        key = controller.observe(state, setup)
        for _ in range(20):
            controller.update(key, ActionType.EXPAND_SEARCH_SPACE, 0.4, key)
        action = controller.select_action(state, setup)
        assert action.action == ActionType.EXPAND_SEARCH_SPACE
        assert action.q_value > 0
