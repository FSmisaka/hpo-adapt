"""Stage 4 tests: rule-based adaptation (pattern table + candidate ranking)."""

import pytest

from hpo_adapt.levels import LevelConfig, global_state_levels
from hpo_adapt.rules import CandidateWeights, RuleEngine
from hpo_adapt.schemas import (
    ActionType,
    BoundaryConcentrationEntry,
    BudgetStatus,
    ConfigurationDiversity,
    CoverageStatus,
    DiagnosticState,
    GoodRegionEntry,
    ImportanceEntry,
    Metric,
    MetricStatus,
    OptimizationStatus,
)
from hpo_adapt.schemas.diagnostic import BoolMetric


def base_state() -> DiagnosticState:
    return DiagnosticState(
        optimization=OptimizationStatus(),
    )


def set_optimization(
    state: DiagnosticState,
    *,
    stagnation=None,
    rel_improvement=None,
    marginal_gain=None,
):
    if stagnation is not None:
        state.optimization.stagnation = BoolMetric.valid(stagnation)
    if rel_improvement is not None:
        state.optimization.relative_improvement = Metric.valid(rel_improvement)
    if marginal_gain is not None:
        state.optimization.marginal_gain = Metric.valid(marginal_gain)


def set_importance(state, lr_global=0.7, dropout_global=0.02, valid=True):
    status = MetricStatus.VALID if valid else MetricStatus.INSUFFICIENT_DATA
    state.search_space.importance = {
        "lr": ImportanceEntry(
            global_importance=lr_global, local_importance=lr_global, rank=1, status=status
        ),
        "dropout": ImportanceEntry(
            global_importance=dropout_global, local_importance=dropout_global, rank=2, status=status
        ),
    }


def set_boundary(state, lr_all=0.4, lr_good=0.9, lr_upper=0.85, lr_lower=0.05):
    state.search_space.boundary_concentration = {
        "lr": BoundaryConcentrationEntry(
            all=lr_all, good=lr_good, upper=lr_upper, lower=lr_lower, status=MetricStatus.VALID
        )
    }


def set_good_region(state, lr_mean=0.008, lr_lower=0.006, lr_upper=0.010, width=0.4):
    state.search_space.good_region = {
        "lr": GoodRegionEntry(
            mean=lr_mean, lower=lr_lower, upper=lr_upper, width_ratio=width, status=MetricStatus.VALID
        )
    }


def set_diversity(state, recent=0.08, glob=0.24):
    state.search_space.configuration_diversity = ConfigurationDiversity(
        recent=recent, global_=glob, ratio=recent / glob, status=MetricStatus.VALID
    )


def set_coverage(state, overall=0.42):
    state.search_space.coverage = CoverageStatus(
        overall=overall, per_hyperparameter={"lr": overall}, status=MetricStatus.VALID
    )


def set_budget(state, consumed=0.2, remaining=0.8):
    state.budget = BudgetStatus(
        total_trials=100,
        completed_trials=int(100 * consumed),
        budget_consumed=Metric.valid(consumed),
        budget_remaining=Metric.valid(remaining),
    )


class TestCandidateRanking:
    def test_candidate_score_formula(self):
        engine = RuleEngine(CandidateWeights(alpha=0.5, beta=0.3, gamma=0.2))
        state = base_state()
        set_importance(state, lr_global=0.8, dropout_global=0.05)
        set_boundary(state, lr_good=0.9)
        set_good_region(state, width=0.2)
        ranked = engine.rank_candidates(state)
        lr = next(c for c in ranked if c.name == "lr")
        expected = 0.5 * 0.8 + 0.3 * 0.9 + 0.2 * (1 - 0.2)
        assert lr.score == pytest.approx(expected, abs=1e-6)
        # dropout has no boundary/good-region data: score = 0.5 * 0.05
        dropout = next(c for c in ranked if c.name == "dropout")
        assert dropout.score == pytest.approx(0.5 * 0.05, abs=1e-6)
        assert ranked[0].name == "lr"

    def test_insufficient_importance_excluded(self):
        engine = RuleEngine()
        state = base_state()
        set_importance(state, valid=False)
        assert engine.rank_candidates(state) == []


class TestPatternTable:
    def test_pattern1_truncation_recommends_expand(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=True)
        set_importance(state, lr_global=0.8)
        set_boundary(state, lr_good=0.91, lr_upper=0.86, lr_lower=0.05)
        action = engine.recommend(state)
        assert action is not None
        assert action.action == ActionType.EXPAND_SEARCH_SPACE
        assert action.target == "lr"
        assert action.parameters["direction"] == "upper"

    def test_pattern1_direction_lower(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=True)
        set_importance(state, lr_global=0.8)
        set_boundary(state, lr_good=0.9, lr_upper=0.10, lr_lower=0.80)
        action = engine.recommend(state)
        assert action.parameters["direction"] == "lower"

    def test_pattern2_promising_region_recommends_shift(self):
        engine = RuleEngine()
        state = base_state()
        set_importance(state, lr_global=0.75)
        set_good_region(state, width=0.15, lr_lower=0.006, lr_upper=0.010)
        action = engine.recommend(state)
        assert action is not None
        assert action.action == ActionType.SHIFT_SEARCH_REGION
        assert action.target == "lr"
        assert action.parameters["target_region"] == [0.006, 0.010]

    def test_pattern2_off_center_good_region_recommends_shift(self):
        from tests.test_diagnostic import make_setup

        engine = RuleEngine()
        state = base_state()
        set_importance(state, lr_global=0.75)
        set_good_region(state, width=0.15, lr_lower=0.006, lr_upper=0.010)
        action = engine.recommend(state, make_setup())
        assert action is not None
        assert action.action == ActionType.SHIFT_SEARCH_REGION

    def test_pattern2_centered_good_region_recommends_shrink(self):
        from tests.test_diagnostic import make_setup

        engine = RuleEngine()
        state = base_state()
        set_importance(state, lr_global=0.75)
        # log-space center of the setup range [1e-5, 1e-2] is ~3.16e-4
        set_good_region(state, lr_mean=3.16e-4, lr_lower=3.0e-4, lr_upper=3.33e-4, width=0.15)
        action = engine.recommend(state, make_setup())
        assert action is not None
        assert action.action == ActionType.SHRINK_SEARCH_SPACE
        assert action.target == "lr"
        assert action.parameters["target_region"] == [3.0e-4, 3.33e-4]

    def test_pattern3_low_importance_freeze(self):
        engine = RuleEngine()
        state = base_state()
        set_importance(state, lr_global=0.03, dropout_global=0.9)
        previous = base_state()
        set_importance(previous, lr_global=0.05, dropout_global=0.9)
        matches = engine.evaluate(state, None, previous)
        freeze = [m for m in matches if m.action == ActionType.FREEZE_HYPERPARAMETER]
        assert freeze and freeze[0].target == "lr"

    def test_pattern3_requires_low_importance_across_rounds(self):
        engine = RuleEngine()
        state = base_state()
        set_importance(state, lr_global=0.03, dropout_global=0.9)
        # first observation only -> no FREEZE (spec: "low importance for multiple rounds")
        assert not [
            m for m in engine.evaluate(state) if m.action == ActionType.FREEZE_HYPERPARAMETER
        ]
        # previous round says lr was important -> not persistent -> no FREEZE
        previous_high = base_state()
        set_importance(previous_high, lr_global=0.8, dropout_global=0.9)
        assert not [
            m
            for m in engine.evaluate(state, None, previous_high)
            if m.action == ActionType.FREEZE_HYPERPARAMETER
        ]

    def test_pattern4_premature_convergence(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=True)
        set_diversity(state, recent=0.02, glob=0.24)  # ratio 0.083 < 0.30
        set_importance(state, lr_global=0.3, dropout_global=0.3)
        action = engine.recommend(state)
        assert action is not None
        assert action.action == ActionType.INCREASE_EXPLORATION
        assert action.target == "optimizer"

    def test_pattern5_insufficient_exploration(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=True)
        set_diversity(state, recent=0.20, glob=0.24)  # ratio 0.83, not low
        set_coverage(state, overall=0.30)  # low
        set_importance(state, lr_global=0.3, dropout_global=0.3)
        action = engine.recommend(state)
        assert action is not None
        assert action.action == ActionType.INCREASE_EXPLORATION

    def test_pattern6_budget_valuable(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=False, rel_improvement=0.05)
        set_budget(state, consumed=0.9, remaining=0.1)
        set_importance(state, lr_global=0.4, dropout_global=0.4)
        action = engine.recommend(state)
        assert action is not None
        assert action.action == ActionType.INCREASE_BUDGET
        assert action.target == "budget"

    def test_pattern7_stop(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=True, marginal_gain=1e-6)
        set_budget(state, consumed=0.9, remaining=0.1)
        set_importance(state, lr_global=0.4, dropout_global=0.4)
        set_diversity(state, recent=0.5, glob=0.6)  # diversity fine
        set_coverage(state, overall=0.9)
        matches = engine.evaluate(state)
        stop = [m for m in matches if m.action == ActionType.STOP]
        assert stop

    def test_no_match_returns_none(self):
        engine = RuleEngine()
        state = base_state()
        set_optimization(state, stagnation=False, rel_improvement=0.02)
        set_importance(state, lr_global=0.4, dropout_global=0.4)
        set_budget(state, consumed=0.3, remaining=0.7)
        assert engine.recommend(state) is None


class TestGlobalLevels:
    def test_levels_extracted(self):
        state = base_state()
        set_optimization(state, stagnation=True, rel_improvement=0.0001, marginal_gain=1e-6)
        set_diversity(state, recent=0.05, glob=0.25)
        set_coverage(state, overall=0.35)
        set_budget(state, consumed=0.9, remaining=0.1)
        levels = global_state_levels(state, LevelConfig())
        assert levels["stagnation"] is True
        assert levels["recent_improvement"] == "low"
        assert levels["configuration_diversity"] == "low"
        assert levels["coverage"] == "low"
        assert levels["budget_remaining"] == "low"
        assert levels["marginal_gain"] == "low"

    def test_levels_none_when_insufficient(self):
        levels = global_state_levels(base_state(), LevelConfig())
        assert levels["stagnation"] is None
        assert levels["coverage"] is None
