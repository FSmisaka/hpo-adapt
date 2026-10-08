"""Stage 2 tests: Diagnostic Engine — determinism, spec formulas, insufficient data."""

import copy

import pytest

from hpo_adapt.diagnostic import DiagnosticEngine, diagnose
from hpo_adapt.schemas import (
    BudgetSpec,
    DiagnosticConfig,
    Distribution,
    EvaluationSpec,
    HyperparameterSpec,
    MetricStatus,
    ObjectiveDirection,
    OptimizerSpec,
    Setup,
    TrialLog,
    TrialState,
)

SPECS = [
    HyperparameterSpec(
        name="lr", type="float", low=1e-5, high=1e-2, distribution=Distribution.LOG_UNIFORM
    ),
    HyperparameterSpec(name="depth", type="int", low=1, high=8),
    HyperparameterSpec(name="opt", type="categorical", choices=["adam", "sgd", "rms"]),
]


def make_setup(max_trials: int = 100) -> Setup:
    return Setup(
        setup_id="setup_diag",
        search_space=list(SPECS),
        optimizer=OptimizerSpec(name="TPE"),
        budget=BudgetSpec(max_trials=max_trials),
        evaluation=EvaluationSpec(protocol="holdout", direction=ObjectiveDirection.MAXIMIZE),
    )


def trial(i, params, value, state=TrialState.COMPLETE, duration=1.0):
    return TrialLog(
        trial_id=i, params=params, value=value, state=state, duration=duration, timestamp=float(i)
    )


class TestPreprocessing:
    def test_minimize_is_negated(self):
        setup = make_setup()
        trials = [
            trial(0, {"lr": 1e-4, "depth": 2, "opt": "adam"}, 0.5),
            trial(1, {"lr": 1e-3, "depth": 4, "opt": "sgd"}, 0.2),
        ]
        ds_max = DiagnosticEngine().run(trials, setup, ObjectiveDirection.MAXIMIZE)
        ds_min = DiagnosticEngine().run(trials, setup, ObjectiveDirection.MINIMIZE)
        assert ds_max.meta.num_complete_trials == 2
        assert ds_min.meta.num_complete_trials == 2

    def test_failed_pruned_excluded_from_stats(self):
        setup = make_setup()
        trials = [
            trial(0, {"lr": 1e-4, "depth": 2, "opt": "adam"}, 0.1),
            trial(1, {"lr": 1e-4, "depth": 2, "opt": "adam"}, None, TrialState.FAIL),
            trial(2, {"lr": 1e-4, "depth": 2, "opt": "adam"}, None, TrialState.PRUNED),
        ]
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        assert ds.meta.num_complete_trials == 1
        assert ds.meta.num_failed_trials == 1
        assert ds.meta.num_pruned_trials == 1
        assert ds.meta.failure_rate == pytest.approx(2 / 3, abs=1e-9)


class TestInsufficientData:
    def test_all_metrics_insufficient_with_few_trials(self):
        setup = make_setup()
        trials = [trial(i, {"lr": 1e-4, "depth": 2, "opt": "adam"}, 0.1 * i) for i in range(4)]
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        assert not ds.meta.data_sufficient
        for entry in ds.search_space.importance.values():
            assert entry.status == MetricStatus.INSUFFICIENT_DATA
            assert entry.global_importance is None
        assert ds.search_space.coverage.status == MetricStatus.INSUFFICIENT_DATA
        assert ds.search_space.boundary_concentration == {}
        assert ds.search_space.good_region == {}

    def test_importance_requires_30_trials(self):
        setup = make_setup()
        trials = [trial(i, {"lr": 1e-4, "depth": 2, "opt": "adam"}, 0.1) for i in range(25)]
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        for entry in ds.search_space.importance.values():
            assert entry.status == MetricStatus.INSUFFICIENT_DATA
        assert ds.search_space.good_region["lr"].status == MetricStatus.VALID


class TestOptimizationFormulas:
    def test_best_improvement_stagnation_marginal_gain(self):
        setup = make_setup()
        cfg = DiagnosticConfig(
            analysis_window=10, min_trials_for_stagnation=10, min_trials_for_basic_metrics=5
        )
        engine = DiagnosticEngine(cfg)
        values = [0.50 + 0.01 * i for i in range(20)] + [0.69] * 10
        trials = [
            trial(i, {"lr": 1e-4, "depth": 1 + i % 8, "opt": ["adam", "sgd", "rms"][i % 3]}, values[i])
            for i in range(30)
        ]
        ds = engine.run(trials, setup, ObjectiveDirection.MAXIMIZE)
        opt = ds.optimization
        assert opt.best_objective.status == MetricStatus.VALID
        assert opt.best_objective.value == pytest.approx(0.69, abs=1e-9)
        assert opt.recent_improvement.status == MetricStatus.VALID
        assert opt.recent_improvement.value == pytest.approx(0.0, abs=1e-12)
        assert opt.stagnation.status == MetricStatus.VALID
        assert opt.stagnation.value is True
        assert opt.marginal_gain.value == pytest.approx(0.0, abs=1e-12)

    def test_no_stagnation_when_improving(self):
        setup = make_setup()
        cfg = DiagnosticConfig(
            analysis_window=10, min_trials_for_stagnation=10, min_trials_for_basic_metrics=5
        )
        engine = DiagnosticEngine(cfg)
        values = [0.5 + 0.02 * i for i in range(30)]
        trials = [
            trial(i, {"lr": 1e-4, "depth": 1 + i % 8, "opt": ["adam", "sgd", "rms"][i % 3]}, values[i])
            for i in range(30)
        ]
        ds = engine.run(trials, setup, ObjectiveDirection.MAXIMIZE)
        assert ds.optimization.stagnation.value is False
        assert ds.optimization.recent_improvement.value == pytest.approx(0.2, abs=1e-9)

    def test_relative_improvement_formula(self):
        setup = make_setup()
        cfg = DiagnosticConfig(
            analysis_window=10, min_trials_for_stagnation=10, min_trials_for_basic_metrics=5
        )
        engine = DiagnosticEngine(cfg)
        values = [1.0] * 19 + [1.1]
        trials = [
            trial(i, {"lr": 1e-4, "depth": 1 + i % 8, "opt": ["adam", "sgd", "rms"][i % 3]}, values[i])
            for i in range(20)
        ]
        ds = engine.run(trials, setup, ObjectiveDirection.MAXIMIZE)
        assert ds.optimization.recent_improvement.value == pytest.approx(0.1, abs=1e-9)
        assert ds.optimization.relative_improvement.value == pytest.approx(0.1, abs=1e-6)


class TestSearchSpaceMetrics:
    def test_boundary_concentration_all_at_upper(self):
        setup = make_setup()
        trials = []
        for i in range(30):
            depth = 8 if i % 10 == 0 else 1 + i % 6
            value = 0.9 if i % 10 == 0 else 0.1
            trials.append(trial(i, {"lr": 1e-4, "depth": depth, "opt": "adam"}, value))
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        bc = ds.search_space.boundary_concentration["depth"]
        assert bc.status == MetricStatus.VALID
        assert bc.good == pytest.approx(1.0, abs=1e-9)
        assert bc.upper == pytest.approx(1.0, abs=1e-9)
        assert bc.lower == pytest.approx(0.0, abs=1e-9)
        # boundary: depth=8 on i%10==0 (3 trials) plus depth=1 on i%6==0 & i%10!=0 (4 trials)
        assert bc.all == pytest.approx(7 / 30, abs=1e-9)

    def test_good_region_and_width_ratio(self):
        setup = make_setup()
        trials = []
        for i in range(40):
            lr = 0.005 if i % 10 == 0 else 0.0001 + (i % 9) * 0.0001
            value = 0.9 if i % 10 == 0 else 0.1
            trials.append(trial(i, {"lr": lr, "depth": 4, "opt": "adam"}, value))
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        gr = ds.search_space.good_region["lr"]
        assert gr.status == MetricStatus.VALID
        assert gr.mean == pytest.approx(0.005, abs=1e-12)
        assert gr.width_ratio == pytest.approx(0.0, abs=1e-9)

    def test_categorical_good_region(self):
        setup = make_setup()
        trials = []
        for i in range(30):
            opt = "sgd" if i % 10 == 0 else ["adam", "rms"][i % 2]
            value = 0.9 if i % 10 == 0 else 0.1
            trials.append(trial(i, {"lr": 1e-4, "depth": 2, "opt": opt}, value))
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        cat = ds.search_space.good_region_categorical["opt"]
        assert cat.status == MetricStatus.VALID
        assert cat.dominant_category == "sgd"
        assert cat.dominant_ratio == pytest.approx(1.0, abs=1e-9)

    def test_diversity_spec_formula(self):
        # d(x_i, x_k) = (1/m) * sum_j d_j; categorical d_j = 1(x_ij != x_kj)
        from hpo_adapt.diagnostic.metrics import mixed_pairwise_distance

        a = trial(0, {"lr": 1e-5, "depth": 1, "opt": "adam"}, 0.5)
        b = trial(1, {"lr": 1e-2, "depth": 1, "opt": "sgd"}, 0.6)
        c = trial(2, {"lr": 1e-5, "depth": 8, "opt": "adam"}, 0.7)
        # d(a,b): lr 1.0 + depth 0.0 + opt 1.0 = 2/3
        assert mixed_pairwise_distance([a, b], SPECS) == pytest.approx(2.0 / 3.0, abs=1e-9)
        # d(a,c): lr 0.0 + depth 7/7 + opt 0.0 = 1/3
        assert mixed_pairwise_distance([a, c], SPECS) == pytest.approx(1.0 / 3.0, abs=1e-9)
        # identical configs -> 0
        assert mixed_pairwise_distance([a, trial(3, dict(a.params), 0.5)], SPECS) == 0.0

    def test_diversity_collapse_detection(self):
        setup = make_setup()
        trials = []
        for i in range(40):
            if i < 20:
                params = {
                    "lr": 10 ** (-5 + (i % 3)),
                    "depth": 1 + i % 8,
                    "opt": ["adam", "sgd", "rms"][i % 3],
                }
            else:
                params = {"lr": 1e-4, "depth": 4, "opt": "adam"}
            trials.append(trial(i, params, 0.1 + 0.001 * i))
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        div = ds.search_space.configuration_diversity
        assert div.status == MetricStatus.VALID
        assert div.recent == pytest.approx(0.0, abs=1e-12)
        assert div.global_ > 0.0
        assert div.ratio == pytest.approx(0.0, abs=1e-9)

    def test_coverage(self):
        setup = make_setup()
        trials = [
            trial(i, {"lr": 1e-4, "depth": 4, "opt": ["adam", "sgd", "rms"][i % 3]}, 0.1 + 0.01 * i)
            for i in range(30)
        ]
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        cov = ds.search_space.coverage
        assert cov.status == MetricStatus.VALID
        assert cov.per_hyperparameter["lr"] == pytest.approx(0.1, abs=1e-9)
        assert cov.per_hyperparameter["depth"] == pytest.approx(0.1, abs=1e-9)
        assert cov.overall == pytest.approx(0.1, abs=1e-9)

    def test_importance_ranks_dominant_parameter_first(self):
        setup = make_setup()
        trials = []
        for i in range(60):
            lr = 10 ** (-5 + (i % 6))
            value = 0.1 + 0.15 * (i % 6)
            trials.append(
                trial(i, {"lr": lr, "depth": 1 + i % 8, "opt": ["adam", "sgd", "rms"][i % 3]}, value)
            )
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        imp = ds.search_space.importance
        assert all(e.status == MetricStatus.VALID for e in imp.values())
        assert imp["lr"].rank == 1
        assert imp["lr"].global_importance > imp["depth"].global_importance
        total = sum(e.global_importance for e in imp.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestBudget:
    def test_budget_consumed_remaining(self):
        setup = make_setup(max_trials=100)
        trials = [trial(i, {"lr": 1e-4, "depth": 2, "opt": "adam"}, 0.1) for i in range(40)]
        ds = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        assert ds.budget.budget_consumed.status == MetricStatus.VALID
        assert ds.budget.budget_consumed.value == pytest.approx(0.4, abs=1e-12)
        assert ds.budget.budget_remaining.value == pytest.approx(0.6, abs=1e-12)
        assert ds.budget.total_trials == 100
        assert ds.budget.completed_trials == 40

    def test_global_budget_semantics(self):
        """In the adaptation loop, budget status must reflect the GLOBAL budget."""
        setup = make_setup(max_trials=30)  # per-phase budget
        trials = [trial(i, {"lr": 1e-4, "depth": 2, "opt": "adam"}, 0.1) for i in range(20)]
        ds = diagnose(
            trials,
            setup,
            ObjectiveDirection.MAXIMIZE,
            total_budget=100,
            trials_used_before=20,
        )
        assert ds.budget.total_trials == 100  # global, not 30
        assert ds.budget.completed_trials == 40  # previous phases + this phase
        assert ds.budget.budget_consumed.value == pytest.approx(0.4, abs=1e-12)
        assert ds.budget.budget_remaining.value == pytest.approx(0.6, abs=1e-12)


class TestDeterminism:
    def test_engine_is_deterministic(self):
        setup = make_setup()
        trials = []
        for i in range(60):
            v = ((i * 37) % 100) / 100.0
            trials.append(
                trial(
                    i,
                    {"lr": 10 ** (-5 + (i % 5)), "depth": 1 + i % 8, "opt": ["adam", "sgd", "rms"][i % 3]},
                    v,
                )
            )
        ds1 = diagnose(trials, setup, ObjectiveDirection.MAXIMIZE)
        ds2 = diagnose(copy.deepcopy(trials), setup, ObjectiveDirection.MAXIMIZE)
        assert ds1.model_dump() == ds2.model_dump()
