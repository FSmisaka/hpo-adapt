"""Stage 1 tests: Classical HPO module."""

import pytest

from hpo_adapt.hpo import get_objective, run_hpo
from hpo_adapt.schemas import (
    BudgetSpec,
    EvaluationSpec,
    HyperparameterSpec,
    ObjectiveDirection,
    OptimizerSpec,
    ResourceConstraints,
    Setup,
    TaskSpec,
    TrialState,
)
from hpo_adapt.schemas.task import HyperparameterMetadata
from hpo_adapt.validator import ValidationError, validate_setup


def make_task(objective: str = "synthetic_mlp") -> TaskSpec:
    return TaskSpec(
        task_id="task_hpo",
        task_type="classification",
        description="synthetic",
        model_family="SyntheticMLP",
        dataset_signature="synthetic_v1",
        objective_name="accuracy",
        direction=ObjectiveDirection.MAXIMIZE,
        objective=objective,
        available_hyperparameters=[
            HyperparameterMetadata(name="learning_rate", type="float", low=1e-6, high=1.0, log=True),
            HyperparameterMetadata(name="batch_size", type="int", default=64, low=8, high=512),
            HyperparameterMetadata(name="optimizer_type", type="categorical", choices=["adam", "sgd"]),
            HyperparameterMetadata(name="dropout", type="float", low=0.0, high=0.9),
        ],
        resource_constraints=ResourceConstraints(max_total_trials=100),
    )


def make_setup(max_trials: int = 20, optimizer: str = "TPE") -> Setup:
    return Setup(
        setup_id="setup_000",
        phase=0,
        search_space=[
            HyperparameterSpec(name="learning_rate", type="float", low=1e-5, high=1e-2),
            HyperparameterSpec(name="batch_size", type="int", low=8, high=256),
            HyperparameterSpec(name="optimizer_type", type="categorical", choices=["adam", "sgd"]),
        ],
        optimizer=OptimizerSpec(name=optimizer),
        budget=BudgetSpec(max_trials=max_trials),
        evaluation=EvaluationSpec(protocol="holdout", direction=ObjectiveDirection.MAXIMIZE),
    )


class TestRunner:
    def test_runs_and_produces_trial_logs(self):
        result = run_hpo(make_task(), make_setup(max_trials=15))
        assert len(result.trials) == 15
        assert all(isinstance(t.params, dict) for t in result.trials)
        assert result.num_completed > 0
        assert result.best_value is not None
        for t in result.trials:
            assert t.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}

    def test_deterministic_with_same_seed(self):
        task, setup = make_task(), make_setup(max_trials=10, optimizer="RANDOM")
        r1 = run_hpo(task, setup)
        r2 = run_hpo(task, setup)
        assert [t.params for t in r1.trials] == [t.params for t in r2.trials]
        assert r1.best_value == r2.best_value

    def test_warm_start_reuses_history_without_consuming_budget(self):
        task, setup = make_task(), make_setup(max_trials=15, optimizer="RANDOM")
        phase1 = run_hpo(task, setup)
        # phase 2: same setup (next phase id) warm-started with phase-1 trials
        setup2 = setup.model_copy(deep=True)
        setup2.phase = 1
        phase2 = run_hpo(task, setup2, warm_start_trials=phase1.trials)
        assert phase2.warm_started == len(phase1.trials)
        # only NEW trials consume budget
        assert len(phase2.trials) == 15
        # the optimizer sees the history: best cannot be worse than phase 1's
        assert phase2.best_value is not None and phase1.best_value is not None
        assert phase2.best_value >= phase1.best_value - 1e-12

    def test_warm_start_skips_out_of_range_trials(self):
        task = make_task()
        setup = make_setup(max_trials=5, optimizer="RANDOM")
        past = run_hpo(task, setup).trials
        # shrink the lr range so that most past lr values fall outside
        narrow = setup.model_copy(deep=True)
        narrow.search_space[0].low = 1e-5
        narrow.search_space[0].high = 1.0001e-5
        result = run_hpo(task, narrow, warm_start_trials=past)
        assert result.warm_started < len(past)

    def test_invalid_setup_rejected(self):
        task, setup = make_task(), make_setup()
        setup.optimizer = OptimizerSpec(name="CMAES")
        try:
            run_hpo(task, setup)
            raise AssertionError("expected ValidationError")
        except ValidationError:
            pass

    def test_frozen_params_are_used(self):
        task, setup = make_task(), make_setup(max_trials=5, optimizer="RANDOM")
        setup.frozen = {"dropout": 0.1}
        result = run_hpo(task, setup)
        assert all(t.params.get("dropout") == 0.1 for t in result.trials)

    def test_minimize_direction(self):
        task = make_task(objective="branin")
        task.available_hyperparameters = [
            HyperparameterMetadata(name="x1", type="float", low=-5.0, high=10.0),
            HyperparameterMetadata(name="x2", type="float", low=0.0, high=15.0),
        ]
        task.direction = ObjectiveDirection.MINIMIZE
        setup = Setup(
            setup_id="setup_branin",
            search_space=[
                HyperparameterSpec(name="x1", type="float", low=-5.0, high=10.0),
                HyperparameterSpec(name="x2", type="float", low=0.0, high=15.0),
            ],
            optimizer=OptimizerSpec(name="RANDOM"),
            budget=BudgetSpec(max_trials=30),
            evaluation=EvaluationSpec(protocol="holdout", direction=ObjectiveDirection.MINIMIZE),
        )
        result = run_hpo(task, setup)
        assert result.best_value is not None
        assert result.best_value < 30.0  # branin < 30 with modest effort


class TestCmaEs:
    def test_cmaes_runs_on_numerical_task(self):
        # Branin-style purely numerical task: CMA-ES is applicable.
        task = TaskSpec(
            task_id="task_cmaes",
            task_type="regression",
            description="numerical only",
            model_family="Analytic",
            dataset_signature="branin",
            objective_name="value",
            direction=ObjectiveDirection.MINIMIZE,
            objective="branin",
            available_hyperparameters=[
                HyperparameterMetadata(name="x1", type="float", low=-5.0, high=10.0),
                HyperparameterMetadata(name="x2", type="float", low=0.0, high=15.0),
            ],
            resource_constraints=ResourceConstraints(max_total_trials=100),
            allowed_optimizers=["RANDOM", "TPE", "CMAES"],
        )
        setup = Setup(
            setup_id="setup_cmaes",
            search_space=[
                HyperparameterSpec(name="x1", type="float", low=-5.0, high=10.0),
                HyperparameterSpec(name="x2", type="float", low=0.0, high=15.0),
            ],
            optimizer=OptimizerSpec(name="CMAES"),
            budget=BudgetSpec(max_trials=12),
            evaluation=EvaluationSpec(protocol="holdout", direction=ObjectiveDirection.MINIMIZE),
        )
        result = run_hpo(task, setup)
        assert len(result.trials) == 12
        assert result.best_value is not None
        assert result.best_value < 50.0

    def test_cmaes_rejected_when_not_allowed(self):
        # Tasks with categorical parameters do not allow CMA-ES; the Tool
        # Validator must reject such a setup before any HPO run.
        setup = make_setup(optimizer="CMAES")
        errors = validate_setup(setup, make_task())
        assert any("not in allowed optimizer list" in e for e in errors)
        with pytest.raises(ValidationError):
            run_hpo(make_task(), setup)


class TestObjectives:
    def test_registry(self):
        assert "synthetic_mlp" in get_objective.__globals__["_REGISTRY"]
        fn = get_objective("synthetic_mlp")
        value = fn(
            {"learning_rate": 0.45, "batch_size": 32, "optimizer_type": "sgd", "dropout": 0.05}
        )
        assert value > 0.9

    def test_failure_paths(self):
        import pytest

        fn = get_objective("synthetic_mlp")
        with pytest.raises(RuntimeError):
            fn({"learning_rate": 1e-3, "batch_size": 32, "optimizer_type": "sgd", "dropout": 0.9})
        with pytest.raises(ValueError):
            fn({"learning_rate": 1e-3, "batch_size": 500, "optimizer_type": "sgd", "dropout": 0.1})
