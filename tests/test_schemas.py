"""Stage 0 tests: schemas + tool validator."""

import pytest

from hpo_adapt.schemas import (
    ActionType,
    BudgetSpec,
    DiagnosticConfig,
    Distribution,
    EvaluationSpec,
    HyperparameterSpec,
    ObjectiveDirection,
    OptimizerSpec,
    RLAction,
    Setup,
    TaskSpec,
    TrialLog,
    TrialState,
)
from hpo_adapt.schemas.task import HyperparameterMetadata, ResourceConstraints
from hpo_adapt.validator import (
    ValidationError,
    parse_setup_json,
    require_valid_setup,
    validate_rl_action,
    validate_setup,
)


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="task_001",
        task_type="classification",
        description="synthetic classification",
        model_family="SyntheticMLP",
        dataset_signature="synthetic_v1",
        objective_name="accuracy",
        direction=ObjectiveDirection.MAXIMIZE,
        objective="synthetic_mlp",
        available_hyperparameters=[
            HyperparameterMetadata(
                name="learning_rate", type="float", default=1e-3, low=1e-6, high=1.0, log=True
            ),
            HyperparameterMetadata(name="batch_size", type="int", default=64, low=8, high=512),
            HyperparameterMetadata(
                name="optimizer_type", type="categorical", default="adam",
                choices=["adam", "sgd"],
            ),
            HyperparameterMetadata(name="dropout", type="float", default=0.1, low=0.0, high=0.9),
        ],
        resource_constraints=ResourceConstraints(max_total_trials=100, max_trials_per_phase=50),
        allowed_optimizers=["RANDOM", "TPE"],
    )


def make_setup() -> Setup:
    return Setup(
        setup_id="setup_000",
        phase=0,
        search_space=[
            HyperparameterSpec(
                name="learning_rate", type="float", low=1e-5, high=1e-2, default=1e-3
            ),
            HyperparameterSpec(name="batch_size", type="int", low=8, high=256, default=64),
            HyperparameterSpec(
                name="optimizer_type",
                type="categorical",
                choices=["adam", "sgd"],
                default="adam",
            ),
        ],
        optimizer=OptimizerSpec(name="TPE", config={"n_startup_trials": 10}),
        budget=BudgetSpec(max_trials=30),
        evaluation=EvaluationSpec(protocol="holdout", direction=ObjectiveDirection.MAXIMIZE),
    )


class TestSchemas:
    def test_setup_defaults_and_log_distribution(self):
        setup = make_setup()
        lr = setup.hyperparameter("learning_rate")
        assert lr is not None and lr.distribution == Distribution.LOG_UNIFORM
        bs = setup.hyperparameter("batch_size")
        assert bs is not None and bs.distribution == Distribution.UNIFORM

    def test_trial_log_matches_spec(self):
        log = TrialLog(
            trial_id=1, params={"learning_rate": 0.1}, value=0.5,
            state=TrialState.COMPLETE, duration=1.2, timestamp=1000.0,
        )
        assert log.state == TrialState.COMPLETE
        assert TrialLog.model_validate(log.model_dump()) == log

    def test_rl_action_alias(self):
        action = RLAction.model_validate(
            {
                "action": "EXPAND_SEARCH_SPACE",
                "candidate_hyperparameter": "learning_rate",
                "parameters": {"direction": "upper"},
                "confidence": 0.87,
                "q_value": 0.63,
            }
        )
        assert action.target == "learning_rate"
        assert action.action == ActionType.EXPAND_SEARCH_SPACE

    def test_diagnostic_config_defaults_from_spec(self):
        cfg = DiagnosticConfig()
        assert cfg.min_trials_for_basic_metrics == 5
        assert cfg.min_trials_for_importance == 30
        assert cfg.min_trials_for_good_region == 20
        assert cfg.min_trials_for_stagnation == 20
        assert cfg.analysis_window == 20
        assert cfg.good_trial_quantile == 0.10
        assert cfg.boundary_threshold == 0.05
        assert cfg.good_region_quantiles == (0.10, 0.90)
        assert cfg.coverage_bins == 10


class TestSetupValidator:
    def test_valid_setup_passes(self):
        assert validate_setup(make_setup(), make_task()) == []

    def test_invented_hyperparameter_rejected(self):
        setup = make_setup()
        setup.search_space.append(
            HyperparameterSpec(name="momentum", type="float", low=0.0, high=1.0)
        )
        errors = validate_setup(setup, make_task())
        assert any("momentum" in e for e in errors)

    def test_bad_bounds_rejected(self):
        setup = make_setup()
        setup.search_space[0] = HyperparameterSpec(
            name="learning_rate", type="float", low=1e-2, high=1e-5
        )
        errors = validate_setup(setup, make_task())
        assert any("low" in e for e in errors)

    def test_log_requires_positive_low(self):
        setup = make_setup()
        setup.search_space[0] = HyperparameterSpec(
            name="learning_rate", type="float", low=-1.0, high=1.0,
            distribution=Distribution.LOG_UNIFORM,
        )
        errors = validate_setup(setup, make_task())
        assert any("log" in e for e in errors)

    def test_unknown_optimizer_rejected(self):
        setup = make_setup()
        setup.optimizer = OptimizerSpec(name="CMAES")
        errors = validate_setup(setup, make_task())
        assert any("optimizer" in e for e in errors)

    def test_direction_mismatch_rejected(self):
        setup = make_setup()
        setup.evaluation = EvaluationSpec(protocol="holdout", direction=ObjectiveDirection.MINIMIZE)
        errors = validate_setup(setup, make_task())
        assert any("direction" in e for e in errors)

    def test_require_valid_setup_raises(self):
        setup = make_setup()
        setup.budget = BudgetSpec(max_trials=100)
        with pytest.raises(ValidationError):
            require_valid_setup(setup, make_task())


class TestOtherValidators:
    def test_rl_action_validator(self):
        task = make_task()
        names = [hp.name for hp in task.available_hyperparameters]
        ok = RLAction(action=ActionType.EXPAND_SEARCH_SPACE, target="learning_rate")
        assert validate_rl_action(ok, names) == []
        bad = RLAction(action=ActionType.EXPAND_SEARCH_SPACE, target="nonexistent")
        assert validate_rl_action(bad, names)
        stop = RLAction(action=ActionType.STOP, target="learning_rate")
        assert validate_rl_action(stop, names)

    def test_parse_setup_json(self):
        setup = make_setup()
        parsed = parse_setup_json(setup.model_dump())
        assert parsed == setup
        with pytest.raises(ValidationError):
            parse_setup_json({"search_space": []})
