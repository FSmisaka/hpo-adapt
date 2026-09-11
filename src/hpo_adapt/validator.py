"""Tool Validator (spec.md: 其它模块和说明 — Tool Validator).

Checks that module outputs conform to the structured schemas. In particular,
every Setup (initial or revised) MUST pass ``validate_setup`` before the
Classical HPO module is allowed to run it.
"""

from typing import Any

from hpo_adapt.schemas.action import HYPERPARAMETER_ACTIONS, ActionType, RLAction
from hpo_adapt.schemas.diagnostic import DiagnosticState
from hpo_adapt.schemas.experience import Experience
from hpo_adapt.schemas.setup import Distribution, Setup
from hpo_adapt.schemas.task import TaskSpec


class ValidationError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def _check(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_setup(
    setup: Setup,
    task: TaskSpec,
    *,
    allowed_optimizers: list[str] | None = None,
) -> list[str]:
    """Return a list of violations; empty list means the Setup is valid."""
    errors: list[str] = []
    allowed = allowed_optimizers if allowed_optimizers is not None else task.allowed_optimizers
    available = {hp.name: hp for hp in task.available_hyperparameters}

    _check(len(setup.search_space) > 0, errors, "search_space must not be empty")

    names: set[str] = set()
    for spec in setup.search_space:
        _check(
            spec.name not in names,
            errors,
            f"duplicate search-space entry for hyperparameter '{spec.name}'",
        )
        names.add(spec.name)
        # Never invent hyperparameters that do not exist in the provided system.
        _check(
            spec.name in available,
            errors,
            f"hyperparameter '{spec.name}' does not exist in the task",
        )
        if spec.name not in available:
            continue
        meta = available[spec.name]
        _check(
            spec.type == meta.type,
            errors,
            f"'{spec.name}': type '{spec.type}' does not match task metadata type '{meta.type}'",
        )
        if spec.type == "categorical":
            _check(
                bool(spec.choices),
                errors,
                f"categorical hyperparameter '{spec.name}' requires non-empty choices",
            )
            if meta.choices:
                invalid = [c for c in (spec.choices or []) if c not in meta.choices]
                _check(
                    not invalid,
                    errors,
                    f"'{spec.name}': choices {invalid} outside task metadata choices",
                )
        else:
            _check(
                spec.low is not None and spec.high is not None,
                errors,
                f"numerical hyperparameter '{spec.name}' requires low and high",
            )
            if spec.low is not None and spec.high is not None:
                _check(
                    spec.low < spec.high,
                    errors,
                    f"'{spec.name}': low ({spec.low}) must be < high ({spec.high})",
                )
                _check(
                    spec.low > 0 if spec.distribution == Distribution.LOG_UNIFORM else True,
                    errors,
                    f"'{spec.name}': log distribution requires low > 0",
                )
                if spec.type == "int":
                    _check(
                        float(spec.low).is_integer() and float(spec.high).is_integer(),
                        errors,
                        f"'{spec.name}': int hyperparameter requires integer bounds",
                    )

    for name in setup.frozen:
        _check(name in available, errors, f"frozen hyperparameter '{name}' does not exist in the task")

    _check(
        setup.optimizer.name in allowed,
        errors,
        f"optimizer '{setup.optimizer.name}' not in allowed optimizer list {allowed}",
    )
    _check(
        setup.budget.max_trials >= 1,
        errors,
        "budget.max_trials must be >= 1",
    )
    _check(
        setup.budget.max_trials <= task.resource_constraints.max_trials_per_phase
        if task.resource_constraints.max_trials_per_phase is not None
        else True,
        errors,
        f"budget.max_trials exceeds max_trials_per_phase "
        f"({task.resource_constraints.max_trials_per_phase})",
    )
    _check(
        setup.evaluation.direction == task.direction,
        errors,
        "evaluation.direction does not match task objective direction",
    )
    return errors


def validate_rl_action(action: RLAction, available_hyperparameters: list[str]) -> list[str]:
    errors: list[str] = []
    if action.action == ActionType.STOP:
        _check(action.target is None, errors, "STOP action must have target = null")
    elif action.action == ActionType.INCREASE_EXPLORATION:
        _check(
            action.target in (None, "optimizer"),
            errors,
            "INCREASE_EXPLORATION target must be 'optimizer' or null",
        )
    elif action.action == ActionType.INCREASE_BUDGET:
        _check(
            action.target in (None, "budget"),
            errors,
            "INCREASE_BUDGET target must be 'budget' or null",
        )
    elif action.action in HYPERPARAMETER_ACTIONS:
        _check(
            action.target is not None and action.target in available_hyperparameters,
            errors,
            f"{action.action.value} requires a valid hyperparameter target",
        )
    return errors


def validate_diagnostic_state(state: DiagnosticState) -> list[str]:
    errors: list[str] = []
    _check(state.meta.num_complete_trials >= 0, errors, "num_complete_trials must be >= 0")
    return errors


def validate_experience(experience: Experience) -> list[str]:
    errors: list[str] = []
    _check(bool(experience.experience_id), errors, "experience_id must be non-empty")
    _check(bool(experience.action.type), errors, "action.type must be non-empty")
    return errors


def parse_setup_json(data: dict[str, Any]) -> Setup:
    """Tool validator for agent JSON output: parse + structural validation of a Setup."""
    if not isinstance(data, dict):
        raise ValidationError(["setup output must be a JSON object"])
    required = ["search_space", "optimizer", "budget", "evaluation"]
    errors: list[str] = []
    for key in required:
        _check(key in data, errors, f"setup output missing required key '{key}'")
    if errors:
        raise ValidationError(errors)
    try:
        return Setup.model_validate(data)
    except Exception as exc:  # pydantic validation errors
        raise ValidationError([f"setup output does not satisfy SetupSchema: {exc}"]) from exc


def require_valid_setup(setup: Setup, task: TaskSpec) -> Setup:
    """Raise unless the Setup passes the validator; gate before running HPO."""
    errors = validate_setup(setup, task)
    if errors:
        raise ValidationError(errors)
    return setup
