"""Optuna-backed HPO runner: Setup -> Study -> TrialLogs."""

from typing import Any, Optional

import optuna
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.trial import TrialState as OptunaTrialState
from pydantic import BaseModel, Field

from hpo_adapt.hpo.objectives import get_objective
from hpo_adapt.schemas import (
    Distribution,
    Setup,
    TaskSpec,
    TrialLog,
    TrialState,
)
from hpo_adapt.validator import require_valid_setup

optuna.logging.set_verbosity(optuna.logging.ERROR)


class HPOResult(BaseModel):
    setup_id: str
    phase: int
    trials: list[TrialLog] = Field(default_factory=list)
    best_value: Optional[float] = None
    best_params: Optional[dict[str, Any]] = None

    @property
    def num_completed(self) -> int:
        return sum(1 for t in self.trials if t.state == TrialState.COMPLETE)


_STATE_MAP = {
    OptunaTrialState.COMPLETE: TrialState.COMPLETE,
    OptunaTrialState.PRUNED: TrialState.PRUNED,
    OptunaTrialState.FAIL: TrialState.FAIL,
}


def _build_sampler(setup: Setup, seed: int) -> BaseSampler:
    name = setup.optimizer.name.upper()
    if name == "RANDOM":
        return RandomSampler(seed=seed)
    if name == "TPE":
        n_startup = int(setup.optimizer.config.get("n_startup_trials", 10))
        return TPESampler(seed=seed, n_startup_trials=n_startup)
    raise ValueError(f"unsupported optimizer '{setup.optimizer.name}'")


def _suggest(trial: optuna.Trial, setup: Setup) -> dict[str, Any]:
    params: dict[str, Any] = dict(setup.frozen)
    for spec in setup.search_space:
        if spec.type == "categorical":
            params[spec.name] = trial.suggest_categorical(spec.name, spec.choices or [])
        elif spec.type == "int":
            params[spec.name] = trial.suggest_int(spec.name, int(spec.low or 0), int(spec.high or 0))
        else:
            log = spec.distribution == Distribution.LOG_UNIFORM
            params[spec.name] = trial.suggest_float(
                spec.name, float(spec.low or 0.0), float(spec.high or 1.0), log=log
            )
    return params


def run_hpo(task: TaskSpec, setup: Setup, *, study_name: Optional[str] = None) -> HPOResult:
    """Run one HPO phase according to a validated Setup and return TrialLogs."""
    require_valid_setup(setup, task)
    objective_fn = get_objective(task.objective)
    seed = setup.evaluation.seed + setup.phase
    sampler = _build_sampler(setup, seed)

    study = optuna.create_study(
        direction=setup.evaluation.direction.value,
        sampler=sampler,
        study_name=study_name or f"{task.task_id}_{setup.setup_id}",
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, setup)
        return objective_fn(params)

    study.optimize(objective, n_trials=setup.budget.max_trials, catch=(RuntimeError, ValueError))

    trials: list[TrialLog] = []
    for t in study.trials:
        trials.append(
            TrialLog(
                trial_id=t.number,
                params={**setup.frozen, **t.params},
                value=t.value if t.state == OptunaTrialState.COMPLETE else None,
                state=_STATE_MAP.get(t.state, TrialState.FAIL),
                duration=(
                    t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete
                else None,
                timestamp=t.datetime_start.timestamp() if t.datetime_start else None,
            )
        )

    best_value: Optional[float] = None
    best_params: Optional[dict[str, Any]] = None
    if study.best_value is not None:
        best_value = float(study.best_value)
        best_params = dict(study.best_params)

    return HPOResult(
        setup_id=setup.setup_id,
        phase=setup.phase,
        trials=trials,
        best_value=best_value,
        best_params=best_params,
    )
