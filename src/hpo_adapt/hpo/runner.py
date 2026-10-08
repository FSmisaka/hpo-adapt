"""Optuna-backed HPO runner: Setup -> Study -> TrialLogs."""

from typing import Any, Optional

import optuna
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler
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
    warm_started: int = 0  # historical trials injected into this phase's study

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
    if name == "CMAES":
        # CMA-ES requires a purely numerical search space (no categorical /
        # conditional parameters); the validator only allows it for such tasks.
        return CmaEsSampler(seed=seed)
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


def _distributions(setup: Setup) -> dict[str, Any]:
    from optuna.distributions import (
        CategoricalDistribution,
        FloatDistribution,
        IntDistribution,
    )

    dists: dict[str, Any] = {}
    for spec in setup.search_space:
        if spec.type == "categorical":
            dists[spec.name] = CategoricalDistribution(list(spec.choices or []))
        elif spec.type == "int":
            dists[spec.name] = IntDistribution(int(spec.low or 0), int(spec.high or 0))
        else:
            dists[spec.name] = FloatDistribution(
                float(spec.low or 0.0),
                float(spec.high or 1.0),
                log=spec.distribution == Distribution.LOG_UNIFORM,
            )
    return dists


def _in_range(spec: Any, value: Any) -> bool:
    if spec.type == "categorical":
        return value in (spec.choices or [])
    low, high = float(spec.low or 0.0), float(spec.high or 1.0)
    if spec.type == "int":
        return low <= int(value) <= high
    return low <= float(value) <= high


def _inject_warm_start(
    study: optuna.study.Study, setup: Setup, past: Optional[list[TrialLog]]
) -> int:
    """Inject historical trials as completed study trials (no re-evaluation).

    Each adaptation phase otherwise starts a fresh optimizer state and would
    waste its random startup budget re-exploring. Warm-starting reuses every
    previously evaluated configuration that is compatible with the current
    search space (out-of-range points are skipped). Only NEW trials consume
    budget, so static and adaptive runs stay directly comparable.
    """
    if not past:
        return 0
    from optuna.trial import create_trial

    dists = _distributions(setup)
    spec_by_name = {s.name: s for s in setup.search_space}
    added = 0
    for t in past:
        if t.state != TrialState.COMPLETE or t.value is None:
            continue
        params: dict[str, Any] = {}
        compatible = True
        for name, spec in spec_by_name.items():
            if name not in t.params or not _in_range(spec, t.params[name]):
                compatible = False
                break
            params[name] = int(t.params[name]) if spec.type == "int" else t.params[name]
        if not compatible:
            continue
        study.add_trial(
            create_trial(
                state=OptunaTrialState.COMPLETE,
                params=params,
                distributions=dists,
                value=float(t.value),
            )
        )
        added += 1
    return added


def run_hpo(
    task: TaskSpec,
    setup: Setup,
    *,
    study_name: Optional[str] = None,
    warm_start_trials: Optional[list[TrialLog]] = None,
) -> HPOResult:
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
    warm_started = _inject_warm_start(study, setup, warm_start_trials)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, setup)
        return objective_fn(params)

    study.optimize(objective, n_trials=setup.budget.max_trials, catch=(RuntimeError, ValueError))

    # study.trials includes the warm-started (injected) trials; only the newly
    # evaluated ones consume budget and belong in this phase's TrialLog.
    new_trials = study.trials[warm_started:]
    trials: list[TrialLog] = []
    for t in new_trials:
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
        warm_started=warm_started,
    )
