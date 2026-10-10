"""Tests for the wandb-style user API (hpo_adapt.api / hpo_adapt.run)."""

import numpy as np
import pytest

import hpo_adapt as ha
from hpo_adapt.api import _normalize_space
from hpo_adapt.hpo.objectives import get_objective
from hpo_adapt.schemas import HyperparameterMetadata


def _toy_objective(params):
    lr = float(params["learning_rate"])
    depth = float(params.get("depth", 4))
    return float(np.exp(-((np.log(lr / 0.05) / 1.4) ** 2)) * 0.9 + 0.1 * depth / 32.0)


SPACE = dict(
    learning_rate=ha.Float(1e-4, 1.0, log=True, default=1e-2),
    depth=ha.Int(2, 32, default=8),
    kernel=ha.Choice(["rbf", "poly"], default="rbf"),
)


class TestSpaceHelpers:
    def test_float_int_choice_build_metadata(self):
        f = ha.Float(0.1, 10.0, log=True, default=1.0, description="reg")
        assert f.type == "float" and f.log and f.default == 1.0
        i = ha.Int(2, 64, default=8)
        assert i.type == "int" and i.low == 2.0 and i.high == 64.0
        c = ha.Choice(["a", "b"], default="a")
        assert c.type == "categorical" and c.choices == ["a", "b"]

    def test_normalize_dict_names_the_specs(self):
        metas = _normalize_space(SPACE)
        assert [m.name for m in metas] == ["learning_rate", "depth", "kernel"]

    def test_normalize_tuple_space(self):
        metas = _normalize_space({"x": (0.0, 1.0)})
        assert metas[0].type == "float" and metas[0].low == 0.0

    def test_normalize_rejects_bad_and_empty(self):
        with pytest.raises(TypeError):
            _normalize_space({"x": 42})
        with pytest.raises(ValueError):
            _normalize_space({})

    def test_normalize_rejects_unnamed_list(self):
        with pytest.raises(TypeError):
            _normalize_space([HyperparameterMetadata(name="", type="float", low=0, high=1)])


class TestRun:
    def test_minimal_run(self):
        run = ha.run(
            _toy_objective,
            SPACE,
            direction="maximize",
            total_trials=24,
            max_rounds=2,
            quiet=True,
            dashboard=False,
        )
        assert run.url is None
        assert run.best_value is not None and run.best_value > 0.5
        assert "learning_rate" in run.best_params
        assert run.rounds >= 1
        # closed-loop bookkeeping survived the wrapping
        pr = run.pipeline_result
        assert pr.total_trials_used <= 24
        assert pr.experiences_recorded >= 1

    def test_objective_registered_and_wrapped(self):
        run = ha.run(
            _toy_objective, SPACE, total_trials=8, max_rounds=1,
            quiet=True, dashboard=False, task_name="regcheck",
        )
        assert get_objective(run.task.objective) is not None
        assert run.task.objective.startswith("user::regcheck::")

    def test_task_built_from_arguments(self):
        run = ha.run(
            _toy_objective, SPACE, total_trials=8, max_rounds=1, quiet=True,
            dashboard=False, task_name="My Task!", model_family="MLP",
            dataset="synth-v1", objective_name="acc",
        )
        task = run.task
        assert task.task_id.startswith("my_task_")
        assert task.model_family == "MLP"
        assert task.dataset_signature == "synth-v1"
        assert task.objective_name == "acc"
        assert len(task.available_hyperparameters) == 3
        assert task.resource_constraints.max_total_trials == 8

    def test_user_exceptions_become_failed_trials(self):
        def broken(params):
            if params["depth"] > 4:
                raise KeyError("boom")  # not RuntimeError/ValueError -> wrapped
            return 0.5

        run = ha.run(
            broken,
            dict(depth=ha.Int(2, 8, default=2)),
            total_trials=6,
            max_rounds=1,
            quiet=True,
            dashboard=False,
        )
        # no exception escaped; failed trials were tolerated by the loop
        assert run.pipeline_result.total_trials_used >= 1

    def test_minimize_direction_round_trip(self):
        def descending(params):
            return -abs(float(params["x"]) - 0.3)

        run = ha.run(
            descending,
            dict(x=ha.Float(-2.0, 2.0, default=0.0)),
            direction="minimize",
            total_trials=12,
            max_rounds=2,
            quiet=True,
            dashboard=False,
        )
        assert run.best_value is not None and run.best_value < -0.0
        assert run.pipeline_result.best_value == pytest.approx(-run.best_value)

    def test_wait_is_interruptible(self):
        run = ha.run(
            _toy_objective, SPACE, total_trials=6, max_rounds=1, quiet=True, dashboard=False
        )
        run.wait()  # EOF on stdin in tests -> returns immediately


class TestSeededReproducibility:
    def test_same_seed_same_trajectory(self):
        kwargs = dict(
            objective=_toy_objective, space=SPACE, total_trials=18, max_rounds=2,
            quiet=True, dashboard=False, seed=7,
        )
        a = ha.run(**kwargs)
        b = ha.run(**kwargs)
        assert a.best_value == b.best_value
        assert a.best_params == b.best_params
