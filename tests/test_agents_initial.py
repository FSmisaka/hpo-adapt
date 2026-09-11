"""Stage 3 tests: Initial Setup Agent."""

import pytest

from hpo_adapt.agents.initial_setup import (
    INITIAL_SETUP_PROMPT,
    HeuristicInitialSetupAgent,
    LLMInitialSetupAgent,
)
from hpo_adapt.agents.llm import extract_json
from hpo_adapt.schemas import Distribution
from hpo_adapt.validator import validate_setup
from tests.test_hpo import make_task


class TestHeuristicInitialSetupAgent:
    def test_generates_valid_setup(self):
        task = make_task()
        setup = HeuristicInitialSetupAgent().generate(task)
        assert validate_setup(setup, task) == []
        assert setup.phase == 0
        assert setup.setup_id == "setup_000"
        assert setup.optimizer.name in task.allowed_optimizers
        assert {s.name for s in setup.search_space} == {
            "learning_rate", "batch_size", "optimizer_type", "dropout"
        }

    def test_log_distribution_for_wide_float(self):
        task = make_task()
        setup = HeuristicInitialSetupAgent().generate(task)
        lr = setup.hyperparameter("learning_rate")
        assert lr is not None
        assert lr.distribution == Distribution.LOG_UNIFORM

    def test_deterministic(self):
        task = make_task()
        s1 = HeuristicInitialSetupAgent().generate(task)
        s2 = HeuristicInitialSetupAgent().generate(task)
        assert s1 == s2

    def test_budget_respects_constraints(self):
        task = make_task()
        setup = HeuristicInitialSetupAgent().generate(task)
        assert setup.budget.max_trials <= task.resource_constraints.max_total_trials

    def test_memory_context_recorded_in_assumptions(self):
        task = make_task()
        setup = HeuristicInitialSetupAgent().generate(task, memory_context="similar task used lr in [1e-4, 1e-2]")
        assert any("memory" in a.lower() for a in setup.assumptions)

    def test_int_hyperparameters_get_integer_bounds(self):
        from hpo_adapt.schemas.task import HyperparameterMetadata

        task = make_task()
        task.available_hyperparameters = [
            HyperparameterMetadata(name="n_estimators", type="int", default=100, low=10, high=300)
        ]
        setup = HeuristicInitialSetupAgent().generate(task)
        spec = setup.hyperparameter("n_estimators")
        assert spec is not None
        assert float(spec.low).is_integer() and float(spec.high).is_integer()
        assert validate_setup(setup, task) == []


class TestLLMInitialSetupAgent:
    def test_parses_valid_llm_output(self):
        task = make_task()
        reply = """
        Here is the setup.
        ```json
        {
          "setup_id": "setup_000",
          "phase": 0,
          "search_space": [
            {"name": "learning_rate", "type": "float", "low": 1e-5, "high": 1e-2,
             "distribution": "log_uniform", "default": 1e-3,
             "rationale": "wide log range", "confidence": 0.8},
            {"name": "optimizer_type", "type": "categorical",
             "choices": ["adam", "sgd"], "default": "adam",
             "rationale": "both supported", "confidence": 0.7}
          ],
          "optimizer": {"name": "TPE", "config": {"n_startup_trials": 10}},
          "budget": {"max_trials": 20},
          "evaluation": {"protocol": "holdout", "direction": "maximize", "seed": 0},
          "assumptions": ["lr most impactful"],
          "confidence": 0.75
        }
        ```
        """

        class FakeClient:
            def complete(self, system: str, user: str) -> str:
                assert system == INITIAL_SETUP_PROMPT
                assert "learning_rate" in user  # hyperparameter metadata was provided
                return reply

        setup = LLMInitialSetupAgent(FakeClient()).generate(task)
        assert validate_setup(setup, task) == []

    def test_rejects_invented_hyperparameters(self):
        task = make_task()
        reply = (
            '{"setup_id": "setup_000", "phase": 0, "search_space": ['
            '{"name": "momentum", "type": "float", "low": 0.0, "high": 1.0}], '
            '"optimizer": {"name": "TPE"}, "budget": {"max_trials": 10}, '
            '"evaluation": {"protocol": "holdout", "direction": "maximize"}}'
        )

        class FakeClient:
            def complete(self, system: str, user: str) -> str:
                return reply

        with pytest.raises(ValueError, match="invalid"):
            LLMInitialSetupAgent(FakeClient()).generate(task)

    def test_extract_json_variants(self):
        assert extract_json('blah {"a": 1} blah') == {"a": 1}
        assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
        with pytest.raises(Exception):
            extract_json("no json here")
