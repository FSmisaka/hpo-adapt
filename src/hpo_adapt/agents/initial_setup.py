"""Initial Setup Agent (spec.md 模块一).

Responsible ONLY for the first HPO setup S_0. No trajectory exists yet, so
the agent reasons from task information, hyperparameter metadata, prior
knowledge and (optionally) memory / web-search context.

Two backends share the same interface:

- ``HeuristicInitialSetupAgent``: deterministic, metadata-driven. Default
  backend so the closed loop is runnable and testable without any API key.
- ``LLMInitialSetupAgent``: sends the spec prompt (verbatim below) together
  with the task inputs to an LLM and validates the JSON output through the
  Tool Validator.
"""

import math
from typing import Any, Optional, Protocol

from hpo_adapt.agents.llm import LLMClient, extract_json
from hpo_adapt.schemas import (
    BudgetSpec,
    Distribution,
    EvaluationSpec,
    HyperparameterSpec,
    ObjectiveDirection,
    OptimizerSpec,
    Setup,
    TaskSpec,
)
from hpo_adapt.validator import parse_setup_json, validate_setup

INITIAL_SETUP_PROMPT = """You are an HPO Initial Setup Agent.

Your goal is to construct a reasonable initial setup for hyperparameter
optimization for the given machine learning task.

You are NOT responsible for executing HPO. You only construct the setup.

INPUTS
------

1. Task description
2. Model / training code description
3. Dataset information
4. Optimization objective and direction
5. Computational constraints
6. Available hyperparameter metadata
7. Relevant prior knowledge retrieved from memory
8. Optional web-search results

YOUR TASK
---------

Step 1. Identify the tunable hyperparameters that are actually available
in the provided model/training code.

Step 2. Construct an initial search space.

For each hyperparameter specify:
- name
- type
- range / choices
- distribution
- default value
- optional conditional dependency
- brief rationale
- confidence in [0, 1]

Step 3. Select one HPO optimizer from the allowed optimizer list.

Step 4. Determine an initial HPO budget under the given computational constraints.

Step 5. Specify the evaluation protocol.

REQUIREMENTS
------------

- Never invent hyperparameters that do not exist in the provided system.
- Prefer a compact and meaningful search space.
- Use log-scale distributions for parameters spanning multiple orders of
  magnitude when appropriate.
- Do not unnecessarily include low-impact or purely deterministic parameters.
- All generated configurations must satisfy the provided constraints.
- Explicitly state assumptions.
- Provide concise rationales, not hidden chain-of-thought reasoning.

OUTPUT
------

Return ONLY a JSON object following the provided SetupSchema.

The output must contain:
- search_space
- optimizer
- budget
- evaluation
- assumptions
- confidence

The JSON must be valid and machine-readable."""


class InitialSetupAgent(Protocol):
    def generate(self, task: TaskSpec, memory_context: Optional[str] = None) -> Setup: ...


def _default_budget(task: TaskSpec) -> int:
    rc = task.resource_constraints
    if rc.max_trials_per_phase is not None:
        return rc.max_trials_per_phase
    quarter = max(10, rc.max_total_trials // 4)
    return max(5, min(quarter, rc.max_total_trials))


def _spec_from_metadata(meta: Any) -> HyperparameterSpec:
    """Derive a search-space entry from task hyperparameter metadata."""
    if meta.type == "categorical":
        return HyperparameterSpec(
            name=meta.name,
            type="categorical",
            choices=list(meta.choices or []),
            default=meta.default,
            rationale=f"Categories taken from task metadata for '{meta.name}'.",
            confidence=0.8 if meta.choices else 0.5,
        )
    low, high = meta.low, meta.high
    low_f: float
    high_f: float
    derived = low is None or high is None
    default = meta.default
    if derived:
        # No metadata bounds: derive a compact range around the default.
        if meta.type == "int":
            d_val = float(int(default) if default is not None else 32)
            low_f, high_f = float(max(1, int(d_val) // 4)), d_val * 4
        else:
            d_val = float(default) if default is not None else 0.5
            if d_val > 0:
                low_f, high_f = d_val / 10.0, d_val * 10.0
            else:
                low_f, high_f = 0.0, 1.0
    elif default is not None and meta.low is not None and meta.high is not None:
        # Compact, default-centered initial range within the metadata domain.
        if meta.log:
            d_val = float(default) if default > 0 else max(float(meta.low), 1e-6)
            low_f = max(float(meta.low), d_val / 10.0)
            high_f = min(float(meta.high), d_val * 10.0)
        else:
            margin = (float(meta.high) - float(meta.low)) / 4.0
            low_f = max(float(meta.low), float(default) - margin)
            high_f = min(float(meta.high), float(default) + margin)
        if low_f >= high_f:
            low_f, high_f = float(meta.low), float(meta.high)
    else:
        low_f, high_f = float(low or 0.0), float(high or 1.0)
    if meta.type == "int":
        # Integer hyperparameters need integer bounds (Tool Validator).
        low_f, high_f = float(math.floor(low_f)), float(math.ceil(high_f))
        if low_f >= high_f:
            high_f = low_f + 1.0
    spec = HyperparameterSpec(
        name=meta.name,
        type=meta.type,
        low=low_f,
        high=high_f,
        default=meta.default,
        rationale=(
            f"Compact range centered on the default value of '{meta.name}'."
            if default is not None
            else f"Range taken from task metadata for '{meta.name}'."
        ),
        confidence=0.6 if derived or default is None else 0.7,
    )
    if meta.log and spec.distribution == Distribution.UNIFORM:
        spec.distribution = Distribution.LOG_UNIFORM
    return spec


class HeuristicInitialSetupAgent:
    """Deterministic metadata-driven initial setup (default backend)."""

    def generate(self, task: TaskSpec, memory_context: Optional[str] = None) -> Setup:
        search_space = [_spec_from_metadata(meta) for meta in task.available_hyperparameters]
        optimizer_name = "TPE" if "TPE" in task.allowed_optimizers else task.allowed_optimizers[0]
        assumptions = [
            "Ranges derived from task metadata",
            "No HPO trajectory yet",
        ]
        if memory_context:
            assumptions.append(f"Prior knowledge from memory: {memory_context}")
        setup = Setup(
            setup_id="setup_000",
            phase=0,
            search_space=search_space,
            optimizer=OptimizerSpec(name=optimizer_name, config={"n_startup_trials": 10}),
            budget=BudgetSpec(max_trials=_default_budget(task)),
            evaluation=EvaluationSpec(
                protocol="holdout", direction=task.direction, seed=0
            ),
            assumptions=assumptions,
            confidence=0.7,
        )
        errors = validate_setup(setup, task)
        if errors:
            raise ValueError(f"heuristic initial setup invalid: {errors}")
        return setup


class LLMInitialSetupAgent:
    """LLM backend using the spec prompt; output parsed and validated."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def generate(self, task: TaskSpec, memory_context: Optional[str] = None) -> Setup:
        user_payload = {
            "task_description": task.description,
            "model_family": task.model_family,
            "dataset_information": task.dataset_signature,
            "objective": {"name": task.objective_name, "direction": task.direction.value},
            "computational_constraints": task.resource_constraints.model_dump(),
            "available_hyperparameter_metadata": [
                m.model_dump() for m in task.available_hyperparameters
            ],
            "allowed_optimizers": task.allowed_optimizers,
            "memory_prior_knowledge": memory_context or "none",
        }
        import json

        reply = self.client.complete(
            INITIAL_SETUP_PROMPT,
            json.dumps(user_payload, indent=2, default=str),
        )
        data = extract_json(reply)
        data.setdefault("setup_id", "setup_000")
        data.setdefault("phase", 0)
        setup = parse_setup_json(data)
        errors = validate_setup(setup, task)
        if errors:
            raise ValueError(f"LLM initial setup invalid: {errors}")
        return setup


def create_initial_setup_agent(
    backend: str = "heuristic", client: Optional[LLMClient] = None
) -> InitialSetupAgent:
    if backend == "heuristic":
        return HeuristicInitialSetupAgent()
    if backend == "llm":
        if client is None:
            from hpo_adapt.agents.llm import OpenAICompatClient

            client = OpenAICompatClient()
        return LLMInitialSetupAgent(client)
    raise ValueError(f"unknown backend '{backend}'")
