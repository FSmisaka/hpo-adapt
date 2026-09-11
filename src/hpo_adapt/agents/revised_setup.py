"""Revised Setup Agent (spec.md 模块四).

Receives Current Setup + Diagnostic State + RL Action + Relevant Memory and
produces S_{t+1}. The RL Controller only decides *what* to do; this agent
determines *how* to execute the intervention and must output a Setup that
again passes the Tool Validator before any HPO run.
"""

import math
from typing import Any, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from hpo_adapt.agents.llm import LLMClient, extract_json
from hpo_adapt.diagnostic.metrics import ParamTransform
from hpo_adapt.schemas import (
    DiagnosticState,
    Distribution,
    HyperparameterSpec,
    RLAction,
    Setup,
    TaskSpec,
    TrialLog,
)
from hpo_adapt.schemas.action import ActionType
from hpo_adapt.validator import parse_setup_json, require_valid_setup

REVISED_SETUP_PROMPT = """You are the Revised HPO Setup Agent.

Your task is to revise the current HPO setup based on structured
diagnostic evidence and a setup intervention selected by an RL controller.

You must NOT execute HPO yourself.

INPUTS
------

1. Current Setup
2. Diagnostic State
3. RL-selected Action
4. Relevant historical experiences retrieved from Memory
5. Allowed optimizers and setup constraints

CURRENT SETUP
-------------

{{CURRENT_SETUP}}

DIAGNOSTIC STATE
----------------

{{DIAGNOSTIC_STATE}}

RL ACTION
---------

{{RL_ACTION}}

RELEVANT MEMORY
---------------

{{MEMORY}}

YOUR TASK
---------

1. Interpret the diagnostic evidence.

2. Determine which component of the current setup should be changed.

3. Apply the requested intervention conservatively.

4. Preserve all unchanged setup components whenever possible.

5. Produce the revised setup.

ACTION DEFINITIONS
------------------

EXPAND_SEARCH_SPACE:
- Identify the target hyperparameter.
- Expand its range toward the side indicated by the good-trial
  distribution.
- Do not expand unrelated parameters.

SHIFT_SEARCH_REGION:
- Move the search interval toward the empirically observed good region.
- Preserve sufficient exploration range.

SHRINK_SEARCH_SPACE:
- Narrow the range around the observed good region.
- Only do this when evidence indicates that focusing the search is
  beneficial.

FREEZE_HYPERPARAMETER:
- Fix a low-importance hyperparameter to a reasonable value.
- Prefer its default value or the value supported by high-performing
  configurations.

INCREASE_EXPLORATION:
- Modify the optimizer or optimizer-specific exploration settings.
- Only select another optimizer from the allowed optimizer list.

INCREASE_BUDGET:
- Increase the remaining trial/time budget within the global resource
  constraint.

STOP:
- Return the current setup and mark the optimization process as terminated.

RULES
-----

- Never introduce hyperparameters that do not exist.
- Never violate hard resource constraints.
- Never create an invalid conditional search space.
- Never change unrelated setup components without evidence.
- Do not blindly trust a single diagnostic metric.
- Prefer changes supported by multiple diagnostic signals.
- Preserve the original setup in the output for auditability.
- Provide concise rationale and confidence, not hidden chain-of-thought.

OUTPUT
------

Return ONLY a valid JSON object following RevisedSetupSchema."""


class InvalidActionError(Exception):
    """Raised when an RLAction cannot be executed structurally (invalid action)."""


class ChangeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hyperparameter: str
    old: Optional[dict[str, Any]] = None
    new: Optional[dict[str, Any]] = None
    reason: str = ""


class RevisedSetupResult(BaseModel):
    """Revised Setup output schema (spec.md 模块四)."""

    model_config = ConfigDict(extra="forbid")

    base_setup_id: str
    action: str
    changes: list[ChangeRecord] = Field(default_factory=list)
    new_setup: Setup
    confidence: Optional[float] = None
    assumptions: list[str] = Field(default_factory=list)


class RevisedSetupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expand_log_factor: float = 10.0
    expand_linear_fraction: float = 0.5
    shrink_margin: float = 0.25
    min_range_ratio: float = 0.05  # never shrink below 5% of current span
    default_budget_increase: int = 10


class RevisedSetupAgent(Protocol):
    def revise(
        self,
        task: TaskSpec,
        current_setup: Setup,
        diagnostic_state: DiagnosticState,
        action: RLAction,
        memory_context: Optional[str] = None,
        remaining_global_trials: Optional[int] = None,
    ) -> RevisedSetupResult: ...


class HeuristicRevisedSetupAgent:
    """Deterministic conservative executor for the 7 intervention types."""

    def __init__(self, config: RevisedSetupConfig | None = None) -> None:
        self.config = config or RevisedSetupConfig()

    # ------------------------------------------------------------------ utils
    def _metadata_bounds(self, task: TaskSpec, name: str) -> tuple[Optional[float], Optional[float]]:
        for meta in task.available_hyperparameters:
            if meta.name == name:
                return meta.low, meta.high
        return None, None

    def _clip(
        self,
        value: float,
        low: Optional[float],
        high: Optional[float],
        spec: HyperparameterSpec,
    ) -> float:
        if low is not None:
            value = max(value, low)
        if high is not None:
            value = min(value, high)
        return value

    def _spec_dict(self, spec: HyperparameterSpec) -> dict[str, Any]:
        return spec.model_dump(
            exclude={"rationale", "confidence"}, exclude_none=True, mode="json"
        )

    def _finalize_int(self, spec: HyperparameterSpec, value: float) -> float:
        return float(math.ceil(value)) if spec.type == "int" else value

    def _next_ids(self, current: Setup) -> tuple[str, int]:
        phase = current.phase + 1
        return f"setup_{phase:03d}", phase

    # ---------------------------------------------------------------- actions
    def _expand(
        self, task: TaskSpec, setup: Setup, action: RLAction, state: DiagnosticState
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        spec = setup.hyperparameter(action.target or "")
        if spec is None or spec.low is None or spec.high is None or spec.type == "categorical":
            raise InvalidActionError(
                f"EXPAND_SEARCH_SPACE requires a numerical hyperparameter target, got '{action.target}'"
            )
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        new_spec = new_setup.hyperparameter(spec.name)
        assert new_spec is not None
        direction = action.parameters.get("direction", "both")
        meta_low, meta_high = self._metadata_bounds(task, spec.name)
        is_log = new_spec.distribution == Distribution.LOG_UNIFORM
        old_low, old_high = float(spec.low), float(spec.high)
        width = old_high - old_low
        reasons: list[str] = []
        new_low, new_high = old_low, old_high
        if is_log:
            if direction in ("upper", "both"):
                new_high = self._clip(old_high * self.config.expand_log_factor, None, meta_high, new_spec)
            if direction in ("lower", "both"):
                new_low = self._clip(old_low / self.config.expand_log_factor, meta_low, None, new_spec)
            reasons.append(
                f"Expanded log-range by factor {self.config.expand_log_factor} ({direction})."
            )
        else:
            extra = width * self.config.expand_linear_fraction
            if direction in ("upper", "both"):
                new_high = self._clip(old_high + extra, None, meta_high, new_spec)
            if direction in ("lower", "both"):
                new_low = self._clip(old_low - extra, meta_low, None, new_spec)
            reasons.append(f"Expanded range by {self.config.expand_linear_fraction:.0%} of width ({direction}).")
        final_low = self._finalize_int(new_spec, new_low)
        final_high = self._finalize_int(new_spec, new_high)
        if final_low >= final_high:
            raise InvalidActionError(f"EXPAND_SEARCH_SPACE collapsed range for '{spec.name}'")
        new_spec.low, new_spec.high = final_low, final_high
        change = ChangeRecord(
            hyperparameter=spec.name,
            old=self._spec_dict(spec),
            new=self._spec_dict(new_spec),
            reason=(action.reason or "good trials concentrated near the boundary") + " " + " ".join(reasons),
        )
        return new_setup, [change], []

    def _shift(
        self, task: TaskSpec, setup: Setup, action: RLAction, state: DiagnosticState
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        spec = setup.hyperparameter(action.target or "")
        if spec is None or spec.type == "categorical":
            raise InvalidActionError(
                f"SHIFT_SEARCH_REGION requires a numerical hyperparameter target, got '{action.target}'"
            )
        gr = state.search_space.good_region.get(spec.name)
        target_region = action.parameters.get("target_region")
        if target_region is None and gr is not None and gr.status.value == "VALID":
            target_region = [gr.lower, gr.upper]
        if not target_region:
            raise InvalidActionError(
                f"SHIFT_SEARCH_REGION for '{spec.name}' lacks a target region (no good-region data)"
            )
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        new_spec = new_setup.hyperparameter(spec.name)
        assert new_spec is not None and spec.low is not None and spec.high is not None

        is_log = new_spec.distribution == Distribution.LOG_UNIFORM
        to_t = (lambda v: math.log(max(v, 1e-300))) if is_log else (lambda v: float(v))
        from_t = (lambda v: float(math.exp(v))) if is_log else (lambda v: float(v))
        center_t = to_t((target_region[0] + target_region[1]) / 2.0) if not is_log else to_t(
            math.sqrt(max(target_region[0], 1e-300) * max(target_region[1], 1e-300))
        )
        half_width_t = (to_t(spec.high) - to_t(spec.low)) / 2.0
        meta_low, meta_high = self._metadata_bounds(task, spec.name)
        new_low_t = center_t - half_width_t
        new_high_t = center_t + half_width_t
        if meta_low is not None:
            new_low_t = max(new_low_t, to_t(meta_low))
        if meta_high is not None:
            new_high_t = min(new_high_t, to_t(meta_high))
        new_spec.low = self._finalize_int(new_spec, from_t(new_low_t))
        new_spec.high = self._finalize_int(new_spec, from_t(new_high_t))
        if new_spec.low >= new_spec.high:
            raise InvalidActionError(f"SHIFT_SEARCH_REGION collapsed range for '{spec.name}'")
        change = ChangeRecord(
            hyperparameter=spec.name,
            old=self._spec_dict(spec),
            new=self._spec_dict(new_spec),
            reason=(action.reason or "moved interval toward the observed good region")
            + f" target_region={list(target_region)}; width preserved.",
        )
        return new_setup, [change], []

    def _shrink(
        self, task: TaskSpec, setup: Setup, action: RLAction, state: DiagnosticState
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        spec = setup.hyperparameter(action.target or "")
        if spec is None or spec.type == "categorical":
            raise InvalidActionError(
                f"SHRINK_SEARCH_SPACE requires a numerical hyperparameter target, got '{action.target}'"
            )
        gr = state.search_space.good_region.get(spec.name)
        target_region = action.parameters.get("target_region")
        if target_region is None and gr is not None and gr.status.value == "VALID":
            target_region = [gr.lower, gr.upper]
        if not target_region:
            raise InvalidActionError(
                f"SHRINK_SEARCH_SPACE for '{spec.name}' lacks a target region (no good-region data)"
            )
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        new_spec = new_setup.hyperparameter(spec.name)
        assert new_spec is not None and spec.low is not None and spec.high is not None
        t = ParamTransform(new_spec)
        span_u = t.to_unit(spec.high) - t.to_unit(spec.low)
        margin = (t.to_unit(target_region[1]) - t.to_unit(target_region[0])) * self.config.shrink_margin
        min_half = span_u * self.config.min_range_ratio / 2.0
        low_u = t.to_unit(target_region[0]) - margin
        high_u = t.to_unit(target_region[1]) + margin
        if high_u - low_u < 2 * min_half:
            mid = (low_u + high_u) / 2.0
            low_u, high_u = mid - min_half, mid + min_half
        new_spec.low = self._finalize_int(new_spec, t.from_unit(max(0.0, low_u)))
        new_spec.high = self._finalize_int(new_spec, t.from_unit(min(1.0, high_u)))
        if new_spec.low >= new_spec.high:
            raise InvalidActionError(f"SHRINK_SEARCH_SPACE collapsed range for '{spec.name}'")
        change = ChangeRecord(
            hyperparameter=spec.name,
            old=self._spec_dict(spec),
            new=self._spec_dict(new_spec),
            reason=(action.reason or "focused the range around the good region")
            + f" target_region={list(target_region)} with {self.config.shrink_margin:.0%} margin.",
        )
        return new_setup, [change], []

    def _freeze(
        self, task: TaskSpec, setup: Setup, action: RLAction, state: DiagnosticState
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        name = action.target or ""
        spec = setup.hyperparameter(name)
        if spec is None:
            if name in setup.frozen:
                raise InvalidActionError(f"hyperparameter '{name}' is already frozen")
            raise InvalidActionError(f"FREEZE_HYPERPARAMETER target '{name}' not in search space")
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        new_setup.search_space = [s for s in new_setup.search_space if s.name != name]
        value = action.parameters.get("value")
        assumptions: list[str] = []
        if value is None:
            # Prefer default value (spec.md FREEZE_HYPERPARAMETER), then good-region evidence.
            if spec.default is not None:
                value = spec.default
            else:
                meta_default = next(
                    (m.default for m in task.available_hyperparameters if m.name == name), None
                )
                if meta_default is not None:
                    value = meta_default
                    assumptions.append(f"Frozen value for '{name}' taken from task metadata default.")
                else:
                    gr = state.search_space.good_region.get(name)
                    cat = state.search_space.good_region_categorical.get(name)
                    if gr is not None and gr.status.value == "VALID" and gr.mean is not None:
                        value = gr.mean
                    elif cat is not None and cat.status.value == "VALID" and cat.dominant_category:
                        value = cat.dominant_category
                    else:
                        raise InvalidActionError(f"FREEZE_HYPERPARAMETER: no value available for '{name}'")
                    assumptions.append(f"Frozen value for '{name}' taken from good-region statistics.")
        if spec.type == "int":
            value = int(round(float(value)))
        new_setup.frozen[name] = value
        change = ChangeRecord(
            hyperparameter=name,
            old=self._spec_dict(spec),
            new={"value": value},
            reason=(action.reason or "low measured importance; frozen to reduce search dimensionality")
            + f" value={value}.",
        )
        if not new_setup.search_space:
            raise InvalidActionError("FREEZE_HYPERPARAMETER cannot freeze the last search-space parameter")
        return new_setup, [change], assumptions

    def _increase_exploration(
        self, task: TaskSpec, setup: Setup, action: RLAction, state: DiagnosticState
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        old = dict(new_setup.optimizer.config)
        assumptions: list[str] = []
        if new_setup.optimizer.name.upper() == "TPE":
            current = int(new_setup.optimizer.config.get("n_startup_trials", 10))
            new_setup.optimizer.config["n_startup_trials"] = min(current * 2, max(10, new_setup.budget.max_trials))
            reason = "Doubled TPE n_startup_trials for stronger random exploration."
        else:
            if "TPE" in task.allowed_optimizers:
                new_setup.optimizer = new_setup.optimizer.model_copy(update={"name": "TPE"})
                new_setup.optimizer.config.setdefault("n_startup_trials", 10)
                reason = "Switched optimizer to TPE with a random startup phase."
            else:
                reason = "Optimizer already maximally exploratory within the allowed list; no change."
                assumptions.append("INCREASE_EXPLORATION had no stronger option available.")
        change = ChangeRecord(
            hyperparameter="optimizer",
            old={"name": setup.optimizer.name, **old},
            new={"name": new_setup.optimizer.name, **new_setup.optimizer.config},
            reason=(action.reason or "increase optimizer exploration") + " " + reason,
        )
        return new_setup, [change], assumptions

    def _increase_budget(
        self,
        task: TaskSpec,
        setup: Setup,
        action: RLAction,
        state: DiagnosticState,
        remaining_global_trials: Optional[int] = None,
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        additional = int(action.parameters.get("additional_trials", self.config.default_budget_increase))
        limit = task.resource_constraints.max_trials_per_phase
        if remaining_global_trials is not None:
            additional = min(additional, max(remaining_global_trials, 0))
        if limit is not None:
            additional = min(additional, max(limit - new_setup.budget.max_trials, 0))
        if additional <= 0:
            raise InvalidActionError("INCREASE_BUDGET: no remaining global trial budget")
        old_budget = new_setup.budget.max_trials
        new_setup.budget.max_trials = old_budget + additional
        change = ChangeRecord(
            hyperparameter="budget",
            old={"max_trials": old_budget},
            new={"max_trials": new_setup.budget.max_trials},
            reason=(action.reason or "clear improvement with budget nearly exhausted")
            + f" +{additional} trials.",
        )
        return new_setup, [change], []

    def _stop(
        self, task: TaskSpec, setup: Setup, action: RLAction, state: DiagnosticState
    ) -> tuple[Setup, list[ChangeRecord], list[str]]:
        new_setup = setup.model_copy(deep=True)
        new_setup.setup_id, new_setup.phase = self._next_ids(setup)
        new_setup.terminated = True
        change = ChangeRecord(
            hyperparameter="process",
            old={"terminated": False},
            new={"terminated": True},
            reason=(action.reason or "low expected benefit of further adaptation") + " Episode terminated.",
        )
        return new_setup, [change], []

    # ----------------------------------------------------------------- entry
    def revise(
        self,
        task: TaskSpec,
        current_setup: Setup,
        diagnostic_state: DiagnosticState,
        action: RLAction,
        memory_context: Optional[str] = None,
        remaining_global_trials: Optional[int] = None,
    ) -> RevisedSetupResult:
        _ = memory_context
        if action.action == ActionType.INCREASE_BUDGET:
            new_setup, changes, assumptions = self._increase_budget(
                task, current_setup, action, diagnostic_state, remaining_global_trials
            )
        else:
            handler = {
                ActionType.EXPAND_SEARCH_SPACE: self._expand,
                ActionType.SHIFT_SEARCH_REGION: self._shift,
                ActionType.SHRINK_SEARCH_SPACE: self._shrink,
                ActionType.FREEZE_HYPERPARAMETER: self._freeze,
                ActionType.INCREASE_EXPLORATION: self._increase_exploration,
                ActionType.STOP: self._stop,
            }[action.action]
            new_setup, changes, assumptions = handler(task, current_setup, action, diagnostic_state)
        assumptions = list(assumptions)
        if memory_context:
            assumptions.append(f"Memory context used: {memory_context}")
        require_valid_setup(new_setup, task)
        return RevisedSetupResult(
            base_setup_id=current_setup.setup_id,
            action=action.action.value,
            changes=changes,
            new_setup=new_setup,
            confidence=action.confidence if action.confidence is not None else 0.6,
            assumptions=assumptions,
        )


class LLMRevisedSetupAgent:
    """LLM backend using the spec prompt; output parsed and validated."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def revise(
        self,
        task: TaskSpec,
        current_setup: Setup,
        diagnostic_state: DiagnosticState,
        action: RLAction,
        memory_context: Optional[str] = None,
        remaining_global_trials: Optional[int] = None,
    ) -> RevisedSetupResult:
        import json

        prompt = (
            REVISED_SETUP_PROMPT.replace("{{CURRENT_SETUP}}", json.dumps(current_setup.model_dump(), default=str))
            .replace("{{DIAGNOSTIC_STATE}}", json.dumps(diagnostic_state.model_dump(), default=str))
            .replace("{{RL_ACTION}}", json.dumps(action.model_dump(by_alias=True), default=str))
            .replace("{{MEMORY}}", memory_context or "none")
        )
        allowed = {
            "allowed_optimizers": task.allowed_optimizers,
            "resource_constraints": task.resource_constraints.model_dump(),
            "available_hyperparameters": [m.name for m in task.available_hyperparameters],
        }
        reply = self.client.complete(prompt, json.dumps(allowed, indent=2))
        data = extract_json(reply)
        if not isinstance(data, dict) or "new_setup" not in data:
            raise ValueError("LLM revised setup missing 'new_setup'")
        new_setup = parse_setup_json(data["new_setup"])
        require_valid_setup(new_setup, task)
        return RevisedSetupResult(
            base_setup_id=data.get("base_setup_id", current_setup.setup_id),
            action=data.get("action", action.action.value),
            changes=[ChangeRecord.model_validate(c) for c in data.get("changes", [])],
            new_setup=new_setup,
            confidence=data.get("confidence"),
            assumptions=list(data.get("assumptions", [])),
        )


def create_revised_setup_agent(
    backend: str = "heuristic", client: Optional[LLMClient] = None
) -> RevisedSetupAgent:
    if backend == "heuristic":
        return HeuristicRevisedSetupAgent()
    if backend == "llm":
        if client is None:
            from hpo_adapt.agents.llm import OpenAICompatClient

            client = OpenAICompatClient()
        return LLMRevisedSetupAgent(client)
    raise ValueError(f"unknown backend '{backend}'")
