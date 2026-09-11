"""Closed-loop adaptation pipeline (spec.md 核心数据流).

    Task -> Initial Setup -> HPO -> Diagnostic -> RL Action -> Revised Setup
         -> HPO -> Memory -> ...

Orchestration rules enforced here:

- Every Setup passes the Tool Validator before HPO runs.
- One Experience is written to Memory only after a complete adaptation step
  (S_t -> HPO -> D_t -> A_t -> S_{t+1} -> Outcome).
- Global trial budget from TaskSpec.resource_constraints is respected across
  all phases.
- Invalid actions receive the fixed penalty reward and do not consume budget.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from hpo_adapt.agents.initial_setup import create_initial_setup_agent
from hpo_adapt.agents.revised_setup import (
    InvalidActionError,
    create_revised_setup_agent,
)
from hpo_adapt.diagnostic import DiagnosticEngine
from hpo_adapt.hpo.runner import run_hpo
from hpo_adapt.memory import MemoryStore, new_experience_id
from hpo_adapt.rl import QController, QLearningConfig, RewardConfig, compute_reward
from hpo_adapt.rules import RuleEngine
from hpo_adapt.schemas import (
    ActionType,
    ActionRecord,
    Diagnosis,
    DiagnosticState,
    Experience,
    ObjectiveDirection,
    Outcome,
    RLAction,
    Setup,
    TaskSpec,
    TaskSignature,
    TrialLog,
)
from hpo_adapt.validator import require_valid_setup


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = 5
    agent_backend: str = "heuristic"  # "heuristic" | "llm"
    memory_path: str = ":memory:"
    qtable_path: Optional[str] = None
    seed: int = 0
    max_invalid_action_attempts: int = 5


class StepRecord(BaseModel):
    """Audit record of one adaptation step."""

    model_config = ConfigDict(extra="forbid")

    round_index: int = 0
    setup_id: str = ""
    trials_used: int = 0
    phase_best: Optional[float] = None  # internal (maximized) space
    cumulative_best: Optional[float] = None
    diagnosis: Optional[str] = None
    action: Optional[dict[str, Any]] = None
    action_valid: bool = True
    reward: Optional[float] = None
    terminated: bool = False
    setup_after_id: Optional[str] = None
    changes: list[dict[str, Any]] = Field(default_factory=list)
    search_space: list[dict[str, Any]] = Field(
        default_factory=list, description="Search space snapshot of this round's setup"
    )
    diagnostic_state: Optional[dict[str, Any]] = Field(
        default=None, description="Full DiagnosticState of this round"
    )


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[StepRecord] = Field(default_factory=list)
    total_trials_used: int = 0
    best_value: Optional[float] = None  # internal space
    best_params: Optional[dict[str, Any]] = None
    best_value_raw: Optional[float] = None  # original task direction
    terminated: bool = False
    q_table_size: int = 0
    experiences_recorded: int = 0


class AdaptationPipeline:
    def __init__(
        self,
        task: TaskSpec,
        config: Optional[PipelineConfig] = None,
        engine: Optional[DiagnosticEngine] = None,
        controller: Optional[QController] = None,
        memory: Optional[MemoryStore] = None,
        rule_engine: Optional[RuleEngine] = None,
    ) -> None:
        self.task = task
        self.config = config or PipelineConfig()
        self.engine = engine or DiagnosticEngine()
        self.rule_engine = rule_engine or RuleEngine()
        self.controller = controller or QController(
            rule_engine=self.rule_engine,
            config=QLearningConfig(seed=self.config.seed),
        )
        self.memory = memory or MemoryStore(self.config.memory_path)
        self.initial_agent = create_initial_setup_agent(self.config.agent_backend)
        self.revised_agent = create_revised_setup_agent(self.config.agent_backend)
        self.reward_config = RewardConfig()

    # ------------------------------------------------------------------ helpers
    def _task_signature(self) -> TaskSignature:
        return TaskSignature(
            task_type=self.task.task_type,
            model_family=self.task.model_family,
            dataset_signature=self.task.dataset_signature,
        )

    def _to_internal(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return value if self.task.direction == ObjectiveDirection.MAXIMIZE else -value

    def _to_raw(self, value: Optional[float]) -> Optional[float]:
        return self._to_internal(value)  # symmetric negation

    def _clamp_phase_budget(self, setup: Setup, remaining: int) -> Setup:
        if setup.budget.max_trials > remaining:
            setup = setup.model_copy(deep=True)
            setup.budget.max_trials = max(remaining, 1)
        return setup

    # --------------------------------------------------------------------- run
    def run(self) -> PipelineResult:
        task = self.task
        total_budget = task.resource_constraints.max_total_trials
        used = 0
        result = PipelineResult()

        # Task -> Memory Retrieval -> Initial Setup Agent
        setup = require_valid_setup(
            self.initial_agent.generate(task, self.memory.format_context(self._task_signature())),
            task,
        )

        cumulative_best: Optional[float] = None
        best_params: Optional[dict[str, Any]] = None
        pending: Optional[dict[str, Any]] = None  # awaiting next-phase outcome
        previous_diagnostic: Optional[DiagnosticState] = None

        for round_index in range(self.config.max_rounds):
            remaining = total_budget - used
            if remaining <= 0:
                break
            exec_setup = self._clamp_phase_budget(setup, remaining)

            # ---- HPO phase with the current (validated) setup
            hpo_result = run_hpo(task, exec_setup)
            used_before = used
            used += len(hpo_result.trials)
            phase_best = self._to_internal(hpo_result.best_value)
            if phase_best is not None and (cumulative_best is None or phase_best > cumulative_best):
                cumulative_best = phase_best
                best_params = dict(hpo_result.best_params or {})

            # ---- Diagnostic Engine (deterministic); budget status is GLOBAL
            diagnostic = self.engine.run(
                hpo_result.trials,
                exec_setup,
                task.direction,
                total_budget=total_budget,
                trials_used_before=used_before,
            )
            current_key = self.controller.observe(diagnostic, exec_setup)

            # ---- close out the previous adaptation step (reward + memory)
            if pending is not None:
                reward = compute_reward(
                    pending["best_before"],
                    cumulative_best if cumulative_best is not None else pending["best_before"],
                    len(hpo_result.trials),
                    total_budget,
                    self.reward_config,
                )
                self.controller.update(
                    pending["key"], pending["action_type"], reward, current_key
                )
                self._write_experience(
                    pending, diagnostic=None, reward=reward,
                    best_after=cumulative_best if cumulative_best is not None else pending["best_before"],
                    trials_used=len(hpo_result.trials),
                )
                result.steps[-1].reward = reward

            # ---- rule diagnosis + memory retrieval for this step
            rule_match = self.rule_engine.primary(diagnostic, exec_setup, previous_diagnostic)
            diagnosis = Diagnosis(
                primary=rule_match.diagnosis if rule_match else "NO_ACTION_RECOMMENDED",
                confidence=rule_match.confidence if rule_match else 0.5,
            )
            memory_context = self.memory.format_context(self._task_signature(), diagnostic)
            rule_recommendation = self.rule_engine.recommend(
                diagnostic, exec_setup, previous_diagnostic
            )
            previous_diagnostic = diagnostic

            # ---- RL Action selection + Revised Setup execution
            action, revision, step = self._act(
                exec_setup, diagnostic, rule_recommendation, memory_context, remaining
            )
            step.round_index = round_index
            step.setup_id = exec_setup.setup_id
            step.trials_used = len(hpo_result.trials)
            step.phase_best = phase_best
            step.cumulative_best = cumulative_best
            step.diagnosis = diagnosis.primary
            step.search_space = [s.model_dump(mode="json") for s in exec_setup.search_space]
            step.diagnostic_state = diagnostic.model_dump(by_alias=True, mode="json")
            result.steps.append(step)

            if not step.action_valid:
                # invalid action: penalized inside _act; retry next round with same setup
                pending = None
                continue

            assert action is not None
            step.action = action.model_dump(by_alias=True)

            if revision is not None and revision.new_setup.terminated:
                reward = 0.0
                self.controller.update(
                    current_key, action.action, reward, None, terminal=True
                )
                step.reward = reward
                step.terminated = True
                step.setup_after_id = revision.new_setup.setup_id
                self._write_experience(
                    {
                        "setup": exec_setup,
                        "diagnostic": diagnostic,
                        "diagnosis": diagnosis,
                        "action": action,
                        "key": current_key,
                        "action_type": action.action,
                        "best_before": cumulative_best,
                    },
                    diagnostic=None,
                    reward=reward,
                    best_after=cumulative_best,
                    trials_used=0,
                )
                result.terminated = True
                break

            assert revision is not None
            step.setup_after_id = revision.new_setup.setup_id
            step.changes = [c.model_dump(mode="json") for c in revision.changes]
            pending = {
                "setup": exec_setup,
                "diagnostic": diagnostic,
                "diagnosis": diagnosis,
                "action": action,
                "key": current_key,
                "action_type": action.action,
                "best_before": cumulative_best,
            }
            setup = revision.new_setup  # S_{t+1}

        result.total_trials_used = used
        result.best_value = cumulative_best
        result.best_params = best_params
        result.best_value_raw = self._to_raw(cumulative_best)
        result.q_table_size = len(self.controller.q_table)
        result.experiences_recorded = self.memory.experience_count()
        if self.config.qtable_path:
            self.controller.save(self.config.qtable_path)
        return result

    # ----------------------------------------------------------------- acting
    def _act(
        self,
        setup: Setup,
        diagnostic: DiagnosticState,
        rule_recommendation: Optional[RLAction],
        memory_context: str,
        remaining_global_trials: int,
    ) -> tuple[Optional[RLAction], Optional[Any], StepRecord]:
        """Select an action and execute it; handles invalid-action retries."""
        for _ in range(self.config.max_invalid_action_attempts):
            action = self.controller.select_action(
                diagnostic, setup, rule_recommendation=rule_recommendation
            )
            key = self.controller.last_key
            assert key is not None
            try:
                revision = self.revised_agent.revise(
                    self.task,
                    setup,
                    diagnostic,
                    action,
                    memory_context=memory_context,
                    remaining_global_trials=remaining_global_trials,
                )
                require_valid_setup(revision.new_setup, self.task)
                return action, revision, StepRecord(round_index=0, setup_id=setup.setup_id, action_valid=True)
            except InvalidActionError:
                reward = compute_reward(0.0, 0.0, 0, 1, self.reward_config, invalid=True)
                self.controller.update(key, action.action, reward, key, terminal=False)
                self._write_experience(
                    {
                        "setup": setup,
                        "diagnostic": diagnostic,
                        "diagnosis": Diagnosis(primary="INVALID_ACTION", confidence=0.0),
                        "action": action,
                        "key": key,
                        "action_type": action.action,
                        "best_before": None,
                    },
                    diagnostic=None,
                    reward=reward,
                    best_after=None,
                    trials_used=0,
                    invalid=True,
                    setup_after=setup,
                )
        return None, None, StepRecord(
            round_index=0, setup_id=setup.setup_id, action_valid=False
        )

    # ----------------------------------------------------------------- memory
    def _write_experience(
        self,
        pending: dict[str, Any],
        diagnostic: Optional[DiagnosticState],
        reward: float,
        best_after: Optional[float],
        trials_used: int,
        invalid: bool = False,
        setup_after: Optional[Setup] = None,
    ) -> None:
        best_before = pending.get("best_before")
        rel = None
        if best_before is not None and best_after is not None:
            rel = (best_after - best_before) / (abs(best_before) + 1e-8)
        experience = Experience(
            experience_id=new_experience_id(),
            task_signature=self._task_signature(),
            setup_before=pending["setup"].model_dump(mode="json"),
            diagnostic_state=pending["diagnostic"].model_dump(by_alias=True, mode="json"),
            diagnosis=pending["diagnosis"],
            action=ActionRecord(
                type=pending["action_type"].value, target=pending["action"].target
            ),
            setup_after=(
                setup_after or pending.get("setup_after") or pending["setup"]
            ).model_dump(mode="json"),
            outcome=Outcome(
                best_before=best_before,
                best_after=best_after,
                relative_improvement=rel,
                budget_consumed=trials_used,
            ),
            reward=reward,
            success=bool(rel is not None and rel > 0),
            rationale=pending["action"].reason or "",
        )
        self.memory.record(experience)


# ----------------------------------------------------------------------- CLI


def _phase_budget(max_total_trials: int) -> int:
    return max(5, min(50, max_total_trials // 3))


def synthetic_mlp_task(max_total_trials: int = 120) -> TaskSpec:
    from hpo_adapt.schemas.task import HyperparameterMetadata, ResourceConstraints

    return TaskSpec(
        task_id="task_synth_001",
        task_type="classification",
        description="Synthetic MLP-style training task with 4 tunable hyperparameters.",
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
                name="optimizer_type", type="categorical", default="adam", choices=["adam", "sgd"]
            ),
            HyperparameterMetadata(name="dropout", type="float", default=0.1, low=0.0, high=0.9),
        ],
        resource_constraints=ResourceConstraints(
            max_total_trials=max_total_trials, max_trials_per_phase=_phase_budget(max_total_trials)
        ),
        allowed_optimizers=["RANDOM", "TPE"],
    )


def branin_task(max_total_trials: int = 120) -> TaskSpec:
    from hpo_adapt.schemas.task import HyperparameterMetadata, ResourceConstraints

    return TaskSpec(
        task_id="task_branin_001",
        task_type="regression",
        description="Branin function minimization benchmark.",
        model_family="Analytic",
        dataset_signature="branin",
        objective_name="branin_value",
        direction=ObjectiveDirection.MINIMIZE,
        objective="branin",
        available_hyperparameters=[
            HyperparameterMetadata(name="x1", type="float", default=-3.0, low=-5.0, high=10.0),
            HyperparameterMetadata(name="x2", type="float", default=7.5, low=0.0, high=15.0),
        ],
        resource_constraints=ResourceConstraints(
            max_total_trials=max_total_trials, max_trials_per_phase=_phase_budget(max_total_trials)
        ),
        allowed_optimizers=["RANDOM", "TPE"],
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnostic-driven adaptive HPO closed loop")
    parser.add_argument("--task", choices=["synthetic_mlp", "branin"], default="synthetic_mlp")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-total-trials", type=int, default=120)
    parser.add_argument("--backend", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--memory-path", default=None, help="SQLite memory file (default: in-memory)")
    parser.add_argument("--qtable-path", default=None, help="Where to save the learned Q-table (JSON)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    args = parser.parse_args(argv)

    task = (
        synthetic_mlp_task(args.max_total_trials)
        if args.task == "synthetic_mlp"
        else branin_task(args.max_total_trials)
    )
    config = PipelineConfig(
        max_rounds=args.max_rounds,
        agent_backend=args.backend,
        memory_path=args.memory_path or ":memory:",
        qtable_path=args.qtable_path,
        seed=args.seed,
    )
    pipeline = AdaptationPipeline(task, config)
    result = pipeline.run()

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"Task: {task.task_id} ({task.direction.value})  backend={config.agent_backend}")
        for step in result.steps:
            action = (step.action or {}).get("action", "INVALID")
            target = (step.action or {}).get("candidate_hyperparameter")
            print(
                f"round {step.round_index}: setup={step.setup_id} trials={step.trials_used} "
                f"phase_best={step.phase_best} diagnosis={step.diagnosis} "
                f"action={action}@{target} reward={step.reward} terminated={step.terminated}"
            )
        print(
            f"total_trials={result.total_trials_used} best(raw)={result.best_value_raw} "
            f"best_params={result.best_params} q_states={result.q_table_size} "
            f"experiences={result.experiences_recorded}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
