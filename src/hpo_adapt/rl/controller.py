"""Tabular Q-Learning RL Controller (spec.md 模块三).

Diagnostic Engine = Observe; RL Controller = Decide; Revised Setup Agent =
Execute. The controller ONLY selects one of the 7 high-level interventions
plus its target; it never modifies a Setup itself.

Update rule:

    Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [r_t + gamma * max_a' Q(s_{t+1}, a') - Q(s_t, a_t)]

Action selection: epsilon-greedy with decay; ties broken by the rule-based
recommendation first (deterministic).
"""

import json
import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from hpo_adapt.rl.state import RLStateConfig, build_state, state_key
from hpo_adapt.rules import RuleEngine
from hpo_adapt.schemas import (
    ActionType,
    DiagnosticState,
    RLAction,
    Setup,
)

ALL_ACTIONS: tuple[ActionType, ...] = (
    ActionType.EXPAND_SEARCH_SPACE,
    ActionType.SHIFT_SEARCH_REGION,
    ActionType.SHRINK_SEARCH_SPACE,
    ActionType.FREEZE_HYPERPARAMETER,
    ActionType.INCREASE_EXPLORATION,
    ActionType.INCREASE_BUDGET,
    ActionType.STOP,
)
_HYPER_ACTIONS = {
    ActionType.EXPAND_SEARCH_SPACE,
    ActionType.SHIFT_SEARCH_REGION,
    ActionType.SHRINK_SEARCH_SPACE,
    ActionType.FREEZE_HYPERPARAMETER,
}


class QLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 0


class QController:
    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        config: Optional[QLearningConfig] = None,
        state_config: Optional[RLStateConfig] = None,
    ) -> None:
        self.rule_engine = rule_engine or RuleEngine()
        self.config = config or QLearningConfig()
        self.state_config = state_config or RLStateConfig()
        self.q_table: dict[tuple, dict[str, float]] = {}
        self.epsilon = self.config.epsilon_start
        self.rng = random.Random(self.config.seed)
        self._last_structured_state: Optional[dict] = None
        self._last_action: Optional[ActionType] = None
        self.last_key: Optional[tuple] = None

    # ------------------------------------------------------------------ state
    def observe(self, state: DiagnosticState, setup: Setup) -> tuple:
        """Build and remember the tabular state for the current diagnostic."""
        structured = build_state(state, setup, self.rule_engine, state_config=self.state_config)
        self._last_structured_state = structured
        self.last_key = state_key(structured)
        return self.last_key

    # ---------------------------------------------------------------- actions
    def _stop_available(self, state: DiagnosticState) -> bool:
        """Evidence gate for STOP (spec pattern table, row 7).

        STOP is only a legal choice when the diagnostic evidence supports
        "low expected benefit of further adaptation": stagnation observed AND
        a majority of the global budget already consumed. This prevents
        unguided (epsilon-exploration) premature termination.
        """
        stag = state.optimization.stagnation
        if stag.status.value != "VALID" or not bool(stag.value):
            return False
        consumed = state.budget.budget_consumed
        if consumed.status.value == "VALID" and consumed.value is not None:
            return float(consumed.value) >= 0.5
        return True

    def _available_actions(self, state: DiagnosticState) -> list[ActionType]:
        candidates = self.rule_engine.rank_candidates(state)
        if not candidates:
            actions = [a for a in ALL_ACTIONS if a not in _HYPER_ACTIONS]
        else:
            actions = list(ALL_ACTIONS)
        if not self._stop_available(state):
            actions = [a for a in actions if a != ActionType.STOP]
        return actions

    def _q_row(self, key: tuple) -> dict[str, float]:
        return self.q_table.setdefault(key, {a.value: 0.0 for a in ALL_ACTIONS})

    def select_action(
        self,
        state: DiagnosticState,
        setup: Setup,
        rule_recommendation: Optional[RLAction] = None,
    ) -> RLAction:
        key = self.observe(state, setup)
        row = self._q_row(key)
        available = self._available_actions(state)

        recommended = rule_recommendation.action if rule_recommendation is not None else None
        if self.rng.random() < self.epsilon:
            chosen = self.rng.choice(available)
        else:
            # arg-max over available actions; ties broken by rule recommendation, then fixed order
            best_q = max(row[a.value] for a in available)
            tied = [a for a in available if abs(row[a.value] - best_q) < 1e-12]
            tied.sort(key=lambda a: (a is not recommended, ALL_ACTIONS.index(a)))
            chosen = tied[0]

        action = self._materialize(chosen, state, setup, row, recommended)
        self._last_action = chosen
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return action

    def _materialize(
        self,
        chosen: ActionType,
        state: DiagnosticState,
        setup: Setup,
        row: dict[str, float],
        recommended: Optional[ActionType],
    ) -> RLAction:
        """Attach target and parameters to the chosen action type."""
        q_value = row[chosen.value]
        values = [row[a.value] for a in ALL_ACTIONS]
        positives = [v for v in values if v > 0]
        if chosen == ActionType.STOP or chosen in (ActionType.INCREASE_EXPLORATION,):
            target = None if chosen == ActionType.STOP else "optimizer"
        elif chosen == ActionType.INCREASE_BUDGET:
            target = "budget"
        else:
            candidates = self.rule_engine.rank_candidates(state)
            if not candidates:
                # no valid hyperparameter context: materialize without target (will be invalid)
                return RLAction(action=chosen, target=None, confidence=1.0 / len(ALL_ACTIONS), q_value=q_value)
            top = candidates[0]
            target = top.name

        parameters: dict = {}
        if chosen == ActionType.EXPAND_SEARCH_SPACE and target is not None:
            bc = state.search_space.boundary_concentration.get(target)
            if bc is not None and bc.status.value == "VALID":
                parameters["direction"] = "upper" if (bc.upper or 0.0) >= (bc.lower or 0.0) else "lower"
        elif chosen in (ActionType.SHIFT_SEARCH_REGION, ActionType.SHRINK_SEARCH_SPACE) and target is not None:
            gr = state.search_space.good_region.get(target)
            if gr is not None and gr.status.value == "VALID":
                parameters["target_region"] = [gr.lower, gr.upper]

        if positives:
            confidence = max(0.0, q_value) / sum(positives)
        else:
            confidence = 1.0 / len(ALL_ACTIONS)
        reason = "tabular Q-learning selection"
        if recommended == chosen:
            reason += "; agrees with rule-based recommendation"
        return RLAction(
            action=chosen,
            target=target,
            parameters=parameters,
            confidence=round(min(max(confidence, 0.0), 1.0), 4),
            q_value=round(q_value, 6),
            reason=reason,
        )

    # ---------------------------------------------------------------- update
    def update(
        self,
        prev_key: tuple,
        action: ActionType,
        reward: float,
        next_key: Optional[tuple],
        terminal: bool = False,
    ) -> float:
        """One tabular Q-learning update; returns the new Q value."""
        cfg = self.config
        row = self._q_row(prev_key)
        old = row[action.value]
        if terminal or next_key is None:
            target = reward
        else:
            next_row = self._q_row(next_key)
            target = reward + cfg.discount_factor * max(next_row.values())
        new_q = old + cfg.learning_rate * (target - old)
        row[action.value] = new_q
        return new_q

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        data = {
            "q_table": {format_key(k): v for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "config": self.config.model_dump(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.q_table = {parse_key(k): v for k, v in data["q_table"].items()}
        self.epsilon = float(data.get("epsilon", self.config.epsilon_end))
        if "config" in data:
            self.config = QLearningConfig.model_validate(data["config"])

    def policy_summary(self) -> dict[str, dict[str, float]]:
        return {format_key(k): dict(v) for k, v in self.q_table.items()}


def format_key(key: tuple) -> str:
    return "|".join(key)


def parse_key(text: str) -> tuple:
    return tuple(text.split("|"))
