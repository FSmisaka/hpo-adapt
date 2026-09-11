"""RL state discretization (spec.md 模块三 — State).

s_t = (GlobalState_t, CandidateContext_t)

GlobalState: stagnation (0/1), recent_improvement / configuration_diversity /
coverage / budget_remaining / marginal_gain (each low/medium/high).

CandidateContext: for the TopK hyperparameters ranked by CandidateScore,
the (importance, boundary_concentration, good_region_width) levels.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from hpo_adapt.levels import LevelConfig, global_state_levels
from hpo_adapt.rules import RuleEngine
from hpo_adapt.schemas import DiagnosticState, MetricStatus, Setup

GLOBAL_KEYS = (
    "stagnation",
    "recent_improvement",
    "configuration_diversity",
    "coverage",
    "budget_remaining",
    "marginal_gain",
)


class RLStateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k_candidates: int = 2


def build_state(
    state: DiagnosticState,
    setup: Setup,
    rule_engine: RuleEngine,
    level_config: Optional[LevelConfig] = None,
    state_config: Optional[RLStateConfig] = None,
) -> dict[str, Any]:
    """Structured RL state: global levels + per-candidate levels."""
    level_config = level_config or LevelConfig()
    state_config = state_config or RLStateConfig()
    global_state = global_state_levels(state, level_config)
    candidates = rule_engine.rank_candidates(state)[: state_config.top_k_candidates]
    candidate_context = [
        (
            c.importance_level,
            c.boundary_level,
            c.width_level,
        )
        for c in candidates
    ]
    return {"global": global_state, "candidates": candidate_context}


def state_key(structured: dict[str, Any]) -> tuple:
    """Hashable flat key (strings only) for the tabular Q-table."""
    glob = structured["global"]
    global_part = tuple(str(glob.get(k)) for k in GLOBAL_KEYS)
    candidate_part = tuple(str(level) for cand in structured["candidates"] for level in cand)
    return global_part + candidate_part


def format_state_key(key: tuple) -> str:
    return "|".join(key)
