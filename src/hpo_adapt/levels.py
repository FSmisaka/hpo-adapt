"""Discretized level extraction from DiagnosticState (spec.md 模块三 State).

The rule-based adaptation (Stage 4) and the RL state representation
(Stage 7) both consume three-level discretizations of the continuous
diagnostic metrics, so the thresholds live in one shared place.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from hpo_adapt.schemas import DiagnosticState, MetricStatus, safe_get


class Level(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WidthLevel(str, Enum):
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"


class LevelConfig(BaseModel):
    """Discretization thresholds (documented defaults; all overridable)."""

    model_config = ConfigDict(extra="forbid")

    importance_low: float = 0.10
    importance_high: float = 0.30
    boundary_low: float = 0.20
    boundary_high: float = 0.50
    width_narrow: float = 0.30
    width_wide: float = 0.70
    improvement_low: float = 0.001  # spec stagnation_relative_threshold
    improvement_high: float = 0.01
    diversity_low: float = 0.30
    diversity_high: float = 0.70
    coverage_low: float = 0.40
    coverage_high: float = 0.70
    budget_low: float = 0.20
    budget_high: float = 0.50
    marginal_gain_low: float = 1e-4
    marginal_gain_high: float = 1e-3
    budget_consumed_high: float = 0.80


def _three_level(value: float | None, low: float, high: float) -> Level | None:
    if value is None:
        return None
    if value < low:
        return Level.LOW
    if value >= high:
        return Level.HIGH
    return Level.MEDIUM


def importance_level(value: float | None, cfg: LevelConfig | None = None) -> Level | None:
    cfg = cfg or LevelConfig()
    return _three_level(value, cfg.importance_low, cfg.importance_high)


def boundary_level(value: float | None, cfg: LevelConfig | None = None) -> Level | None:
    cfg = cfg or LevelConfig()
    return _three_level(value, cfg.boundary_low, cfg.boundary_high)


def width_level(value: float | None, cfg: LevelConfig | None = None) -> WidthLevel | None:
    cfg = cfg or LevelConfig()
    if value is None:
        return None
    if value < cfg.width_narrow:
        return WidthLevel.NARROW
    if value >= cfg.width_wide:
        return WidthLevel.WIDE
    return WidthLevel.MEDIUM


def global_state_levels(
    state: DiagnosticState, cfg: LevelConfig | None = None
) -> dict[str, str | bool | None]:
    """Discretized Global State (spec.md 模块三):

    stagnation, recent_improvement, configuration_diversity, coverage,
    budget_remaining, marginal_gain — each low/medium/high (stagnation: 0/1).
    """
    cfg = cfg or LevelConfig()
    opt, ss, bud = state.optimization, state.search_space, state.budget
    div = ss.configuration_diversity
    return {
        "stagnation": opt.stagnation.value if opt.stagnation.status == MetricStatus.VALID else None,
        "recent_improvement": _lvl_name(
            _three_level(
                opt.relative_improvement.value
                if opt.relative_improvement.status == MetricStatus.VALID
                else None,
                cfg.improvement_low,
                cfg.improvement_high,
            )
        ),
        "configuration_diversity": _lvl_name(
            _three_level(
                div.ratio if div.status == MetricStatus.VALID else None,
                cfg.diversity_low,
                cfg.diversity_high,
            )
        ),
        "coverage": _lvl_name(
            _three_level(
                ss.coverage.overall if ss.coverage.status == MetricStatus.VALID else None,
                cfg.coverage_low,
                cfg.coverage_high,
            )
        ),
        "budget_remaining": _lvl_name(
            _three_level(
                bud.budget_remaining.value
                if bud.budget_remaining.status == MetricStatus.VALID
                else None,
                cfg.budget_low,
                cfg.budget_high,
            )
        ),
        "marginal_gain": _lvl_name(
            _three_level(
                opt.marginal_gain.value if opt.marginal_gain.status == MetricStatus.VALID else None,
                cfg.marginal_gain_low,
                cfg.marginal_gain_high,
            )
        ),
    }


def _lvl_name(level: Level | WidthLevel | None) -> str | None:
    return level.value if level is not None else None


def diagnostic_signature(state: DiagnosticState, cfg: LevelConfig | None = None) -> dict:
    """Compact signature used for Memory retrieval (TaskSignature + DiagnosticSignature)."""
    levels = global_state_levels(state, cfg)
    sig: dict = dict(levels)
    # add dominant candidate context
    best_name, best = None, None
    for name, entry in state.search_space.importance.items():
        if entry.status != MetricStatus.VALID or entry.global_importance is None:
            continue
        if best is None or (entry.global_importance or 0.0) > (best.global_importance or 0.0):
            best_name, best = name, entry
    if best is not None and best_name is not None:
        bc = state.search_space.boundary_concentration.get(best_name)
        gr = state.search_space.good_region.get(best_name)
        sig["top_hyperparameter"] = best_name
        sig["importance"] = importance_level(best.global_importance, cfg)
        sig["boundary_concentration"] = (
            boundary_level(bc.good, cfg) if bc and bc.status == MetricStatus.VALID else None
        )
        sig["good_region_width"] = width_level(gr.width_ratio, cfg) if gr and gr.status == MetricStatus.VALID else None
    return sig
