"""DiagnosticState: structured state extracted from Trial Logs (spec.md core data object 4).

Every metric carries a status of VALID or INSUFFICIENT_DATA so that downstream
modules never consume unreliable numbers (spec.md: 数据充分性与 Metric Status).
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class MetricStatus(str, Enum):
    VALID = "VALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class Metric(BaseModel):
    """A scalar metric with its validity status."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA

    @staticmethod
    def valid(value: float) -> "Metric":
        return Metric(value=float(value), status=MetricStatus.VALID)

    @staticmethod
    def insufficient() -> "Metric":
        return Metric(value=None, status=MetricStatus.INSUFFICIENT_DATA)


class BoolMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Optional[bool] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA

    @staticmethod
    def valid(value: bool) -> "BoolMetric":
        return BoolMetric(value=bool(value), status=MetricStatus.VALID)

    @staticmethod
    def insufficient() -> "BoolMetric":
        return BoolMetric(value=None, status=MetricStatus.INSUFFICIENT_DATA)


class OptimizationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_objective: Metric = Field(default_factory=Metric.insufficient)
    recent_improvement: Metric = Field(default_factory=Metric.insufficient)
    relative_improvement: Metric = Field(default_factory=Metric.insufficient)
    stagnation: BoolMetric = Field(default_factory=BoolMetric.insufficient)
    marginal_gain: Metric = Field(default_factory=Metric.insufficient)


class ImportanceEntry(BaseModel):
    """Per-hyperparameter importance: fANOVA (global) + LPI (local), spec.md."""

    model_config = ConfigDict(extra="forbid")

    global_importance: Optional[float] = None
    local_importance: Optional[float] = None
    rank: Optional[int] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA


class BoundaryConcentrationEntry(BaseModel):
    """Boundary concentration over all trials and top-q% good trials."""

    model_config = ConfigDict(extra="forbid")

    all: Optional[float] = None
    good: Optional[float] = None
    upper: Optional[float] = None
    lower: Optional[float] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA


class GoodRegionEntry(BaseModel):
    """Good region for numerical parameters (mean/quantiles/width ratio)."""

    model_config = ConfigDict(extra="forbid")

    mean: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    width_ratio: Optional[float] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA


class CategoricalGoodRegionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_frequency: dict[str, float] = Field(default_factory=dict)
    dominant_category: Optional[str] = None
    dominant_ratio: Optional[float] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA


class SearchSpaceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    importance: dict[str, ImportanceEntry] = Field(default_factory=dict)
    boundary_concentration: dict[str, BoundaryConcentrationEntry] = Field(default_factory=dict)
    good_region: dict[str, GoodRegionEntry] = Field(default_factory=dict)
    good_region_categorical: dict[str, CategoricalGoodRegionEntry] = Field(default_factory=dict)
    configuration_diversity: "ConfigurationDiversity" = Field(
        default_factory=lambda: ConfigurationDiversity()
    )
    coverage: "CoverageStatus" = Field(default_factory=lambda: CoverageStatus())


class ConfigurationDiversity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent: Optional[float] = None
    global_: Optional[float] = Field(default=None, alias="global")
    ratio: Optional[float] = None
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CoverageStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: Optional[float] = None
    per_hyperparameter: dict[str, Optional[float]] = Field(default_factory=dict)
    status: MetricStatus = MetricStatus.INSUFFICIENT_DATA


class BudgetStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_trials: Optional[int] = None
    completed_trials: Optional[int] = None
    budget_consumed: Metric = Field(default_factory=Metric.insufficient)
    budget_remaining: Metric = Field(default_factory=Metric.insufficient)
    marginal_gain_per_trial: Metric = Field(default_factory=Metric.insufficient)
    marginal_gain_per_second: Metric = Field(default_factory=Metric.insufficient)


class DiagnosticMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_complete_trials: int = 0
    num_failed_trials: int = 0
    num_pruned_trials: int = 0
    failure_rate: Optional[float] = None
    analysis_window: int = 0
    data_sufficient: bool = False


class DiagnosticState(BaseModel):
    """Structured diagnostic state (spec.md core data object 4)."""

    model_config = ConfigDict(extra="forbid")

    optimization: OptimizationStatus = Field(default_factory=OptimizationStatus)
    search_space: SearchSpaceStatus = Field(default_factory=SearchSpaceStatus)
    budget: BudgetStatus = Field(default_factory=BudgetStatus)
    meta: DiagnosticMeta = Field(default_factory=DiagnosticMeta)


class DiagnosticConfig(BaseModel):
    """All thresholds and defaults from spec.md."""

    model_config = ConfigDict(extra="forbid")

    min_trials_for_basic_metrics: int = 5
    min_trials_for_importance: int = 30
    min_trials_for_good_region: int = 20
    min_trials_for_stagnation: int = 20
    analysis_window: int = 20
    epsilon_y: float = 1e-8
    stagnation_relative_threshold: float = 0.001
    good_trial_quantile: float = 0.10
    boundary_threshold: float = 0.05
    good_region_quantiles: tuple[float, float] = (0.10, 0.90)
    coverage_bins: int = 10
    surrogate_random_state: int = 0
    surrogate_n_estimators: int = 100
    lpi_grid_size: int = 20
    lpi_top_configs: int = 10


def safe_get(data: dict[str, Any], path: str) -> Any:
    """Helper for reading nested diagnostic dicts, e.g. safe_get(ds, 'optimization.stagnation')."""
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


SearchSpaceStatus.model_rebuild()
