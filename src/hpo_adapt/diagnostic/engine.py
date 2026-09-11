"""Diagnostic Engine (spec.md 模块二).

Deterministic Python module turning Trial Logs into a structured
DiagnosticState that directly supports the next Setup decision. No LLM is
involved in computing any metric.
"""

import math

import numpy as np

from hpo_adapt.diagnostic.importance import compute_importance
from hpo_adapt.diagnostic.metrics import (
    PreparedData,
    bin_occupancy,
    boundary_mask,
    good_indices,
    is_numeric,
    mixed_pairwise_distance,
    numeric_values,
    param_vector,
    prepare,
    unit_values,
)
from hpo_adapt.schemas import (
    BoundaryConcentrationEntry,
    BudgetStatus,
    CategoricalGoodRegionEntry,
    ConfigurationDiversity,
    CoverageStatus,
    DiagnosticConfig,
    DiagnosticMeta,
    DiagnosticState,
    GoodRegionEntry,
    ImportanceEntry,
    Metric,
    MetricStatus,
    ObjectiveDirection,
    OptimizationStatus,
    Setup,
    TrialLog,
)


class DiagnosticEngine:
    """Deterministic TrialLog + Setup -> DiagnosticState."""

    def __init__(self, config: DiagnosticConfig | None = None) -> None:
        self.config = config or DiagnosticConfig()

    def run(
        self,
        trials: list[TrialLog],
        setup: Setup,
        direction: ObjectiveDirection,
        *,
        total_budget: int | None = None,
        trials_used_before: int = 0,
    ) -> DiagnosticState:
        """Compute the DiagnosticState.

        Budget semantics (spec.md: Budget = Initial Setup + Trial Logs):
        when ``total_budget`` is given (adaptation loop), consumed/remaining
        are computed against the GLOBAL trial budget, counting the trials of
        all previous phases via ``trials_used_before``. When it is omitted,
        the per-phase Setup budget is used (single-phase usage).
        """
        cfg = self.config
        data = prepare(trials, direction)
        n = len(data.completed)
        w = min(cfg.analysis_window, max(n - 1, 0))

        optimization = self._optimization_status(data)
        search_space = self._search_space_status(data, setup)
        budget = self._budget_status(data, setup, w, total_budget, trials_used_before)
        meta = DiagnosticMeta(
            num_complete_trials=n,
            num_failed_trials=data.num_failed,
            num_pruned_trials=data.num_pruned,
            failure_rate=data.failure_rate,
            analysis_window=cfg.analysis_window,
            data_sufficient=n >= cfg.min_trials_for_basic_metrics,
        )
        return DiagnosticState(
            optimization=optimization,
            search_space=search_space,
            budget=budget,
            meta=meta,
        )

    # ------------------------------------------------------------- optimization
    def _optimization_status(self, data: PreparedData) -> OptimizationStatus:
        cfg = self.config
        n = data.y_tilde.size
        status = OptimizationStatus()
        if n >= cfg.min_trials_for_basic_metrics and n > 0:
            status.best_objective = Metric.valid(float(data.best_series[-1]))
        w = cfg.analysis_window
        if n > w and n >= cfg.min_trials_for_stagnation:
            b_t = float(data.best_series[-1])
            b_tw = float(data.best_series[-1 - w])
            improvement = b_t - b_tw
            rel = improvement / (abs(b_tw) + cfg.epsilon_y)
            status.recent_improvement = Metric.valid(improvement)
            status.relative_improvement = Metric.valid(rel)
            status.stagnation.value = bool(rel < cfg.stagnation_relative_threshold)
            status.stagnation.status = MetricStatus.VALID
            status.marginal_gain = Metric.valid(improvement / w)
        elif n >= cfg.min_trials_for_stagnation:
            # t < w: stagnation is explicitly false (spec.md Stagnation).
            status.stagnation = BoolMetricValid(False)
        return status

    # ------------------------------------------------------------- search space
    def _search_space_status(self, data: PreparedData, setup: Setup):
        cfg = self.config
        n = len(data.completed)
        search_space = setup.search_space

        importance: dict[str, ImportanceEntry] = {}
        if n >= cfg.min_trials_for_importance:
            raw = compute_importance(data.completed, data.y_tilde, search_space, cfg)
            ranked = sorted(
                search_space, key=lambda s: (-raw[s.name]["global_importance"], s.name)
            )
            for rank, spec in enumerate(ranked, start=1):
                entry = raw[spec.name]
                importance[spec.name] = ImportanceEntry(
                    global_importance=round(entry["global_importance"], 6),
                    local_importance=round(entry["local_importance"], 6),
                    rank=rank,
                    status=MetricStatus.VALID,
                )
        else:
            for spec in search_space:
                importance[spec.name] = ImportanceEntry()

        boundary: dict[str, BoundaryConcentrationEntry] = {}
        good_region: dict[str, GoodRegionEntry] = {}
        good_region_cat: dict[str, CategoricalGoodRegionEntry] = {}
        if n >= cfg.min_trials_for_good_region:
            good_idx = good_indices(data.y_tilde, cfg.good_trial_quantile)
            good_trials = [data.completed[i] for i in good_idx]
            for spec in search_space:
                if is_numeric(spec):
                    z_all = unit_values(data.completed, spec)
                    z_good = unit_values(good_trials, spec)
                    low_all, up_all = boundary_mask(z_all, cfg.boundary_threshold)
                    low_good, up_good = boundary_mask(z_good, cfg.boundary_threshold)
                    boundary[spec.name] = BoundaryConcentrationEntry(
                        all=float(np.mean(low_all | up_all)),
                        good=float(np.mean(low_good | up_good)),
                        upper=float(np.mean(up_good)),
                        lower=float(np.mean(low_good)),
                        status=MetricStatus.VALID,
                    )
                    vals_good = numeric_values(good_trials, spec)
                    q_low, q_high = np.quantile(vals_good, cfg.good_region_quantiles)
                    width_ratio = _width_ratio(spec, q_low, q_high)
                    good_region[spec.name] = GoodRegionEntry(
                        mean=float(np.mean(vals_good)),
                        lower=float(q_low),
                        upper=float(q_high),
                        width_ratio=width_ratio,
                        status=MetricStatus.VALID,
                    )
                else:
                    cats = list(dict.fromkeys(spec.choices or []))
                    counts = {c: 0 for c in cats}
                    for v in param_vector(good_trials, spec.name):
                        if v in counts:
                            counts[v] += 1
                    freq = {str(c): counts[c] / len(good_trials) for c in cats}
                    dominant = max(cats, key=lambda c: (counts[c], str(c))) if cats else None
                    good_region_cat[spec.name] = CategoricalGoodRegionEntry(
                        category_frequency=freq,
                        dominant_category=dominant,
                        dominant_ratio=freq.get(str(dominant), 0.0) if dominant else None,
                        status=MetricStatus.VALID,
                    )

        diversity = ConfigurationDiversity()
        if n >= cfg.min_trials_for_basic_metrics and n >= 2:
            window = min(cfg.analysis_window, n)
            recent_trials = data.completed[-window:]
            div_recent = mixed_pairwise_distance(recent_trials, search_space)
            div_global = mixed_pairwise_distance(data.completed, search_space)
            diversity.recent = div_recent
            diversity.global_ = div_global
            diversity.ratio = div_recent / (div_global + cfg.epsilon_y)
            diversity.status = MetricStatus.VALID

        coverage = CoverageStatus()
        numeric_specs = [s for s in search_space if is_numeric(s)]
        if numeric_specs and n >= cfg.min_trials_for_basic_metrics:
            per_hp: dict[str, float | None] = {}
            for spec in numeric_specs:
                per_hp[spec.name] = bin_occupancy(
                    unit_values(data.completed, spec), cfg.coverage_bins
                )
            coverage.per_hyperparameter = per_hp
            coverage.overall = float(np.mean([v or 0.0 for v in per_hp.values()]))
            coverage.status = MetricStatus.VALID

        from hpo_adapt.schemas import SearchSpaceStatus

        return SearchSpaceStatus(
            importance=importance,
            boundary_concentration=boundary,
            good_region=good_region,
            good_region_categorical=good_region_cat,
            configuration_diversity=diversity,
            coverage=coverage,
        )

    # ------------------------------------------------------------------ budget
    def _budget_status(
        self,
        data: PreparedData,
        setup: Setup,
        w: int,
        total_budget: int | None = None,
        trials_used_before: int = 0,
    ) -> BudgetStatus:
        cfg = self.config
        n = len(data.completed)
        if total_budget is not None:
            total = total_budget
            completed = trials_used_before + n
        else:
            total = setup.budget.max_trials
            completed = n
        status = BudgetStatus(total_trials=total, completed_trials=completed)
        if total > 0 and completed > 0:
            consumed = completed / total
            status.budget_consumed = Metric.valid(consumed)
            status.budget_remaining = Metric.valid(max(0.0, 1.0 - consumed))
        if data.y_tilde.size > cfg.analysis_window and data.y_tilde.size >= cfg.min_trials_for_stagnation:
            b_t = float(data.best_series[-1])
            b_tw = float(data.best_series[-1 - cfg.analysis_window])
            status.marginal_gain_per_trial = Metric.valid((b_t - b_tw) / cfg.analysis_window)
            durations = [t.duration for t in data.completed if t.duration is not None]
            if len(durations) == n and n > cfg.analysis_window:
                t_now = sum(durations)
                t_prev = sum(durations[: n - cfg.analysis_window])
                denom = (t_now - t_prev) + 1e-8
                if denom > 0:
                    status.marginal_gain_per_second = Metric.valid((b_t - b_tw) / denom)
        return status


def BoolMetricValid(value: bool):  # noqa: N802 (small helper)
    from hpo_adapt.schemas import BoolMetric

    return BoolMetric(value=value, status=MetricStatus.VALID)


def _width_ratio(spec, q_low: float, q_high: float) -> float:
    """WidthRatio_j = (q_high - q_low) / (u - l), computed in log space for log params."""
    if spec.distribution is not None and spec.distribution.value == "log_uniform":
        num = math.log(max(q_high, q_low)) - math.log(max(min(q_low, q_high), 1e-300))
        den = math.log(float(spec.high)) - math.log(float(spec.low))
    else:
        num = q_high - q_low
        den = float(spec.high) - float(spec.low)
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def diagnose(
    trials: list[TrialLog],
    setup: Setup,
    direction: ObjectiveDirection,
    config: DiagnosticConfig | None = None,
    *,
    total_budget: int | None = None,
    trials_used_before: int = 0,
) -> DiagnosticState:
    """Functional entry point."""
    return DiagnosticEngine(config).run(
        trials,
        setup,
        direction,
        total_budget=total_budget,
        trials_used_before=trials_used_before,
    )
