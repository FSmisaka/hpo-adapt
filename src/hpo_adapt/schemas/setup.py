"""Setup: current HPO setup — Search Space, Optimizer, Budget, Evaluation (spec.md core data object 2)."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from hpo_adapt.schemas.task import ObjectiveDirection


class Distribution(str, Enum):
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    CHOICE = "choice"


class HyperparameterSpec(BaseModel):
    """One entry of the search space."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = Field(description="'float' | 'int' | 'categorical'")
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[list[Any]] = None
    distribution: Optional[Distribution] = None
    default: Optional[Any] = None
    depends_on: Optional[dict[str, Any]] = Field(
        default=None, description="Optional conditional dependency, e.g. {'optimizer_type': 'sgd'}"
    )
    rationale: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.distribution is None:
            if self.type == "categorical":
                self.distribution = Distribution.CHOICE
            elif self.low is not None and self.low > 0 and self.high is not None:
                ratio = self.high / max(self.low, 1e-12)
                if ratio >= 100.0:
                    self.distribution = Distribution.LOG_UNIFORM
                else:
                    self.distribution = Distribution.UNIFORM
            else:
                self.distribution = Distribution.UNIFORM


class OptimizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    config: dict[str, Any] = Field(default_factory=dict, description="Optimizer-specific configuration")


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_trials: int = Field(gt=0, description="N_budget: maximum number of trials for this phase")
    max_time_seconds: Optional[float] = None


class EvaluationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: str = Field(description="e.g. 'holdout' | 'cv'")
    direction: ObjectiveDirection
    seed: int = 0
    params: dict[str, Any] = Field(default_factory=dict)


class Setup(BaseModel):
    """Structured HPO setup (spec.md core data object 2)."""

    model_config = ConfigDict(extra="forbid")

    setup_id: str
    phase: int = Field(default=0, ge=0)
    search_space: list[HyperparameterSpec]
    optimizer: OptimizerSpec
    budget: BudgetSpec
    evaluation: EvaluationSpec
    frozen: dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters fixed by FREEZE_HYPERPARAMETER actions"
    )
    assumptions: list[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    terminated: bool = Field(default=False, description="Set to True by the STOP action")

    def hyperparameter(self, name: str) -> Optional[HyperparameterSpec]:
        for spec in self.search_space:
            if spec.name == name:
                return spec
        return None

    def search_space_dict(self) -> dict[str, HyperparameterSpec]:
        return {spec.name: spec for spec in self.search_space}
