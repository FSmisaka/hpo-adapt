"""TaskSpec: task description, model, dataset, objective and resource constraints."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class HyperparameterMetadata(BaseModel):
    """Metadata describing one hyperparameter that actually exists in the task."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = Field(description="'float' | 'int' | 'categorical'")
    default: Optional[Any] = None
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[list[Any]] = None
    log: bool = False
    description: str = ""


class ResourceConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_total_trials: int = Field(gt=0, description="Global trial budget across all phases")
    max_time_seconds: Optional[float] = None
    max_trials_per_phase: Optional[int] = None


class TaskSpec(BaseModel):
    """Structured task specification (spec.md core data object 1)."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: str
    description: str
    model_family: str
    dataset_signature: str
    objective_name: str
    direction: ObjectiveDirection
    objective: str = Field(description="Registry key of the objective function to optimize")
    available_hyperparameters: list[HyperparameterMetadata] = Field(default_factory=list)
    resource_constraints: ResourceConstraints
    allowed_optimizers: list[str] = Field(default_factory=lambda: ["RANDOM", "TPE"])

    @property
    def objective_direction(self) -> ObjectiveDirection:
        return self.direction
