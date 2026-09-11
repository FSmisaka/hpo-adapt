"""RLAction: setup intervention selected by the RL Controller (spec.md core data object 5).

A_t = (ActionType_t, Target_t, Parameters_t)
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    EXPAND_SEARCH_SPACE = "EXPAND_SEARCH_SPACE"
    SHIFT_SEARCH_REGION = "SHIFT_SEARCH_REGION"
    SHRINK_SEARCH_SPACE = "SHRINK_SEARCH_SPACE"
    FREEZE_HYPERPARAMETER = "FREEZE_HYPERPARAMETER"
    INCREASE_EXPLORATION = "INCREASE_EXPLORATION"
    INCREASE_BUDGET = "INCREASE_BUDGET"
    STOP = "STOP"


HYPERPARAMETER_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.EXPAND_SEARCH_SPACE,
        ActionType.SHIFT_SEARCH_REGION,
        ActionType.SHRINK_SEARCH_SPACE,
        ActionType.FREEZE_HYPERPARAMETER,
    }
)


class RLAction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    action: ActionType
    target: Optional[str] = Field(
        default=None,
        alias="candidate_hyperparameter",
        description="Hyperparameter name, 'optimizer', 'budget', or None for STOP",
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    q_value: Optional[float] = None
    reason: Optional[str] = None
