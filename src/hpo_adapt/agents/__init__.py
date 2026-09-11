"""LLM-driven Setup Agents (spec.md 模块一 / 模块四)."""

from hpo_adapt.agents.initial_setup import (
    INITIAL_SETUP_PROMPT,
    HeuristicInitialSetupAgent,
    LLMInitialSetupAgent,
    create_initial_setup_agent,
)
from hpo_adapt.agents.llm import (
    LLMClient,
    LLMError,
    OpenAICompatClient,
    extract_json,
    is_llm_configured,
)
from hpo_adapt.agents.revised_setup import (
    REVISED_SETUP_PROMPT,
    ChangeRecord,
    HeuristicRevisedSetupAgent,
    InvalidActionError,
    LLMRevisedSetupAgent,
    RevisedSetupConfig,
    RevisedSetupResult,
    create_revised_setup_agent,
)

__all__ = [
    "ChangeRecord",
    "HeuristicInitialSetupAgent",
    "HeuristicRevisedSetupAgent",
    "INITIAL_SETUP_PROMPT",
    "InvalidActionError",
    "LLMClient",
    "LLMError",
    "LLMInitialSetupAgent",
    "LLMRevisedSetupAgent",
    "OpenAICompatClient",
    "REVISED_SETUP_PROMPT",
    "RevisedSetupConfig",
    "RevisedSetupResult",
    "create_initial_setup_agent",
    "create_revised_setup_agent",
    "extract_json",
    "is_llm_configured",
]
