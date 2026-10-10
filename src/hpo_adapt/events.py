"""Live-run event layer.

The closed loop emits lifecycle events so that external observers (the local
dashboard, custom loggers, ...) can follow a run in real time without
touching the pipeline's return values::

    run_start -> initial_setup -> round_start -> trial* -> round_end
              -> diagnostic -> [reward] -> action -> revision
              -> round_start -> ... -> episode_end

An event sink is any callable ``(event_type: str, payload: dict) -> None``.
Sinks must never raise into the pipeline; ``emit`` swallows and records such
errors so observability can not break an experiment.
"""

import sys
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


EventPayload = dict[str, Any]


@runtime_checkable
class EventSinkLike(Protocol):
    def __call__(self, event_type: str, payload: EventPayload) -> None: ...


class EventSink(BaseModel):
    """Optional observer attached to AdaptationPipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handler: Optional[Callable[[str, EventPayload], None]] = None
    errors: list[str] = Field(default_factory=list)

    def __call__(self, event_type: str, payload: EventPayload) -> None:
        self.emit(event_type, payload)

    def emit(self, event_type: str, payload: EventPayload) -> None:
        if self.handler is None:
            return
        try:
            self.handler(event_type, payload)
        except Exception as exc:  # observability must never break the run
            msg = f"event sink error on '{event_type}': {exc!r}"
            self.errors.append(msg)
            print(f"[hpo-adapt] {msg}", file=sys.stderr)


def null_sink(event_type: str, payload: EventPayload) -> None:
    """Default no-op sink."""
