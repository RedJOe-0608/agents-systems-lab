"""Event vocabulary used for tracing and replay."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4


class EventKind(StrEnum):
    TASK_STARTED = "task.started"
    AGENT_DELEGATED = "agent.delegated"
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    CHECKPOINTED = "runtime.checkpointed"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Immutable event suitable for tracing or durable replay."""

    task_id: UUID
    kind: EventKind
    payload: Mapping[str, Any] = field(default_factory=dict)
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
