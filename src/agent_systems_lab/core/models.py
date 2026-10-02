"""Core task models shared by framework and custom implementations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4


class TaskStatus(StrEnum):
    """Lifecycle state for an agent task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A provider-independent request submitted to an agent runtime."""

    objective: str
    task_id: UUID = field(default_factory=uuid4)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("objective must not be empty")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Terminal output returned by an agent runtime."""

    task_id: UUID
    status: TaskStatus
    summary: str
    artifacts: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            raise ValueError("a task result must have a terminal status")
        object.__setattr__(self, "metrics", _immutable_mapping(self.metrics))
