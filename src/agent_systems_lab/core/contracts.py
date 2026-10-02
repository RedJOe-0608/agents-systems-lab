"""Replaceable boundaries around orchestration, context, execution, and tracing."""

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol

from agent_systems_lab.core.events import RuntimeEvent
from agent_systems_lab.core.models import TaskResult, TaskSpec


class AgentRuntime(Protocol):
    """Execute a task without exposing a framework-specific state object."""

    async def run(self, task: TaskSpec) -> TaskResult: ...


class Sandbox(Protocol):
    """Execute untrusted commands behind explicit capability boundaries."""

    async def execute(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> tuple[int, str, str]: ...


class ContextEngine(Protocol):
    """Select and recursively process external context."""

    async def query(self, objective: str, sources: Sequence[str]) -> str: ...


class TraceSink(Protocol):
    """Receive framework-neutral runtime events."""

    async def emit(self, event: RuntimeEvent) -> None: ...

    def stream(self, task_id: str) -> AsyncIterator[RuntimeEvent]: ...


class ModelGateway(Protocol):
    """Minimal model boundary used by future framework adapters."""

    async def invoke(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        tools: Sequence[Mapping[str, object]] = (),
    ) -> Mapping[str, object]: ...

