from uuid import uuid4

import pytest

from agent_systems_lab.core.models import TaskResult, TaskSpec, TaskStatus


def test_task_spec_requires_an_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        TaskSpec(objective="  ")


def test_task_spec_copies_metadata() -> None:
    metadata = {"source": "fixture"}
    task = TaskSpec(objective="Inspect the repository", metadata=metadata)
    metadata["source"] = "mutated"

    assert task.metadata["source"] == "fixture"


def test_task_result_requires_terminal_status() -> None:
    with pytest.raises(ValueError, match="terminal"):
        TaskResult(task_id=uuid4(), status=TaskStatus.RUNNING, summary="still working")

