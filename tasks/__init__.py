"""
Task definitions — easy → hard difficulty progression.

Each task configures the DataCenter (zones, CRACs, loads) and provides
step-by-step dynamics (variable IT loads, ambient changes, events).
"""

from __future__ import annotations

from .base_task import BaseTask, EpisodeMetrics
from .task1_single_zone import Task1SingleZone
from .task2_variable_workload import Task2VariableWorkload
from .task3_random_events import Task3RandomEvents

# ── Registry ────────────────────────────────────────────

_TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "task_1_single_zone": Task1SingleZone,
    "task_2_variable_workload": Task2VariableWorkload,
    "task_3_random_events": Task3RandomEvents,
}


def load_task(task_id: str, seed: int | None = None) -> BaseTask:
    """
    Instantiate a task by its string identifier.

    Raises ``ValueError`` if the *task_id* is not registered.
    """
    cls = _TASK_REGISTRY.get(task_id)
    if cls is None:
        available = ", ".join(sorted(_TASK_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task_id {task_id!r}.  Available tasks: {available}"
        )
    return cls(seed=seed)


__all__ = [
    "BaseTask",
    "EpisodeMetrics",
    "Task1SingleZone",
    "Task2VariableWorkload",
    "Task3RandomEvents",
    "load_task",
]
