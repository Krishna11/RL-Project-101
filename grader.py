"""
Per-task grading logic.

Delegates to each task's ``grade()`` method and wraps it with
standardised output formatting.
"""

from __future__ import annotations

from .tasks import load_task
from .tasks.base_task import EpisodeMetrics


def grade_episode(
    task_id: str,
    episode_metrics: EpisodeMetrics,
) -> dict:
    """
    Grade a completed episode.

    Returns a dict::

        {
            "task_id": str,
            "difficulty": str,
            "score": float,          # 0.0 – 1.0
            "total_steps": int,
            "avg_pue": float,
            "max_temp": float,
            "thermal_violations": int,
        }
    """
    task = load_task(task_id)
    score = task.grade(episode_metrics)
    score = max(0.0, min(1.0, score))

    return {
        "task_id": task_id,
        "difficulty": task.difficulty,
        "score": round(score, 4),
        "total_steps": episode_metrics.total_steps,
        "avg_pue": round(episode_metrics.avg_pue, 4),
        "max_temp": round(episode_metrics.max_temp, 2),
        "thermal_violations": episode_metrics.thermal_violations,
    }


def openenv_grader(stdout: str) -> float:
    """
    OpenEnv-spec compatible grader.
    Parses the final output string to extract the episode score.
    """
    import re
    match = re.search(r"\[END\].*?score=([0-9.]+)", stdout)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return 0.0
