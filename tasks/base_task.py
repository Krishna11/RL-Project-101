"""
Abstract base class for all task definitions.

A *task* configures:
  1. The DataCenter topology (zones, CRACs, zone-CRAC mapping).
  2. Step-by-step dynamics (IT loads, ambient temp, events).
  3. Reward weights and grading logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..thermal.datacenter import DataCenter


@dataclass
class EpisodeMetrics:
    """Accumulated episode statistics used by the grader."""

    total_steps: int = 0
    total_reward: float = 0.0
    avg_pue: float = 1.0
    max_temp: float = 0.0
    min_temp: float = 100.0
    thermal_violations: int = 0     # count of steps with any zone > 27 °C
    consecutive_violations: int = 0  # longest streak
    _current_streak: int = 0

    def record_step(
        self,
        zone_temps: list[float],
        pue: float,
        reward: float,
    ) -> None:
        """Update metrics after one step."""
        self.total_steps += 1
        self.total_reward += reward

        # Running PUE average
        self.avg_pue += (pue - self.avg_pue) / self.total_steps

        for t in zone_temps:
            self.max_temp = max(self.max_temp, t)
            self.min_temp = min(self.min_temp, t)

        violation = any(t > 27.0 for t in zone_temps)
        if violation:
            self.thermal_violations += 1
            self._current_streak += 1
            self.consecutive_violations = max(
                self.consecutive_violations, self._current_streak
            )
        else:
            self._current_streak = 0


class BaseTask(ABC):
    """
    Abstract task interface.

    Subclasses must implement:
      • ``build_datacenter()``   — create initial topology
      • ``get_it_load(step)``    — return per-zone IT loads
      • ``grade(metrics)``       — return a 0.0–1.0 score
    """

    task_id: str = "base"
    difficulty: str = "easy"
    num_zones: int = 1
    num_cracs: int = 1
    episode_length: int = 60
    reward_weights: dict[str, float] = {"safety": 0.5, "energy": 0.5}

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed

    @abstractmethod
    def build_datacenter(self) -> DataCenter:
        """Create and return a fully initialised DataCenter."""
        ...

    @abstractmethod
    def get_it_load(self, step: int) -> list[float]:
        """Return IT power (Watts) for each zone at the given step."""
        ...

    def get_ambient_temp(self, step: int) -> float:
        """Override in tasks with dynamic ambient temperature."""
        return 35.0

    def get_tou_price(self, step: int) -> float:
        """Override in tasks with time-of-use energy pricing ($/kWh)."""
        return 0.08

    def maybe_trigger_event(self, step: int) -> Optional[dict]:
        """Override to inject random events (CRAC failures, load spikes)."""
        return None

    @abstractmethod
    def grade(self, episode_metrics: EpisodeMetrics) -> float:
        """Return a 0.0–1.0 grade for the completed episode."""
        ...
