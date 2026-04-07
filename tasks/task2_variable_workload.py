"""
Task 2 — Variable Workload (Medium).

4 zones · 2 CRACs · sinusoidal IT loads · variable ambient · 120 steps.
"""

from __future__ import annotations

import math

from ..thermal.datacenter import CRACUnit, DataCenter, RackZone
from .base_task import BaseTask, EpisodeMetrics


class Task2VariableWorkload(BaseTask):
    task_id = "task_2_variable_workload"
    difficulty = "medium"
    num_zones = 4
    num_cracs = 2
    episode_length = 120
    reward_weights = {"safety": 0.40, "energy": 0.35, "stability": 0.25}

    def build_datacenter(self) -> DataCenter:
        zones = [
            RackZone(zone_id=i, it_power_w=10_000.0)
            for i in range(4)
        ]
        cracs = [
            CRACUnit(crac_id=0, serves_zones=[0, 1]),
            CRACUnit(crac_id=1, serves_zones=[2, 3]),
        ]
        return DataCenter(zones=zones, cracs=cracs)

    def get_it_load(self, step: int) -> list[float]:
        """Sinusoidal IT load: 5–15 kW per zone, phase-shifted."""
        base = 10_000.0
        amplitude = 5_000.0
        return [
            base + amplitude * math.sin(2 * math.pi * (step + i * 15) / 120)
            for i in range(self.num_zones)
        ]

    def get_ambient_temp(self, step: int) -> float:
        """25 °C → 40 °C → 25 °C over the episode."""
        return 32.5 + 7.5 * math.sin(2 * math.pi * step / 120)

    def grade(self, episode_metrics: EpisodeMetrics) -> float:
        if episode_metrics.consecutive_violations > 5:
            return 0.0
        if episode_metrics.avg_pue < 1.4 and episode_metrics.thermal_violations == 0:
            return 1.0
        if episode_metrics.avg_pue < 1.6:
            return 0.5
        return 0.25
