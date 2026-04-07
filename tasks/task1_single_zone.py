"""
Task 1 — Single Zone Steady State (Easy).

1 zone · 1 CRAC · constant 10 kW IT load · 60 steps.
"""

from __future__ import annotations

from ..thermal.datacenter import CRACUnit, DataCenter, RackZone
from .base_task import BaseTask, EpisodeMetrics


class Task1SingleZone(BaseTask):
    task_id = "task_1_single_zone"
    difficulty = "easy"
    num_zones = 1
    num_cracs = 1
    episode_length = 60
    reward_weights = {"safety": 0.5, "energy": 0.5}

    def build_datacenter(self) -> DataCenter:
        zones = [RackZone(zone_id=0, it_power_w=10_000.0)]
        cracs = [CRACUnit(crac_id=0, serves_zones=[0])]
        return DataCenter(zones=zones, cracs=cracs)

    def get_it_load(self, step: int) -> list[float]:
        return [10_000.0]

    def grade(self, episode_metrics: EpisodeMetrics) -> float:
        if episode_metrics.max_temp > 27.0:
            return 0.0
        if episode_metrics.avg_pue > 1.6:
            return 0.5
        return 1.0
