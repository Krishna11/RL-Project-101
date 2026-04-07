"""
Task 3 — Random Events & TOU Pricing (Hard).

8 zones · 3 CRACs · random IT spikes · CRAC failures · TOU pricing · 180 steps.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from ..thermal.datacenter import CRACUnit, DataCenter, RackZone
from .base_task import BaseTask, EpisodeMetrics


class Task3RandomEvents(BaseTask):
    task_id = "task_3_random_events"
    difficulty = "hard"
    num_zones = 8
    num_cracs = 3
    episode_length = 180
    reward_weights = {
        "safety": 0.30,
        "energy": 0.30,
        "cost": 0.20,
        "stability": 0.10,
        "resilience": 0.10,
    }

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._rng = random.Random(seed if seed is not None else 42)
        self._crac_failure_start: Optional[int] = None
        self._failed_crac_id: Optional[int] = None

    def build_datacenter(self) -> DataCenter:
        zones = [
            RackZone(zone_id=i, it_power_w=10_000.0)
            for i in range(8)
        ]
        cracs = [
            CRACUnit(crac_id=0, serves_zones=[0, 1, 2]),
            CRACUnit(crac_id=1, serves_zones=[3, 4, 5]),
            CRACUnit(crac_id=2, serves_zones=[6, 7]),
        ]
        return DataCenter(zones=zones, cracs=cracs)

    def get_it_load(self, step: int) -> list[float]:
        base_loads = [
            10_000 + 5_000 * math.sin(2 * math.pi * (step + i * 10) / 180)
            for i in range(8)
        ]
        # 10 % chance of a random spike on one zone per step
        if self._rng.random() < 0.10:
            spike_zone = self._rng.randint(0, 7)
            base_loads[spike_zone] += self._rng.uniform(3_000, 8_000)
        return base_loads

    def get_ambient_temp(self, step: int) -> float:
        """Diurnal curve + stochastic noise (±3 °C)."""
        base = 32.5 + 7.5 * math.sin(2 * math.pi * step / 180)
        noise = self._rng.gauss(0, 1.5)
        return max(15.0, min(50.0, base + noise))

    def get_tou_price(self, step: int) -> float:
        """$/kWh — peak during steps 60–120."""
        if 60 <= step <= 120:
            return 0.25
        return 0.08

    def maybe_trigger_event(self, step: int) -> Optional[dict]:
        """
        Possibly trigger (or resolve) a CRAC failure.

        Returns a dict `{"type": "crac_failure", "crac_id": int}` when
        a failure starts, or `{"type": "crac_restored", "crac_id": int}`
        when it ends.  Returns ``None`` otherwise.
        """
        # Check if an existing failure should resolve (lasts 10 steps)
        if self._crac_failure_start is not None:
            if step - self._crac_failure_start >= 10:
                restored_id = self._failed_crac_id
                self._crac_failure_start = None
                self._failed_crac_id = None
                return {"type": "crac_restored", "crac_id": restored_id}
            return None  # failure still in progress — no new event

        # 2 % chance per step after step 30
        if step > 30 and self._rng.random() < 0.02:
            self._crac_failure_start = step
            self._failed_crac_id = self._rng.randint(0, self.num_cracs - 1)
            return {"type": "crac_failure", "crac_id": self._failed_crac_id}

        return None

    def grade(self, episode_metrics: EpisodeMetrics) -> float:
        """Continuous 0.0–1.0 grade — weighted average of sub-scores."""
        if episode_metrics.total_steps == 0:
            return 0.0

        avg_reward = episode_metrics.total_reward / episode_metrics.total_steps

        # Penalty for sustained violations
        violation_penalty = min(
            1.0, episode_metrics.consecutive_violations / 10.0
        )

        return max(0.0, min(1.0, avg_reward - 0.3 * violation_penalty))
