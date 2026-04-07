"""
Core OpenEnv environment — implements reset / step / state.

This is the *server-side* logic.  The client connects via the FastAPI
endpoints defined in ``app.py``.

Every public method guards against:
  • reset() not having been called yet.
  • Steps taken after episode completion.
  • Mismatched action CRAC count.
  • Thermal simulation divergence (clamped).
  • Non-finite intermediate values.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..models import (
    Action,
    CRACObservation,
    Observation,
    State,
    ZoneObservation,
)
from ..rewards import composite_reward
from ..tasks import load_task
from ..tasks.base_task import BaseTask, EpisodeMetrics
from ..thermal.datacenter import DataCenter
from ..thermal.physics import (
    compute_effective_h,
    compute_pue,
    crac_power_draw,
    newton_cooling_step,
)
from ..thermal.constants import CRITICAL_TEMP

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Request models (only used server-side by FastAPI)
# ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_id: str = "task_1_single_zone"


class StepRequest(BaseModel):
    """Wraps the Action so FastAPI can deserialise the body."""
    cracs: list[dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────

class CoolPilotEnvironment:
    """
    Server-side OpenEnv Environment.

    Lifecycle:  ``reset()``  →  ``step()`` × N  →  ``state()`` at any time.
    """

    def __init__(self) -> None:
        self._dc: Optional[DataCenter] = None
        self._task: Optional[BaseTask] = None
        self._episode_id: str = ""
        self._prev_temps: list[float] = []
        self._metrics: Optional[EpisodeMetrics] = None
        self._is_done: bool = False

    # ── OpenEnv API ─────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: str = "task_1_single_zone",
    ) -> Observation:
        """
        Initialise a new episode.

        Edge cases:
          • Unknown task_id → ValueError (caught by FastAPI → 422).
          • Called while previous episode running → silently replaces.
        """
        self._task = load_task(task_id, seed=seed)
        self._dc = self._task.build_datacenter()
        self._episode_id = str(uuid.uuid4())
        self._prev_temps = [z.temperature for z in self._dc.zones]
        self._metrics = EpisodeMetrics()
        self._is_done = False

        logger.info(
            "Reset → task=%s  zones=%d  cracs=%d  episode=%s",
            task_id,
            len(self._dc.zones),
            len(self._dc.cracs),
            self._episode_id[:8],
        )

        return self._build_observation(
            reward=0.5, terminated=False, truncated=False
        )

    def step(self, action: Action) -> Observation:
        """
        Apply agent action, advance thermal sim, compute reward.

        Edge cases:
          • step() before reset()        → RuntimeError.
          • step() after episode done     → returns last obs with done=True.
          • Action has wrong CRAC count  → pad with current settings or truncate.
          • CRAC failure event           → failed unit is skipped.
          • Temperature divergence       → clamped by physics.py.
        """
        if self._dc is None or self._task is None:
            raise RuntimeError(
                "step() called before reset().  Call reset() first."
            )

        if self._is_done:
            logger.warning("step() called after episode ended — returning final obs.")
            return self._build_observation(
                reward=0.0, terminated=True, truncated=True
            )

        dc = self._dc
        task = self._task

        # ── 1. Apply agent actions to CRAC units ───────
        for i, crac in enumerate(dc.cracs):
            if i < len(action.cracs):
                ca = action.cracs[i]
                if crac.is_online:
                    crac.fan_speed = ca.fan_speed
                    crac.chilled_water_flow = ca.chilled_water_flow
                    crac.supply_temp = ca.supply_temp
                # If CRAC is offline, ignore the action for it
            # If action has fewer CRACs than env, keep current settings

        # ── 2. Trigger task events (CRAC failures, etc.) ─
        event = task.maybe_trigger_event(dc.current_step)
        if event is not None:
            if event["type"] == "crac_failure":
                crac_id = event["crac_id"]
                for c in dc.cracs:
                    if c.crac_id == crac_id:
                        c.is_online = False
                        logger.warning("CRAC %d FAILED at step %d", crac_id, dc.current_step)
            elif event["type"] == "crac_restored":
                crac_id = event["crac_id"]
                for c in dc.cracs:
                    if c.crac_id == crac_id:
                        c.is_online = True
                        logger.info("CRAC %d RESTORED at step %d", crac_id, dc.current_step)

        # ── 3. Update IT loads for current step ────────
        it_loads = task.get_it_load(dc.current_step)
        for i, zone in enumerate(dc.zones):
            zone.it_power_w = it_loads[i] if i < len(it_loads) else zone.it_power_w

        # ── 4. Advance thermal simulation ──────────────
        for zone in dc.zones:
            serving_cracs = dc.cracs_serving(zone.zone_id)
            if not serving_cracs:
                # No cooling → temperature drifts up from IT heat
                # Use a minimal h_eff (natural convection only)
                zone.temperature = newton_cooling_step(
                    t_zone=zone.temperature,
                    q_it=zone.it_power_w,
                    c_zone=zone.thermal_cap,
                    h_eff=5.0,  # minimal natural convection
                    t_supply=dc.ambient_temp,
                    dt=dc.time_step_s,
                )
                continue

            h_total = sum(
                compute_effective_h(c.fan_speed, c.chilled_water_flow)
                for c in serving_cracs
            )
            t_supply_avg = (
                sum(c.supply_temp for c in serving_cracs) / len(serving_cracs)
            )
            zone.temperature = newton_cooling_step(
                t_zone=zone.temperature,
                q_it=zone.it_power_w,
                c_zone=zone.thermal_cap,
                h_eff=h_total,
                t_supply=t_supply_avg,
                dt=dc.time_step_s,
            )

        # ── 5. Update ambient if task supports it ──────
        dc.ambient_temp = task.get_ambient_temp(dc.current_step)

        # ── 6. Advance time ────────────────────────────
        dc.current_step += 1
        dc.elapsed_seconds += dc.time_step_s

        # ── 7. Compute reward ──────────────────────────
        zone_temps = dc.zone_temps
        total_it = dc.total_it_power_w
        total_cooling = sum(
            crac_power_draw(
                c.fan_speed, c.chilled_water_flow,
                c.max_fan_power_w, c.max_pump_power_w,
            )
            for c in dc.cracs if c.is_online
        )
        pue = compute_pue(total_it, total_cooling)

        tou_price = task.get_tou_price(dc.current_step)

        reward = composite_reward(
            zone_temps=zone_temps,
            prev_zone_temps=self._prev_temps,
            pue=pue,
            cooling_power_w=total_cooling,
            tou_price=tou_price,
            dt_hours=dc.time_step_s / 3600.0,
            active_cracs=len(dc.online_cracs),
            total_cracs=len(dc.cracs),
            weights=task.reward_weights,
        )

        # ── 8. Record metrics ──────────────────────────
        self._metrics.record_step(zone_temps, pue, reward)
        self._prev_temps = zone_temps[:]

        # ── 9. Check termination ───────────────────────
        terminated = any(t > CRITICAL_TEMP for t in zone_temps)
        truncated = dc.current_step >= task.episode_length
        self._is_done = terminated or truncated

        return self._build_observation(reward, terminated, truncated)

    def state(self) -> State:
        """
        Return episode-level metadata.

        Edge cases:
          • Called before reset() → returns empty default state.
        """
        if self._dc is None or self._task is None:
            return State()

        return State(
            episode_id=self._episode_id,
            task_id=self._task.task_id,
            step_count=self._dc.current_step,
            max_steps=self._task.episode_length,
            elapsed_simulated_seconds=self._dc.elapsed_seconds,
            is_done=self._is_done,
        )

    # ── Helpers ─────────────────────────────────────────

    def _build_observation(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Observation:
        dc = self._dc
        total_it = dc.total_it_power_w
        total_cooling = sum(
            crac_power_draw(
                c.fan_speed, c.chilled_water_flow,
                c.max_fan_power_w, c.max_pump_power_w,
            )
            for c in dc.cracs if c.is_online
        )
        return Observation(
            zones=[
                ZoneObservation(
                    zone_id=z.zone_id,
                    temperature=z.temperature,
                    it_power_w=z.it_power_w,
                )
                for z in dc.zones
            ],
            cracs=[
                CRACObservation(
                    crac_id=c.crac_id,
                    fan_speed=c.fan_speed,
                    chilled_water_flow=c.chilled_water_flow,
                    supply_temp=c.supply_temp,
                    power_draw_w=round(
                        crac_power_draw(
                            c.fan_speed, c.chilled_water_flow,
                            c.max_fan_power_w, c.max_pump_power_w,
                        ),
                        2,
                    ),
                    is_online=c.is_online,
                )
                for c in dc.cracs
            ],
            ambient_temp=dc.ambient_temp,
            pue=compute_pue(total_it, total_cooling),
            total_it_power_w=total_it,
            total_cooling_power_w=round(total_cooling, 2),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            step_number=dc.current_step,
        )
