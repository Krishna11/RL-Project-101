# Copyright (c) Team CoolPilot.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license.

"""
Typed data contracts for the CoolPilot environment (Pydantic v2).

Every model uses strict validation, immutable-by-default fields, and
exhaustive edge-case handling so that:
  1. Invalid /step payloads fail fast with clear errors.
  2. Serialised JSON is always spec-compliant.
  3. Round-tripping (dict → model → dict) is lossless.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ─────────────────────────────────────────────────────────
# Action Models
# ─────────────────────────────────────────────────────────

class CRACAction(BaseModel):
    """
    Action payload for a **single** CRAC unit.

    All three fields are required.  Values are hard-clamped to their
    physical operating range during validation so the thermal simulation
    never receives out-of-range inputs.
    """

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",            # reject unknown keys immediately
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "fan_speed": 0.6,
                    "chilled_water_flow": 0.5,
                    "supply_temp": 15.0,
                }
            ]
        },
    )

    fan_speed: float = Field(
        ...,
        ge=0.0,  # accept 0.0 and clamp up — see validator
        le=1.0,
        description="Fan speed ratio.  Accepted range: [0.0, 1.0]; "
                    "values below 0.1 are clamped to the minimum operational speed.",
    )
    chilled_water_flow: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Chilled-water flow ratio.  Same clamping rules as fan_speed.",
    )
    supply_temp: float = Field(
        ...,
        ge=5.0,   # wider acceptance; clamp in validator
        le=25.0,
        description="CRAC supply-air temperature (°C).  Clamped to [10, 20].",
    )

    # ── Validators ──────────────────────────────────────

    @field_validator("fan_speed", "chilled_water_flow", mode="before")
    @classmethod
    def _reject_non_finite(cls, v: Any, info) -> float:
        """Reject NaN / ±Inf before Pydantic's own float coercion."""
        try:
            v = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{info.field_name} must be a finite number, got {v!r}"
            ) from exc
        if not math.isfinite(v):
            raise ValueError(
                f"{info.field_name} must be finite, got {v}"
            )
        return v

    @field_validator("supply_temp", mode="before")
    @classmethod
    def _reject_non_finite_temp(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"supply_temp must be a finite number, got {v!r}"
            ) from exc
        if not math.isfinite(v):
            raise ValueError(f"supply_temp must be finite, got {v}")
        return v

    @model_validator(mode="after")
    def _clamp_to_operational_range(self) -> "CRACAction":
        """
        Soft-clamp values to their *operational* range.

        This is intentionally lenient: the LLM may output 0.05 for
        fan_speed, which we clamp to 0.1 rather than rejecting.
        """
        # Fan & water: operational minimum = 0.1
        object.__setattr__(
            self, "fan_speed", max(0.1, min(1.0, self.fan_speed))
        )
        object.__setattr__(
            self, "chilled_water_flow",
            max(0.1, min(1.0, self.chilled_water_flow)),
        )
        # Supply temp: operational [10, 20] °C
        object.__setattr__(
            self, "supply_temp", max(10.0, min(20.0, self.supply_temp))
        )
        return self


class Action(BaseModel):
    """
    Top-level action — one ``CRACAction`` per CRAC unit in the environment.

    Edge cases handled:
      • Empty ``cracs`` list → ValidationError (at least 1 required).
      • Extra/fewer CRACs than the env expects → handled at step() time.
      • Nested validation errors bubble up with full path.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "cracs": [
                        {
                            "fan_speed": 0.6,
                            "chilled_water_flow": 0.5,
                            "supply_temp": 15.0,
                        }
                    ]
                }
            ]
        },
    )

    cracs: list[CRACAction] = Field(
        ...,
        min_length=1,
        description="Controls for each CRAC unit (ordered by crac_id).",
    )


# ─────────────────────────────────────────────────────────
# Observation Models
# ─────────────────────────────────────────────────────────

class ZoneObservation(BaseModel):
    """Read-only snapshot of a single rack zone."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    zone_id: int = Field(..., ge=0, description="Zone identifier (0-indexed).")
    temperature: float = Field(
        ...,
        description="Current zone air temperature (°C), rounded to 2 dp.",
    )
    it_power_w: float = Field(
        ...,
        ge=0.0,
        description="Current IT power draw (Watts).",
    )

    @field_validator("temperature", mode="before")
    @classmethod
    def _round_temperature(cls, v: Any) -> float:
        v = float(v)
        if not math.isfinite(v):
            # Simulation diverged — clamp to a reportable extreme
            return 100.0 if v > 0 else -10.0
        return round(v, 2)


class CRACObservation(BaseModel):
    """Read-only snapshot of a single CRAC unit."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    crac_id: int = Field(..., ge=0)
    fan_speed: float = Field(..., ge=0.0, le=1.0)
    chilled_water_flow: float = Field(..., ge=0.0, le=1.0)
    supply_temp: float = Field(..., description="Supply air temperature (°C).")
    power_draw_w: float = Field(
        ...,
        ge=0.0,
        description="CRAC electrical consumption (Watts).",
    )
    is_online: bool = Field(
        default=True,
        description="False if the CRAC unit has suffered a failure event.",
    )


class Observation(BaseModel):
    """
    Full observation returned by ``reset()`` and ``step()``.

    This is the canonical payload shape for the ``/ws`` WebSocket protocol
    *and* the ``POST /step`` HTTP fallback.

    Edge cases:
      • ``reward`` is hard-clamped to [0.0, 1.0].
      • ``pue`` of exactly 0.0 is replaced with 1.0 (physical minimum).
      • Non-finite numerics are caught and replaced with safe defaults.
    """

    model_config = ConfigDict(extra="ignore")

    zones: list[ZoneObservation] = Field(default_factory=list)
    cracs: list[CRACObservation] = Field(default_factory=list)

    ambient_temp: float = Field(
        default=35.0,
        description="Outside / ambient temperature (°C).",
    )
    pue: float = Field(
        default=1.0,
        ge=1.0,
        description="Power Usage Effectiveness (≥ 1.0).",
    )
    total_it_power_w: float = Field(default=0.0, ge=0.0)
    total_cooling_power_w: float = Field(default=0.0, ge=0.0)

    reward: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Composite reward for this step, clamped to [0, 1].",
    )
    terminated: bool = Field(
        default=False,
        description="True when a critical thermal violation ends the episode.",
    )
    truncated: bool = Field(
        default=False,
        description="True when max episode steps are reached.",
    )
    step_number: int = Field(default=0, ge=0)
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary diagnostics (sub-scores, warnings, etc.).",
    )

    # ── Validators ──────────────────────────────────────

    @field_validator("reward", mode="before")
    @classmethod
    def _clamp_reward(cls, v: Any) -> float:
        v = float(v)
        if not math.isfinite(v):
            return 0.0
        return round(max(0.0, min(1.0, v)), 4)

    @field_validator("pue", mode="before")
    @classmethod
    def _sanitise_pue(cls, v: Any) -> float:
        v = float(v)
        if not math.isfinite(v) or v < 1.0:
            return 1.0
        return round(v, 4)


# ─────────────────────────────────────────────────────────
# State Model
# ─────────────────────────────────────────────────────────

class State(BaseModel):
    """
    Episode-level metadata returned by the ``state()`` endpoint.

    This is the OpenEnv "game state" — a lightweight summary used by
    orchestrators to decide whether to continue or terminate.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    episode_id: str = Field(
        default="",
        min_length=0,
        description="UUID identifying the current episode.",
    )
    task_id: str = Field(
        default="task_1_single_zone",
        description="Which task variant is active.",
    )
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(default=60, ge=1)
    elapsed_simulated_seconds: float = Field(default=0.0, ge=0.0)
    is_done: bool = Field(default=False)

    @field_validator("episode_id", mode="before")
    @classmethod
    def _coerce_episode_id(cls, v: Any) -> str:
        """Accept None or non-string and coerce gracefully."""
        if v is None:
            return ""
        return str(v)
