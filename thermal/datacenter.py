"""
DataCenter model — mutable runtime state for the thermal simulation.

Uses plain dataclasses (not Pydantic) to keep the hot-loop lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RackZone:
    """A logical group of server racks with a uniform air temperature."""

    zone_id: int
    it_power_w: float = 10_000.0        # Current IT load (Watts)
    temperature: float = 22.0           # Current zone air temp (°C)
    thermal_cap: float = 50_000.0       # J/°C  (air mass × specific heat)
    area: float = 10.0                  # m²  effective exchange area

    def __post_init__(self) -> None:
        if self.thermal_cap <= 0:
            raise ValueError(f"thermal_cap must be > 0, got {self.thermal_cap}")
        if self.it_power_w < 0:
            self.it_power_w = 0.0


@dataclass
class CRACUnit:
    """Computer Room Air Conditioning unit — agent-controllable."""

    crac_id: int
    fan_speed: float = 0.5              # 0.0 – 1.0
    chilled_water_flow: float = 0.5     # 0.0 – 1.0
    supply_temp: float = 15.0           # °C  [10, 20] range
    max_fan_power_w: float = 5_000.0
    max_pump_power_w: float = 3_000.0
    serves_zones: list[int] = field(default_factory=list)
    is_online: bool = True              # False during failure events

    def __post_init__(self) -> None:
        self.fan_speed = max(0.0, min(1.0, self.fan_speed))
        self.chilled_water_flow = max(0.0, min(1.0, self.chilled_water_flow))
        self.supply_temp = max(10.0, min(20.0, self.supply_temp))


@dataclass
class DataCenter:
    """Top-level container holding the full mutable simulation state."""

    zones: list[RackZone] = field(default_factory=list)
    cracs: list[CRACUnit] = field(default_factory=list)
    ambient_temp: float = 35.0          # Outside temp °C
    time_step_s: float = 60.0           # 1-minute ticks
    current_step: int = 0
    elapsed_seconds: float = 0.0

    # ── Lookup helpers ──────────────────────────────────

    def zone_by_id(self, zone_id: int) -> Optional[RackZone]:
        """Return the zone with *zone_id*, or None if not found."""
        for z in self.zones:
            if z.zone_id == zone_id:
                return z
        return None

    def cracs_serving(self, zone_id: int) -> list[CRACUnit]:
        """Return all *online* CRAC units that serve *zone_id*."""
        return [
            c for c in self.cracs
            if zone_id in c.serves_zones and c.is_online
        ]

    @property
    def total_it_power_w(self) -> float:
        return sum(z.it_power_w for z in self.zones)

    @property
    def zone_temps(self) -> list[float]:
        return [z.temperature for z in self.zones]

    @property
    def online_cracs(self) -> list[CRACUnit]:
        return [c for c in self.cracs if c.is_online]
