"""
Pure-Python thermal dynamics — zero external dependencies.

Every function is deterministic, side-effect-free, and guards against
non-finite inputs (NaN, ±Inf, division-by-zero).
"""

from __future__ import annotations

import math

from .constants import MISC_OVERHEAD_W, TEMP_CLAMP_MAX, TEMP_CLAMP_MIN


# ─────────────────────────────────────────────────────────
# Core thermal step
# ─────────────────────────────────────────────────────────

def newton_cooling_step(
    t_zone: float,       # current zone temp (°C)
    q_it: float,         # IT heat load (W)
    c_zone: float,       # thermal capacitance (J/°C)
    h_eff: float,        # effective heat transfer coeff (W/°C)
    t_supply: float,     # CRAC supply air temp (°C)
    dt: float,           # time step (s)
) -> float:
    """
    Advance zone temperature by one time-step using Newton's law of cooling.

    Formula:
        T(t+dt) = T(t) + dt × [ Q_IT / C_zone  −  h_eff × (T_zone − T_supply) / C_zone ]

    Edge cases:
      • c_zone ≤ 0 → returns t_zone unchanged (avoids div-by-zero).
      • dt ≤ 0     → returns t_zone unchanged (no time has passed).
      • Any non-finite input → returns t_zone unchanged.
      • Result clamped to [TEMP_CLAMP_MIN, TEMP_CLAMP_MAX] to prevent divergence.
    """
    # Guard: all inputs must be finite
    if not all(math.isfinite(x) for x in (t_zone, q_it, c_zone, h_eff, t_supply, dt)):
        return t_zone

    # Guard: degenerate cases
    if c_zone <= 0.0 or dt <= 0.0:
        return t_zone

    heat_in = q_it / c_zone
    heat_out = h_eff * (t_zone - t_supply) / c_zone
    d_temp = (heat_in - heat_out) * dt

    new_temp = t_zone + d_temp

    # Clamp to prevent simulation divergence
    return max(TEMP_CLAMP_MIN, min(TEMP_CLAMP_MAX, new_temp))


# ─────────────────────────────────────────────────────────
# PUE computation
# ─────────────────────────────────────────────────────────

def compute_pue(
    total_it_power_w: float,
    cooling_power_w: float,
    misc_overhead_w: float = MISC_OVERHEAD_W,
) -> float:
    """
    PUE = Total Facility Power / IT Equipment Power.

    Returns:
        PUE ratio (ideal = 1.0, typical DC = 1.2–1.8).

    Edge cases:
      • IT power ≤ 0   → returns 1.0 (no meaningful PUE).
      • Cooling < 0     → treated as 0.
      • Non-finite       → returns 1.0.
      • Result > 10.0    → clamped (something is very wrong; don't let it explode).
    """
    if not math.isfinite(total_it_power_w) or total_it_power_w <= 0.0:
        return 1.0
    cooling_power_w = max(0.0, cooling_power_w) if math.isfinite(cooling_power_w) else 0.0
    misc_overhead_w = max(0.0, misc_overhead_w) if math.isfinite(misc_overhead_w) else 0.0

    total_facility = total_it_power_w + cooling_power_w + misc_overhead_w
    pue = total_facility / total_it_power_w

    return min(10.0, max(1.0, pue))


# ─────────────────────────────────────────────────────────
# CRAC power draw
# ─────────────────────────────────────────────────────────

def crac_power_draw(
    fan_speed_pct: float,       # 0.0 – 1.0
    chilled_water_flow: float,  # 0.0 – 1.0
    max_fan_power_w: float = 5000.0,
    max_pump_power_w: float = 3000.0,
) -> float:
    """
    Estimate CRAC unit electrical consumption.

    Fan power ∝ (speed)³   (affinity law).
    Pump power ∝ flow      (linear simplification).

    Edge cases:
      • Negative inputs → clamped to 0.
      • Non-finite → returns 0.0.
    """
    if not all(
        math.isfinite(x)
        for x in (fan_speed_pct, chilled_water_flow, max_fan_power_w, max_pump_power_w)
    ):
        return 0.0

    fan_speed_pct = max(0.0, min(1.0, fan_speed_pct))
    chilled_water_flow = max(0.0, min(1.0, chilled_water_flow))
    max_fan_power_w = max(0.0, max_fan_power_w)
    max_pump_power_w = max(0.0, max_pump_power_w)

    fan_power = max_fan_power_w * (fan_speed_pct ** 3)
    pump_power = max_pump_power_w * chilled_water_flow

    return fan_power + pump_power


# ─────────────────────────────────────────────────────────
# Effective heat-transfer coefficient
# ─────────────────────────────────────────────────────────

def compute_effective_h(
    fan_speed_pct: float,
    chilled_water_flow: float,
    base_h: float = 1500.0,   # W/°C at 100 % fan + 100 % water
) -> float:
    """
    Effective heat-transfer coefficient as function of agent controls.

    h_eff = base_h × (0.6 × fan + 0.4 × water)

    Edge cases:
      • Inputs clamped to [0, 1] range.
      • Non-finite → returns 0.0.
    """
    if not all(math.isfinite(x) for x in (fan_speed_pct, chilled_water_flow, base_h)):
        return 0.0

    fan_speed_pct = max(0.0, min(1.0, fan_speed_pct))
    chilled_water_flow = max(0.0, min(1.0, chilled_water_flow))
    base_h = max(0.0, base_h)

    return base_h * (0.6 * fan_speed_pct + 0.4 * chilled_water_flow)
