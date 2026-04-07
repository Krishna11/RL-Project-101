"""
Composite reward function — all sub-scores normalised to [0.0, 1.0].

Every function guards against empty lists, non-finite values, and
division-by-zero.
"""

from __future__ import annotations

import math

from .thermal.constants import SAFE_TEMP_MAX, SAFE_TEMP_MIN


def _finite_or(value: float, default: float = 0.0) -> float:
    return value if math.isfinite(value) else default


# ─────────────────────────────────────────────────────────
# Sub-scores
# ─────────────────────────────────────────────────────────

def safety_score(zone_temps: list[float]) -> float:
    """
    1.0 if all zones in [18, 27] °C.  Linearly degrades with violations.

    Each zone contributes equally.  A zone at exactly the boundary scores
    1.0.  A zone 5 °C beyond scores 0.0.

    Edge: empty list → 1.0 (no zones to violate).
    """
    if not zone_temps:
        return 1.0
    scores: list[float] = []
    for t in zone_temps:
        t = _finite_or(t, 30.0)  # assume hot if NaN
        if SAFE_TEMP_MIN <= t <= SAFE_TEMP_MAX:
            scores.append(1.0)
        elif t < SAFE_TEMP_MIN:
            scores.append(max(0.0, 1.0 - (SAFE_TEMP_MIN - t) / 5.0))
        else:
            scores.append(max(0.0, 1.0 - (t - SAFE_TEMP_MAX) / 5.0))
    return sum(scores) / len(scores)


def energy_score(pue: float) -> float:
    """
    1.0 at PUE = 1.0 (ideal), 0.0 at PUE ≥ 2.0.

    Edge: PUE < 1.0 → clamped to 1.0.  Non-finite → 0.0.
    """
    pue = _finite_or(pue, 2.0)
    pue = max(1.0, pue)
    return max(0.0, min(1.0, 1.0 - (pue - 1.0)))


def stability_score(
    zone_temps: list[float],
    prev_zone_temps: list[float],
) -> float:
    """
    Penalises large temperature swings between steps.

    1.0 = no change, 0.0 = average swing ≥ 3 °C.

    Edge: mismatched lengths → uses min length.
    Edge: either list empty → 1.0.
    """
    if not zone_temps or not prev_zone_temps:
        return 1.0
    pairs = zip(zone_temps, prev_zone_temps)
    swings = [abs(_finite_or(t, 22.0) - _finite_or(pt, 22.0)) for t, pt in pairs]
    if not swings:
        return 1.0
    avg_swing = sum(swings) / len(swings)
    return max(0.0, 1.0 - avg_swing / 3.0)


def cost_score(
    cooling_power_w: float,
    tou_price: float,
    dt_hours: float,
    budget_per_step: float = 0.50,
) -> float:
    """
    1.0 if energy cost ≤ 0, 0.0 if cost ≥ budget.

    Edge: negative power → 0.0 cost.  Non-finite → 0.0 score.
    """
    cooling_power_w = max(0.0, _finite_or(cooling_power_w))
    tou_price = max(0.0, _finite_or(tou_price, 0.08))
    dt_hours = max(0.0, _finite_or(dt_hours, 1 / 60))

    cost = (cooling_power_w / 1000.0) * tou_price * dt_hours
    if budget_per_step <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - cost / budget_per_step))


def resilience_score(
    active_cracs: int,
    total_cracs: int,
    zone_temps: list[float],
) -> float:
    """
    Bonus for maintaining safe temps during CRAC failures.

    1.0 if all zones safe despite reduced capacity.
    Only "active" during failure events (active < total).

    Edge: total_cracs ≤ 0 → 0.0.
    """
    if total_cracs <= 0:
        return 0.0
    if active_cracs >= total_cracs:
        return 1.0

    safe = all(
        SAFE_TEMP_MIN <= _finite_or(t, 30.0) <= SAFE_TEMP_MAX
        for t in zone_temps
    )
    capacity_ratio = max(0.0, active_cracs / total_cracs)
    return 1.0 if safe else capacity_ratio


# ─────────────────────────────────────────────────────────
# Composite
# ─────────────────────────────────────────────────────────

def composite_reward(
    zone_temps: list[float],
    prev_zone_temps: list[float],
    pue: float,
    cooling_power_w: float = 0.0,
    tou_price: float = 0.08,
    dt_hours: float = 1 / 60,
    active_cracs: int = 1,
    total_cracs: int = 1,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute final reward in [0.0, 1.0].

    Default weights (Task 1): safety = 0.5, energy = 0.5.
    Task-specific weights override via *weights* dict.

    Edge: weights summing to 0 → returns 0.0.
    Edge: unknown weight keys → silently ignored.
    """
    if weights is None:
        weights = {"safety": 0.5, "energy": 0.5}

    scores = {
        "safety": safety_score(zone_temps),
        "energy": energy_score(pue),
        "stability": stability_score(zone_temps, prev_zone_temps),
        "cost": cost_score(cooling_power_w, tou_price, dt_hours),
        "resilience": resilience_score(active_cracs, total_cracs, zone_temps),
    }

    total = sum(
        weights.get(k, 0.0) * scores.get(k, 0.0)
        for k in weights
    )
    return round(max(0.0, min(1.0, total)), 4)
