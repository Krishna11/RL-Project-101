"""
Default physical and environment constants.

All values are based on ASHRAE TC 9.9 guidelines for data-center
thermal management.  They can be overridden per-task or per-test.
"""

from __future__ import annotations

# ── Temperature Thresholds (°C) ─────────────────────────
SAFE_TEMP_MIN: float = 18.0       # ASHRAE recommended lower bound
SAFE_TEMP_MAX: float = 27.0       # ASHRAE recommended upper bound
CRITICAL_TEMP: float = 35.0       # Hardware auto-shutdown threshold

# ── Ambient / Outside ──────────────────────────────────
AMBIENT_TEMP_DEFAULT: float = 35.0  # Worst-case summer day

# ── CRAC Operating Ranges ──────────────────────────────
SUPPLY_TEMP_MIN: float = 10.0     # Minimum supply-air temp (°C)
SUPPLY_TEMP_MAX: float = 20.0     # Maximum supply-air temp (°C)
FAN_SPEED_MIN: float = 0.1        # Minimum operational fan speed ratio
FAN_SPEED_MAX: float = 1.0
CHILLED_WATER_FLOW_MIN: float = 0.1
CHILLED_WATER_FLOW_MAX: float = 1.0

# ── Power / Overhead ───────────────────────────────────
MISC_OVERHEAD_W: float = 500.0    # Lighting, PDU losses, etc.

# ── Simulation Defaults ────────────────────────────────
DEFAULT_DT: float = 60.0          # Seconds per simulation step
DEFAULT_EPISODE_LENGTH: int = 120  # Steps (= 2 hours simulated)

# ── Temperature Clamping (safety net for divergent sims) ─
TEMP_CLAMP_MIN: float = -10.0
TEMP_CLAMP_MAX: float = 100.0
