"""Tests for thermal/physics.py — edge-case focused."""

import math
import pytest

from coolpilot.thermal.physics import (
    compute_effective_h,
    compute_pue,
    crac_power_draw,
    newton_cooling_step,
)


class TestNewtonCoolingStep:
    def test_basic_cooling(self):
        """Zone should cool down when h_eff is large enough."""
        result = newton_cooling_step(
            t_zone=30.0, q_it=5000, c_zone=50000,
            h_eff=500, t_supply=15.0, dt=60,
        )
        assert result < 30.0

    def test_basic_heating(self):
        """Zone should heat up when IT load dominates cooling."""
        result = newton_cooling_step(
            t_zone=20.0, q_it=50000, c_zone=50000,
            h_eff=10, t_supply=15.0, dt=60,
        )
        assert result > 20.0

    def test_zero_dt_unchanged(self):
        assert newton_cooling_step(25.0, 10000, 50000, 150, 15.0, dt=0) == 25.0

    def test_zero_capacitance_unchanged(self):
        assert newton_cooling_step(25.0, 10000, 0, 150, 15.0, 60) == 25.0

    def test_negative_capacitance_unchanged(self):
        assert newton_cooling_step(25.0, 10000, -1, 150, 15.0, 60) == 25.0

    def test_nan_input_returns_original(self):
        assert newton_cooling_step(25.0, float("nan"), 50000, 150, 15.0, 60) == 25.0

    def test_inf_input_returns_original(self):
        assert newton_cooling_step(25.0, float("inf"), 50000, 150, 15.0, 60) == 25.0

    def test_result_clamped_high(self):
        """Even with extreme heat, temp shouldn't exceed TEMP_CLAMP_MAX."""
        result = newton_cooling_step(
            t_zone=99.0, q_it=1e9, c_zone=100, h_eff=0.001, t_supply=15.0, dt=60,
        )
        assert result <= 100.0

    def test_result_clamped_low(self):
        """Extreme overcooling should be clamped at TEMP_CLAMP_MIN."""
        result = newton_cooling_step(
            t_zone=0.0, q_it=0, c_zone=100, h_eff=1000, t_supply=-100.0, dt=60,
        )
        assert result >= -10.0


class TestComputePUE:
    def test_ideal_pue(self):
        """With no cooling, PUE should be just above 1.0 (overhead only)."""
        pue = compute_pue(10000, 0)
        assert 1.0 <= pue < 1.1

    def test_zero_it_power(self):
        assert compute_pue(0, 5000) == 1.0

    def test_negative_it_power(self):
        assert compute_pue(-100, 5000) == 1.0

    def test_nan_returns_default(self):
        assert compute_pue(float("nan"), 5000) == 1.0

    def test_typical_pue(self):
        pue = compute_pue(10000, 5000)
        assert 1.5 <= pue <= 1.6

    def test_pue_capped_at_10(self):
        pue = compute_pue(1, 1000000)
        assert pue <= 10.0


class TestCracPowerDraw:
    def test_zero_speed_zero_power(self):
        assert crac_power_draw(0, 0) == 0.0

    def test_full_speed_max_power(self):
        assert crac_power_draw(1.0, 1.0) == 5000.0 + 3000.0

    def test_cubic_fan_law(self):
        half = crac_power_draw(0.5, 0, max_pump_power_w=0)
        full = crac_power_draw(1.0, 0, max_pump_power_w=0)
        assert abs(half - full * 0.125) < 1.0  # 0.5^3 = 0.125

    def test_nan_returns_zero(self):
        assert crac_power_draw(float("nan"), 0.5) == 0.0

    def test_negative_clamped(self):
        result = crac_power_draw(-0.5, -0.5)
        assert result == 0.0


class TestComputeEffectiveH:
    def test_full_controls(self):
        h = compute_effective_h(1.0, 1.0, base_h=150)
        assert h == 150.0

    def test_zero_controls(self):
        assert compute_effective_h(0.0, 0.0) == 0.0

    def test_nan_returns_zero(self):
        assert compute_effective_h(float("nan"), 0.5) == 0.0
