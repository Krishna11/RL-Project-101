"""Tests for rewards.py — edge-case focused."""

import pytest

from coolpilot.rewards import (
    composite_reward,
    cost_score,
    energy_score,
    resilience_score,
    safety_score,
    stability_score,
)


class TestSafetyScore:
    def test_all_safe(self):
        assert safety_score([20.0, 22.0, 25.0]) == 1.0

    def test_one_hot(self):
        score = safety_score([20.0, 30.0])
        assert 0.0 < score < 1.0

    def test_way_too_hot(self):
        assert safety_score([35.0]) == 0.0

    def test_too_cold(self):
        score = safety_score([15.0])
        assert 0.0 < score < 1.0

    def test_empty_list(self):
        assert safety_score([]) == 1.0


class TestEnergyScore:
    def test_ideal_pue(self):
        assert energy_score(1.0) == 1.0

    def test_terrible_pue(self):
        assert energy_score(2.0) == 0.0

    def test_nan_pue(self):
        assert energy_score(float("nan")) == 0.0


class TestStabilityScore:
    def test_no_change(self):
        assert stability_score([22.0], [22.0]) == 1.0

    def test_big_swing(self):
        score = stability_score([25.0], [22.0])
        assert score == 0.0

    def test_empty_prev(self):
        assert stability_score([22.0], []) == 1.0

    def test_empty_current(self):
        assert stability_score([], [22.0]) == 1.0


class TestCostScore:
    def test_no_cooling(self):
        assert cost_score(0, 0.08, 1/60) == 1.0

    def test_expensive(self):
        score = cost_score(100_000, 0.25, 1/60)
        assert score < 1.0

    def test_zero_budget(self):
        assert cost_score(1000, 0.08, 1/60, budget_per_step=0) == 0.0


class TestResilienceScore:
    def test_no_failure(self):
        assert resilience_score(3, 3, [22.0]) == 1.0

    def test_failure_with_safe_temps(self):
        assert resilience_score(2, 3, [22.0, 23.0]) == 1.0

    def test_failure_with_violation(self):
        score = resilience_score(1, 3, [30.0])
        assert 0.0 < score < 1.0

    def test_zero_total_cracs(self):
        assert resilience_score(0, 0, [22.0]) == 0.0


class TestCompositeReward:
    def test_default_weights(self):
        r = composite_reward([22.0], [22.0], pue=1.2)
        assert 0.0 <= r <= 1.0

    def test_custom_weights(self):
        r = composite_reward(
            [22.0], [22.0], pue=1.2,
            weights={"safety": 1.0},
        )
        assert r == 1.0

    def test_empty_zones(self):
        r = composite_reward([], [], pue=1.5)
        assert 0.0 <= r <= 1.0
