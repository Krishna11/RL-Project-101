"""Tests for models.py — Pydantic v2 validation edge cases."""

import math
import pytest
from pydantic import ValidationError

from coolpilot.models import (
    Action,
    CRACAction,
    CRACObservation,
    Observation,
    State,
    ZoneObservation,
)


class TestCRACAction:
    def test_valid_action(self):
        a = CRACAction(fan_speed=0.6, chilled_water_flow=0.5, supply_temp=15.0)
        assert a.fan_speed == 0.6

    def test_clamps_fan_below_minimum(self):
        a = CRACAction(fan_speed=0.05, chilled_water_flow=0.5, supply_temp=15.0)
        assert a.fan_speed == 0.1

    def test_clamps_supply_temp(self):
        a = CRACAction(fan_speed=0.5, chilled_water_flow=0.5, supply_temp=8.0)
        assert a.supply_temp == 10.0

    def test_clamps_high_supply_temp(self):
        a = CRACAction(fan_speed=0.5, chilled_water_flow=0.5, supply_temp=22.0)
        assert a.supply_temp == 20.0

    def test_rejects_nan(self):
        with pytest.raises(ValidationError):
            CRACAction(fan_speed=float("nan"), chilled_water_flow=0.5, supply_temp=15.0)

    def test_rejects_inf(self):
        with pytest.raises(ValidationError):
            CRACAction(fan_speed=float("inf"), chilled_water_flow=0.5, supply_temp=15.0)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            CRACAction(fan_speed=0.5, chilled_water_flow=0.5, supply_temp=15.0, bogus=1)

    def test_coerces_string_to_float(self):
        a = CRACAction(fan_speed="0.5", chilled_water_flow="0.5", supply_temp="15.0")
        assert a.fan_speed == 0.5


class TestAction:
    def test_valid(self):
        a = Action(cracs=[CRACAction(fan_speed=0.5, chilled_water_flow=0.5, supply_temp=15.0)])
        assert len(a.cracs) == 1

    def test_rejects_empty_cracs(self):
        with pytest.raises(ValidationError):
            Action(cracs=[])

    def test_model_validate_dict(self):
        d = {"cracs": [{"fan_speed": 0.5, "chilled_water_flow": 0.5, "supply_temp": 15.0}]}
        a = Action.model_validate(d)
        assert a.cracs[0].fan_speed == 0.5


class TestObservation:
    def test_reward_clamped(self):
        obs = Observation(reward=1.5)
        assert obs.reward <= 1.0

    def test_negative_reward_clamped(self):
        obs = Observation(reward=-0.5)
        assert obs.reward >= 0.0

    def test_nan_reward_becomes_zero(self):
        obs = Observation(reward=float("nan"))
        assert obs.reward == 0.0

    def test_pue_below_one_clamped(self):
        obs = Observation(pue=0.5)
        assert obs.pue >= 1.0

    def test_defaults(self):
        obs = Observation()
        assert obs.step_number == 0
        assert obs.terminated is False
        assert obs.zones == []


class TestState:
    def test_none_episode_id(self):
        s = State(episode_id=None)
        assert s.episode_id == ""

    def test_int_episode_id(self):
        s = State(episode_id=12345)
        assert s.episode_id == "12345"

    def test_defaults(self):
        s = State()
        assert s.is_done is False
        assert s.step_count == 0
