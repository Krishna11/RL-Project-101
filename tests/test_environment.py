"""Tests for server/environment.py — full reset/step/state lifecycle."""

import pytest

from coolpilot.models import Action, CRACAction
from coolpilot.server.environment import CoolPilotEnvironment


@pytest.fixture
def env():
    return CoolPilotEnvironment()


class TestReset:
    def test_basic_reset(self, env):
        obs = env.reset(task_id="task_1_single_zone")
        assert len(obs.zones) == 1
        assert len(obs.cracs) == 1
        assert obs.step_number == 0

    def test_unknown_task(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent_task")

    def test_reset_clears_previous(self, env):
        env.reset(task_id="task_1_single_zone")
        obs = env.reset(task_id="task_2_variable_workload")
        assert len(obs.zones) == 4


class TestStep:
    def test_single_step(self, env):
        env.reset(task_id="task_1_single_zone")
        action = Action(cracs=[CRACAction(fan_speed=0.7, chilled_water_flow=0.6, supply_temp=14.0)])
        obs = env.step(action)
        assert obs.step_number == 1
        assert 0.0 <= obs.reward <= 1.0

    def test_step_before_reset_raises(self, env):
        action = Action(cracs=[CRACAction(fan_speed=0.5, chilled_water_flow=0.5, supply_temp=15.0)])
        with pytest.raises(RuntimeError, match="reset"):
            env.step(action)

    def test_episode_terminates(self, env):
        env.reset(task_id="task_1_single_zone")
        action = Action(cracs=[CRACAction(fan_speed=0.1, chilled_water_flow=0.1, supply_temp=20.0)])

        done = False
        for _ in range(200):
            obs = env.step(action)
            if obs.terminated or obs.truncated:
                done = True
                break
        assert done

    def test_step_after_done_safe(self, env):
        env.reset(task_id="task_1_single_zone")
        action = Action(cracs=[CRACAction(fan_speed=0.5, chilled_water_flow=0.5, supply_temp=15.0)])

        # Run to completion
        for _ in range(70):
            obs = env.step(action)
            if obs.terminated or obs.truncated:
                break

        # Extra step after done should not crash
        obs = env.step(action)
        assert obs.terminated or obs.truncated


class TestState:
    def test_state_before_reset(self, env):
        s = env.state()
        assert s.is_done is False

    def test_state_after_reset(self, env):
        env.reset(task_id="task_1_single_zone")
        s = env.state()
        assert s.task_id == "task_1_single_zone"
        assert s.step_count == 0
