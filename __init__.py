# Copyright (c) Team CoolPilot.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license.

"""
CoolPilot — Data Center Cooling RL Environment.

An OpenEnv-compliant environment for training RL agents to optimise
HVAC/CRAC controls in a simulated data center.

Public API
----------
Models (Pydantic v2):
    CRACAction, Action, ZoneObservation, CRACObservation,
    Observation, State

Client:
    CoolPilotEnv  — WebSocket-based ``EnvClient`` subclass

Example (async):
    >>> from CoolPilot import CoolPilotEnv, Action, CRACAction
    >>>
    >>> async with CoolPilotEnv(base_url="ws://localhost:7860") as env:
    ...     result = await env.reset(task_id="task_1_single_zone")
    ...     action = Action(cracs=[CRACAction(fan_speed=0.6, chilled_water_flow=0.5, supply_temp=15.0)])
    ...     result = await env.step(action)

Example (sync wrapper):
    >>> with CoolPilotEnv(base_url="ws://localhost:7860").sync() as env:
    ...     result = env.reset(task_id="task_1_single_zone")
"""

from .models import (
    Action,
    CRACAction,
    CRACObservation,
    Observation,
    State,
    ZoneObservation,
)
from .client import CoolPilotEnv

__all__ = [
    # ── Models ──
    "CRACAction",
    "Action",
    "ZoneObservation",
    "CRACObservation",
    "Observation",
    "State",
    # ── Client ──
    "CoolPilotEnv",
]
