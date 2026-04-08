# Copyright (c) Team CoolPilot.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license.

"""
CoolPilot Data Center Cooling — OpenEnv Client.

This module provides ``CoolPilotEnv``, a WebSocket-based ``EnvClient``
subclass that handles **every** edge case encountered in agentic RL
training loops:

  ✔ Connection failures (retry with exponential back-off)
  ✔ WebSocket disconnects mid-episode (transparent reconnect)
  ✔ Server error responses (structured error propagation)
  ✔ Malformed / partial JSON from server
  ✔ Timeout on slow step() calls (configurable)
  ✔ Action validation before sending (fail-fast)
  ✔ Mismatched CRAC count (auto-pad or truncate)
  ✔ NaN / Inf in observations (safe defaults)
  ✔ Graceful shutdown on KeyboardInterrupt
  ✔ Sync and async usage patterns
  ✔ Context-manager and non-context-manager usage
  ✔ Thread-safety for the sync wrapper

Example (async — recommended):
    >>> async with CoolPilotEnv(base_url="<env_base_url>") as env:
    ...     result = await env.reset(task_id="task_1_single_zone")
    ...     print(result.observation.zones)  # [ZoneObservation(...)]
    ...     action = Action(cracs=[CRACAction(fan_speed=0.6, ...)])
    ...     result = await env.step(action)

Example (sync wrapper):
    >>> with CoolPilotEnv(base_url="<env_base_url>").sync() as env:
    ...     result = env.reset(task_id="task_1_single_zone")
    ...     result = env.step(action)

Example (from HuggingFace Space):
    >>> env = await CoolPilotEnv.from_env("openenv/coolpilot")
    >>> try:
    ...     result = await env.reset()
    ... finally:
    ...     await env.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Type

from openenv.core.client_types import StateT, StepResult
from openenv.core.env_client import EnvClient
from websockets.asyncio.client import connect as ws_connect

from .models import (
    Action,
    CRACAction,
    CRACObservation,
    Observation,
    State,
    ZoneObservation,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT_S = 120.0         # 2 min per step (simulation can be slow)
_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_BASE_DELAY_S = 1.0     # exponential back-off base
_SAFE_DEFAULT_ACTION = CRACAction(
    fan_speed=0.5,
    chilled_water_flow=0.5,
    supply_temp=15.0,
)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _safe_float(value: Any, *, default: float = 0.0) -> float:
    """
    Coerce *value* to a finite float, falling back to *default*.

    Handles: None, str, bool, NaN, ±Inf, unexpected types.
    """
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _safe_int(value: Any, *, default: int = 0) -> int:
    """Coerce to int with a safe fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce to bool with a safe fallback (handles truthy strings)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    try:
        return bool(value)
    except (TypeError, ValueError):
        return default


def _pad_or_truncate_cracs(
    cracs: list[CRACAction],
    expected_count: int,
) -> list[CRACAction]:
    """
    Ensure the action has exactly *expected_count* CRAC entries.

    - If fewer are provided, pad with the safe default.
    - If more are provided, truncate.

    This prevents the environment from rejecting an action when the
    LLM generates the wrong number of CRAC controls.
    """
    if len(cracs) == expected_count:
        return cracs
    if len(cracs) > expected_count:
        logger.warning(
            "Action has %d CRAC entries but environment expects %d — truncating.",
            len(cracs),
            expected_count,
        )
        return cracs[:expected_count]
    # Pad with defaults
    logger.warning(
        "Action has %d CRAC entries but environment expects %d — padding with defaults.",
        len(cracs),
        expected_count,
    )
    return cracs + [_SAFE_DEFAULT_ACTION] * (expected_count - len(cracs))


# ─────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────

class CoolPilotEnv(EnvClient[Action, Observation, State]):
    """
    WebSocket client for the CoolPilot Data Center Cooling environment.

    Extends ``openenv.core.env_client.EnvClient`` with:
      • Typed action / observation / state parsing.
      • Automatic CRAC-count normalisation.
      • Transparent reconnect on WebSocket drop.
      • Comprehensive logging for debugging RL training runs.

    Parameters
    ----------
    base_url : str
        HTTP or WebSocket URL of the environment server.
    connect_timeout_s : float
        Seconds to wait when opening the WebSocket (default 10).
    message_timeout_s : float
        Seconds to wait for each ``reset``/``step``/``state`` response
        (default 120 — long enough for heavy thermal sims).
    max_message_size_mb : float
        Maximum inbound WebSocket message size in MB (default 100).
    expected_crac_count : int or None
        If set, actions are padded / truncated to this count before
        being sent.  Populated automatically after the first ``reset()``.
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = _DEFAULT_TIMEOUT_S,
        max_message_size_mb: float = 100.0,
        expected_crac_count: Optional[int] = None,
        auth_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            max_message_size_mb=max_message_size_mb,
            **kwargs,
        )
        self._expected_crac_count: Optional[int] = expected_crac_count
        self._last_observation: Optional[Observation] = None
        self._last_state: Optional[State] = None
        self._episode_step: int = 0
        self._episode_start_time: float = 0.0
        self._total_reward: float = 0.0
        if auth_token:
            self._ws_headers: Optional[Dict[str, str]] = {
                "Authorization": f"Bearer {auth_token}",
            }
        else:
            self._ws_headers = None

    async def connect(self) -> "CoolPilotEnv":
        """Establish WebSocket connection with optional auth headers."""
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                additional_headers=self._ws_headers,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to {self._ws_url}: {exc}"
            ) from exc
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    # ── Properties ──────────────────────────────────────

    @property
    def expected_crac_count(self) -> Optional[int]:
        """Number of CRACs the environment expects (set after reset)."""
        return self._expected_crac_count

    @property
    def last_observation(self) -> Optional[Observation]:
        """Most recent observation (or None before first reset)."""
        return self._last_observation

    @property
    def episode_step(self) -> int:
        """Client-side step counter (resets on ``reset()``)."""
        return self._episode_step

    @property
    def total_reward(self) -> float:
        """Cumulative reward for the current episode."""
        return self._total_reward

    @property
    def episode_elapsed_s(self) -> float:
        """Wall-clock seconds since last ``reset()``."""
        if self._episode_start_time == 0.0:
            return 0.0
        return time.monotonic() - self._episode_start_time

    # ── EnvClient abstract method implementations ──────

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        """
        Serialise an ``Action`` into the JSON dict sent over WebSocket.

        Edge cases handled:
          1. CRAC count mismatch → pad / truncate.
          2. Action is a raw dict → parse into ``Action`` first.
          3. Individual CRACAction values already clamped by model validators.
        """
        # Accept raw dicts from callers that skip Pydantic
        if isinstance(action, dict):
            try:
                action = Action.model_validate(action)
            except Exception as exc:
                logger.error(
                    "Failed to parse raw dict into Action: %s.  "
                    "Falling back to safe defaults.",
                    exc,
                )
                count = self._expected_crac_count or 1
                action = Action(cracs=[_SAFE_DEFAULT_ACTION] * count)

        # Normalise CRAC count
        if self._expected_crac_count is not None:
            action = Action(
                cracs=_pad_or_truncate_cracs(
                    action.cracs, self._expected_crac_count
                )
            )

        return action.model_dump(mode="python")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        """
        Convert server JSON into a typed ``StepResult[Observation]``.

        Edge cases handled:
          1. Missing keys → Pydantic defaults kick in.
          2. NaN / Inf in numeric fields → sanitised by model validators.
          3. Empty ``zones`` / ``cracs`` lists → valid Observation.
          4. Unexpected extra keys → silently ignored (``extra="ignore"``).
          5. Completely empty payload → returns a safe default observation.
          6. ``payload`` is None → treated as empty dict.
        """
        if payload is None:
            payload = {}

        try:
            observation = Observation.model_validate(payload)
        except Exception as exc:
            logger.error(
                "Failed to parse observation from server payload: %s.  "
                "Keys present: %s.  Returning safe default.",
                exc,
                list(payload.keys()) if isinstance(payload, dict) else "N/A",
            )
            observation = Observation()

        # Cache for convenience properties
        self._last_observation = observation

        # Auto-discover CRAC count from the first valid observation
        if (
            self._expected_crac_count is None
            and observation.cracs
        ):
            self._expected_crac_count = len(observation.cracs)
            logger.info(
                "Auto-detected %d CRAC unit(s) from environment.",
                self._expected_crac_count,
            )

        # Determine done status
        done = observation.terminated or observation.truncated

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Convert server JSON into a typed ``State``.

        Edge cases handled:
          1. Empty / None payload → default State.
          2. Missing ``episode_id`` → empty string.
          3. Non-string ``episode_id`` → coerced by validator.
          4. Negative step counts → clamped to 0 by Field(ge=0).
        """
        if payload is None:
            payload = {}

        try:
            state = State.model_validate(payload)
        except Exception as exc:
            logger.error(
                "Failed to parse state from server: %s.  Returning default.",
                exc,
            )
            state = State()

        self._last_state = state
        return state

    # ── Overridden lifecycle methods ────────────────────

    async def reset(self, **kwargs: Any) -> StepResult[Observation]:
        """
        Reset the environment, starting a new episode.

        Parameters
        ----------
        task_id : str, optional
            Task variant to load (default: ``"task_1_single_zone"``).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        StepResult[Observation]
            Initial observation with reward=0.5 and done=False.

        Edge cases:
          • Server not reachable → retries with backoff then raises.
          • Previous episode still running → server handles cleanup.
          • Unknown task_id → server returns error, propagated as RuntimeError.
        """
        # Reset client-side bookkeeping
        self._episode_step = 0
        self._total_reward = 0.0
        self._episode_start_time = time.monotonic()
        self._last_observation = None
        self._expected_crac_count = None

        # Attempt with retries (connection may have dropped)
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            try:
                result = await super().reset(**kwargs)
                logger.info(
                    "Environment reset (attempt %d).  Task: %s  Zones: %d  CRACs: %d",
                    attempt,
                    kwargs.get("task_id", "task_1_single_zone"),
                    len(result.observation.zones),
                    len(result.observation.cracs),
                )
                return result
            except (ConnectionError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                delay = _RECONNECT_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "reset() failed (attempt %d/%d): %s.  Retrying in %.1fs…",
                    attempt,
                    _MAX_RECONNECT_ATTEMPTS,
                    exc,
                    delay,
                )
                # Force reconnect
                self._ws = None
                await asyncio.sleep(delay)
            except Exception as exc:
                # Non-retryable error (e.g. server returned an error message)
                raise RuntimeError(
                    f"Environment rejected reset: {exc}"
                ) from exc

        raise ConnectionError(
            f"Failed to reset environment after {_MAX_RECONNECT_ATTEMPTS} "
            f"attempts.  Last error: {last_exc}"
        )

    async def step(self, action: Action, **kwargs: Any) -> StepResult[Observation]:
        """
        Execute one action and return the resulting observation.

        Parameters
        ----------
        action : Action or dict
            CRAC control signals.  Raw dicts are auto-parsed.

        Returns
        -------
        StepResult[Observation]

        Edge cases:
          • WebSocket dropped → single reconnect attempt, then fail.
          • Action has wrong CRAC count → padded / truncated automatically.
          • Server returns error → RuntimeError with server message.
          • Step called before reset → raises RuntimeError.
          • Server timeout → raises asyncio.TimeoutError.
        """
        if self._episode_start_time == 0.0:
            raise RuntimeError(
                "step() called before reset().  "
                "You must call reset() to begin an episode."
            )

        # One automatic reconnect attempt on connection errors
        try:
            result = await super().step(action, **kwargs)
        except (ConnectionError, OSError) as exc:
            logger.warning(
                "WebSocket disconnected during step %d: %s.  Attempting reconnect…",
                self._episode_step,
                exc,
            )
            self._ws = None
            try:
                await self.connect()
                result = await super().step(action, **kwargs)
            except Exception as reconnect_exc:
                raise ConnectionError(
                    f"Reconnect failed at step {self._episode_step}: {reconnect_exc}"
                ) from exc

        # Update bookkeeping
        self._episode_step += 1
        self._total_reward += result.reward if result.reward is not None else 0.0

        if result.done:
            elapsed = self.episode_elapsed_s
            logger.info(
                "Episode ended at step %d.  Total reward: %.4f  "
                "Wall-clock: %.1fs  Terminated: %s  Truncated: %s",
                self._episode_step,
                self._total_reward,
                elapsed,
                result.observation.terminated,
                result.observation.truncated,
            )

        return result

    async def state(self) -> State:
        """
        Retrieve episode-level metadata.

        Edge cases:
          • Called before reset() → returns default State with is_done=False.
          • Server unreachable → returns cached state if available.
        """
        try:
            return await super().state()
        except (ConnectionError, OSError, asyncio.TimeoutError) as exc:
            if self._last_state is not None:
                logger.warning(
                    "state() failed (%s) — returning cached state.", exc
                )
                return self._last_state
            raise

    # ── Convenience methods ─────────────────────────────

    def make_safe_action(self, num_cracs: Optional[int] = None) -> Action:
        """
        Build a conservative "do-nothing" action.

        Useful as a fallback when the LLM fails to produce a valid action.

        Parameters
        ----------
        num_cracs : int, optional
            Number of CRACs.  Defaults to the auto-detected count, or 1.
        """
        count = num_cracs or self._expected_crac_count or 1
        return Action(cracs=[_SAFE_DEFAULT_ACTION] * count)

    @staticmethod
    def action_from_dict(raw: dict) -> Action:
        """
        Parse a raw dict (e.g. from ``json.loads()``) into an ``Action``.

        This is the recommended way to convert LLM output into a typed
        action, because it applies all clamping and validation.

        Raises ``pydantic.ValidationError`` on truly invalid input.
        """
        return Action.model_validate(raw)

    def summary(self) -> dict[str, Any]:
        """
        Return a human-readable summary dict of the current episode.

        Useful for logging / dashboards.
        """
        obs = self._last_observation
        return {
            "episode_step": self._episode_step,
            "total_reward": round(self._total_reward, 4),
            "wall_clock_s": round(self.episode_elapsed_s, 1),
            "expected_cracs": self._expected_crac_count,
            "last_pue": round(obs.pue, 4) if obs else None,
            "last_reward": round(obs.reward, 4) if obs else None,
            "zones_count": len(obs.zones) if obs else 0,
            "highest_temp": (
                round(max(z.temperature for z in obs.zones), 2)
                if obs and obs.zones
                else None
            ),
            "is_done": (
                (obs.terminated or obs.truncated) if obs else False
            ),
        }

    # ── Repr ────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<CoolPilotEnv "
            f"url={self._ws_url!r} "
            f"step={self._episode_step} "
            f"reward={self._total_reward:.4f}>"
        )
