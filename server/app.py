"""
FastAPI entry point — wraps the CoolPilot environment for OpenEnv.

Endpoints:
  POST  /reset   — start a new episode
  POST  /step    — take one action
  GET   /state   — episode metadata
  GET   /health  — liveness probe
  GET   /schema  — JSON Schema for Action / Observation
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from ..models import Action, CRACAction, Observation
from .environment import CoolPilotEnvironment, ResetRequest

logger = logging.getLogger(__name__)

# ── App + Environment ───────────────────────────────────

app = FastAPI(
    title="CoolPilot — DC Cooling RL Env",
    version="1.0.0",
    description=(
        "OpenEnv-compliant Data Center Cooling RL Environment.  "
        "Control HVAC/CRAC units to minimise PUE."
    ),
)

env = CoolPilotEnvironment()


# ── Endpoints ───────────────────────────────────────────

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> dict[str, Any]:
    """
    Reset the environment to begin a new episode.

    Edge cases:
      • No body → defaults to task_1_single_zone.
      • Invalid task_id → 400 error.
    """
    try:
        obs = env.reset(
            seed=request.seed if request else None,
            task_id=request.task_id if request else "task_1_single_zone",
        )
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /reset")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {exc}"
        ) from exc


@app.post("/step")
async def step(body: dict[str, Any]) -> dict[str, Any]:
    """
    Execute one step in the environment.

    Accepts raw JSON and validates it into an Action.

    Edge cases:
      • Invalid action JSON  → 422 with validation details.
      • step() before reset() → 400.
      • Episode already done  → returns final obs.
    """
    try:
        action = Action.model_validate(body)
    except ValidationError as exc:
        # Try to salvage: maybe the body IS the action (no wrapper)
        try:
            if "cracs" not in body:
                # Wrap bare CRAC list
                action = Action(cracs=[CRACAction.model_validate(body)])
            else:
                raise
        except Exception:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid action: {exc.error_count()} error(s).  "
                       f"Expected {{'cracs': [{{fan_speed, chilled_water_flow, supply_temp}}]}}",
            ) from exc

    try:
        obs = env.step(action)
        return obs.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /step")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {exc}"
        ) from exc


@app.get("/state")
async def state() -> dict[str, Any]:
    """Return episode-level metadata."""
    return env.state().model_dump()


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for container orchestrators."""
    return {"status": "ok"}


@app.get("/schema")
async def schema() -> dict[str, Any]:
    """Return JSON Schema for Action and Observation models."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
    }


# ── Direct run ──────────────────────────────────────────

def main() -> None:
    """Entry point for ``uv run server`` or ``python -m coolpilot.server.app``."""
    uvicorn.run(
        "coolpilot.server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
