#!/usr/bin/env python3
"""
CoolPilot Inference Script — OpenEnv Hackathon Submission.

MANDATORY STDOUT FORMAT:
    [START] task=<task_name> env=coolpilot model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Environment Variables:
    API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Hugging Face / API key
    IMAGE_NAME     Docker image name for from_docker_image() (optional)
    TASK_ID        Task to run (default: task_1_single_zone)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import textwrap
import time
from typing import Any, List, Optional

from openai import OpenAI

from coolpilot import CoolPilotEnv, Action, CRACAction

# ── Config ──────────────────────────────────────────────

if "API_BASE_URL" not in os.environ:
    os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = os.environ.get("HF_TOKEN", "dummy")

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_ID = os.getenv("TASK_ID", "task_1_single_zone")
BENCHMARK = "coolpilot"

MAX_STEPS = 200          # safety cap
TIME_BUDGET_S = 900.0    # 15 min
TEMPERATURE = 0.2
MAX_TOKENS = 512

# Task-specific max steps for score normalization
TASK_MAX_STEPS = {
    "task_1_single_zone": 60,
    "task_2_variable_workload": 120,
    "task_3_random_events": 180,
}

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert data center cooling controller.

You control HVAC/CRAC units to minimise energy (PUE) while keeping
all server rack zones within safe temperature bounds (18–27°C).

Each step you receive zone temperatures, CRAC settings, PUE, and ambient temp.

Respond with ONLY a JSON object (no markdown, no explanation):
{
  "cracs": [
    {"fan_speed": 0.1-1.0, "chilled_water_flow": 0.1-1.0, "supply_temp": 10.0-20.0}
  ]
}

Strategy:
- If any zone > 25°C: increase fan_speed and chilled_water_flow, lower supply_temp
- If all zones < 22°C: reduce cooling to save energy
- Make gradual adjustments to avoid oscillation
""").strip()


# ─────────────────────────────────────────────────────────
# Logging helpers (EXACT required format)
# ─────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────────────────
# JSON extraction from LLM output
# ─────────────────────────────────────────────────────────

def parse_action_json(content: str) -> dict:
    """Extract JSON from LLM response, handling fences and trailing text."""
    content = content.strip()

    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1]

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    start = content.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Cannot extract JSON from: {content[:200]}")


# ─────────────────────────────────────────────────────────
# Observation formatting
# ─────────────────────────────────────────────────────────

def format_observation(obs: dict) -> str:
    """Format observation dict for LLM prompt."""
    lines = [
        f"Step {obs.get('step_number', '?')} | "
        f"PUE: {obs.get('pue', 0):.3f} | "
        f"Ambient: {obs.get('ambient_temp', 0):.1f}°C | "
        f"Reward: {obs.get('reward', 0):.4f}"
    ]
    for z in obs.get("zones", []):
        t = z.get("temperature", 0)
        flag = "SAFE" if 18 <= t <= 27 else "DANGER"
        lines.append(
            f"  Zone {z.get('zone_id')}: {t:.1f}°C "
            f"({z.get('it_power_w', 0)/1000:.1f}kW) [{flag}]"
        )
    for c in obs.get("cracs", []):
        status = "ONLINE" if c.get("is_online", True) else "OFFLINE"
        lines.append(
            f"  CRAC {c.get('crac_id')}: fan={c.get('fan_speed', 0):.2f} "
            f"water={c.get('chilled_water_flow', 0):.2f} "
            f"supply={c.get('supply_temp', 0):.1f}°C [{status}]"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# PID fallback controller
# ─────────────────────────────────────────────────────────

def pid_action(obs: dict) -> dict:
    """Proportional controller targeting 22.5°C — always works."""
    zones = obs.get("zones", [])
    if not zones:
        return {
            "cracs": [
                {"fan_speed": 0.5, "chilled_water_flow": 0.5, "supply_temp": 15.0}
            ]
        }

    avg_temp = sum(z.get("temperature", 22.0) for z in zones) / len(zones)
    error = avg_temp - 22.5

    fan = max(0.1, min(1.0, 0.5 + error * 0.10))
    water = max(0.1, min(1.0, 0.5 + error * 0.08))
    supply = max(10.0, min(20.0, 15.0 - error * 0.50))

    return {
        "cracs": [
            {
                "fan_speed": round(fan, 2),
                "chilled_water_flow": round(water, 2),
                "supply_temp": round(supply, 1),
            }
            for _ in obs.get("cracs", [{"crac_id": 0}])
        ]
    }


def action_to_short_str(action: dict) -> str:
    """Compact string representation of an action for [STEP] logging."""
    cracs = action.get("cracs", [])
    if not cracs:
        return "no-op"
    c = cracs[0]
    return (
        f"fan={c.get('fan_speed', 0):.2f},"
        f"water={c.get('chilled_water_flow', 0):.2f},"
        f"supply={c.get('supply_temp', 0):.1f}"
    )


# ─────────────────────────────────────────────────────────
# Main async inference loop
# ─────────────────────────────────────────────────────────

async def run_episode(task_id: str) -> dict:
    """
    Run one full episode on the given task.

    Returns a result dict with score in [0, 1].
    """
    start_time = time.monotonic()
    llm_client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )

    # ── Connect to environment ──────────────────────────
    if LOCAL_IMAGE_NAME:
        env = await CoolPilotEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = CoolPilotEnv(base_url=ENV_BASE_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env:
            # ── Reset ───────────────────────────────────
            result = await env.reset(task_id=task_id)
            obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else {}
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            use_pid = False

            # ── Step loop ───────────────────────────────
            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                elapsed = time.monotonic() - start_time
                if elapsed > TIME_BUDGET_S * 0.85 and not use_pid:
                    use_pid = True

                # ── Get action ──────────────────────────
                action_dict: Optional[dict] = None
                last_error = None

                if not use_pid:
                    obs_text = format_observation(obs_dict)
                    messages.append({
                        "role": "user",
                        "content": f"Current state:\n{obs_text}\n\nProvide action JSON.",
                    })

                    try:
                        completion = llm_client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=TEMPERATURE,
                            max_tokens=MAX_TOKENS,
                            stream=False,
                        )
                        reply = (completion.choices[0].message.content or "").strip()
                        messages.append({"role": "assistant", "content": reply})
                        action_dict = parse_action_json(reply)
                    except Exception as exc:
                        last_error = str(exc)
                        # Remove unanswered user message
                        if messages[-1]["role"] == "user":
                            messages.pop()

                # Fallback to PID
                if action_dict is None:
                    action_dict = pid_action(obs_dict)

                # ── Execute step ────────────────────────
                try:
                    action = Action.model_validate(action_dict)
                    result = await env.step(action)
                except Exception as exc:
                    last_error = str(exc)
                    # Emergency PID retry
                    action_dict = pid_action(obs_dict)
                    action = Action.model_validate(action_dict)
                    result = await env.step(action)

                obs_dict = result.observation.model_dump() if hasattr(result.observation, 'model_dump') else {}
                reward = result.reward if result.reward is not None else 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_to_short_str(action_dict),
                    reward=reward,
                    done=done,
                    error=last_error,
                )

                # Sliding window for LLM context
                if len(messages) > 20:
                    messages = messages[:1] + messages[-10:]

                if done:
                    break

            # ── Compute score ───────────────────────────
            if rewards:
                score = sum(rewards) / len(rewards)  # average reward
            score = max(0.0, min(1.0, score))
            success = score >= 0.5

    except Exception as exc:
        last_error = str(exc)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "steps": steps_taken,
        "score": round(score, 2),
        "success": success,
        "rewards": rewards,
    }


async def main() -> None:
    """Run inference on the specified task (or all 3)."""
    tasks = os.getenv("TASKS", TASK_ID).split(",")

    for task_id in tasks:
        task_id = task_id.strip()
        await run_episode(task_id)


if __name__ == "__main__":
    asyncio.run(main())
