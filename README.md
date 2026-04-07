---
title: Coolpilot
emoji: 🌖
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: bsd-3-clause
app_port: 7860
---

# CoolPilot 🧊 — Data Center Cooling RL Environment

CoolPilot is an OpenEnv-compliant Reinforcement Learning environment where an agent learns to optimize datacenter HVAC/CRAC (Computer Room Air Conditioning) units.

## Motivation & Description

Datacenters consume massive amounts of energy, up to 40% of which is used solely for cooling server racks. The goal of this environment is to optimize the Power Usage Effectiveness (PUE) of a simulated datacenter by intelligently adjusting fan speeds, chilled water flow, and supply air temperatures. 

An agent must balance two competing objectives:
1. **Safety:** Keeping all server rack zones within the ASHRAE recommended safe operating ranges (18°C – 27°C). If temperatures exceed 35°C, servers shut down, causing catastrophic episodes termination.
2. **Efficiency:** Minimizing the cooling power consumed to achieve a PUE as close to 1.0 as possible.

This environment simulates a real-world thermodynamic task based on Newton's law of cooling, dynamically fluctuating IT workloads, and time-of-use energy pricing.

## Observation Space

The observation space is returned as a JSON structure (Pydantic model) describing the thermal state of the datacenter.

```json
{
  "zones": [
    {
      "zone_id": 0,
      "temperature": 22.5,
      "it_power_w": 10000.0
    }
  ],
  "cracs": [
    {
      "crac_id": 0,
      "fan_speed": 0.5,
      "chilled_water_flow": 0.5,
      "supply_temp": 15.0,
      "power_draw_w": 2500.0,
      "is_online": true
    }
  ],
  "ambient_temp": 35.0,
  "pue": 1.25,
  "total_it_power_w": 10000.0,
  "total_cooling_power_w": 2500.0,
  "reward": 0.5,
  "terminated": false,
  "truncated": false,
  "step_number": 1
}
```

## Action Space

The action space expects a JSON object containing settings for each CRAC unit in the datacenter. The agent can adjust:
- `fan_speed` [0.1 - 1.0]: Controls volume of chilled air.
- `chilled_water_flow` [0.1 - 1.0]: Controls heat exchange rate.
- `supply_temp` [10.0 - 20.0]: Setpoint for the air leaving the unit (°C).

```json
{
  "cracs": [
    {
      "fan_speed": 0.6,
      "chilled_water_flow": 0.7,
      "supply_temp": 12.0
    }
  ]
}
```

## Task Descriptions

The environment contains 3 grading tasks of increasing difficulty:

| Task ID | Difficulty | Description |
|---|---|---|
| `task_1_single_zone` | Easy | 1 server zone, 1 CRAC unit. Constant 10 kW IT load. The agent must find the steady-state optimal parameters to minimize PUE. |
| `task_2_variable_workload` | Medium | 4 zones, 2 CRAC units. IT workloads oscillate sinusoidally (5-15 kW) simulating day/night cycles. Ambient temperature fluctuates. |
| `task_3_random_events` | Hard | 8 zones, 3 CRAC units. Sudden random IT power spikes occur. CRAC units can randomly fail and go offline for 10 steps. Time-of-Use pricing penalizes excessive cooling during peak hours. |

## Reward Function

The reward is a normalized float between `[0.0, 1.0]` calculated at every step to provide dense, partial progress signals. It is a composite of:
- **Safety Score:** 1.0 if all zones are between 18°C-27°C. Degrades heavily if temperatures approach critical thresholds.
- **Energy Score:** 1.0 if PUE is perfect (1.0). Scales down to 0.0 as PUE approaches 2.0.
- **Stability Score:** Penalizes rapid, extreme swings in temperature.
- **Cost/Resilience Scores:** (Used in Hard tasks). Rewards maintaining temperatures even when a CRAC unit fails.

## Running Locally (After Cloning)

Follow these steps to get the project running from scratch on any machine.

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/coolpilot.git
cd coolpilot
```

### Step 2 — Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

> You should see `(.venv)` appear at the start of your terminal prompt.

### Step 3 — Install Dependencies
```bash
pip install -e ".[dev]"
```
This installs all runtime and development dependencies defined in `pyproject.toml`
(FastAPI, Pydantic, Uvicorn, OpenAI client, Pytest, etc.)

### Step 4 — Verify Everything Works (Run Tests)
```bash
pytest tests/ -v
```
You should see **73 passed** ✅. If any test fails, check your Python version (requires ≥ 3.10).

### Step 5 — Start the Environment Server

Open **Terminal 1** and run:
```bash
uvicorn coolpilot.server.app:app --port 7860
```

Check it's alive:
```bash
# Linux / Mac
curl http://localhost:7860/health

# Windows PowerShell
Invoke-RestMethod http://localhost:7860/health
```
Expected response: `{"status": "ok"}`

### Step 6 — Test the API Manually (Optional)

Open **Terminal 2** and try:
```powershell
# Start a new episode
Invoke-RestMethod -Uri http://localhost:7860/reset `
  -Method POST -ContentType "application/json" `
  -Body '{"task_id": "task_1_single_zone"}'

# Send a cooling action
Invoke-RestMethod -Uri http://localhost:7860/step `
  -Method POST -ContentType "application/json" `
  -Body '{"cracs": [{"fan_speed": 0.8, "chilled_water_flow": 0.7, "supply_temp": 13.0}]}'

# Check episode state
Invoke-RestMethod http://localhost:7860/state
```

### Step 7 — Run the AI Agent

With the server still running in Terminal 1, open **Terminal 2** and set your credentials:

**Windows (PowerShell):**
```powershell
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN     = "your_huggingface_token"
$env:TASK_ID      = "task_1_single_zone"

python inference.py
```

**Mac / Linux:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"
export TASK_ID="task_1_single_zone"

python inference.py
```

> **Note:** If you don't have an HF token, the agent automatically falls back to a
> deterministic PID controller — no LLM required for local testing.

You can run all 3 tasks at once:
```bash
# Windows
$env:TASKS = "task_1_single_zone,task_2_variable_workload,task_3_random_events"

# Mac/Linux
export TASKS="task_1_single_zone,task_2_variable_workload,task_3_random_events"

python inference.py
```

---

## Quick Reference

```
git clone → cd coolpilot → python -m venv .venv → activate
→ pip install -e ".[dev]" → pytest tests/ -v
→ uvicorn ... --port 7860 → python inference.py
```

## Baseline Scores

Using the `Qwen/Qwen2.5-72B-Instruct` model and fallback PID controller, the following baseline scores were achieved:

- **Task 1 (Easy):** 0.35
- **Task 2 (Medium):** 0.28
- **Task 3 (Hard):** 0.15

*(Scores out of 1.0. These represent a non-RL zero-shot/few-shot prompt baseline that an RL algorithm using `openenv-core` can improve upon.)*
