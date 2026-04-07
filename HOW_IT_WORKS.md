# 🧊 CoolPilot — How It Works (Simple Guide)

## What is this project?

Imagine a **data center** — a big room full of servers (computers) that get
really hot. If they overheat (above 35°C), they shut down and bad things happen.

**CoolPilot** is a game where an AI agent learns to control the air conditioning
(called CRAC units) to:
- ✅ Keep servers cool (18–27°C safe zone)
- ✅ Use as little energy as possible (save electricity)

It's like a thermostat, but way smarter.

---

## The Big Picture (5 parts)

```
   🤖 AI Agent                    🖥️ Server                    🌡️ Simulation
  (inference.py)              (server/app.py)           (thermal/physics.py)
       |                            |                          |
       |--- "set fan to 60%" ------>|                          |
       |                            |--- runs physics -------->|
       |                            |<-- new temperature ------|
       |<-- "zone is now 24°C" ----|                          |
       |                            |                          |
       |--- "set fan to 40%" ------>|  ... repeats 60 times ...|
```

**That's it.** The AI sends cooling settings → physics simulates what happens →
AI sees the result → AI adjusts → repeat.

---

## File by File (what each file does)

### 📦 Config files (boring but necessary)
| File | What it does |
|------|-------------|
| `openenv.yaml` | Tells OpenEnv "hey, I'm a CoolPilot environment, run me on port 7860" |
| `pyproject.toml` | Lists what Python packages we need (fastapi, pydantic, etc.) |

### 🧠 The Brain
| File | What it does |
|------|-------------|
| `models.py` | Defines the **shape** of data. Like a form: "fan_speed must be 0.1 to 1.0" |
| `client.py` | How to **talk** to the server from Python code. Handles connection drops, retries, etc. |
| `inference.py` | The **AI agent**. Asks an LLM "what should I do?" or falls back to a simple math formula (PID) |

### 🌡️ The Physics (thermal/ folder)
| File | What it does |
|------|-------------|
| `constants.py` | Magic numbers: "safe temp is 18-27°C", "time step is 60 seconds" |
| `physics.py` | The actual math: Newton's cooling law. "If fan is at 60%, temperature drops by X degrees" |
| `datacenter.py` | The data center model: zones (groups of servers) and CRAC units (air conditioners) |

### 🎮 The Game Levels (tasks/ folder)
| File | Difficulty | What's different |
|------|-----------|-----------------|
| `task1_single_zone.py` | 🟢 Easy | 1 zone, 1 AC unit, constant load |
| `task2_variable_workload.py` | 🟡 Medium | 4 zones, 2 ACs, load goes up and down like a wave |
| `task3_random_events.py` | 🔴 Hard | 8 zones, 3 ACs, random spikes, AC units can break! |

### 🖥️ The Server (server/ folder)
| File | What it does |
|------|-------------|
| `app.py` | The **web server**. Has 3 buttons: `/reset` (start new game), `/step` (take action), `/state` (check score) |
| `environment.py` | Connects everything: takes your action → runs physics → calculates reward → sends back result |
| `Dockerfile` | Recipe to package everything in a container for Hugging Face Spaces |

### 📊 Scoring
| File | What it does |
|------|-------------|
| `rewards.py` | Calculates how well you did. Are zones safe? Is energy usage low? Score from 0 to 1. |
| `grader.py` | Final grade at the end of an episode |

---

## How One "Game" Plays Out

```
Step 1: RESET
   → Server creates a data center with zones at 22°C
   → Sends back: "here's what the data center looks like"

Step 2-60: STEP (repeat)
   → AI looks at temperatures and says:
     "Set fan to 70%, water flow to 50%, supply air to 14°C"
   → Server runs the physics for 60 seconds of simulated time
   → Temperatures change based on:
       HEATING: servers produce heat (10,000 watts!)
       COOLING: AC removes heat (depends on fan speed)
   → Server calculates reward:
       safety_score: all zones in 18-27°C? → 1.0 (good!) or 0.0 (bad!)
       energy_score: PUE close to 1.0? → 1.0 (efficient!) or 0.0 (wasteful!)
   → Sends back new temperatures + reward

Step 61: DONE
   → Episode ends (truncated after 60 steps)
   → Final grade calculated
```

---

## The Reward Formula (how scoring works)

The AI gets a score from **0.0** (terrible) to **1.0** (perfect) each step:

```
reward = (weight₁ × safety) + (weight₂ × energy) + ...
```

| Sub-score | What it measures | Perfect score when... |
|-----------|-----------------|----------------------|
| **Safety** | Are zones in 18-27°C? | All zones in safe range |
| **Energy** | Is PUE low? | PUE = 1.0 (impossible, but closer is better) |
| **Stability** | Are temps jumping around? | Temps barely change between steps |
| **Cost** | Is electricity cheap right now? | Using less power during peak hours |
| **Resilience** | Can you handle AC breakdowns? | Zones stay safe even when an AC dies |

**PUE** = Power Usage Effectiveness = Total Power / IT Power.
- PUE of 1.0 = perfect (all power goes to computing)
- PUE of 2.0 = bad (half the power is wasted on cooling)

---

## How to Run It

### Quick test (30 seconds)
```powershell
cd "c:\Users\kkuna\Desktop\meta x sst hackthon\coolpilot"

# Run all tests
python -m pytest tests/ -v
```

### Run the full thing (2 minutes)

**Terminal 1 — Start the server:**
```powershell
cd "c:\Users\kkuna\Desktop\meta x sst hackthon"
python -m uvicorn coolpilot.server.app:app --port 7860
```

**Terminal 2 — Test the API:**
```powershell
# Check server is alive
Invoke-RestMethod http://localhost:7860/health

# Start a new game
Invoke-RestMethod -Uri http://localhost:7860/reset -Method POST -ContentType "application/json" -Body '{"task_id":"task_1_single_zone"}'

# Take one action
Invoke-RestMethod -Uri http://localhost:7860/step -Method POST -ContentType "application/json" -Body '{"cracs":[{"fan_speed":0.8,"chilled_water_flow":0.7,"supply_temp":13.0}]}'

# Check game state
Invoke-RestMethod http://localhost:7860/state
```

**Terminal 2 — Run the AI agent:**
```powershell
cd "c:\Users\kkuna\Desktop\meta x sst hackthon"
$env:API_BASE_URL="http://localhost:7860"
python coolpilot/inference.py
```

---

## Common Questions

**Q: Why does the AI use a "PID fallback"?**
A: The LLM (like ChatGPT) might be slow, crash, or give bad JSON. The PID
controller is a simple math formula that always works — it's our safety net.

**Q: What's a CRAC unit?**
A: Computer Room Air Conditioning. Think of it as a powerful AC unit with
a fan and chilled water. The AI controls the fan speed, water flow, and
air temperature.

**Q: Why 3 difficulty levels?**
A: The hackathon judges test on all 3. Easy = can your AI control 1 AC?
Hard = can it handle 8 zones, random failures, and electricity price changes?

**Q: What's OpenEnv?**
A: A framework by Meta (PyTorch team) for creating RL training environments.
It standardizes the API (`reset`, `step`, `state`) so any RL agent can plug in.
