# 🧊 CoolPilot  
### *Intelligent Data Center Cooling with Reinforcement Learning*

<div align="center">

[![Python](https://img.shields.io/badge/Python-≥3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗_Live_Demo-heykunal123/coolpilot-yellow)](https://huggingface.co/spaces/heykunal123/coolpilot)

**Meta × Scaler School of Technology · OpenEnv Hackathon 2026**

</div>

---

## 🚀 Overview

Modern data centers consume **~1–2% of global electricity**, with **30–40% dedicated solely to cooling**. Even small efficiency gains can yield massive cost savings and carbon reductions.

**CoolPilot** explores a powerful idea:

> **Can a Reinforcement Learning agent outperform traditional rule-based cooling systems?**

This project provides an **OpenEnv-compliant RL environment** where an AI agent learns to dynamically control HVAC/CRAC systems in real time.

---

## 🎯 Objective

The agent continuously observes system conditions and adjusts:

| Parameter | Range | Role |
|----------|------|------|
| **Fan Speed** | `0.1 – 1.0` | Controls airflow volume |
| **Chilled Water Flow** | `0.1 – 1.0` | Determines cooling capacity |
| **Supply Air Temperature** | `10°C – 20°C` | Sets output air temperature |

### Goal:
- Maintain **safe server temperatures (18°C – 27°C)**  
- Minimize **Power Usage Effectiveness (PUE → 1.0)**  

---

## 📊 Why It Matters

| Metric | Traditional Systems | CoolPilot |
|-------|-------------------|----------|
| Cooling Strategy | Static rules | Adaptive RL |
| Low Load Handling | Overcooling | Energy-efficient |
| Failure Response | None | Real-time compensation |
| Typical PUE | 1.4 – 2.0 | **1.1 – 1.2 target** |

---

## 📖 Research Inspiration

Inspired by:

> **DeepMind × Google (2016)**  
> *“AI reduces data center cooling cost by 40%”*

- ~40% reduction in cooling energy  
- ~15% improvement in PUE  

🔗 https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/

---

## 🏗️ System Architecture

```text
Agent (LLM / RL Policy)
        │
        ▼
CoolPilotEnv (OpenEnv Client)
        │
        ▼ HTTP (REST API)
FastAPI Server (Simulation Engine)
        │
        ├── Thermal Model (Physics-based)
        ├── Reward Engine
        ├── Task Manager
        └── Validation Layer (Pydantic)
```

---

## 🛠 Tech Stack

| Layer | Technology |
|------|-----------|
| Language | Python ≥ 3.10 |
| Backend | FastAPI + Uvicorn |
| RL Interface | OpenEnv Core |
| Validation | Pydantic |
| HTTP Client | HTTPX |
| LLM | OpenAI SDK |
| Model | Qwen2.5-72B-Instruct |
| Deployment | Hugging Face Spaces |
| Container | Docker |
| Testing | Pytest |

---

## 🧠 Core Concepts

### ⚡ Power Usage Effectiveness (PUE)

```
PUE = Total Facility Power / IT Equipment Power
```

---

### 🌡 Thermal Model

```
dT/dt = (Q_it - Q_cooling) / (m × c)
```

---

### 🎯 Reward Function

| Component | Weight |
|----------|--------|
| Safety | 40% |
| Energy Efficiency | 40% |
| Stability | 20% |

---

## ⚙️ API Reference

| Method | Endpoint | Description |
|-------|--------|------------|
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute action |
| GET | `/state` | Current state |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## 🚀 Getting Started

```bash
git clone https://github.com/Krishna11/RL-Project-101.git
cd RL-Project-101

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

---

## ☁️ Deployment

- https://huggingface.co/spaces/heykunal123/coolpilot  
- https://heykunal123-coolpilot.hf.space  

---

## 🏆 Tasks & Scoring

| Task | Difficulty | Zones |
|------|----------|------|
| Task 1 | Easy | 1 |
| Task 2 | Medium | 4 |
| Task 3 | Hard | 8 |

---

## 📄 License

BSD 3-Clause License

---

<div align="center">

### ❤️ Built for Global Impact

Meta × SST OpenEnv Hackathon 2026

</div>
