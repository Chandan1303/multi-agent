---
title: Smart City OpenEnv
emoji: 🏙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Smart City AI Control Node

Advanced Multi-Agent Reinforcement Learning City Simulation built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv).

## Live Demo

Open the dashboard at `/dashboard` after deployment.

## What it does

Simulates a city managed by 4 AI agents that **communicate with each other** in real time:

| Agent | Role | Actions |
|---|---|---|
| Industry | Factory production | reduce_emission / increase_production |
| Government | City policy | impose_penalty / invest_green / healthcare_fund |
| Traffic | Road management | reduce_congestion / smart_routing / normal_traffic |
| Citizen | Public opinion | dissatisfied / eco_initiative / normal |

## Metrics tracked

- Pollution (0-100) — lower is better
- Economy (0-100) — higher is better  
- Satisfaction (0-100) — higher is better
- Health (0-100) — affected by pollution over time
- Energy % — renewable energy share
- Budget ($) — city finances

## Reward system

```
total = base_reward + penalties + bonuses + balance_bonus
```

- Pollution crisis (≥75): **-50 penalty**
- Economy collapse (≤20): **-40 penalty**
- Citizen revolt (≤25): **-40 penalty**
- Perfect city balance: **+80 bonus**

## RL Training

TinyLlama 1.1B trained with REINFORCE algorithm on 100 episodes of real environment interactions.

## Run locally

```bash
docker build -t smart-city-openenv .
docker run -p 8000:8000 smart-city-openenv
# Open http://localhost:8000/dashboard
```

Or without Docker:
```bash
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/simulate` | Run full 30-step simulation |
| POST | `/rl-episode` | Run trained LLM agent |
| GET | `/dashboard` | Web dashboard |
| POST | `/reset` | Reset environment |
| POST | `/step` | Single environment step |
| WS | `/ws` | WebSocket session |
