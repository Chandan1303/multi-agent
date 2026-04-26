---
title: Smart City OpenEnv
emoji: 🏙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🏙️ Smart City AI Control Node

> **Multi-Agent Reinforcement Learning for Urban Management** — built on Meta's [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework for the OpenEnv Hackathon 2026.

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 HuggingFace Space Live Demo | [https://chandan1303-smart-city-openenv.hf.space/dashboard](https://chandan1303-smart-city-openenv.hf.space/dashboard) |
| 📝 Blog Post | https://huggingface.co/spaces/chandan1303/smart-city-openenv-blogs   |
| 📊 Presentation | https://docs.google.com/presentation/d/1T_UYsPdsUp7m2t61hEuoIA0IgAT-14_D/edit?usp=sharing&ouid=114324658033694579315&rtpof=true&sd=true  |
| 💻 GitHub | [https://github.com/Chandan1303/multi-agent](https://github.com/Chandan1303/multi-agent) |

---

## 🌆 The Problem

Real cities face impossible trade-offs every single day:

- **Build factories** → jobs and money, but pollution rises
- **Restrict traffic** → cleaner air, but citizens get frustrated  
- **Fund hospitals** → healthier people, but drains the budget
- **Invest in green energy** → long-term clean city, but expensive now

No single rule works for all situations. Cities need AI that **learns the right balance from experience** — and that's exactly what this project builds.

---

## 🤖 How It Works

This environment simulates a city managed by **4 AI agents that communicate with each other** in real time via a Message Bus system.

### The 4 Agents

| Agent | Real-world role | Actions |
|---|---|---|
| 🏭 Industry | Factory owners | `reduce_emission` / `increase_production` |
| 🏛️ Government | City council | `impose_penalty` / `invest_green` / `healthcare_fund` / `no_action` |
| 🚦 Traffic | Transport authority | `reduce_congestion` / `smart_routing` / `normal_traffic` |
| 👥 Citizen | Public opinion | `dissatisfied` / `eco_initiative` / `normal` |

### Agent Communication (Message Bus)

Agents don't just read the environment — they **talk to each other**:

- Industry alerts Government when pollution is critical
- Government sends penalty notices to Industry
- Citizens protest to Government when health is poor
- Traffic requests budget from Government for smart routing
- Government confirms healthcare funding to Citizens

Each message has a type: `ALERT` 🚨 / `REQUEST` 📋 / `REPORT` 📊 / `CONFIRM` ✅

### City Metrics (6 dimensions)

| Metric | Range | Goal |
|---|---|---|
| Pollution | 0–100 | Keep LOW (< 35) |
| Economy | 0–100 | Keep HIGH (> 70) |
| Satisfaction | 0–100 | Keep HIGH (> 70) |
| Health | 0–100 | Degrades from pollution, restored by healthcare |
| Energy % | 0–100 | Increase renewable share |
| Budget ($) | 0–200 | Every action costs or earns money |

### Random Disasters

Every step has a chance of a disaster event:
- 🏭 Industrial Accident (+20 pollution, -10 economy)
- 🌊 Flood (+5 pollution, -15 economy)
- 🌡️ Heatwave (+10 pollution, -10 health)
- ⚡ Power Outage (-12 economy, -20 energy)
- 🦠 Pandemic Scare (-20 health, -15 satisfaction)

---

## 🏆 Reward System

```
total_reward = base_reward + penalties + bonuses + balance_bonus
```

### Base Formula
```
base = (-pollution × 2.0) + (economy × 1.5) + (satisfaction × 2.0)
       + (health × 1.0) + (energy × 0.5) - (budget_pressure × 0.3)
```

### Penalties
| Situation | Penalty |
|---|---|
| Pollution ≥ 75 (crisis) | **-50** |
| Industry produces during crisis | extra **-30** |
| Economy ≤ 20 (collapse) | **-40** |
| Citizen revolt (satisfaction ≤ 25) | **-40** |
| Health crisis (≤ 30) | **-35** |
| Budget bankrupt (≤ 10) | **-20** |

### Bonuses
| Situation | Bonus |
|---|---|
| Pollution ≤ 20 (ideal) | **+30** |
| Economy ≥ 85 (thriving) | **+25** |
| Citizens very happy (≥ 85) | **+25** |
| Health excellent (≥ 85) | **+20** |
| Green energy ≥ 80% | **+15** |

### Balance Bonus (hardest to achieve)
| Situation | Bonus |
|---|---|
| All metrics healthy simultaneously | **+80 (Perfect City 🏆)** |
| Good balance across all metrics | **+60** |

---

## 🧠 RL Training

**Model:** TinyLlama 1.1B Chat  
**Algorithm:** REINFORCE with baseline (pure PyTorch, no TRL dependency)  
**Training:** 100 episodes on Google Colab T4 GPU (~20 minutes)  
**Data:** 100% real environment interactions — zero synthetic data

The LLM reads the city state as a structured text prompt and outputs one of 4 actions. It gets rewarded for achieving balance across all 6 metrics simultaneously.

### Training Prompt Format
```
[Smart City State]
Pollution: 72/100 (HIGH)
Economy: 45/100 (STABLE)
Satisfaction: 38/100 (LOW)
Weather: rain

[Action Effects]
reduce_emission: pollution -8, economy -2 (best when HIGH)
increase_production: pollution +7, economy +5 (best when LOW)
apply_policy: pollution -10, economy -3, satisfaction +2 (best for CRITICAL)
reduce_traffic: pollution -3, economy +5 (balanced)

Choose the single best action. Reply with only the action name.
```

---

## 📊 City Grade System

After each simulation, the city receives a letter grade (A+ to F) based on a weighted score across all 6 metrics:

| Grade | Score | Meaning |
|---|---|---|
| A+ | ≥ 85 | Perfect city — all metrics thriving |
| A | ≥ 75 | Excellent management |
| B | ≥ 65 | Good balance |
| C | ≥ 55 | Moderate — some issues |
| D | ≥ 40 | Poor — multiple crises |
| F | < 40 | City failing |

---

## 🎮 Scenario Presets

| Scenario | Starting State | Challenge |
|---|---|---|
| Standard City | Balanced (50/50/50) | Learn the basics |
| Crisis Mode | Pollution 85, Economy 25 | Recover from disaster |
| Economic Boom | Economy 90, Pollution 70 | Manage growth vs environment |
| Green City | Pollution 20, Energy 85 | Maintain sustainability |
| Balanced City | All metrics healthy | Achieve Perfect City bonus |

---

## 🔌 OpenEnv API

This environment is fully compatible with the OpenEnv standard:

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Reset the environment |
| POST | `/step` | Execute agent actions |
| GET | `/state` | Current city state |
| GET | `/schema` | Action/observation schemas |
| POST | `/simulate` | Run full 30-step simulation |
| POST | `/rl-episode` | Run trained LLM agent |
| WS | `/ws` | WebSocket persistent session |
| GET | `/dashboard` | Web dashboard |

---

## 🌍 Real-World Applications

| Application | How this maps |
|---|---|
| Urban pollution control | Industry + Government + Traffic managing air quality |
| Smart grid management | Energy metric + green investment decisions |
| Public health response | Health metric + healthcare funding + disaster events |
| Municipal budget allocation | Budget system with cost/benefit trade-offs |
| Traffic optimisation | Smart routing reducing emissions and congestion |
| Disaster response | Random events requiring multi-agent coordination |

---

## 🚀 Run Locally

```bash
# With Docker
docker build -t smart-city-openenv .
docker run -p 7860:7860 smart-city-openenv

# Without Docker
pip install -r rl/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/dashboard
```

---

## 📁 Project Structure

```
smart_city_openenv/
├── server/
│   ├── app.py                          # FastAPI server + all endpoints
│   └── smart_city_openenv_environment.py  # Core environment (reset/step)
├── agents/
│   ├── industry_agent.py               # Factory production decisions
│   ├── government_agent.py             # Policy and investment decisions
│   ├── traffic_agent.py                # Traffic management decisions
│   ├── citizen_agent.py                # Public opinion and protests
│   └── message_bus.py                  # Inter-agent communication system
├── rl/
│   ├── train_ppo.py                    # REINFORCE training (local)
│   ├── colab_train.py                  # Google Colab training script
│   ├── evaluate.py                     # Evaluation vs baselines
│   └── env_wrapper.py                  # Standalone env for RL training
├── ui/
│   ├── index.html                      # Dashboard (4 tabs)
│   ├── style.css                       # Dark theme styling
│   └── script.js                       # Charts and interactivity
├── models.py                           # Action/Observation schemas
├── Dockerfile                          # Container for HF Spaces
└── README.md                           # This file
```
