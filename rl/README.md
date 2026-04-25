---
title: Smart City RL Agent
emoji: 🏙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: gradio_demo.py
pinned: false
---

# Smart City RL Agent

LLM agent trained with PPO on a Smart City environment.
The agent controls pollution, economy, and citizen satisfaction.

## Run locally

```bash
pip install -r requirements.txt

# Train
python train_ppo.py

# Visualize results
python visualize.py

# Launch demo
python gradio_demo.py
```

## Action space
| Action | Effect |
|---|---|
| `reduce_emission` | ↓ Pollution, ↓ Economy |
| `increase_production` | ↑ Pollution, ↑ Economy |
| `apply_policy` | ↓↓ Pollution, ↑ Satisfaction |
| `reduce_traffic` | ↓ Pollution, ↑ Economy |
