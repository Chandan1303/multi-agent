"""
Evaluation: Trained model vs Untrained baseline vs Rule-based oracle.

Runs N_EVAL episodes for each agent, reports:
  - Mean / std total reward
  - Mean final pollution / economy / satisfaction
  - Action distribution
  - Step-by-step comparison on a fixed seed episode

Run:  python rl/evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from env_wrapper import SmartCityEnvWrapper, ACTION_LIST

# ── Config ────────────────────────────────────────────────────────────────────
TRAINED_PATH   = "trained_model"
BASE_MODEL     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_EVAL         = 20       # episodes per agent
MAX_STEPS      = 30
FIXED_SEED     = 42       # for the head-to-head comparison episode
EVAL_FILE      = "eval_results.json"
EVAL_PLOT      = "eval_results.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ── Load tokenizer (shared) ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    TRAINED_PATH if os.path.exists(TRAINED_PATH) else BASE_MODEL
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

action_token_ids = {
    a: tokenizer.encode(a, add_special_tokens=False)[0]
    for a in ACTION_LIST
}
action_idx_tensor = torch.tensor([action_token_ids[a] for a in ACTION_LIST])


# ── Agent definitions ─────────────────────────────────────────────────────────

class LLMAgent:
    """Greedy LLM agent — picks highest-probability action (no sampling)."""
    def __init__(self, model_path: str, name: str):
        self.name = name
        print(f"Loading {name} from {model_path} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32
        ).to(device)
        self.model.eval()

    def act(self, state: dict) -> str:
        prompt = SmartCityEnvWrapper.state_to_prompt(state)
        enc    = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=320).to(device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1, :]
        action_logits = logits[action_idx_tensor.to(device)]
        idx = action_logits.argmax().item()   # greedy — no randomness
        return ACTION_LIST[idx]


class RuleBasedAgent:
    """
    Deterministic rule-based oracle — always makes the textbook-correct choice.
    Used as an upper-bound reference.
    """
    name = "Rule-Based Oracle"

    def act(self, state: dict) -> str:
        p = state["pollution"]
        e = state["economy"]
        if p >= 75:
            return "apply_policy"       # crisis: maximum pollution reduction
        elif p >= 60:
            return "reduce_emission"    # high: reduce emissions
        elif e <= 30:
            return "reduce_traffic"     # economy weak: grow without big pollution
        else:
            return "increase_production"  # safe: grow economy


class RandomAgent:
    """Uniform random baseline — lower bound reference."""
    name = "Random Baseline"

    def act(self, state: dict) -> str:
        return random.choice(ACTION_LIST)


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_episodes(agent, n: int, seed_offset: int = 0) -> dict:
    """Run N episodes, return aggregated stats."""
    rewards, pollutions, economies, satisfactions = [], [], [], []
    all_actions = []

    for i in range(n):
        random.seed(seed_offset + i)   # reproducible per-agent
        env   = SmartCityEnvWrapper(max_steps=MAX_STEPS)
        state = env.reset()
        total_reward = 0.0
        ep_actions   = []
        done = False

        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            ep_actions.append(action)

        rewards.append(total_reward)
        pollutions.append(state["pollution"])
        economies.append(state["economy"])
        satisfactions.append(state["satisfaction"])
        all_actions.extend(ep_actions)

    return {
        "reward_mean":       float(np.mean(rewards)),
        "reward_std":        float(np.std(rewards)),
        "reward_min":        float(np.min(rewards)),
        "reward_max":        float(np.max(rewards)),
        "pollution_mean":    float(np.mean(pollutions)),
        "economy_mean":      float(np.mean(economies)),
        "satisfaction_mean": float(np.mean(satisfactions)),
        "action_dist":       dict(Counter(all_actions)),
        "all_rewards":       rewards,
        "all_pollutions":    pollutions,
    }


def run_fixed_episode(agent, seed: int) -> list[dict]:
    """Run one episode with a fixed seed, return step-by-step log."""
    random.seed(seed)
    env   = SmartCityEnvWrapper(max_steps=MAX_STEPS)
    state = env.reset()
    log   = []
    done  = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        log.append({
            "step":         info["step"],
            "action":       action,
            "pollution":    next_state["pollution"],
            "economy":      next_state["economy"],
            "satisfaction": next_state["satisfaction"],
            "reward":       reward,
            "shaped_bonus": info["shaped_bonus"],
        })
        state = next_state

    return log


# ── Build agent list ──────────────────────────────────────────────────────────
agents = [RuleBasedAgent(), RandomAgent()]

if os.path.exists(TRAINED_PATH):
    agents.insert(0, LLMAgent(TRAINED_PATH, "Trained LLM"))
else:
    print(f"⚠  No trained model found at '{TRAINED_PATH}'. Run train_ppo.py first.")

agents.append(LLMAgent(BASE_MODEL, "Untrained LLM"))

# ── Run evaluation ────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  EVALUATION  |  {N_EVAL} episodes per agent  |  {MAX_STEPS} steps/ep")
print(f"{'='*70}\n")

results = {}
for i, agent in enumerate(agents):
    stats = run_episodes(agent, N_EVAL, seed_offset=i * 1000)
    results[agent.name] = stats
    print(f"  {agent.name}")
    print(f"    Reward:       {stats['reward_mean']:8.1f}  ±{stats['reward_std']:.1f}"
          f"  (min:{stats['reward_min']:.0f}  max:{stats['reward_max']:.0f})")
    print(f"    Pollution:    {stats['pollution_mean']:8.1f}")
    print(f"    Economy:      {stats['economy_mean']:8.1f}")
    print(f"    Satisfaction: {stats['satisfaction_mean']:8.1f}")
    top_actions = sorted(stats["action_dist"].items(), key=lambda x: -x[1])[:3]
    print(f"    Top actions:  {', '.join(f'{a}({c})' for a,c in top_actions)}")
    print()

# ── Fixed-seed head-to-head ───────────────────────────────────────────────────
print(f"{'='*70}")
print(f"  HEAD-TO-HEAD  (fixed seed={FIXED_SEED}, same environment sequence)")
print(f"{'='*70}")
print(f"  {'Step':<5} ", end="")
for agent in agents:
    print(f"  {agent.name[:18]:<20}", end="")
print()
print(f"  {'-'*5} ", end="")
for _ in agents:
    print(f"  {'-'*20}", end="")
print()

fixed_logs = {agent.name: run_fixed_episode(agent, FIXED_SEED) for agent in agents}

for step_i in range(MAX_STEPS):
    print(f"  {step_i+1:<5} ", end="")
    for agent in agents:
        log = fixed_logs[agent.name][step_i]
        print(f"  {log['action'][:10]:<10} P:{log['pollution']:4.0f} R:{log['reward']:6.0f}", end="")
    print()

print()
for agent in agents:
    total = sum(s["reward"] for s in fixed_logs[agent.name])
    final = fixed_logs[agent.name][-1]
    print(f"  {agent.name:<22}  Total reward: {total:8.1f}  "
          f"Final pollution: {final['pollution']:.1f}  "
          f"Economy: {final['economy']:.1f}")

# ── Save results ──────────────────────────────────────────────────────────────
with open(EVAL_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {EVAL_FILE}")

# ── Plot ──────────────────────────────────────────────────────────────────────
def style(ax):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

COLORS = {"Trained LLM": "#10b981", "Untrained LLM": "#ef4444",
          "Rule-Based Oracle": "#38bdf8", "Random Baseline": "#94a3b8"}

fig = plt.figure(figsize=(16, 10), facecolor="#0f172a")
fig.suptitle("Smart City RL — Evaluation Results", color="white",
             fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

agent_names = list(results.keys())
colors      = [COLORS.get(n, "#f59e0b") for n in agent_names]

# 1. Mean reward bar
ax1 = fig.add_subplot(gs[0, 0])
means = [results[n]["reward_mean"] for n in agent_names]
stds  = [results[n]["reward_std"]  for n in agent_names]
bars  = ax1.bar(range(len(agent_names)), means, color=colors, yerr=stds,
                capsize=5, error_kw={"ecolor": "white", "alpha": 0.6})
for bar, val in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val:.0f}", ha="center", color="white", fontsize=8, fontweight="bold")
ax1.set_xticks(range(len(agent_names)))
ax1.set_xticklabels([n.replace(" ", "\n") for n in agent_names], fontsize=7)
ax1.set_title("Mean Total Reward ± std")
ax1.set_ylabel("Reward")
style(ax1)

# 2. Mean pollution bar
ax2 = fig.add_subplot(gs[0, 1])
pollutions = [results[n]["pollution_mean"] for n in agent_names]
bars2 = ax2.bar(range(len(agent_names)), pollutions, color=colors)
ax2.axhline(75, color="#fbbf24", linestyle=":", linewidth=1.2, label="Danger (75)")
ax2.axhline(35, color="#10b981", linestyle=":", linewidth=1.2, label="Safe (35)")
for bar, val in zip(bars2, pollutions):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.0f}", ha="center", color="white", fontsize=8, fontweight="bold")
ax2.set_xticks(range(len(agent_names)))
ax2.set_xticklabels([n.replace(" ", "\n") for n in agent_names], fontsize=7)
ax2.set_title("Mean Final Pollution")
ax2.set_ylabel("Pollution")
ax2.legend(facecolor="#1e293b", labelcolor="white", fontsize=7)
style(ax2)

# 3. Economy + Satisfaction grouped bar
ax3 = fig.add_subplot(gs[0, 2])
x   = np.arange(len(agent_names))
w   = 0.35
eco = [results[n]["economy_mean"]      for n in agent_names]
sat = [results[n]["satisfaction_mean"] for n in agent_names]
ax3.bar(x - w/2, eco, w, color="#38bdf8", label="Economy")
ax3.bar(x + w/2, sat, w, color="#c084fc", label="Satisfaction")
ax3.set_xticks(x)
ax3.set_xticklabels([n.replace(" ", "\n") for n in agent_names], fontsize=7)
ax3.set_title("Economy & Satisfaction")
ax3.legend(facecolor="#1e293b", labelcolor="white", fontsize=7)
style(ax3)

# 4. Reward distribution box plot
ax4 = fig.add_subplot(gs[1, 0:2])
data = [results[n]["all_rewards"] for n in agent_names]
bp   = ax4.boxplot(data, patch_artist=True, notch=False,
                   medianprops={"color": "white", "linewidth": 2})
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ["whiskers", "caps", "fliers"]:
    for item in bp[element]:
        item.set_color("#94a3b8")
ax4.set_xticks(range(1, len(agent_names) + 1))
ax4.set_xticklabels(agent_names, fontsize=8)
ax4.set_title("Reward Distribution across Episodes")
ax4.set_ylabel("Total Reward")
style(ax4)

# 5. Fixed-seed pollution trajectory
ax5 = fig.add_subplot(gs[1, 2])
for agent in agents:
    log = fixed_logs[agent.name]
    steps = [s["step"] for s in log]
    pols  = [s["pollution"] for s in log]
    ax5.plot(steps, pols, color=COLORS.get(agent.name, "#f59e0b"),
             linewidth=2, label=agent.name)
ax5.axhline(75, color="#fbbf24", linestyle=":", linewidth=1, label="Danger")
ax5.axhline(35, color="#10b981", linestyle=":", linewidth=1, label="Safe")
ax5.set_xlabel("Step"); ax5.set_ylabel("Pollution")
ax5.set_title(f"Pollution Trajectory (seed={FIXED_SEED})")
ax5.legend(facecolor="#1e293b", labelcolor="white", fontsize=6)
style(ax5)

plt.savefig(EVAL_PLOT, dpi=150, bbox_inches="tight", facecolor="#0f172a")
print(f"Plot saved → {EVAL_PLOT}")
plt.show()
