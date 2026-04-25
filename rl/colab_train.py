# ============================================================
# Smart City RL Training — Google Colab Script
# Copy this ENTIRE file into a Colab cell and run it.
# Runtime > Change runtime type > T4 GPU
# ============================================================

# ── Cell 1: Install dependencies ─────────────────────────────
import subprocess
subprocess.run(["pip", "install", "transformers", "accelerate", "matplotlib", "-q"])

# ── Cell 2: Environment wrapper (inline) ─────────────────────
import random
from typing import Dict, Tuple, Any

ACTION_MAP = {
    "reduce_emission":     {"industry_action": "reduce_emission",    "government_action": "impose_penalty", "traffic_action": "reduce_congestion", "citizen_action": "normal"},
    "increase_production": {"industry_action": "increase_production","government_action": "no_action",      "traffic_action": "normal_traffic",    "citizen_action": "normal"},
    "apply_policy":        {"industry_action": "reduce_emission",    "government_action": "impose_penalty", "traffic_action": "normal_traffic",    "citizen_action": "normal"},
    "reduce_traffic":      {"industry_action": "increase_production","government_action": "no_action",      "traffic_action": "reduce_congestion", "citizen_action": "normal"},
}
ACTION_LIST = list(ACTION_MAP.keys())

POLLUTION_DANGER    = 75.0
POLLUTION_GOOD      = 35.0
ECONOMY_DANGER      = 25.0
ECONOMY_GOOD        = 70.0
SATISFACTION_DANGER = 25.0
SATISFACTION_GOOD   = 70.0

class SmartCityEnvWrapper:
    def __init__(self, max_steps=30):
        self.max_steps  = max_steps
        self.step_count = 0
        self.env_state  = {}

    def reset(self):
        self.step_count = 0
        self.env_state  = {"pollution": 50.0, "economy": 50.0, "satisfaction": 50.0, "weather": "sunny"}
        return dict(self.env_state)

    def step(self, flat_action):
        self.step_count += 1
        actions = ACTION_MAP.get(flat_action, ACTION_MAP["increase_production"])
        prev    = dict(self.env_state)
        self.env_state["weather"] = random.choice(["sunny", "rain"])

        if actions["industry_action"] == "reduce_emission":
            self.env_state["pollution"] -= 5;  self.env_state["economy"] -= 2
        else:
            self.env_state["pollution"] += 5;  self.env_state["economy"] += 5

        if actions["government_action"] == "impose_penalty":
            self.env_state["pollution"] -= 5;  self.env_state["economy"] -= 3;  self.env_state["satisfaction"] += 2

        if actions["traffic_action"] == "reduce_congestion":
            self.env_state["pollution"] -= 3;  self.env_state["satisfaction"] -= 2
        else:
            self.env_state["pollution"] += 2

        for k in ["pollution", "economy", "satisfaction"]:
            self.env_state[k] = max(0.0, min(100.0, float(self.env_state[k])))

        base_reward = (-self.env_state["pollution"] * 2.0
                       + self.env_state["economy"]  * 1.5
                       + self.env_state["satisfaction"] * 2.0)
        shaped = self._shape(prev, self.env_state, flat_action)
        done   = self.step_count >= self.max_steps
        return dict(self.env_state), base_reward + shaped, done, {"step": self.step_count, "shaped_bonus": shaped}

    def _shape(self, prev, curr, action):
        b = 0.0
        dp = curr["pollution"] - prev["pollution"]
        if curr["pollution"] >= POLLUTION_DANGER:
            b -= 30.0
            if action == "increase_production": b -= 20.0
        if prev["pollution"] > 60 and dp > 0:   b -= dp * 1.5
        if curr["economy"]      <= ECONOMY_DANGER:      b -= 25.0
        if curr["satisfaction"] <= SATISFACTION_DANGER: b -= 25.0
        if curr["pollution"] < 30 and action in ("reduce_emission","apply_policy"): b -= 10.0
        if prev["pollution"] > 60 and dp < 0:   b += abs(dp) * 2.0
        if curr["pollution"]    <= POLLUTION_GOOD:      b += 20.0
        if curr["economy"]      >= ECONOMY_GOOD:        b += 15.0
        if curr["satisfaction"] >= SATISFACTION_GOOD:   b += 15.0
        if curr["pollution"] <= 50 and curr["economy"] >= 50 and curr["satisfaction"] >= 50: b += 25.0
        if prev["pollution"] > 70 and action in ("reduce_emission","apply_policy"): b += 15.0
        return b

    @staticmethod
    def state_to_prompt(state):
        p, e, s, w = state["pollution"], state["economy"], state["satisfaction"], state["weather"]
        p_s = "CRITICAL" if p>=75 else ("HIGH" if p>=60 else ("LOW" if p<=35 else "MODERATE"))
        e_s = "COLLAPSING" if e<=25 else ("WEAK" if e<=40 else ("STRONG" if e>=70 else "STABLE"))
        s_s = "REVOLT RISK" if s<=25 else ("LOW" if s<=40 else ("HIGH" if s>=70 else "STABLE"))
        return (f"[Smart City State]\nPollution: {p:.0f}/100 ({p_s})\n"
                f"Economy: {e:.0f}/100 ({e_s})\nSatisfaction: {s:.0f}/100 ({s_s})\nWeather: {w}\n\n"
                f"[Action Effects]\n"
                f"reduce_emission: pollution -8, economy -2 (best when HIGH)\n"
                f"increase_production: pollution +7, economy +5 (best when LOW)\n"
                f"apply_policy: pollution -10, economy -3, satisfaction +2 (best for CRITICAL)\n"
                f"reduce_traffic: pollution -3, economy +5 (balanced)\n\n"
                f"Choose the single best action. Reply with only the action name.")

print("✅ Environment loaded")

# ── Cell 3: Load model ────────────────────────────────────────
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device != "cuda":
    print("⚠ WARNING: Running on CPU — will be slow. Go to Runtime > Change runtime type > T4 GPU for 10x speed.")
else:
    print("✅ GPU ready!")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
model.to(device)
model.train()
print("✅ Model loaded on", device)

# ── Cell 4: Training setup ────────────────────────────────────
NUM_EPISODES = 20 if device == "cpu" else 100  # 20 on CPU (~30 min), 100 on GPU (~20 min)
MAX_STEPS    = 30
LR           = 2e-5
GAMMA        = 0.97
ENTROPY_COEF = 0.02
SAVE_PATH    = "/content/trained_model"

optimizer = AdamW(model.parameters(), lr=LR)
env       = SmartCityEnvWrapper(max_steps=MAX_STEPS)

action_token_ids  = {a: tokenizer.encode(a, add_special_tokens=False)[0] for a in ACTION_LIST}
action_idx_tensor = torch.tensor([action_token_ids[a] for a in ACTION_LIST], device=device)

metrics = {"episode_rewards": [], "pollution_history": [], "economy_history": [],
           "satisfaction_history": [], "policy_loss": []}
action_counts   = {a: 0 for a in ACTION_LIST}
reward_baseline = deque(maxlen=20)

def compute_returns(rewards, gamma):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)

def sample_action(prompt):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :].float()  # float32 prevents NaN with fp16
    action_logits = logits[action_idx_tensor]
    # Guard against NaN/inf after model updates
    action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    action_logits = torch.clamp(action_logits, -50.0, 50.0)
    probs = F.softmax(action_logits / 0.8, dim=-1)
    # Fallback to uniform if probs are still bad
    if torch.isnan(probs).any() or probs.sum() < 0.99:
        probs = torch.ones(len(ACTION_LIST), device=device) / len(ACTION_LIST)
    return ACTION_LIST[torch.multinomial(probs, 1).item()]

def compute_loss(prompts, actions, advantages):
    """
    Accumulate gradients one step at a time to avoid OOM.
    Each forward pass is done, loss.backward() called immediately,
    then tensors freed before the next step.
    """
    total_loss_val = 0.0
    entropies      = []
    n              = len(prompts)

    for prompt, action, adv in zip(prompts, actions, advantages):
        enc    = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=256).to(device)
        logits = model(**enc).logits[0, -1, :].float()  # float32 for stable gradients
        al     = logits[action_idx_tensor]
        al     = torch.nan_to_num(al, nan=0.0, posinf=10.0, neginf=-10.0)
        lp     = F.log_softmax(al.float(), dim=-1)       # float32 for stability
        p      = lp.exp()
        lp_chosen = lp[ACTION_LIST.index(action)]
        entropy   = -(p * lp).sum()

        adv_t      = torch.tensor(adv, dtype=torch.float32, device=device)
        step_loss  = (-lp_chosen * adv_t - ENTROPY_COEF * entropy) / n

        # Accumulate gradients immediately, then free the graph
        step_loss.backward()
        total_loss_val += step_loss.item()
        entropies.append(entropy.item())

        # Free GPU memory after each step
        del enc, logits, al, lp, p, lp_chosen, entropy, adv_t, step_loss
        torch.cuda.empty_cache()

    # Return a dummy scalar (gradients already accumulated)
    return total_loss_val, float(np.mean(entropies))

print("✅ Training setup complete")

# ── Cell 5: TRAIN ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Training {NUM_EPISODES} episodes on {device.upper()}")
print(f"{'='*65}\n")

for ep in range(NUM_EPISODES):
    state = env.reset()
    ep_prompts, ep_actions, ep_rewards, ep_shaped = [], [], [], []
    total_reward = 0.0
    done = False

    # Collect episode
    while not done:
        prompt = SmartCityEnvWrapper.state_to_prompt(state)
        action = sample_action(prompt)
        state, reward, done, info = env.step(action)
        ep_prompts.append(prompt);  ep_actions.append(action)
        ep_rewards.append(reward);  ep_shaped.append(info["shaped_bonus"])
        total_reward += reward;     action_counts[action] += 1

    # Compute advantages
    returns  = compute_returns(ep_rewards, GAMMA)
    reward_baseline.append(total_reward)
    baseline = float(np.mean(reward_baseline))
    advantages = (returns - baseline) / (returns.std() + 1e-8)

    # Update — gradients already accumulated inside compute_loss
    model.train()
    optimizer.zero_grad()
    loss_val, entropy = compute_loss(ep_prompts, ep_actions, advantages)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.empty_cache()

    metrics["episode_rewards"].append(total_reward)
    metrics["pollution_history"].append(state["pollution"])
    metrics["economy_history"].append(state["economy"])
    metrics["satisfaction_history"].append(state["satisfaction"])
    metrics["policy_loss"].append(loss_val)

    ac_str = " | ".join(f"{k[:6]}:{v}" for k,v in Counter(ep_actions).most_common())
    print(f"Ep {ep+1:3d}/{NUM_EPISODES}  Reward:{total_reward:8.1f}  "
          f"Pollution:{state['pollution']:5.1f}  Economy:{state['economy']:5.1f}  "
          f"Loss:{loss_val:.3f}  [{ac_str}]")

    if (ep+1) % 10 == 0:
        print(f"  ↳ avg reward: {np.mean(metrics['episode_rewards'][-10:]):.1f}  "
              f"avg pollution: {np.mean(metrics['pollution_history'][-10:]):.1f}\n")

# Save
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\n✅ Model saved to {SAVE_PATH}")

n = 10
print(f"\nFirst {n} avg reward : {np.mean(metrics['episode_rewards'][:n]):.1f}")
print(f"Last  {n} avg reward : {np.mean(metrics['episode_rewards'][-n:]):.1f}")
print(f"First {n} avg pollution: {np.mean(metrics['pollution_history'][:n]):.1f}")
print(f"Last  {n} avg pollution: {np.mean(metrics['pollution_history'][-n:]):.1f}")

# ── Cell 6: Plot results ──────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def smooth(arr, w=7):
    return list(np.convolve(arr, np.ones(w)/w, mode="valid")) if len(arr)>=w else arr

rewards    = metrics["episode_rewards"]
pollution  = metrics["pollution_history"]
episodes   = list(range(1, len(rewards)+1))
split      = len(rewards)//2

fig = plt.figure(figsize=(14, 8), facecolor="#0f172a")
fig.suptitle("Smart City RL — Training Results (Colab GPU)", color="white", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

def style(ax):
    ax.set_facecolor("#1e293b"); ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8"); ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#334155")

ax1 = fig.add_subplot(gs[0,:])
ax1.plot(episodes, rewards, color="#38bdf8", alpha=0.25, linewidth=1)
s = smooth(rewards)
ax1.plot(range(4, len(s)+4), s, color="#38bdf8", linewidth=2.5, label="Smoothed")
ax1.axvline(split, color="#f59e0b", linestyle="--", linewidth=1.5, label="Before/After")
ax1.fill_between(episodes[:split], rewards[:split], alpha=0.08, color="#ef4444")
ax1.fill_between(episodes[split:], rewards[split:], alpha=0.08, color="#10b981")
ax1.set_title("Reward vs Episodes"); ax1.set_ylabel("Total Reward")
ax1.legend(facecolor="#1e293b", labelcolor="white"); style(ax1)

ax2 = fig.add_subplot(gs[1,0])
ax2.plot(episodes, pollution, color="#ef4444", alpha=0.3)
sp = smooth(pollution)
ax2.plot(range(4, len(sp)+4), sp, color="#ef4444", linewidth=2.5)
ax2.axhline(75, color="#fbbf24", linestyle=":", label="Danger"); ax2.axhline(35, color="#10b981", linestyle=":", label="Safe")
ax2.set_title("Pollution Trend"); ax2.legend(facecolor="#1e293b", labelcolor="white", fontsize=8); style(ax2)

ax3 = fig.add_subplot(gs[1,1])
before_r = np.mean(rewards[:split]); after_r = np.mean(rewards[split:])
before_p = np.mean(pollution[:split]); after_p = np.mean(pollution[split:])
bars = ax3.bar(["Before\nTraining","After\nTraining"], [before_r, after_r], color=["#ef4444","#10b981"])
for bar, val in zip(bars, [before_r, after_r]):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{val:.0f}", ha="center", color="white", fontweight="bold")
ax3.set_title("Before vs After Reward"); style(ax3)

plt.savefig("/content/training_results.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
plt.show()
print("✅ Plot saved to /content/training_results.png")

# ── Cell 7: Download trained model ───────────────────────────
import shutil
shutil.make_archive("/content/trained_model_export", "zip", "/content/trained_model")
from google.colab import files
files.download("/content/trained_model_export.zip")
files.download("/content/training_results.png")
print("✅ Download started!")
