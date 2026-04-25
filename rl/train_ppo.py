"""
REINFORCE with baseline — SmartCity LLM agent training.
Pure PyTorch + HuggingFace transformers. No TRL, no TensorFlow.

Algorithm:
  1. Collect a full episode (30 steps) of REAL env experience
  2. Compute discounted returns G_t = r_t + γ*r_{t+1} + ...
  3. Normalise advantages: A_t = (G_t - baseline) / std
  4. Policy gradient loss: L = -Σ log π(a_t|s_t) * A_t  - β*H(π)
  5. Backprop + gradient clip + Adam step

Run:  python rl/train_ppo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from env_wrapper import SmartCityEnvWrapper, ACTION_LIST

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_EPISODES = 100
MAX_STEPS    = 30
LR           = 2e-5
GAMMA        = 0.97       # discount — future rewards matter
ENTROPY_COEF = 0.02       # exploration bonus weight
MAX_GRAD_NORM= 1.0        # gradient clipping
BASELINE_WIN = 20         # running mean window for advantage baseline
SAVE_PATH    = "trained_model"
METRICS_FILE = "training_metrics.json"

# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=LR)
env       = SmartCityEnvWrapper(max_steps=MAX_STEPS)

# ── Pre-compute action token ids ──────────────────────────────────────────────
# We only score the first token of each action keyword — fast and sufficient
action_token_ids = {
    a: tokenizer.encode(a, add_special_tokens=False)[0]
    for a in ACTION_LIST
}
action_idx_tensor = torch.tensor(
    [action_token_ids[a] for a in ACTION_LIST], dtype=torch.long
)
print("Action tokens:", {a: tokenizer.decode([action_token_ids[a]]) for a in ACTION_LIST})

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = {
    "episode_rewards":      [],
    "pollution_history":    [],
    "economy_history":      [],
    "satisfaction_history": [],
    "shaped_bonus_history": [],
    "policy_loss":          [],
    "entropy":              [],
}
action_counts  = {a: 0 for a in ACTION_LIST}
reward_baseline = deque(maxlen=BASELINE_WIN)

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Discounted return G_t for each timestep."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)


def sample_action(prompt: str) -> tuple[str, torch.Tensor]:
    """
    Forward pass (no grad) → sample action from softmax over action tokens.
    Returns (action_str, action_logits_over_action_space).
    """
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=320
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]          # [vocab_size]

    action_logits = logits[action_idx_tensor.to(device)] # [4]
    probs = F.softmax(action_logits / 0.8, dim=-1)       # temperature=0.8
    idx   = torch.multinomial(probs, 1).item()
    return ACTION_LIST[idx], action_logits


def compute_episode_loss(
    prompts: list[str],
    actions: list[str],
    advantages: np.ndarray,
) -> tuple[torch.Tensor, float]:
    """
    Recompute log-probs WITH gradients for the collected episode,
    then compute REINFORCE loss + entropy bonus.
    """
    loss_terms = []
    entropies  = []

    for prompt, action, adv in zip(prompts, actions, advantages):
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=320
        ).to(device)

        # Full forward pass WITH grad tracking
        logits = model(**enc).logits[0, -1, :]           # [vocab_size]

        # Log-probs over action space only
        action_logits = logits[action_idx_tensor.to(device)]  # [4]
        log_probs     = F.log_softmax(action_logits, dim=-1)  # [4]
        probs         = log_probs.exp()

        # Log-prob of the chosen action
        chosen_idx = ACTION_LIST.index(action)
        log_prob   = log_probs[chosen_idx]

        # Entropy H(π) = -Σ p log p  (over action space)
        entropy = -(probs * log_probs).sum()

        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)

        # REINFORCE: maximise log_prob * advantage  →  minimise negative
        loss_terms.append(-log_prob * adv_t - ENTROPY_COEF * entropy)
        entropies.append(entropy.item())

    total_loss = torch.stack(loss_terms).mean()
    mean_entropy = float(np.mean(entropies))
    return total_loss, mean_entropy


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  REINFORCE Training  |  {NUM_EPISODES} episodes  |  {MAX_STEPS} steps/ep")
print(f"{'='*70}\n")

for episode in range(NUM_EPISODES):
    state = env.reset()

    ep_prompts      = []
    ep_actions      = []
    ep_raw_rewards  = []
    ep_shaped_bonus = []
    total_reward    = 0.0

    # ── Phase 1: Collect episode from REAL env interactions ───────────────────
    done = False
    while not done:
        prompt = SmartCityEnvWrapper.state_to_prompt(state)
        action, _ = sample_action(prompt)

        next_state, reward, done, info = env.step(action)

        ep_prompts.append(prompt)
        ep_actions.append(action)
        ep_raw_rewards.append(reward)
        ep_shaped_bonus.append(info["shaped_bonus"])

        total_reward += reward
        action_counts[action] += 1
        state = next_state

    # ── Phase 2: Compute advantages ───────────────────────────────────────────
    returns = compute_returns(ep_raw_rewards, GAMMA)

    # Running baseline = mean total reward over last BASELINE_WIN episodes
    reward_baseline.append(total_reward)
    baseline = float(np.mean(reward_baseline))

    # Normalise: zero-mean, unit-variance advantages
    advantages = (returns - baseline) / (returns.std() + 1e-8)

    # ── Phase 3: Policy gradient update ──────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    loss, mean_entropy = compute_episode_loss(ep_prompts, ep_actions, advantages)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    # ── Logging ───────────────────────────────────────────────────────────────
    final = state
    metrics["episode_rewards"].append(total_reward)
    metrics["pollution_history"].append(final["pollution"])
    metrics["economy_history"].append(final["economy"])
    metrics["satisfaction_history"].append(final["satisfaction"])
    metrics["shaped_bonus_history"].append(float(np.sum(ep_shaped_bonus)))
    metrics["policy_loss"].append(loss.item())
    metrics["entropy"].append(mean_entropy)

    ac_str = " | ".join(f"{k[:6]}:{v}" for k, v in Counter(ep_actions).most_common())
    print(
        f"Ep {episode+1:3d}/{NUM_EPISODES}  "
        f"Reward:{total_reward:8.1f}  "
        f"Pollution:{final['pollution']:5.1f}  "
        f"Economy:{final['economy']:5.1f}  "
        f"Sat:{final['satisfaction']:5.1f}  "
        f"Loss:{loss.item():6.3f}  "
        f"H:{mean_entropy:.3f}  [{ac_str}]"
    )

    if (episode + 1) % 10 == 0:
        n = 10
        avg_r = np.mean(metrics["episode_rewards"][-n:])
        avg_p = np.mean(metrics["pollution_history"][-n:])
        avg_e = np.mean(metrics["economy_history"][-n:])
        print(f"  ↳ [10-ep avg]  reward:{avg_r:.1f}  pollution:{avg_p:.1f}  economy:{avg_e:.1f}\n")

# ── Save model + metrics ──────────────────────────────────────────────────────
print(f"\nSaving model → '{SAVE_PATH}' ...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

metrics["action_counts"] = action_counts
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved → {METRICS_FILE}")

# ── Final summary ─────────────────────────────────────────────────────────────
n = min(10, NUM_EPISODES // 2)
print(f"\n{'='*55}")
print(f"  TRAINING SUMMARY")
print(f"{'='*55}")
print(f"  Avg reward   first {n} eps : {np.mean(metrics['episode_rewards'][:n]):8.1f}")
print(f"  Avg reward   last  {n} eps : {np.mean(metrics['episode_rewards'][-n:]):8.1f}")
print(f"  Avg pollution first {n} eps : {np.mean(metrics['pollution_history'][:n]):8.1f}")
print(f"  Avg pollution last  {n} eps : {np.mean(metrics['pollution_history'][-n:]):8.1f}")
print(f"\n  Action distribution (total {NUM_EPISODES * MAX_STEPS} steps):")
for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
    pct = 100 * c / (NUM_EPISODES * MAX_STEPS)
    print(f"    {a:25s}: {c:4d}  ({pct:.1f}%)")
print(f"{'='*55}")
print("\nNext: python rl/evaluate.py")
