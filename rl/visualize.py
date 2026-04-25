"""
Visualise training results from training_metrics.json
Run: python rl/visualize.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

with open("training_metrics.json") as f:
    m = json.load(f)

rewards       = m["episode_rewards"]
pollution     = m["pollution_history"]
economy       = m["economy_history"]
satisfaction  = m["satisfaction_history"]
shaped_bonus  = m["shaped_bonus_history"]
action_counts = m.get("action_counts", {})
episodes      = list(range(1, len(rewards) + 1))
split         = len(rewards) // 2

def smooth(arr, w=7):
    if len(arr) < w:
        return arr
    return list(np.convolve(arr, np.ones(w) / w, mode="valid"))

def style(ax):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

fig = plt.figure(figsize=(16, 12), facecolor="#0f172a")
fig.suptitle("Smart City RL — Training Results", color="white",
             fontsize=20, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

# 1. Reward over episodes (full width)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(episodes, rewards, color="#38bdf8", alpha=0.25, linewidth=1)
s = smooth(rewards)
ax1.plot(range(4, len(s) + 4), s, color="#38bdf8", linewidth=2.5, label="Smoothed reward")
ax1.axvline(split, color="#f59e0b", linestyle="--", linewidth=1.5, label="Before / After split")
ax1.fill_between(episodes[:split], rewards[:split], alpha=0.08, color="#ef4444")
ax1.fill_between(episodes[split:], rewards[split:], alpha=0.08, color="#10b981")
ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward (shaped)")
ax1.set_title("Reward vs Episodes  (red=before training  |  green=after training)")
ax1.legend(facecolor="#1e293b", labelcolor="white")
style(ax1)

# 2. Pollution trend
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(episodes, pollution, color="#ef4444", alpha=0.3, linewidth=1)
sp = smooth(pollution)
ax2.plot(range(4, len(sp) + 4), sp, color="#ef4444", linewidth=2.5)
ax2.axhline(75, color="#fbbf24", linestyle=":", linewidth=1.2, label="Danger (75)")
ax2.axhline(35, color="#10b981", linestyle=":", linewidth=1.2, label="Safe (35)")
ax2.axvline(split, color="#f59e0b", linestyle="--", linewidth=1.2)
ax2.set_xlabel("Episode"); ax2.set_ylabel("Final Pollution")
ax2.set_title("Pollution Trend")
ax2.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
style(ax2)

# 3. Economy & Satisfaction
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(episodes, economy, color="#38bdf8", linewidth=1.5, label="Economy")
ax3.plot(episodes, satisfaction, color="#c084fc", linewidth=1.5, label="Satisfaction")
ax3.axhline(25, color="#ef4444", linestyle=":", linewidth=1, label="Danger (25)")
ax3.axhline(70, color="#10b981", linestyle=":", linewidth=1, label="Good (70)")
ax3.axvline(split, color="#f59e0b", linestyle="--", linewidth=1.2)
ax3.set_xlabel("Episode"); ax3.set_ylabel("Value")
ax3.set_title("Economy & Satisfaction")
ax3.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
style(ax3)

# 4. Before vs After bar chart
ax4 = fig.add_subplot(gs[2, 0])
before_r = np.mean(rewards[:split])
after_r  = np.mean(rewards[split:])
before_p = np.mean(pollution[:split])
after_p  = np.mean(pollution[split:])
x = np.arange(2)
w = 0.35
b1 = ax4.bar(x - w/2, [before_r, after_r], w, color=["#ef4444","#10b981"], label="Reward")
b2 = ax4.bar(x + w/2, [before_p, after_p], w, color=["#f97316","#38bdf8"], label="Pollution")
for bar in list(b1) + list(b2):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{bar.get_height():.0f}", ha="center", color="white", fontsize=9, fontweight="bold")
ax4.set_xticks(x); ax4.set_xticklabels(["Before Training", "After Training"])
ax4.set_title("Before vs After  (Reward & Pollution)")
ax4.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
style(ax4)

# 5. Action distribution pie
ax5 = fig.add_subplot(gs[2, 1])
if action_counts:
    labels = list(action_counts.keys())
    sizes  = list(action_counts.values())
    colors = ["#38bdf8", "#ef4444", "#10b981", "#c084fc"]
    wedges, texts, autotexts = ax5.pie(
        sizes, labels=labels, colors=colors[:len(labels)],
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "white", "fontsize": 8},
    )
    for at in autotexts:
        at.set_color("white")
ax5.set_title("Action Distribution (all episodes)")
ax5.title.set_color("white")

plt.savefig("training_results.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
print("Saved → training_results.png")
plt.show()
