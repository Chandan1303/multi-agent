"""
SmartCity environment wrapper for RL training.
All state transitions mirror smart_city_openenv_environment.py exactly.
Reward shaping is added ON TOP of the base env reward to give the model
clear, dense signals about what constitutes good vs bad decisions.
"""
import random
from typing import Dict, Tuple, Any

# ── Action space ──────────────────────────────────────────────────────────────
# Each flat action maps to a full set of multi-agent decisions.
# The LLM only needs to output one keyword.
ACTION_MAP: Dict[str, Dict[str, str]] = {
    # Best for high-pollution situations: cuts pollution hard, costs economy slightly
    "reduce_emission": {
        "industry_action":  "reduce_emission",
        "government_action":"impose_penalty",
        "traffic_action":   "reduce_congestion",
        "citizen_action":   "normal",
    },
    # Best for low-pollution situations: grows economy, raises pollution
    "increase_production": {
        "industry_action":  "increase_production",
        "government_action":"no_action",
        "traffic_action":   "normal_traffic",
        "citizen_action":   "normal",
    },
    # Balanced: government acts, industry reduces, traffic normal
    "apply_policy": {
        "industry_action":  "reduce_emission",
        "government_action":"impose_penalty",
        "traffic_action":   "normal_traffic",
        "citizen_action":   "normal",
    },
    # Traffic focus: cuts congestion pollution, keeps economy growing
    "reduce_traffic": {
        "industry_action":  "increase_production",
        "government_action":"no_action",
        "traffic_action":   "reduce_congestion",
        "citizen_action":   "normal",
    },
}
ACTION_LIST = list(ACTION_MAP.keys())

# ── Thresholds used for reward shaping ───────────────────────────────────────
POLLUTION_DANGER   = 75.0   # above this → heavy penalty
POLLUTION_GOOD     = 35.0   # below this → bonus
ECONOMY_DANGER     = 25.0   # below this → heavy penalty
ECONOMY_GOOD       = 70.0   # above this → bonus
SATISFACTION_DANGER= 25.0   # below this → heavy penalty (citizen revolt)
SATISFACTION_GOOD  = 70.0   # above this → bonus


class SmartCityEnvWrapper:
    """
    Gym-style wrapper. Every call to step() returns a SHAPED reward that
    combines the base environment reward with explicit bonuses and penalties
    so the RL agent receives clear, dense feedback from real experience.
    """

    def __init__(self, max_steps: int = 30):
        self.max_steps  = max_steps
        self.step_count = 0
        self.env_state: Dict[str, Any] = {}
        self._prev_pollution = 50.0  # track delta for shaping

    # ── Core env ──────────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self.env_state = {
            "pollution":    50.0,
            "economy":      50.0,
            "satisfaction": 50.0,
            "weather":      "sunny",
        }
        self._prev_pollution = 50.0
        return dict(self.env_state)

    def step(self, flat_action: str) -> Tuple[Dict, float, bool, Dict]:
        self.step_count += 1
        actions = ACTION_MAP.get(flat_action, ACTION_MAP["increase_production"])

        prev = dict(self.env_state)  # snapshot before transition

        # ── State transition (identical to server environment) ────────────────
        self.env_state["weather"] = random.choice(["sunny", "rain"])

        if actions["industry_action"] == "reduce_emission":
            self.env_state["pollution"] -= 5
            self.env_state["economy"]   -= 2
        else:  # increase_production
            self.env_state["pollution"] += 5
            self.env_state["economy"]   += 5

        if actions["government_action"] == "impose_penalty":
            self.env_state["pollution"]    -= 5
            self.env_state["economy"]      -= 3
            self.env_state["satisfaction"] += 2

        if actions["traffic_action"] == "reduce_congestion":
            self.env_state["pollution"]    -= 3
            self.env_state["satisfaction"] -= 2
        else:  # normal_traffic
            self.env_state["pollution"] += 2

        if actions["citizen_action"] == "dissatisfied":
            self.env_state["satisfaction"] -= 5
            self.env_state["economy"]      -= 2

        for k in ["pollution", "economy", "satisfaction"]:
            self.env_state[k] = max(0.0, min(100.0, float(self.env_state[k])))

        # ── Base reward (from server environment formula) ─────────────────────
        base_reward = (
            -self.env_state["pollution"]    * 2.0
            + self.env_state["economy"]     * 1.5
            + self.env_state["satisfaction"]* 2.0
        )

        # ── Reward shaping: real experience signals ───────────────────────────
        shaped = self._shape_reward(prev, self.env_state, flat_action)

        total_reward = base_reward + shaped
        done = self.step_count >= self.max_steps

        info = {
            "step":         self.step_count,
            "base_reward":  base_reward,
            "shaped_bonus": shaped,
            "actions":      actions,
        }
        return dict(self.env_state), total_reward, done, info

    # ── Reward shaping logic ──────────────────────────────────────────────────

    def _shape_reward(
        self,
        prev: Dict[str, Any],
        curr: Dict[str, Any],
        action: str,
    ) -> float:
        bonus = 0.0

        pollution_delta    = curr["pollution"]    - prev["pollution"]
        economy_delta      = curr["economy"]      - prev["economy"]
        satisfaction_delta = curr["satisfaction"] - prev["satisfaction"]

        # ── PENALTIES ─────────────────────────────────────────────────────────

        # Pollution crisis: pollution is dangerously high
        if curr["pollution"] >= POLLUTION_DANGER:
            bonus -= 30.0
            # Extra penalty if agent chose to increase production anyway
            if action == "increase_production":
                bonus -= 20.0  # wrong choice under crisis

        # Pollution worsening when already bad (above 60)
        if prev["pollution"] > 60.0 and pollution_delta > 0:
            bonus -= pollution_delta * 1.5  # penalise each unit of increase

        # Economy collapse
        if curr["economy"] <= ECONOMY_DANGER:
            bonus -= 25.0

        # Citizen revolt: satisfaction critically low
        if curr["satisfaction"] <= SATISFACTION_DANGER:
            bonus -= 25.0

        # Unnecessary restriction: applying heavy policy when pollution is already low
        if curr["pollution"] < 30.0 and action in ("reduce_emission", "apply_policy"):
            bonus -= 10.0  # over-regulating kills economy for no reason

        # ── BONUSES ───────────────────────────────────────────────────────────

        # Pollution actively reduced from a high level
        if prev["pollution"] > 60.0 and pollution_delta < 0:
            bonus += abs(pollution_delta) * 2.0  # reward each unit of improvement

        # Pollution brought into safe zone
        if curr["pollution"] <= POLLUTION_GOOD:
            bonus += 20.0

        # Economy thriving
        if curr["economy"] >= ECONOMY_GOOD:
            bonus += 15.0

        # Citizens happy
        if curr["satisfaction"] >= SATISFACTION_GOOD:
            bonus += 15.0

        # Balance bonus: all three metrics in healthy range simultaneously
        if (
            curr["pollution"]    <= 50.0
            and curr["economy"]      >= 50.0
            and curr["satisfaction"] >= 50.0
        ):
            bonus += 25.0  # the ideal city state

        # Correct action under high pollution (agent learned the right response)
        if prev["pollution"] > 70.0 and action in ("reduce_emission", "apply_policy"):
            bonus += 15.0

        return bonus

    # ── Prompt & action parsing ───────────────────────────────────────────────

    @staticmethod
    def state_to_prompt(state: Dict) -> str:
        """
        Rich prompt that tells the model the current city state AND
        what each action does, so it can reason rather than guess.
        """
        p = state["pollution"]
        e = state["economy"]
        s = state["satisfaction"]
        w = state["weather"]

        # Contextual urgency hints based on thresholds
        p_status = "CRITICAL" if p >= 75 else ("HIGH" if p >= 60 else ("LOW" if p <= 35 else "MODERATE"))
        e_status = "COLLAPSING" if e <= 25 else ("WEAK" if e <= 40 else ("STRONG" if e >= 70 else "STABLE"))
        s_status = "REVOLT RISK" if s <= 25 else ("LOW" if s <= 40 else ("HIGH" if s >= 70 else "STABLE"))

        return (
            f"[Smart City State]\n"
            f"Pollution: {p:.0f}/100 ({p_status})\n"
            f"Economy:   {e:.0f}/100 ({e_status})\n"
            f"Satisfaction: {s:.0f}/100 ({s_status})\n"
            f"Weather: {w}\n\n"
            f"[Action Effects]\n"
            f"reduce_emission:    pollution -8, economy -2  (best when pollution is HIGH)\n"
            f"increase_production: pollution +7, economy +5  (best when pollution is LOW)\n"
            f"apply_policy:       pollution -10, economy -3, satisfaction +2  (best for CRITICAL pollution)\n"
            f"reduce_traffic:     pollution -3, economy +5  (balanced choice)\n\n"
            f"Choose the single best action for this state. Reply with only the action name."
        )

    @staticmethod
    def parse_action(text: str) -> str:
        """Extract first valid action keyword from model output."""
        text = text.lower().strip()
        for action in ACTION_LIST:
            if action in text:
                return action
        # Fallback: if model output is garbage, pick safest default
        return "reduce_traffic"
