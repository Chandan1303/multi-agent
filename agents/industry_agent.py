"""
IndustryAgent — controls factory production.

Inter-agent interactions:
  - LISTENS to Government: if government imposes penalty → reduce emissions cooperatively
  - LISTENS to Citizen: if citizens protest → reduce emissions to restore trust
  - BROADCASTS to all: pollution level alerts so others can react
  - RESPONDS to Traffic: if traffic requests emission cut → comply when possible
"""
from .message_bus import MessageBus, Message


class IndustryAgent:
    def __init__(self):
        self.consecutive_high_pollution = 0
        self.government_penalty_active = False
        self.citizen_protest_active = False

    def act(self, state: dict, bus: MessageBus = None) -> str:
        p = state["pollution"]
        e = state["economy"]
        s = state["satisfaction"]

        # ── Read messages from other agents ───────────────────────────────────
        if bus:
            # Government imposed penalty → industry should cooperate
            self.government_penalty_active = bus.has_request("industry", "reduce")
            # Citizens protesting → industry feels social pressure
            self.citizen_protest_active = bus.has_alert("industry", "protest")

        # Track pollution trend
        self.consecutive_high_pollution = (self.consecutive_high_pollution + 1) if p > 60 else 0

        # ── Decision logic ────────────────────────────────────────────────────
        action = self._decide(p, e, s)

        # ── Broadcast to other agents ─────────────────────────────────────────
        if bus:
            if p >= 75:
                bus.broadcast(Message("industry", "all", "ALERT",
                    f"POLLUTION CRISIS at {p:.0f} — industry cutting emissions immediately",
                    {"pollution": p}))
            elif p >= 60:
                bus.broadcast(Message("industry", "government", "REPORT",
                    f"Pollution elevated at {p:.0f} — requesting government support",
                    {"pollution": p}))
            if action == "reduce_emission":
                bus.broadcast(Message("industry", "traffic", "CONFIRM",
                    "Industry reducing emissions — traffic can ease restrictions if needed",
                    {"action": action}))
            if e <= 25:
                bus.broadcast(Message("industry", "government", "ALERT",
                    f"Economy collapsing at {e:.0f} — need government stimulus",
                    {"economy": e}))

        return action

    def _decide(self, p, e, s) -> str:
        if p >= 75:
            return "reduce_emission"
        if self.government_penalty_active and p >= 55:
            return "reduce_emission"   # cooperate with government regulation
        if self.citizen_protest_active and p >= 50:
            return "reduce_emission"   # respond to social pressure
        if p >= 60 and self.consecutive_high_pollution >= 2:
            return "reduce_emission"
        if e <= 20 and p < 65:
            return "increase_production"
        if s <= 25 and p >= 55:
            return "reduce_emission"
        if p >= 60:
            return "reduce_emission"
        return "increase_production"

    def reset(self):
        self.consecutive_high_pollution = 0
        self.government_penalty_active = False
        self.citizen_protest_active = False
