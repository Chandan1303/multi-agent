"""
CitizenAgent — collective voice of city residents.

Inter-agent interactions:
  - LISTENS to Government: if government acts → citizens feel heard, less protest
  - LISTENS to Traffic: if smart routing active → citizens appreciate it
  - BROADCASTS to Government: protest signals demanding action
  - BROADCASTS to Industry: social pressure to reduce emissions
  - BROADCASTS to all: satisfaction alerts
"""
from .message_bus import MessageBus, Message


class CitizenAgent:
    def __init__(self):
        self.dissatisfied_steps = 0
        self.government_responded = False
        self.traffic_improved = False

    def act(self, state: dict, bus: MessageBus = None) -> str:
        p = state["pollution"]
        e = state["economy"]
        s = state["satisfaction"]
        h = state.get("health", 70)
        energy = state.get("energy", 60)
        disaster = state.get("disaster")

        # ── Read messages from other agents ───────────────────────────────────
        if bus:
            # Government confirmed action → citizens feel heard
            self.government_responded = (
                bus.has_confirm("citizen", "healthcare") or
                bus.has_confirm("citizen", "green") or
                bus.has_report("citizen", "penalty")
            )
            # Traffic improved → citizens appreciate it
            self.traffic_improved = bus.has_confirm("citizen", "routing") or bus.has_confirm("citizen", "smart")

        # ── Decision logic ────────────────────────────────────────────────────
        action = self._decide(p, e, s, h, energy, disaster)

        # ── Broadcast to other agents ─────────────────────────────────────────
        if bus:
            if action == "dissatisfied":
                bus.broadcast(Message("citizen", "government", "ALERT",
                    f"Citizens protesting — satisfaction at {s:.0f}, health at {h:.0f}",
                    {"satisfaction": s, "health": h}))
                bus.broadcast(Message("citizen", "industry", "ALERT",
                    f"Citizens demand cleaner industry — pollution at {p:.0f}",
                    {"pollution": p}))
            elif action == "eco_initiative":
                bus.broadcast(Message("citizen", "government", "REPORT",
                    "Citizens launching eco-initiative — government should match with green investment",
                    {"energy": energy}))
                bus.broadcast(Message("citizen", "all", "CONFIRM",
                    "Community eco-initiative active — pollution and energy improving",
                    {}))
            if s >= 75:
                bus.broadcast(Message("citizen", "all", "REPORT",
                    f"Citizens very satisfied ({s:.0f}) — city is thriving",
                    {"satisfaction": s}))

        return action

    def _decide(self, p, e, s, h, energy, disaster) -> str:
        if disaster and s >= 40:
            return "eco_initiative"
        if s <= 20:
            self.dissatisfied_steps += 1
            return "dissatisfied"
        if h <= 30:
            self.dissatisfied_steps += 1
            return "dissatisfied"
        if p >= 80:
            self.dissatisfied_steps += 1
            return "dissatisfied"
        if p >= 65 and e <= 35:
            self.dissatisfied_steps += 1
            return "dissatisfied"
        if e <= 10:
            self.dissatisfied_steps += 1
            return "dissatisfied"
        # Government responded → citizens calm down faster
        if self.government_responded:
            self.dissatisfied_steps = max(0, self.dissatisfied_steps - 2)
        # Traffic improved → citizens appreciate it
        if self.traffic_improved and s >= 35:
            self.dissatisfied_steps = max(0, self.dissatisfied_steps - 1)
        if energy <= 45 and p >= 40 and s >= 45:
            return "eco_initiative"
        if p >= 65 and self.dissatisfied_steps >= 3:
            return "dissatisfied"
        if p < 50 and e >= 40 and h >= 50:
            self.dissatisfied_steps = max(0, self.dissatisfied_steps - 1)
        return "normal"

    def reset(self):
        self.dissatisfied_steps = 0
        self.government_responded = False
        self.traffic_improved = False
