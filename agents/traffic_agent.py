"""
TrafficAgent — manages city traffic and transport.

Inter-agent interactions:
  - LISTENS to Industry: if industry is cutting emissions → traffic can relax
  - LISTENS to Government: if government invested in smart routing → use it
  - LISTENS to Citizen: if citizens are protesting → avoid adding restrictions
  - BROADCASTS to Government: requests smart routing budget when needed
  - BROADCASTS to all: congestion alerts
"""
from .message_bus import MessageBus, Message


class TrafficAgent:
    def __init__(self):
        self.restriction_steps = 0
        self.industry_reducing = False
        self.citizen_protesting = False
        self.government_invested = False

    def act(self, state: dict, bus: MessageBus = None) -> str:
        p = state["pollution"]
        s = state["satisfaction"]
        budget = state.get("budget", 100)
        weather = state.get("weather", "sunny")
        disaster = state.get("disaster")

        # ── Read messages from other agents ───────────────────────────────────
        if bus:
            # Industry already cutting emissions → traffic can be less aggressive
            self.industry_reducing = bus.has_confirm("traffic", "reducing") or bus.has_confirm("traffic", "emission")
            # Citizens protesting → don't add more restrictions
            self.citizen_protesting = bus.has_alert("traffic", "protest")
            # Government invested in green/smart → more budget available
            self.government_invested = bus.has_report("traffic", "invest")

        # ── Decision logic ────────────────────────────────────────────────────
        action = self._decide(p, s, budget, weather, disaster)

        # ── Broadcast to other agents ─────────────────────────────────────────
        if bus:
            if p >= 65:
                bus.broadcast(Message("traffic", "government", "REQUEST",
                    f"Congestion contributing to pollution ({p:.0f}) — requesting smart routing budget",
                    {"pollution": p}))
            if action == "smart_routing":
                bus.broadcast(Message("traffic", "citizen", "CONFIRM",
                    "Smart routing active — traffic flowing efficiently, less pollution",
                    {"action": "smart_routing"}))
            elif action == "reduce_congestion":
                bus.broadcast(Message("traffic", "citizen", "REPORT",
                    "Traffic restrictions in place to reduce pollution",
                    {"action": "reduce_congestion"}))
            if disaster:
                bus.broadcast(Message("traffic", "all", "ALERT",
                    f"Emergency routing active due to {disaster}",
                    {"disaster": disaster}))

        return action

    def _decide(self, p, s, budget, weather, disaster) -> str:
        if disaster and budget >= 4:
            return "smart_routing"
        if p >= 70:
            return "smart_routing" if budget >= 4 else "reduce_congestion"
        # Citizens protesting — avoid adding restrictions
        if self.citizen_protesting and s <= 30 and p < 65:
            self.restriction_steps = 0
            return "normal_traffic"
        # Industry already reducing — traffic can ease off
        if self.industry_reducing and p < 65:
            self.restriction_steps = 0
            return "normal_traffic"
        if s <= 25 and p < 65:
            self.restriction_steps = 0
            return "normal_traffic"
        if p >= 55:
            self.restriction_steps += 1
            return "smart_routing" if budget >= 4 else "reduce_congestion"
        if weather == "storm":
            return "reduce_congestion"
        if p >= 45:
            if self.restriction_steps < 2:
                self.restriction_steps += 1
                return "reduce_congestion"
            self.restriction_steps = 0
            return "normal_traffic"
        self.restriction_steps = 0
        return "normal_traffic"

    def reset(self):
        self.restriction_steps = 0
        self.industry_reducing = False
        self.citizen_protesting = False
        self.government_invested = False
