"""
GovernmentAgent — city policy, regulation, investment.

Inter-agent interactions:
  - LISTENS to Industry: if industry reports pollution → impose penalty
  - LISTENS to Citizen: if citizens protest → respond with healthcare or green investment
  - LISTENS to Traffic: if traffic reports congestion crisis → fund smart routing
  - BROADCASTS to Industry: penalty notices so industry knows to reduce emissions
  - BROADCASTS to all: policy announcements
"""
from .message_bus import MessageBus, Message

HEALTH_CRITICAL = 30.0


class GovernmentAgent:
    def __init__(self):
        self.steps_since_action = 0
        self.industry_alert_received = False
        self.citizen_protest_received = False

    def act(self, state: dict, bus: MessageBus = None) -> str:
        p = state["pollution"]
        e = state["economy"]
        s = state["satisfaction"]
        h = state.get("health", 70)
        budget = state.get("budget", 100)
        energy = state.get("energy", 60)

        self.steps_since_action += 1

        # ── Read messages from other agents ───────────────────────────────────
        if bus:
            self.industry_alert_received = bus.has_report("government", "pollution") or bus.has_alert("government")
            self.citizen_protest_received = bus.has_alert("government", "protest") or bus.has_request("government", "health")

        # ── Decision logic ────────────────────────────────────────────────────
        action = self._decide(p, e, s, h, budget, energy)

        # ── Broadcast to other agents ─────────────────────────────────────────
        if bus:
            if action == "impose_penalty":
                bus.broadcast(Message("government", "industry", "REQUEST",
                    f"Penalty imposed — industry must reduce emissions (pollution: {p:.0f})",
                    {"action": "reduce_emission"}))
                bus.broadcast(Message("government", "all", "REPORT",
                    f"Government imposing pollution penalties — budget +5",
                    {"policy": "impose_penalty"}))
            elif action == "invest_green":
                bus.broadcast(Message("government", "all", "REPORT",
                    f"Government investing in green energy — energy will improve",
                    {"policy": "invest_green"}))
                bus.broadcast(Message("government", "citizen", "CONFIRM",
                    "Green investment underway — city improving sustainability",
                    {}))
            elif action == "healthcare_fund":
                bus.broadcast(Message("government", "citizen", "CONFIRM",
                    f"Healthcare funded — health will recover from {h:.0f}",
                    {"policy": "healthcare_fund"}))
            if budget <= 20:
                bus.broadcast(Message("government", "all", "ALERT",
                    f"Budget critical at ${budget:.0f} — all agents must minimise costs",
                    {"budget": budget}))

        return action

    def _decide(self, p, e, s, h, budget, energy) -> str:
        if budget <= 15:
            return "impose_penalty" if p >= 60 else "no_action"
        if h <= HEALTH_CRITICAL and budget >= 8:
            return "healthcare_fund"
        if p >= 75:
            return "impose_penalty"
        # Citizen protest received → respond visibly
        if self.citizen_protest_received and budget >= 8:
            if h <= 55:
                return "healthcare_fund"
            if energy <= 50:
                return "invest_green"
        if energy <= 40 and budget >= 10 and self.steps_since_action >= 2:
            self.steps_since_action = 0
            return "invest_green"
        if p >= 60 and self.steps_since_action >= 2:
            self.steps_since_action = 0
            return "impose_penalty"
        # Industry reported pollution → act proactively
        if self.industry_alert_received and p >= 50 and self.steps_since_action >= 1:
            self.steps_since_action = 0
            return "impose_penalty"
        if s <= 35 and h <= 55 and budget >= 8:
            return "healthcare_fund"
        if p >= 45 and budget >= 12 and energy <= 60 and self.steps_since_action >= 3:
            self.steps_since_action = 0
            return "invest_green"
        return "no_action"

    def reset(self):
        self.steps_since_action = 0
        self.industry_alert_received = False
        self.citizen_protest_received = False
