"""
MessageBus — inter-agent communication system.

Agents broadcast signals each step. Other agents READ these signals
before making their own decisions, creating real coordination.

Signal types:
  - ALERT:     urgent situation requiring immediate response
  - REQUEST:   asking another agent to take a specific action
  - REPORT:    sharing information about current conditions
  - CONFIRM:   acknowledging another agent's action
"""
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Message:
    sender: str       # "industry" | "government" | "traffic" | "citizen"
    target: str       # "all" | specific agent name
    signal: str       # ALERT | REQUEST | REPORT | CONFIRM
    content: str      # human-readable message
    data: Dict = field(default_factory=dict)  # structured payload


class MessageBus:
    """Shared communication channel for all agents."""

    def __init__(self):
        self._messages: List[Message] = []
        self._history: List[Dict] = []   # full log for UI display

    def broadcast(self, msg: Message):
        self._messages.append(msg)
        self._history.append({
            "sender":  msg.sender,
            "target":  msg.target,
            "signal":  msg.signal,
            "content": msg.content,
        })

    def get_messages(self, target: str) -> List[Message]:
        """Get all messages addressed to a specific agent or 'all'."""
        return [m for m in self._messages if m.target in (target, "all")]

    def has_alert(self, target: str, keyword: str = "") -> bool:
        """Check if there's an active ALERT for this agent."""
        for m in self.get_messages(target):
            if m.signal == "ALERT":
                if not keyword or keyword.lower() in m.content.lower():
                    return True
        return False

    def has_request(self, target: str, keyword: str = "") -> bool:
        """Check if another agent is requesting this agent to act."""
        for m in self.get_messages(target):
            if m.signal == "REQUEST":
                if not keyword or keyword.lower() in m.content.lower():
                    return True
        return False

    def has_report(self, target: str, keyword: str = "") -> bool:
        """Check if another agent sent a report to this agent."""
        for m in self.get_messages(target):
            if m.signal == "REPORT":
                if not keyword or keyword.lower() in m.content.lower():
                    return True
        return False

    def has_confirm(self, target: str, keyword: str = "") -> bool:
        """Check if another agent confirmed an action to this agent."""
        for m in self.get_messages(target):
            if m.signal == "CONFIRM":
                if not keyword or keyword.lower() in m.content.lower():
                    return True
        return False

    def clear(self):
        """Clear messages at start of each step."""
        self._messages.clear()

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def get_last_n(self, n: int = 5) -> List[Dict]:
        return self._history[-n:] if self._history else []
