from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict, Any, Optional


class SmartCityOpenenvAction(Action):
    """Action for the Advanced Smart City environment."""

    industry_action: str = Field(
        default="increase_production",
        description="reduce_emission | increase_production"
    )
    government_action: str = Field(
        default="no_action",
        description="impose_penalty | invest_green | healthcare_fund | no_action"
    )
    traffic_action: str = Field(
        default="normal_traffic",
        description="reduce_congestion | smart_routing | normal_traffic"
    )
    citizen_action: str = Field(
        default="normal",
        description="dissatisfied | eco_initiative | normal"
    )


class SmartCityOpenenvObservation(Observation):
    """Observation from the Advanced Smart City environment."""

    state: Dict[str, Any] = Field(default_factory=dict)
    weather: str = Field(default="sunny")
    disaster: Optional[str] = Field(default=None)
