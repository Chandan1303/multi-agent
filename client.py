from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SmartCityOpenenvAction, SmartCityOpenenvObservation


class SmartCityOpenenvEnv(
    EnvClient[SmartCityOpenenvAction, SmartCityOpenenvObservation, State]
):
    """
    Client for the Smart City Openenv Environment.
    """

    def _step_payload(self, action: SmartCityOpenenvAction) -> Dict:
        """
        Convert SmartCityOpenenvAction to JSON payload for step message.
        """
        return {
            "industry_action": action.industry_action,
            "government_action": action.government_action,
            "traffic_action": action.traffic_action,
            "citizen_action": action.citizen_action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SmartCityOpenenvObservation]:
        """
        Parse server response into StepResult[SmartCityOpenenvObservation].
        """
        obs_data = payload.get("observation", {})
        observation = SmartCityOpenenvObservation(
            state=obs_data.get("state", {}),
            weather=obs_data.get("weather", "sunny"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
