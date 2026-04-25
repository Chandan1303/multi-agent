from uuid import uuid4
import random
from typing import Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SmartCityOpenenvAction, SmartCityOpenenvObservation
    from .rubrics import SmartCityRubric
except ImportError:
    from models import SmartCityOpenenvAction, SmartCityOpenenvObservation
    try:
        from server.rubrics import SmartCityRubric
    except ImportError:
        from .rubrics import SmartCityRubric


class SmartCityOpenenvEnvironment(Environment):
    """
    Advanced Smart City Openenv Environment implementation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the smart_city_openenv environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.env_state: Dict[str, Any] = {}
        self.max_steps = 30
        self.rubric = SmartCityRubric()
        self.current_scenario = "standard"
        
    def set_scenario(self, scenario: str):
        self.current_scenario = scenario

    def reset(self) -> SmartCityOpenenvObservation:
        """
        Reset the environment with the selected scenario.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Base state
        self.env_state = {
            "pollution": 50.0,
            "economy": 50.0,
            "satisfaction": 50.0,
            "health": 70.0,
            "energy": 60.0,
            "budget": 100.0,
            "weather": "sunny"
        }
        
        # Apply scenario modifiers
        if self.current_scenario == "climate_crisis":
            self.env_state["pollution"] = 90.0
            self.env_state["health"] = 30.0
        elif self.current_scenario == "economic_collapse":
            self.env_state["economy"] = 20.0
            self.env_state["budget"] = 0.0
        elif self.current_scenario == "energy_failure":
            self.env_state["energy"] = 0.0
            self.env_state["satisfaction"] = 30.0
        
        return SmartCityOpenenvObservation(
            state=self.env_state.copy(),
            weather=self.env_state["weather"],
            disaster=None,
            done=False,
            reward=0.0,
        )

    def step(self, action: SmartCityOpenenvAction) -> SmartCityOpenenvObservation:
        """
        Execute a step in the advanced real-world simulation environment.
        """
        self._state.step_count += 1
        disaster_event = None
        
        # 1. Weather and Disasters (Real-world stochasticity)
        weather_roll = random.random()
        if weather_roll < 0.2:
            self.env_state["weather"] = "rain"
        elif weather_roll < 0.3:
            self.env_state["weather"] = "storm"
        else:
            self.env_state["weather"] = "sunny"
            
        disaster_roll = random.random()
        if disaster_roll < 0.05:
            disaster_event = "Power Outage"
            self.env_state["energy"] -= 20
            self.env_state["satisfaction"] -= 10
            self.env_state["economy"] -= 10
        elif disaster_roll < 0.10:
            disaster_event = "Smog Wave"
            self.env_state["pollution"] += 15
            self.env_state["health"] -= 15
            self.env_state["satisfaction"] -= 5
            
        # 2. Industry Actions
        if action.industry_action == "reduce_emission":
            self.env_state["pollution"] -= 6
            self.env_state["economy"] -= 3
            self.env_state["energy"] -= 5 # Uses less energy
        else: # increase_production
            if self.env_state["energy"] > 20: # Only if energy is available
                self.env_state["pollution"] += 8
                self.env_state["economy"] += 6
                self.env_state["energy"] -= 10
            else:
                self.env_state["economy"] -= 5 # Penalized for production attempt without energy

        # 3. Government Actions
        if action.government_action == "impose_penalty":
            self.env_state["pollution"] -= 5
            self.env_state["economy"] -= 3
            self.env_state["satisfaction"] += 2
            self.env_state["budget"] += 5 # Penalties increase budget
        elif action.government_action == "invest_green":
            if self.env_state["budget"] >= 10:
                self.env_state["budget"] -= 10
                self.env_state["energy"] += 20 # Add renewable energy
                self.env_state["pollution"] -= 4
                self.env_state["satisfaction"] += 3
        elif action.government_action == "healthcare_fund":
            if self.env_state["budget"] >= 5:
                self.env_state["budget"] -= 5
                self.env_state["health"] += 10
                self.env_state["satisfaction"] += 4

        # 4. Traffic Actions
        if action.traffic_action == "reduce_congestion":
            self.env_state["pollution"] -= 4
            self.env_state["satisfaction"] -= 3
        elif action.traffic_action == "smart_routing":
            if self.env_state["budget"] >= 2:
                self.env_state["budget"] -= 2
                self.env_state["pollution"] -= 2
                self.env_state["economy"] += 2 # Efficient logistics
                self.env_state["satisfaction"] += 2
        else: # normal_traffic
            self.env_state["pollution"] += 3

        # 5. Citizen Actions
        if action.citizen_action == "dissatisfied":
            self.env_state["satisfaction"] -= 5
            self.env_state["economy"] -= 2 # Protests hurt economy
        elif action.citizen_action == "eco_initiative":
            self.env_state["pollution"] -= 3
            self.env_state["health"] += 2
            self.env_state["satisfaction"] += 2
            
        # 6. Natural Environment Dynamics (Passive loops)
        if self.env_state["pollution"] > 70:
            self.env_state["health"] -= 5
        
        # Energy naturally replenishes slightly
        self.env_state["energy"] += 5
        
        # Budget tax collection based on economy
        self.env_state["budget"] += (self.env_state["economy"] / 20)
            
        # Constrain variables between 0 and 100 (except budget which can just grow)
        self.env_state["pollution"] = max(0.0, min(100.0, float(self.env_state["pollution"])))
        self.env_state["economy"] = max(0.0, min(100.0, float(self.env_state["economy"])))
        self.env_state["satisfaction"] = max(0.0, min(100.0, float(self.env_state["satisfaction"])))
        self.env_state["health"] = max(0.0, min(100.0, float(self.env_state["health"])))
        self.env_state["energy"] = max(0.0, min(100.0, float(self.env_state["energy"])))
        self.env_state["budget"] = max(0.0, float(self.env_state["budget"]))
        
        done = self._state.step_count >= self.max_steps
        
        obs = SmartCityOpenenvObservation(
            state=self.env_state.copy(),
            weather=self.env_state["weather"],
            disaster=disaster_event,
            done=done,
            reward=0.0, # Placeholder
            metadata={
                "step": self._state.step_count,
                "weather": self.env_state["weather"],
                "disaster": disaster_event
            },
        )
        
        # Calculate Reward using OpenEnv Rubric system
        reward = self.rubric(action, obs)
        obs.reward = reward
        
        return obs

    @property
    def state(self) -> State:
        """
        Get the current environment state.
        """
        return self._state
