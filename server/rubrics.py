from openenv.core.rubrics.base import Rubric

class PollutionRubric(Rubric):
    def forward(self, action, observation) -> float:
        pollution = observation.state.get("pollution", 50.0)
        return -float(pollution) * 2.0

class EconomyRubric(Rubric):
    def forward(self, action, observation) -> float:
        economy = observation.state.get("economy", 50.0)
        return float(economy) * 1.5

class SatisfactionRubric(Rubric):
    def forward(self, action, observation) -> float:
        satisfaction = observation.state.get("satisfaction", 50.0)
        return float(satisfaction) * 2.0

class HealthRubric(Rubric):
    def forward(self, action, observation) -> float:
        health = observation.state.get("health", 70.0)
        # Big penalty if health drops too low
        return float(health) * 1.5 if health > 40 else -100.0

class EnergyBudgetRubric(Rubric):
    def forward(self, action, observation) -> float:
        energy = observation.state.get("energy", 60.0)
        budget = observation.state.get("budget", 100.0)
        reward = 0.0
        # Penalize running out of energy
        if energy < 10:
            reward -= 50.0
        # Small reward for maintaining healthy budget
        reward += (budget / 50.0)
        return reward

class SmartCityRubric(Rubric):
    """
    Advanced Composite rubric utilizing OpenEnv's Rubric system to provide a rich,
    informative reward signal that captures multiple competing real-world objectives
    (pollution, economy, satisfaction, health, energy grid, city budget) 
    in a complex multi-agent setting.
    """
    def __init__(self):
        super().__init__()
        self.pollution_rubric = PollutionRubric()
        self.economy_rubric = EconomyRubric()
        self.satisfaction_rubric = SatisfactionRubric()
        self.health_rubric = HealthRubric()
        self.energy_budget_rubric = EnergyBudgetRubric()

    def forward(self, action, observation) -> float:
        return (
            self.pollution_rubric(action, observation) + 
            self.economy_rubric(action, observation) + 
            self.satisfaction_rubric(action, observation) +
            self.health_rubric(action, observation) +
            self.energy_budget_rubric(action, observation)
        )
