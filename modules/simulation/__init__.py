"""
What-If Scenario Simulation Engine for AI Crop Doctor.
Simulates environmental vulnerability shifts under user-specified weather scenario overrides.
"""

from .simulator import ScenarioSimulator, ScenarioComparison

__all__ = [
    "ScenarioSimulator",
    "ScenarioComparison",
]
