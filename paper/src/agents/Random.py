import numpy as np
from typing import Dict
from PyExpUtils.utils.Collector import Collector
from agents.BaseAgent import BaseAgent
from utils.policies import Policy

class RandomAgent(BaseAgent):
    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        super().__init__(features, actions, params, seed, collector)

        def probabilities(x: np.ndarray):
            return np.ones(actions) / actions

        self.policy = Policy(probabilities, rng=self.policy_rng)
