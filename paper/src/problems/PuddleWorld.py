from problems.BaseProblem import BaseProblem
from environments.PuddleWorld import PuddleWorld as PWEnv
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class PuddleWorld(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = PWEnv(self.env_params, self.seed)
        self.actions = 4

        self.rep = IdentityRep()

        self.rep_params['input_ranges'] = [
            [0, 1],
            [0, 1],
        ]

        self.features = 2
        self.gamma = 1
