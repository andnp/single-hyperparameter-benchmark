from problems.BaseProblem import BaseProblem
from environments.Gym import Gym
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class LunarLander(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.actions = 4

        self.rep = IdentityRep()

        self.features = 8
        self.gamma = 0.99

        # I'm actually going to encode the last 2 binary features as a single [0, 3] feature
        self.rep_params['input_ranges'] = [
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [0, 3],
        ]

        self.env = Gym('LunarLander-v2', self.run, None)
