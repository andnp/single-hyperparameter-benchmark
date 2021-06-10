from problems.BaseProblem import BaseProblem
from environments.Minatar import Minatar
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class MinBreakout(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = Minatar('breakout', self.run)
        self.actions = self.env.env.num_actions()

        self.rep = IdentityRep()

        # where features stands for "channels" here
        self.features = self.env.env.state_shape()[2]
        self.gamma = 0.99
