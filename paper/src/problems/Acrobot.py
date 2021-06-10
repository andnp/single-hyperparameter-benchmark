import numpy as np
from problems.BaseProblem import BaseProblem
from PyRlEnvs.domains.Acrobot import Acrobot as Env
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class Acrobot(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.actions = 3

        self.rep = IdentityRep()

        self.features = 6
        self.gamma = 0.99

        ma_vel1 = 4 * np.pi
        ma_vel2 = 9 * np.pi

        self.rep_params['input_ranges'] = [
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-ma_vel1, ma_vel1],
            [-ma_vel2, ma_vel2],
        ]

        # trick gym into thinking the max number of steps is a bit longer
        # that way we get to control the termination at max steps
        self.env = Env(self.run)
