import numpy as np
from problems.BaseProblem import BaseProblem
from environments.Gym import Gym
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class CartPole(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.actions = 2

        self.rep = IdentityRep()

        self.features = 4
        self.gamma = 0.99

        x_thresh = 4.8
        theta_thresh = 12 * 2 * np.pi / 360

        # grabbed these ranges from here: https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/FiniteTrackCartPole.py
        self.rep_params['input_ranges'] = [
            [-x_thresh, x_thresh],
            [-6, 6],
            [-theta_thresh, theta_thresh],
            [-2.0, 2.0],
        ]

        # trick gym into thinking the max number of steps is a bit longer
        # that way we get to control the termination at max steps
        self.env = Gym('CartPole-v1', self.run, None)
