from problems.BaseProblem import BaseProblem
from PyRlEnvs.domains.MountainCar import MountainCar as Env
from PyFixedReps.BaseRepresentation import BaseRepresentation
from utils.random import getControlledSeed

def minMax(x, mi, ma):
    return (x - mi) / (ma - mi)

class MCScaledRep(BaseRepresentation):
    def encode(self, s, a=None):
        p, v = s

        sp = minMax(p, -1.2, 0.5)
        sv = minMax(v, -0.07, 0.07)

        return sp, sv

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        seed = getControlledSeed(self.env_params.get('control_seed', True), self.seed)

        self.env = Env(seed)
        self.actions = 3

        self.rep = MCScaledRep()

        self.rep_params['input_ranges'] = [
            [-1, 1],
            [-1, 1],
        ]

        self.features = 2
        self.gamma = 0.99
