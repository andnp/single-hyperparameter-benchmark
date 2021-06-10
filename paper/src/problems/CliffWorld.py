from problems.BaseProblem import BaseProblem
from PyRlEnvs.domains.CliffWorld import CliffWorld as Env
from PyFixedReps.BaseRepresentation import BaseRepresentation
from utils.random import getControlledSeed

class GridRep(BaseRepresentation):
    def __init__(self, transform):
        self.transform = transform

    def encode(self, s, a=None):
        return self.transform(s)

class CliffWorld(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        seed = getControlledSeed(self.env_params.get('control_seed', True), self.seed)

        self.env = Env(seed)
        self.actions = 4

        self.rep = GridRep(Env.getCoords)

        self.rep_params['input_ranges'] = [
            [0, Env.shape[0]],
            [0, Env.shape[1]],
        ]

        self.features = 2
        self.gamma = 0.99
