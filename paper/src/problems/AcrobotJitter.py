from PyRlEnvs.domains.Acrobot import JitterAcrobot
from problems.Acrobot import Acrobot
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class AcrobotJitter(Acrobot):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)

        # noise is the variance of a mean-zero normal distribution
        # non_normal is the percent of samples which are drawn from a gamma distribution
        self.env = JitterAcrobot(noise=0.1, non_normal=0.1, seed=self.seed)
