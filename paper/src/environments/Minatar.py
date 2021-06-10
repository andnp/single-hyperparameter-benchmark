import numpy as np
from RlGlue import BaseEnvironment
from minatar import Environment

class Minatar(BaseEnvironment):
    def __init__(self, name, seed):
        self.env = Environment(name, random_seed=seed)

    def start(self):
        self.env.reset()
        s = self.env.state()
        s = s.transpose(2, 0, 1)
        return s

    def step(self, a):
        r, t = self.env.act(a)
        sp = self.env.state()
        sp = sp.transpose(2, 0, 1)

        return (r, sp, t)
