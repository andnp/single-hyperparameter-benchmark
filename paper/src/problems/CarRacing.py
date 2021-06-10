import numpy as np
from typing import Optional
from problems.BaseProblem import BaseProblem
from environments.Gym import Gym
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class Env(Gym):
    def __init__(self, seed: int, max_steps: Optional[int]):
        super().__init__('CarRacing-v0', seed, max_steps=max_steps)

    def start(self):
        s = super().start()

        return s.transpose(2, 0, 1)

    # grabbed the re-encoded actions from the human-playable version of this domain
    # in the openai-gym repo here: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py#L603
    def step(self, a):
        car_a = np.array([0., 0., 0.])

        if a == 0:
            car_a[0] = -1.0

        elif a == 1:
            car_a[0] = 1.0

        elif a == 2:
            car_a[1] = 1.0

        elif a == 3:
            car_a[2] = 0.8

        r, sp, t = super().step(car_a)
        sp = sp.transpose(2, 0, 1)
        self.env.render()
        return (r, sp, t)

class CarRacing(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.actions = 5

        self.rep = IdentityRep()

        # features stands for channels here
        self.features = 3
        self.gamma = 0.99

        # trick gym into thinking the max number of steps is a bit longer
        # that way we get to control the termination at max steps
        self.env = Env(self.run, 500)
