import numpy as np
from typing import Dict, Union
from agents.BaseAgent import BaseAgent
from PyExpUtils.utils.Collector import Collector
from numba import njit

@njit(cache=True)
def _update(w, x, a, r, xp, ap, gamma, alpha):
    qsa = w[a].dot(x)
    qspap = w[ap].dot(xp)

    delta = r + gamma * qspap - qsa

    w[a] = w[a] + alpha * delta * x

class SARSA(BaseAgent):
    ap: Union[int, None]

    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        super().__init__(features, actions, params, seed, collector)

        # TODO: convert this to use indices as features similar to the ESARSA agent
        raise Exception('This is momentarily not working')

        self.ap = None

    def selectAction(self, x: np.ndarray):
        if self.ap is None:
            self.ap = self.policy.selectAction(x)

        return self.ap

    # where the learning magic happens
    # uses state-based gamma (so no need to handle terminal states specially)
    def update(self, x: np.ndarray, a: int, xp: np.ndarray, r: float, gamma: float):
        self.ap = self.policy.selectAction(x)

        _update(self.w, x, a, r, xp, self.ap, gamma, self.alpha)

    def cleanup(self):
        self.ap = None
