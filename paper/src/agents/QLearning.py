import numpy as np
from agents.BaseAgent import BaseAgent
from numba import njit

@njit(cache=True)
def _update(w, x, a, xp, r, gamma, alpha):
    qsa = w[a][x].sum()

    qsp = w.T[xp].sum(axis=0)

    delta = r + gamma * qsp.max() - qsa

    w[a][x] = w[a][x] + alpha / len(x) * delta

class QLearning(BaseAgent):
    # where the learning magic happens
    # uses state-based gamma (so no need to handle terminal states specially)
    def update(self, x, a, xp, r, gamma):
        _update(self.w, x, a, xp, r, gamma, self.alpha)

    def values(self, x: np.ndarray):
        return self.w[:, x].sum(axis=1)
