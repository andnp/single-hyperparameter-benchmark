import numpy as np
from typing import Any, Callable, Sequence
from PyExpUtils.utils.types import NpList
from PyExpUtils.utils.random import sample

class Policy:
    def __init__(self, probs: Callable[[Any], NpList], rng = np.random):
        self.probs = probs
        self.random = rng

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(action_probabilities, rng=self.random)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: Sequence[NpList]):
    return Policy(lambda s: probs[s])

def fromActionArray(probs: NpList):
    return Policy(lambda s: probs)
