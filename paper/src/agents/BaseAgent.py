from abc import abstractmethod
from typing import Dict
import numpy as np
from numba import njit
from representations.registry import getRepresentation

from utils.policies import Policy
from PyExpUtils.utils.arrays import argsmax
from PyExpUtils.utils.Collector import Collector
from utils.random import getControlledSeed

@njit(cache=True)
def egreedy_probabilities(qs: np.ndarray, actions: int, epsilon: float):
    # compute the greedy policy
    max_acts = argsmax(qs)
    pi: np.ndarray = np.zeros(actions)
    for a in max_acts:
        pi[a] = 1. / len(max_acts)

    # compute a uniform random policy
    uniform: np.ndarray = np.ones(actions) / actions

    # epsilon greedy is a mixture of greedy + uniform random
    return (1. - epsilon) * pi + epsilon * uniform

class BaseAgent:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        self.actions = actions
        self.params = params
        self.collector = collector
        self.policy_rng = np.random.RandomState(getControlledSeed(self.params.get('policy_controlled', True), seed))
        self.seed = seed

        # define parameter contract
        self.alpha: float = params.get('alpha', 0.0)
        self.epsilon: float = params.get('epsilon', 0.0)

        # have the agent build the representation
        # but let the "agent_wrapper" deal with using it
        # that way we can save a bit of compute by caching things
        self.rep_params: Dict = params.get('representation', { 'type': 'identity' })

        Rep = getRepresentation(self.rep_params['type'])
        self.rep = Rep(self.rep_params, features)

        self.features = self.rep.features()

        # learnable parameters
        self.w: np.ndarray = np.zeros((actions, self.features))

        # compute the action probabilities based on the q-values
        def probabilities(x: np.ndarray):
            qs = self.values(x)
            return egreedy_probabilities(qs, self.actions, self.epsilon)

        # create a policy utility object to help keep track of sampling from the probabilities
        self.policy = Policy(probabilities, rng=self.policy_rng)

    # compute the value function given a numpy input
    # returns an np.array of size <actions>
    def values(self, x: np.ndarray):
        return self.w.dot(x)

    # just to conform to rlglue interface
    # passes action selection on to the policy utility object
    def selectAction(self, x: np.ndarray):
        return self.policy.selectAction(x)

    # where the learning magic happens
    # uses state-based gamma (so no need to handle terminal states specially)
    @abstractmethod
    def update(self, x: np.ndarray, a: int, xp: np.ndarray, r: float, gamma: float):
        pass

    def cleanup(self):
        pass
