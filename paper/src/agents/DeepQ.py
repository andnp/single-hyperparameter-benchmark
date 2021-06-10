from typing import Dict

from PyExpUtils.utils.Collector import Collector
from PyExpUtils.utils.dict import merge
from agents.DQN import DQN

class DeepQ(DQN):
    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        super().__init__(features, actions, merge(params, {'loss': 'mse'}), seed, collector)
