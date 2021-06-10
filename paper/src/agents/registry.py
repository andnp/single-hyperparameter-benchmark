from agents.DeepQ import DeepQ
from agents.DQN import DQN
from agents.ESARSA import ESARSA
from agents.QLearning import QLearning
from agents.Random import RandomAgent
from agents.SARSA import SARSA

def getAgent(name):
    if name == 'QLearning':
        return QLearning

    if name == 'SARSA':
        return SARSA

    if name == 'SARSA2':
        return SARSA

    if name == 'ESARSA':
        return ESARSA

    if name == 'DQN':
        return DQN

    if name == 'DeepQ':
        return DeepQ

    if name == 'Random':
        return RandomAgent

    raise NotImplementedError()
