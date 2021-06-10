from problems.AcrobotJitter import AcrobotJitter
from problems.CarRacing import CarRacing
from problems.CliffWorld import CliffWorld
from problems.PuddleWorld import PuddleWorld
from problems.Acrobot import Acrobot
from problems.CartPole import CartPole
from problems.LunarLander import LunarLander
from problems.MinBreakout import MinBreakout
from problems.MountainCar import MountainCar

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    if name == 'Cartpole':
        return CartPole

    if name == 'Acrobot':
        return Acrobot

    if name == 'JitterAcrobot':
        return AcrobotJitter

    if name == 'LunarLander':
        return LunarLander

    if name == 'MinBreakout':
        return MinBreakout

    if name == 'PuddleWorld':
        return PuddleWorld

    if name == 'CarRacing':
        return CarRacing

    if name == 'CliffWorld':
        return CliffWorld

    raise NotImplementedError()
