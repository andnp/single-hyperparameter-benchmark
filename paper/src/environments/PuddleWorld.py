import numpy as np
from RlGlue import BaseEnvironment
from utils.random import getControlledSeed

from numba import njit

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

@njit(cache=True)
def l2_dist(x1, y1, x2, y2):
    return np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))

class puddle():
    def __init__(self, headX, headY, tailX, tailY, radius, length, axis):
        self.headX = headX
        self.headY = headY
        self.tailX = tailX
        self.tailY = tailY
        self.radius = radius
        self.length = length
        self.axis = axis

    def get_distance(self, xCoor, yCoor):

        if self.axis == 0:
            u = (xCoor - self.tailX)/self.length
        else:
            u = (yCoor - self.tailY)/self.length

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = l2_dist(self.tailX, self.tailY, xCoor, yCoor)
            else:
                dist = l2_dist(self.headX, self.headY, xCoor, yCoor)
        else:
            x = self.tailX + u * (self.headX - self.tailX)
            y = self.tailY + u * (self.headY - self.tailY)

            dist = l2_dist(x, y, xCoor, yCoor)

        if dist < self.radius:
            return (self.radius - dist)
        else:
            return 0

class PuddleWorld(BaseEnvironment):
    def __init__(self, params, seed):
        self.easy_prob = params.get('easy_prob', 0)
        self.start_pos = params.get('start_pos', 0)

        env_seed = getControlledSeed(params.get('control_seed', True), seed)
        self.random = np.random.RandomState(env_seed)

        self.episodic = True

        self.state = None
        self.puddle1 = puddle(0.45,0.75,0.1,0.75,0.1,0.35,0)
        self.puddle2 = puddle(0.45,0.8,0.45,0.4,0.1,0.4,1)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0
        self.pworld_mid_x = 0.5
        self.pworld_mid_y = 0.5

        self.goal_dimension = 0.05
        self.def_displacement = 0.05

        self.sigma = 0.01

        self.goal_x_coor = self.pworld_max_x - self.goal_dimension
        self.goal_y_coor = self.pworld_max_y - self.goal_dimension

        self.easy_hard = np.array([[0.9,0.9],  [self.start_pos, self.start_pos]])

        # self.easy_hard = [np.array([0.7,0.85]),np.array([0.35,0.6])]
        # self.easy_hard = [np.array([0.9,0.9]),np.array([0.4,0.7])] #5th November
        # self.easy_hard = [np.array([0.97,0.1]),np.array([0.4,0.7])] #8th November
        # self.easy_hard = [np.array([0.9,0.9]),np.array([0.01,0.01])] #9th November
        # self.easy_hard = [np.array([0.9,0.9]),np.array([0.0,0.0])] #12th November
        # self.easy_hard = [np.array([0.4,0.7]),np.array([0.0,0.0])] #17th November
        # self.easy_hard = [np.array([0.2,0.2]),np.array([0.0,0.0])] #18th November
        # self.easy_hard = [np.array([0.0,0.2]),np.array([0.0,0.0])] #18th November-2
        #maybe any state starting on the edge is hard?
        # self.easy_hard = [np.array([0.9,0.9]), np.array([0.0,0.0])] #20th November

    def actions(self, s):
        return [UP, DOWN, RIGHT, LEFT]

    def start(self):
        self.state = np.zeros(2)
        if self.random.random() < self.easy_prob:
            self.state[:] = self.easy_hard[0]
        else:
            self.state[:] = self.easy_hard[1]

        return np.copy(self.state)


    def _terminal(self):
        s = self.state
        return (s[0] >= self.goal_x_coor) and (s[1] >= self.goal_y_coor)

    def _reward(self, x, y, terminal):
        if terminal:
            return -1.

        reward = -1.
        dist = self.puddle1.get_distance(x, y)
        reward += (-400. * dist)
        dist = self.puddle2.get_distance(x, y)
        reward += (-400. * dist)

        return reward

    def step(self, a):
        s = self.state

        xpos = s[0]
        ypos = s[1]

        n = self.random.normal(scale=self.sigma)

        if a == UP: #up
            ypos += (self.def_displacement+n)
        elif a == DOWN: #down
            ypos -= (self.def_displacement+n)
        elif a == RIGHT: #right
            xpos += (self.def_displacement+n)
        elif a == LEFT: #left
            xpos -= (self.def_displacement+n)
        else:
            raise Exception()

        if xpos > self.pworld_max_x:
            xpos = self.pworld_max_x
        elif xpos < self.pworld_min_x:
            xpos = self.pworld_min_x

        if ypos > self.pworld_max_y:
            ypos = self.pworld_max_y
        elif ypos < self.pworld_min_y:
            ypos = self.pworld_min_y

        s[0] = xpos
        s[1] = ypos
        self.state = s

        t = self._terminal()
        r = self._reward(xpos, ypos, t)

        return (r, np.copy(self.state), t)
