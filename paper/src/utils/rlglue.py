import numpy as np
from RlGlue import BaseAgent

# we have two representations
# the first is *the* representation that the agent controls
# the other is the "partial observability" representation that is part of the problem spec
def chainReps(agent_rep, problem_rep):
    return lambda s: agent_rep.encode(problem_rep.encode(s))


# keeps a one step memory for TD based agents
class OneStepWrapper(BaseAgent):
    def __init__(self, agent, gamma, rep):
        self.agent = agent
        self.gamma = gamma
        self.rep = rep

        self.encode = chainReps(agent.rep, self.rep)

        self.s = None
        self.a = None
        self.x = None

    def start(self, s):
        self.s = s
        self.x = self.encode(s)

        self.a = self.agent.selectAction(self.x)
        return self.a

    def step(self, r, sp):
        xp = self.encode(sp)

        self.agent.update(self.x, self.a, xp, r, self.gamma)

        self.a = self.agent.selectAction(xp)

        self.s = sp
        self.x = xp

        return self.a

    def end(self, r):
        self.agent.update(self.x, self.a, np.zeros_like(self.x), r, 0)

        # let the agent do some end of episode clean-up
        # like resetting traces
        self.agent.cleanup()
