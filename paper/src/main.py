import numpy as np
import logging
import socket
import torch
import time
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from experiment import ExperimentModel
from problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector
from utils.rlglue import OneStepWrapper

if len(sys.argv) < 2:
    print('run again with:')
    print('python3 src/main.py <path/to/description.json> <idx> <opt:prod>')
    exit(1)

prod = len(sys.argv) == 4 or 'cdr' in socket.gethostname()
# prod = True
if not prod:
    logging.basicConfig(level=logging.DEBUG)
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)


torch.set_num_threads(1)

exp = ExperimentModel.load(sys.argv[1])
idx = int(sys.argv[2])

max_steps = exp.max_steps
run = exp.getRun(idx)

collector = Collector()

# perform initial rejection downsampling to reduce memory costs
# collector.setSampleRate('update_interference', int(max_steps // 100))

# set random seeds accordingly
np.random.seed(run)
torch.manual_seed(run)

Problem = getProblem(exp.problem)
problem = Problem(exp, idx)

# parameters of the experiment that might be manipulated
# things like episode cutoffs
exp_params = problem.exp_params

agent = problem.getAgent(collector)
env = problem.getEnvironment()

wrapper = OneStepWrapper(agent, problem.getGamma(), problem.rep)

glue = RlGlue(wrapper, env)

# Run the experiment
glue.start()
start_time = time.time()
episode = 0

# track the reward over episode boundaries
# whenever we "teleport" due to cutoff
# then terminate this "meta-episode" only when we actually hit the goal state
running_reward = 0
running_steps = 0
for step in range(exp.max_steps):
    r, _, _, t = glue.step()

    # on the terminal state or max steps
    # go back to start state
    # not that agent.end() is **not** called when we hit episode max
    # so we just teleport and do not bootstrap off of the "teleport" transition
    should_limit_episode_steps = exp_params.get('cutoff', -1) != -1
    is_episode_max = should_limit_episode_steps and glue.num_steps >= exp_params['cutoff']
    if t or is_episode_max:
        if is_episode_max and exp_params.get('cutoff_update', False):
            wrapper.end(r)

        episode += 1

        # make sure traces still get cleared up
        agent.cleanup()

        # collect an array of rewards that is the length of the number of steps in episode
        # effectively we count the whole episode as having received the same final reward
        collector.concat('step_return', [glue.total_reward] * glue.num_steps)
        # also track the reward per episode (this may not have the same length for all agents!)
        collector.collect('episodic_return', glue.total_reward)

        running_reward += glue.total_reward
        running_steps += glue.num_steps
        if t:
            collector.collect('big_episodic_return', running_reward)
            collector.concat('big_step_return', [running_reward] * running_steps)
            running_reward = 0
            running_steps = 0

        # compute the average time-per-step in ms
        avg_time = 1000 * (time.time() - start_time) / step
        logging.debug(f' {episode} {step} {glue.total_reward} {avg_time:.4}ms')

        glue.start()


# try to detect if a run never finished
# if we have no data in the 'step_return' key, then the termination condition was never hit
if collector.run_data.get('step_return') is None:
    # collect an array of rewards that is the length of the number of steps in episode
    # effectively we count the whole episode as having received the same final reward
    collector.concat('step_return', [glue.total_reward] * glue.num_steps)
    # also track the reward per episode (this may not have the same length for all agents!)
    collector.collect('episodic_return', glue.total_reward)

collector.fillRest('step_return', exp.max_steps)

from PyExpUtils.results.backends.h5 import saveResults

for key in collector.run_data:
    data = collector.run_data[key]
    saveResults(exp, idx, key, data)
