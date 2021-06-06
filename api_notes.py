# This script is intended to define what I would expect an shb api to look like
# I'll develop the code according to this contract, then delete this script

from shb import SHB

env_specific_params = {
    'MountainCar': {
        'layers': [
            { 'units': 32, 'act': 'relu' },
            { 'units': 32, 'act': 'relu' },
        ]
    },
    'CartPole': {
        'layers': [
            { 'units': 64, 'act': 'relu' },
            { 'units': 64, 'act': 'relu' },
        ]
    }
}

# sweep these params for all environments
param_sweeps = {
    'alpha': [2**-i for i in range(6, 13)],
    'gamma': [0.9, 0.95, 0.99],
    'epsilon': [0.05, 0.1, 0.15],
}

shb = SHB()

shb.registerAlg('DQN', param_sweeps, env_specific_params)
shb.registerAlg('QRC', param_sweeps, env_specific_params)

shb.registerEnvPool(['MountainCar', 'CartPole'])

# make defaults selectionRuns=3, evalRuns=100
# let's allow the ability to use repeated_measures, for instance if they choose to make selectionRuns pretty big
jobs = shb.iterateJobs(alg='DQN', selectionRuns=3, evalRuns=250, repeated_measures=False)

print(len(jobs))  # => 2 envs * (63 params * 3 runs + 250 runs) = 878

for job in jobs:
    # let's handle generating the correct seeds
    # we'll hand back a global seed that is the bare minimum
    # this should be consistent from alg to alg
    # should handle no repeated measures for selectionRuns
    # and different seeds for evalRuns than selectionRuns
    seed = job.seed

    # as well as optional specific seeds
    alg_seed = job.alg_seed  # no repeated measures over envs, consistent across algs
    env_seed = job.env_seed  # consistent across algs

    # get the names of alg/env pair to run
    env = job.env
    alg = job.alg

    # we will handle iterating params
    params = job.params
    print(params)  # =>
    expected = {
        'alpha': 2**-6,
        'gamma': 0.9,
        'epsilon': 0.05,
        'layers': [
            { 'units': 32, 'act': 'relu' },
            { 'units': 32, 'act': 'relu' },
        ]
    }

    job.record(result)

# -----------------------------------------------------------
# if results are already recorded and saved elsewhere on disk
# -----------------------------------------------------------

shb.registerAlg(...)
shb.registerEnvPool(...)

for result in dataset:
    # I don't really care what their dataset looks like, just need to know env, alg, params, result and I will store this myself.
    shb.record(result.alg, result.env, result.params, result.data)
