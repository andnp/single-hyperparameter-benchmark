PROBLEMS = ['Acrobot', 'Cartpole', 'CliffWorld', 'JitterAcrobot', 'LunarLander', 'MountainCar', 'PuddleWorld']

def getCutoff(problem: str):
    if problem == 'CliffWorld':
        return [-1, 50, 100]

    return [-1, 500, 1000]

def getMaxSteps(problem: str):
    if problem == 'LunarLander':
        return 250000

    return 200000

def getBufferSize(problem: str):
    if problem == 'LunarLander':
        return 10000

    return 4000

def getRepType(problem: str):
    if problem == 'CliffWorld':
        return "OneLayerRelu"

    return "TwoLayerRelu"

def getHidden(problem: str):
    if problem == 'LunarLander':
        return 128

    if problem == 'Cartpole':
        return 64

    if problem == 'CliffWorld':
        return 16

    return 32

def getNNSpec(problem: str, agent: str):
    return f"""{{
    "agent": "{agent}",
    "problem": "{problem}",
    "max_steps": {getMaxSteps(problem)},
    "metaParameters": {{
        "epsilon": 0.1,
        "alpha": {[2**-i for i in range(6, 13)]},

        "target_refresh": [1, 8, 32],
        "buffer_size": {getBufferSize(problem)},
        "batch": 32,
        "control_seed": true,
        "buffer_controlled": true,

        "optimizer": {{
            "name": "ADAM",
            "beta1": 0.9,
            "beta2": 0.999
        }},

        "representation": {{
            "type": "{getRepType(problem)}",
            "hidden": {getHidden(problem)}
        }},

        "environment": {{
            "control_seed": true
        }},

        "experiment": {{
            "cutoff": {getCutoff(problem)}
        }}
    }}
}}
"""

def getTC(problem: str):
    if problem == 'LunarLander':
        return 'll-tile-coder'

    return 'tile-coder'

def getTCSpec(problem: str, agent: str):
    return f"""{{
    "agent": "{agent}",
    "problem": "{problem}",
    "max_steps": {getMaxSteps(problem)},
    "metaParameters": {{
        "epsilon": 0.1,
        "alpha": {[2**-i for i in range(2, 10)]},

        "control_seed": true,

        "representation": {{
            "type": "{getTC(problem)}",
            "tiles": [2, 4, 8],
            "tilings": [8, 16, 32]
        }},

        "environment": {{
            "control_seed": true
        }},

        "experiment": {{
            "cutoff": {getCutoff(problem)}
        }}
    }}
}}
"""

NN_AGENTS = ['DQN', 'DeepQ']
TC_AGENTS = ['ESARSA', 'QLearning']

for problem in PROBLEMS:
    for agent in NN_AGENTS + TC_AGENTS:

        if agent in NN_AGENTS:
            spec = getNNSpec(problem, agent)
        else:
            spec = getTCSpec(problem, agent)

        path = f'experiments/benchmark/{problem}/{agent}.json'
        with open(path, 'w') as f:
            f.write(spec)
