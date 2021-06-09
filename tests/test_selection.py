import unittest
import numpy as np
from shb.shb import SHB

def buildFakeSHBTrial():
    per_env_params = {
        'MountainCar': { 'representation': { 'type': 'TwoLayerNetwork', 'units': 32 }},
        'CartPole': { 'representation': { 'type': 'TwoLayerNetwork', 'units': 64 }},
    }

    algs = [
        ('DQN', {
            'optimizer': { 'stepsize': [0.1, 0.01, 0.001] },
            'epsilon': [0.05, 0.1],
            'gamma': 0.99,
        }, per_env_params),
        ('DeepQ', {
            'optimizer': { 'stepsize': [0.1, 0.01, 0.001] },
            'epsilon': [0.05, 0.1],
            'gamma': 0.99,
        }, per_env_params)
    ]

    shb = SHB(
        selection_runs=3,
        eval_runs=250,
        repeated_measures=False,
        algs=algs,
        envs=['MountainCar', 'CartPole'],
    )

    return shb

class TestSelection(unittest.TestCase):
    def test_cdfScale(self):
        shb = buildFakeSHBTrial()

        jobs = shb.iterateModelSelectionJobs()
        for i, job in enumerate(jobs):
            # pretend to run an experiment on this job
            # then save some results
            job.record(i)

        cdf = shb.cdfScale('CartPole', 10)
        self.assertAlmostEqual(cdf, 0.27777, places=4)

        cdf = shb.cdfScale('MountainCar', 10)
        self.assertAlmostEqual(cdf, 0)

        cdf = shb.cdfScale('MountainCar', 19)
        self.assertAlmostEqual(cdf, 1 / 36)

    def test_pickParameters(self):
        np.random.seed(0)
        shb = buildFakeSHBTrial()
        shb._setUpDataStorage()

        # let's construct some fake data
        # let this be the best param
        shb.data['DeepQ'][0, 2] = [40, 45, 50]
        shb.data['DeepQ'][0, 0:2] = -3
        shb.data['DeepQ'][0, 2:6] = 30
        shb.data['DeepQ'][1] = 50

        # and some random data for the other alg
        shb.data['DQN'] = np.random.normal(0, 3, size=(2, 6, 3))

        params = shb.pickParameters()

        expected = {
            'DeepQ': {
                'epsilon': 0.05,
                'gamma': 0.99,
                'optimizer': { 'stepsize': 0.01 },
            },
            'DQN': {
                'epsilon': 0.1,
                'gamma': 0.99,
                'optimizer': { 'stepsize': 0.1 },
            },
        }

        self.assertDictEqual(params, expected)
