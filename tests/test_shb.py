import unittest
from shb.shb import SHB

def buildFakeSHBTrial():
    per_env_params = {
        'MountainCar': { 'representation': { 'type': 'TwoLayerNetwork', 'units': 32 }},
        'PuddleWorld': { 'representation': { 'type': 'TwoLayerNetwork', 'units': 48 }},
        'CartPole': { 'representation': { 'type': 'TwoLayerNetwork', 'units': 64 }},
    }

    algs = [
        ('DQN', {
            'optimizer': { 'stepsize': [0.1, 0.01, 0.001] },
            'epsilon': [0.05, 0.1, 0.15],
            'gamma': [0.99, 0.999],
        }, per_env_params),
        ('DeepQ', {
            'optimizer': { 'stepsize': [0.1, 0.01, 0.001] },
            'epsilon': [0.05, 0.1, 0.15],
            'gamma': [0.99, 0.999],
        }, per_env_params)
    ]

    shb = SHB(
        selection_runs=3,
        eval_runs=250,
        repeated_measures=False,
        algs=algs,
        envs=['MountainCar', 'PuddleWorld', 'CartPole'],
    )

    return shb

class TestSHB(unittest.TestCase):
    def test_modelSelection(self):
        shb = buildFakeSHBTrial()

        jobs = shb.iterateModelSelectionJobs()
        jobs = list(jobs)

        # 2 algs * 18 params * 3 envs = 108
        # 108 things * 3 runs == 324
        self.assertEqual(len(jobs), 324)

        # very first job will be CartPole and DeepQ (alphabetical)
        job = jobs[0]

        # check seeds
        self.assertEqual(job.seed, 0)
        self.assertEqual(job.alg_seed, 0)
        self.assertEqual(job.env_seed, 0)

        # check params
        self.assertDictEqual(job.params, {
            'optimizer': { 'stepsize': 0.1 },
            'epsilon': 0.05,
            'gamma': 0.99,
            'representation': { 'type': 'TwoLayerNetwork', 'units': 64 },
        })

        self.assertEqual(job.alg, 'DeepQ')
        self.assertEqual(job.env, 'CartPole')
        self.assertEqual(job.type, 'selection')
        self.assertEqual(job.idx, 0)

        # should see 3*18 Cartpoles in a row
        job = jobs[54]

        # check seeds
        self.assertEqual(job.seed, 3)
        self.assertEqual(job.env_seed, 0)
        self.assertEqual(job.alg_seed, 3)

        # check params
        self.assertDictEqual(job.params, {
            'optimizer': { 'stepsize': 0.1 },
            'epsilon': 0.05,
            'gamma': 0.99,
            'representation': { 'type': 'TwoLayerNetwork', 'units': 32 },
        })

        self.assertEqual(job.alg, 'DeepQ')
        self.assertEqual(job.env, 'MountainCar')
        self.assertEqual(job.type, 'selection')
        self.assertEqual(job.idx, 0)

        # check some seeds to make sure they match expectations
        self.assertEqual(jobs[1].alg_seed, 0)  # diff params see same nn (or at least assuming same layer sizes etc.)
        self.assertEqual(jobs[1].env_seed, 0)  # diff params see same env

        self.assertEqual(jobs[18].alg_seed, 1)  # same param sees diff nns on next run
        self.assertEqual(jobs[18].env_seed, 1)  # same param sees diff envs on next run

        self.assertEqual(jobs[54].seed, 3)
        self.assertEqual(jobs[54].alg_seed, 3)
        self.assertEqual(jobs[54].env_seed, 0)

        self.assertEqual(jobs[162].seed, 0)  # new alg, first set of params
        self.assertEqual(jobs[162].alg_seed, 0)
        self.assertEqual(jobs[162].env_seed, 0)

    def test_repeatedMeasures(self):
        shb = buildFakeSHBTrial()
        # bypass the warning about repeated measures and small runs :)
        shb.repeated_measures = True

        jobs = shb.iterateModelSelectionJobs()
        jobs = list(jobs)

        # sanity check first alg/env/param
        self.assertEqual(jobs[0].seed, 0)
        self.assertEqual(jobs[0].alg_seed, 0)
        self.assertEqual(jobs[0].env_seed, 0)

        # next param should *always* use same seed
        self.assertEqual(jobs[1].seed, 0)
        self.assertEqual(jobs[1].alg_seed, 0)
        self.assertEqual(jobs[1].env_seed, 0)

        # next run should use a different seed
        self.assertEqual(jobs[18].seed, 1)
        self.assertEqual(jobs[18].alg_seed, 1)
        self.assertEqual(jobs[18].env_seed, 1)

        # next env should reset seeds with repeated measures on
        self.assertEqual(jobs[54].seed, 0)
        self.assertEqual(jobs[54].alg_seed, 0)
        self.assertEqual(jobs[54].env_seed, 0)

    def test_evaluation(self):
        shb = buildFakeSHBTrial()

        # I need some fake parameters which are the "shb" params
        params = {
            'DeepQ': {
                'optimizer': { 'stepsize': 0.1 },
                'epsilon': 0.05,
                'gamma': 0.999,
            },
            'DQN': {
                'optimizer': { 'stepsize': 0.01 },
                'epsilon': 0.15,
                'gamma': 0.99,
            },
        }

        jobs = shb.iterateEvaluationJobs(params)
        jobs = list(jobs)

        # 250 * 2 algs * 3 envs = 1500
        self.assertEqual(len(jobs), 1500)

        # very first job will be CartPole and DeepQ (alphabetical)
        job = jobs[0]

        # check seeds
        self.assertEqual(job.seed, 324)
        self.assertEqual(job.alg_seed, 324)
        self.assertEqual(job.env_seed, 324)

        # check params
        self.assertDictEqual(job.params, {
            'optimizer': { 'stepsize': 0.1 },
            'epsilon': 0.05,
            'gamma': 0.999,
            'representation': { 'type': 'TwoLayerNetwork', 'units': 64 },
        })

        self.assertEqual(job.alg, 'DeepQ')
        self.assertEqual(job.env, 'CartPole')
        self.assertEqual(job.type, 'evaluation')
        self.assertEqual(job.idx, 0)

        # next run of same alg, env pair. New seed
        job = jobs[1]
        self.assertEqual(job.seed, 325)
        self.assertEqual(job.alg_seed, 325)
        self.assertEqual(job.env_seed, 325)

        # first run of next env
        job = jobs[250]
        self.assertEqual(job.seed, 574)
        self.assertEqual(job.alg_seed, 574)
        self.assertEqual(job.env_seed, 324)

        # first run of next alg
        job = jobs[750]
        self.assertEqual(job.seed, 324)
        self.assertEqual(job.alg_seed, 324)
        self.assertEqual(job.env_seed, 324)
