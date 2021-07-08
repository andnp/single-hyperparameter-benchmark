import numpy as np
from logging import warn
from typing import Any, Callable, Dict, List, Generator, Optional, Tuple, Union
from PyExpUtils.utils.dict import merge
from PyExpUtils.utils.permute import getParameterPermutation, getNumberOfPermutations

# Type aliases
Params = Dict[str, Any]
AlgDescription = Tuple[str, Params, Dict[str, Params]]

class Job:
    """
    A data class representing meta-data for a single job: (alg, env, run, hyper-setting)-tuple

    ...
    Attributes
    ----------
    seed : int
        A global random seed
    alg_seed : int
        A random seed for the algorithm specifically
    env_seed : int
        A random seed for the environment specifically

    idx : int
        The parameter combination index (supplied by PyExpUtils)
    run : int
        The run number for this particular tuple of (alg, env, hyper-setting)

    params : Dict[str, Any]
        A dictionary mapping from a hyper name to a single value for that hyper

    type : 'selection' | 'evaluation'
        A string literal indicating which stage of the SHB is being executed

    Methods
    -------

    record(data: float) -> None
        saves a single result of this particular job to the parent `shb` object
    """
    def __init__(self, idx: int, alg: str, env: str, params: Params, run: int, _type: str):
        self.seed: int
        self.alg_seed: int
        self.env_seed: int

        self.idx = idx
        self.run = run

        self.env = env
        self.alg = alg

        self.params = params

        self.type = _type

        self._storeData: Callable

    def record(self, result: float):
        if self.type == 'evaluation':
            raise Exception("Sorry, don't know how to store evaluation data")

        self._storeData(self.alg, self.env, self.idx, self.run, result)


class SHB:
    def __init__(self, selection_runs: int = 3, eval_runs: int = 250, repeated_measures: bool = False, algs: Optional[List[AlgDescription]] = None, envs: Optional[List[str]] = None) -> None:
        self._algs: Dict[str, AlgDescription] = {}
        self._envs = envs if envs is not None else []

        if algs is not None:
            for alg in algs:
                self._algs[alg[0]] = alg

        self.repeated_measures = repeated_measures
        self.selection_runs = selection_runs
        self.eval_runs = eval_runs

        self.data: Union[None, Dict[str, np.ndarray]] = None

        if repeated_measures and selection_runs < 30:
            warn('Using repeated measures with a small number of runs will result in high bias.')

    def registerAlg(self, alg: str, params: Params, per_env_params: Optional[Dict[str, Params]] = None):
        # sanity check, make sure this isn't already registered
        if alg in self._algs:
            raise Exception('Algorithm has already been registered', alg)

        # otherwise, register it
        # let's always make the per_env_params a dict for convenience
        if per_env_params is None:
            per_env_params = {}

        self._algs[alg] = (alg, params, per_env_params)

    def registerEnvPool(self, envs: List[str]):
        # sanity check, make sure not registered yet
        for env in envs:
            if env in self._envs:
                raise Exception('Environment has already been registered', env)

        # otherwise, register them
        self._envs += envs

    def iterateModelSelectionJobs(self) -> Generator[Job, None, None]:
        # let's guarantee some orderings
        algs = sorted(self._algs.keys(), key=str.casefold)
        envs = sorted(self._envs, key=str.casefold)

        for alg in algs:
            _, param_sweeps, per_env = self._algs[alg]
            for e, env in enumerate(envs):
                env_params = per_env.get(env, {})
                num_perm = getNumberOfPermutations(param_sweeps)

                seed = e * self.selection_runs
                alg_seed = e * self.selection_runs
                env_seed = 0

                if self.repeated_measures:
                    seed = 0
                    alg_seed = 0

                for sr in range(self.selection_runs):
                    for idx in range(num_perm):
                        swept_params = getParameterPermutation(param_sweeps, idx)
                        params = merge(swept_params, env_params)

                        job = Job(idx, alg, env, params, sr, _type='selection')

                        job.seed = seed
                        job.alg_seed = alg_seed
                        job.env_seed = env_seed

                        job._storeData = self.record

                        yield job

                    alg_seed += 1
                    env_seed += 1
                    seed += 1

    def iterateEvaluationJobs(self, alg_params: Dict[str, Params]) -> Generator[Job, None, None]:
        algs = sorted(self._algs.keys(), key=str.casefold)
        envs = sorted(self._envs, key=str.casefold)

        # sanity check that we have params specified for each alg
        # also ensure there is only one setting specified for each alg
        for alg in algs:
            params = alg_params[alg]
            num_perm = getNumberOfPermutations(params)
            assert num_perm == 1

        # we need to know how many seeds we've tranversed so far
        # so that we use fresh seeds for the evaluation runs
        # otherwise we suffer a *large* amount of maximization bias
        # for now just use a lazy heuristic: we've definitely used less seeds than num selection jobs
        selection_jobs = self.iterateModelSelectionJobs()
        seed_offset = len(list(selection_jobs))

        # if we've gotten here, things are appropriately specified
        for alg in algs:
            seed = seed_offset
            alg_seed = seed_offset
            env_seed = seed_offset

            params = alg_params[alg]
            _, _, per_env = self._algs[alg]
            for env in envs:
                env_params = per_env.get(env, {})
                all_params = merge(params, env_params)
                for run in range(self.eval_runs):
                    job = Job(0, alg, env, all_params, run, _type='evaluation')

                    job.seed = seed
                    job.alg_seed = alg_seed
                    job.env_seed = env_seed
                    yield job

                    seed += 1
                    alg_seed += 1
                    env_seed += 1

                # if using repeated measures, reset seeds for each env
                if self.repeated_measures:
                    alg_seed = seed_offset
                    seed = seed_offset

                # always reset the env seed for each new env
                env_seed = seed_offset

    def _setUpDataStorage(self):
        if self.data is not None:
            raise Exception('We have already setup the data storage')

        n_envs = len(self._envs)

        # don't assume each alg has same number of parameters
        # so use separate storage for each
        self.data = {}

        for alg in self._algs:
            _, sweepable, _ = self._algs[alg]
            n_params = getNumberOfPermutations(sweepable)

            self.data[alg] = np.zeros((n_envs, n_params, self.selection_runs))

        # for type inference purposes
        return self.data

    def record(self, alg: str, env: str, param_idx: int, run: int, result: float):
        envs = sorted(self._envs, key=str.casefold)

        env_idx = envs.index(env)

        # if we've not stored any data yet, first initialize some storage
        if self.data is None:
            self.data = self._setUpDataStorage()

        storage = self.data[alg]

        storage[env_idx, param_idx, run] = result

    def cdfScale(self, env: Union[str, int], data: float):
        if self.data is None:
            raise Exception("Can't cdfScale without data")

        if type(env) is str:
            envs = sorted(self._envs, key=str.casefold)
            env_idx = envs.index(env)

        else:
            env_idx = env

        cdfs = np.empty(len(self._algs))

        for i, alg in enumerate(self._algs):
            env_data = self.data[alg][env_idx]
            count = np.sum(env_data < data)
            cdfs[i] = count / (env_data.shape[0] * env_data.shape[1])

        return np.mean(cdfs)

    def pickParameters(self):
        if self.data is None:
            raise Exception("Can't pick parameters without data")

        out: Dict[str, Params] = {}
        envs = sorted(self._envs, key=str.casefold)
        for alg in self._algs:
            param_vals = np.zeros(self.data[alg].shape[0:2])
            for i, env in enumerate(envs):
                data = self.data[alg][i]

                # take cdfScaling of each run/param combo
                vf = np.vectorize(lambda x: self.cdfScale(env, x))
                scaled = vf(data)

                # average over runs
                scaled = np.mean(scaled, axis=1)
                param_vals[i] = scaled

            # average over environments
            param_vals = np.mean(param_vals, axis=0)

            # max over parameters
            param_idx = int(np.argmax(param_vals))

            # save params
            _, sweepable, _ = self._algs[alg]
            out[alg] = getParameterPermutation(sweepable, param_idx)

        return out
