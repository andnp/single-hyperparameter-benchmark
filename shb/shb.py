from logging import warn
from typing import Any, Dict, List, Generator, Optional, Tuple
from PyExpUtils.utils.dict import merge
from PyExpUtils.utils.permute import getParameterPermutation, getNumberOfPermutations

# Type aliases
Result = Any
Params = Dict[str, Any]
AlgDescription = Tuple[str, Params, Dict[str, Params]]

class Job:
    def __init__(self, idx: int, alg: str, env: str, params: Params, _type: str):
        self.seed: int
        self.alg_seed: int
        self.env_seed: int

        self.idx = idx

        self.env = env
        self.alg = alg

        self.params = params

        self.type = _type

    def record(self, result: Result):
        pass

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
                    for i in range(num_perm):
                        swept_params = getParameterPermutation(param_sweeps, i)
                        params = merge(swept_params, env_params)

                        idx = sr * num_perm + i
                        job = Job(idx, alg, env, params, _type='selection')

                        job.seed = seed
                        job.alg_seed = alg_seed
                        job.env_seed = env_seed

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
                    job = Job(run, alg, env, all_params, _type='evaluation')

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

    def record(self, alg: str, env: str, params: Params, result: Result):
        pass
