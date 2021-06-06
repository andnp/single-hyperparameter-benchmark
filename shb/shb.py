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
        for a, alg in enumerate(self._algs):
            _, sweepable, per_env = self._algs[alg]
            for e, env in enumerate(self._envs):
                # we want the global seed to be different for each env.
                # this way if I am randomly sampling over nn initializations, I get a different set of NNs for each env
                seed = e
                alg_seed = e
                env_seed = 0

                env_params = per_env.get(env, {})
                param_sweeps: Params = merge(sweepable, env_params)

                num_perm = getNumberOfPermutations(param_sweeps)

                for sr in range(self.selection_runs):
                    for i in range(num_perm):
                        params = getParameterPermutation(param_sweeps, i)

                        idx = sr * num_perm + i
                        job = Job(idx, alg, env, params, _type='selection')

                        job.seed = seed
                        job.alg_seed = alg_seed
                        job.env_seed = env_seed

                        yield job

                        if not self.repeated_measures:
                            seed += 1
                            env_seed += 1
                            alg_seed += 1

                    if self.repeated_measures:
                        seed += 1
                        env_seed += 1
                        alg_seed += 1

    def record(self, alg: str, env: str, params: Params, result: Result):
        pass
