# Single Hyperparameter Benchmark

## Organization
**shb/**: We include a small library to (a) demonstrate the ideas presented in the paper and (b) assist in utilizing the benchmark in future experiments.
We include documentation and example usage both in the code, in `tests/`, and below in this README.

**paper/**: We include the full (research) code repository used to generate the results presented in the paper for reproducibility.
Relevant documentation for running this code can be found in [paper/README.md](paper/README.md).
We note that, due to the "research" nature of the code, several extraneous scripts, abstractions, and unnecessary generality can be found in the reproduction code; documenting the journey taken to arrive at the finally proposed benchmark.

## Example Usage

```python
from SHB import SHB

shb = SHB(
    # how many independent runs per parameter setting
    # for selecting the SHB parameter
    # per environment
    # [default=3]
    selection_runs=3,
    # how many runs to use to evaluate the selected param
    # per environment
    # [default=250]
    eval_runs=250,
    # Whether to use a variance reduction method across environments
    # only useful if environments share some amount of variance
    # adds a small amount of bias (negligible as selection_runs gets larger)
    # [default=False]
    repeated_measures=False,
    # List of tuples specifying an algorithm, a set of hyperparameters to sweep
    # and an optional mapping from environment name -> (non-sweepable) parameters
    # If these are not specified here, they need to be specified by calling
    # `shb.registerAlg(...)`
    # [default=None]
    algs=[
        ('DQN', {
            'optimizer': { 'stepsize': [0.1, 0.01, 0.001]},
            'epsilon': [0.05, 0.1, 0.15],
        }, None),
        ('DeepQ', {
            'optimizer': { 'stepsize': [0.1, 0.01, 0.001]},
            'epsilon': [0.05, 0.1, 0.15],
        }, None),
    ],
    # List of environment names to test on
    # If not specified here, needs to be registered
    # `shb.registerEnvPool(...)`
    # [default=None]
    envs=['Cartpole', 'MountainCar', 'Acrobot'],
)

for job in shb.iterateModelSelectionJobs():
    alg = constructAlgFromName(job.alg)
    env = constructEnvFromName(job.env)

    # control some seeds
    np.random.seed(job.seed)
    alg.seed(job.alg_seed)
    env.seed(job.env_seed)

    # set the hypers
    alg.setParameters(job.params)

    # run the experiment (left as an exercise to the reader)
    result = runExperiment(alg, env)

    # save results
    job.record(result)

# do the scaling and maximizing
# gives back a dict like:
# shb_params = {
#   'DQN': {
#     'optimizer': { 'stepsize': 0.001 },
#     'epsilon': 0.1,
#   },
#   'DeepQ': {
#     'optimizer': { 'stepsize': 0.1 },
#     'epsilon': 0.05,
#   },
# }
shb_params = shb.pickParameters()

for job in shb.iterateEvaluationJobs(shb_params):
    alg = constructAlgFromName(job.alg)
    env = constructEnvFromName(job.env)

    # control some seeds
    np.random.seed(job.seed)
    alg.seed(job.alg_seed)
    env.seed(job.env_seed)

    # set the hypers
    alg.setParameters(job.params)

    # run the experiment (left as an exercise to the reader)
    result = runExperiment(alg, env)

    # TODO: do something with these results
```
