# Cross-environment Hyperparameter Setting Benchmark

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

## Documentation
This library exposes one class for performing the SHB as well as a data-class for describing a single experiment.
The design philosophies of this library are:
1. Provide a minimal implementation (no opinionated agent/environment APIs, we leave that to RlGlue or OpenAI Gym).
2. Minimize statefulness in classes (if it can be recomputed instead of memorized, prefer that). This way the SHB experimental process can be easily broken into individual scripts for performing each stage of the procedure and heavily parallelized.
3. Prefer clarity to performance. The functions provided should be called no more than once each for an entire experiment and should take no longer than a few milliseconds to run. These functions **are not** designed to be called within a hot loop. We prefer readability so consumers of the library can alternatively elect to use their own implementation, using ours as reference.

### Order of operations
The SHB performs experiments in two stages: (1) hyperparameter selection and (2) evaluation of selected parameters.
The code is structured similarly, with a couple of additional bookkeeping stages.

Steps:
1. Register algorithm names and environment names, register hyperparameters to sweep, select desired number of runs.
2. Generate a list of experiments that need to be run, including the random seeds to use for both (alg, env, global), and the hyperparameters to test.
3. Perform the first stage of experiments by running the produced list of jobs. Record data (likely to disk).
4. Read data from disk and register it to the corresponding job from the SHB class.
5. Perform CDF scaling, averaging over environments, and maximizing over hyperparameters for each algorithm.
6. Generate a list of evaluation jobs to be run. Run them. Do what you want with the data, the SHB no longer needs to be involved.

### Registration
Registration of algorithms, environments, and hyperparameters should preferably be done when the `SHB` object is created.
We provide an alternative API for modifying these registrations post-hoc to match some procedural workflows.

**Registration at construction-time.** Environment registration is simple, we need only a list of environment names.
Algorithm registration is paired with hyperparameter registration, allowing different sets of hyperparameters to be swept for different algorithms.
In addition, optional environment-specific hyperparameters can be provided to the agent at this time (for example, perhaps neural network architecture is environment-specific).
These environment-specific hyperparameters *cannot* contain swept parameters.
```python
dqn_params = {
    # parameters can contain arbitrarily nested objects
    'optimizer': {
        # lists *at the lowest level* specify a sweep over a parameter
        # i.e. this line indicates 3 different stepsizes should be tested
        'stepsize': [0.1, 0.01, 0.001],
    },
    # for all values of previous parameters,
    # we should also try these 3 value of epsilon
    'epsilon': [0.05, 0.1, 0.15]
}
# ultimately, this object specified 9 hyperparameter settings (all combinations).

env_specific = {
    # The first level of this object should contain keys of registered environments
    # these are optional, and if a particular environment is not mentioned here a default
    # {} empty object will instead be used.
    'MountainCar': {
        'network': {
            # notice this list is *not* at the lowest level (there are child objects)
            # this will *not* be swept and so indicates a 2-layer network (not 2 different 1 layer networks)
            'layers': [
                { 'units': 64, 'activation': 'relu' },
                { 'units': 32, 'activation': 'relu' },
            ]
        },
    },
    'CartPole': {
        'network': {
            'layers': [
                { 'units': 64, 'activation': 'relu' },
                { 'units': 64, 'activation': 'relu' },
            ]
        },
    },
}

shb = SHB(
    ...,
    algs=[
        # each alg registration is a tuple containing:
        # Name, sweepable parameters, environment-specific parameters
        ('DQN', dqn_params, env_specific),
        ('DeepQ', deepq_params, env_specific)
    ],
    envs=['MountainCar', 'CartPole', 'Acrobot'],
    ...,
)
```

### Generating Selection Jobs
After registration, the `shb` object can now generate a list of jobs.
A job is an object containing some meta-data necessary to run a single run of a single experiment:
```python
# a global random seed
# At bare minimum, this must be specified for all rngs
job.seed

# random seeds specific to the agent and environment
# if possible, setting specific rngs for the agent and environment can help reduce variance
job.alg_seed
job.env_seed

# The index for a given hyperparameter setting
# can use PyExpUtils to obtain specific combination
job.idx

# Which run number is this for this specific env, alg, hyperparam setting combo?
job.run

# a single hyperparameter combination.
# sweepable lists have been turned into a single element (List[T] -> T)
job.params

# is this a 'selection' or 'evaluation' job (e.g. which stage?)
job.type

# a shortcut utility function provided by the `shb` object to help
# organize / store data.
job.record(data: float)
```

To generate the complete list of these jobs, one can call
```python
for job in shb.iterateModelSelectionJobs():
    # do something with the job
```

It is possible this function will need to be called twice.
Once to generate and schedule the jobs.
A second time to record the outcome of those jobs (i.e. this decoupling allows jobs to be run asynchronously from two independent script invocations).
A consistent ordering of jobs is guaranteed and sufficient meta-data is provided to uniquely identify each job minimally (e.g. with `idx` and `run`).

### Analyzing model selection results
Once data has been recorded into an `shb` object, then the `shb` can perform the scaling and analysis; providing selected hypers as an artifact.
```python
# note this function can be expensive (a few seconds for very large datasets)
# it is best to save the results to disk and avoid calling in a hot loop
params = shb.pickParameters()
print(params) # =>

# an example object produced
{
    # each set of params are indexed by the algorithm name
    'DQN': {
        # object structure is respected and maintained
        # sweepable parameters now include a single element of the list
        'optimizer': { 'stepsize': 0.1 },
        'epsilon': 0.05,
    },
    'DeepQ': {
        'optimizer': { 'stepsize': 0.001 },
        'epsilon': 0.05,
    },
}
```

### Generating evaluation jobs
To complete the final stage of the SHB requires rerunning the selected hypers for many runs.
This can be done by calling `iterateEvaluationJobs`, which takes as argument the specific hypers selected by the SHB.
Due to the design philosophy of minimizing statefulness of the `shb` object, we do not store the results of `pickParameters()`
and instead require these results to be provided back to us.
The `job` api is the same as that provided by `iterateModelSelectionJobs`.

```python
for job in shb.iterateEvaluationJobs(params):
    # do something with the job
```
