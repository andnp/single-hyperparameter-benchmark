# Reproduction Code

## Setting up repo
**This codebase only works with python 3.6 and above.**

Packages are stored in a `requirements.txt` file.
To install:
```
pip install -r requirements.txt
```

On machines that you do not have root access to (like compute canada machines), you will need to install in the user directory.
You can do this with:
```
pip install --user -r requirements.txt
```
Or you need to set up a virtual environment:
```
virtualenv -p python3 env
```

---
## Running the code
**Everything should be run from the paper/ directory of the repo!**

Let's say you want to generate a learning curve for a single run of an algorithm.
```
python src/main.py <path/to/experiment.json> <parameter_setting_idx>
```
It isn't super easy to know which `parameter_setting_idx` to use.
It is more simple to make an experiment description `.json` that only contains one possible parameter permutation (i.e. has no arrays in it).
This will save the results in the results folder as specified above.

For example to generate a single run of a benchmark experiment,
```
python src/main.py experiments/benchmark/Acrobot/DeepQ.json 0
```
To run this over all indices requires some simple scripting, for instance using gnu-parallel
```
parallel python src/main.py experiments/benchmark/Acrobot/DeepQ.json ::: 0..1000
```

---
## Dependencies
This template repo depends on a few other shared libraries to make code-splitting and sharing a little easier (for me).
The documentation and source code can be found at the following links.
* [RLGlue](https://github.com/andnp/rlglue) - my own minimal implementation of RLGlue
* [PyExpUtils](https://github.com/andnp/pyexputils) - a library containing the experiment running framework
* [PyFixedReps](https://github.com/andnp/pyfixedreps) - a few fixed representation algorithms implemented in python (e.g. tile-coding, rbfs, etc.)


---
## Organization Patterns

### Experiments
All experiments are described as completely as possible within static data files.
I choose to use `.json` files for human readability and because I am most comfortable with them.
These are stored in the `experiments` folder, usually in a subdirectory with a short name for the experiment being run (e.g. `experiments/idealH` would specify an experiment that tests the effects of using h*).

Experiment `.json` files look something like:
```jsonc
{
    "agent": "gtd2", // <-- name of your agent. these names are defined in agents/registry.py
    "problem": "randomwalk", // <-- name of the problem you're solving. these are defined in problems/registry.py
    "metaParameters": { // <-- a dictionary containing all of the meta-parameters for this particular algorithm
        "alpha": [1, 0.5, 0.25], // <-- sweep over these 3 values of alpha
        "beta": 1.0, // <-- don't sweep over beta, always use 1.0
        "use_ideal_h": true,
        "lambda": [0.0, 0.1]
    }
}
```

### Problems
I define a **problem** as a combination of:
1) environment
2) representation
3) target/behavior policies
4) number of steps
5) gamma
6) starting conditions for the agent (like in Baird's)

### results
The results are saved in a path that is defined by the experiment definition used.
The configuration for the results is specified in `config.json`.
Using the current `config.json` yields results paths that look like:
```
<base_path>/results/<experiment short name>/<agent name>/<parameter values>/errors_summary.npy
```
Where `<base_path>` is defined when you run an experiment.

### src
This is where the source code is stored.
The only `.py` files it contains are "top-level" scripts that actually run an experiment.
No utility files or shared logic at the top-level.

**agents:** contains each of the agents.
Preferably, these would be one agent per file.

**analysis:** contains shared utility code for analysing the results.
This *does not* contain scripts for analysing results, only shared logic (e.g. plotting code or results filtering).

**environments:** contains minimal implementations of just the environment dynamics.

**utils:** various utility code snippets for doing things like manipulating file paths or getting the last element of an array.
These are just reusable code chunks that have no other clear home.
I try to sort them into files that roughly name how/when they will be used (e.g. things that manipulate files paths goes in `paths.py`, things that manipulate arrays goes in `arrays.py`, etc.).
