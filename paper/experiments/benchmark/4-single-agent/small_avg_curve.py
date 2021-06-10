import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')

import numpy as np
from PyExpPlotting.learning_curves import lineplot
from PyExpPlotting.tools import findExperiments
from PyExpPlotting.matplot import save, setDefaultConference

from PyExpUtils.utils.path import up, fileName
from PyExpUtils.utils.dict import get
from PyExpUtils.results.results import loadResults, getBest
from analysis.results import findExpPath
from analysis.colors import colors
from experiment.tools import parseCmdLineArgs
from experiment import ExperimentModel

from multiprocessing.pool import Pool
from functools import partial

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

def smooth(lines, beta=0.9):
    def inner(lines):
        d = lines[0]
        for x in lines:
            d = beta * d + (1 - beta) * x
            yield d

    return np.array(list(inner(lines)))

def whereParametersEqual(results, vals):
    def check(r):
        for key in vals:
            val = vals[key]

            if get(r.params, key, val) != val:
                return False

        return True

    return filter(check, results)

ALG_ORDER = ['DQN', 'DeepQ']

def generatePlot(ax, exp_paths, result_file, alg, num_samples, cutoff, target):
    exp_path = findExpPath(exp_paths, alg)

    if 'CliffWorld' in exp_path and cutoff > 0:
        cutoff = cutoff / 10

    exp = ExperimentModel.load(exp_path)
    results = loadResults(exp, f'{result_file}.h5', cache=False)
    results = whereParametersEqual(results, {
        'experiment.cutoff': cutoff,
        'target_refresh': target,
    })

    best = getBest(results, prefer='big', reducer=np.nanmean)
    data = best.load()
    data = smooth(data)

    # set up a resampling thing
    # lets draw 30 samples and see where that takes us
    lines = []
    stderrs = []
    for s in range(30):
        samples = np.random.permutation(len(data))[0:num_samples]
        sample = [data[idx] for idx in samples]
        mean = np.nanmean(sample, axis=0)
        lineplot(ax, mean, None, { 'color': 'black', 'alpha_main': 0.25, 'width': 0.2, 'legend': False })
        lines.append(mean)

        stderr = np.nanstd(sample, axis=0, ddof=1) / np.sqrt(num_samples)
        stderrs.append(stderr)

    stat = np.nanmean(lines, axis=1)
    best_idx = np.argmax(stat)
    worst_idx = np.argmin(stat)

    lineplot(ax, lines[best_idx], stderrs[best_idx], { 'color': 'blue', 'alpha_main': 1, 'width': .5, 'legend': False })
    lineplot(ax, lines[worst_idx], stderrs[worst_idx], { 'color': 'red', 'alpha_main': 1, 'width': .5, 'legend': False })


def iterateDomains(exps, path, should_save, save_type, section, domain):
    np.random.seed(0)
    print(domain)
    for alg in ALG_ORDER:
        # for result_file in ['episodic_return']:
        for result_file in ['step_return', 'big_step_return']:
            for samples in [3, 10, 30]:
                for cutoff in [-1, 500, 1000]:
                    for target in [1, 8, 32]:
                        f, axes = plt.subplots(1)
                        generatePlot(axes, exps[domain], result_file, alg, samples, cutoff, target)

                        name = f'{alg}_{samples}_{cutoff}_{target}'

                        if should_save:
                            save(
                                save_path=f'{up(path)}/plots/{section}/{result_file}/{domain}',
                                plot_name=name,
                                save_type=save_type,
                                width=.33,
                                f=f,
                            )

                            plt.close()

                        else:
                            plt.show()
                            exit()


if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    section = fileName(path)

    pool = Pool()
    exps = findExperiments(key='{domain}', path=up(path))
    pool.map(partial(iterateDomains, exps, path, should_save, save_type, section), exps)
