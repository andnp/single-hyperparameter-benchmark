import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')

import numpy as np
from PyExpPlotting.distributions import plot
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

def whereParametersEqual(results, vals):
    def check(r):
        for key in vals:
            val = vals[key]

            if get(r.params, key, val) != val:
                return False

        return True

    return filter(check, results)

ALG_ORDER = ['DQN', 'DeepQ']

def generatePlot(ax, exp_paths, result_file, alg, cutoff, target):
    exp_path = findExpPath(exp_paths, alg)

    if 'CliffWorld' in exp_path and cutoff > 0:
        cutoff = cutoff / 10

    exp = ExperimentModel.load(exp_path)
    results = loadResults(exp, f'{result_file}.h5')
    results = whereParametersEqual(results, {
        'experiment.cutoff': cutoff,
        'target_refresh': target,
    })

    best = getBest(results, prefer='big', reducer=np.nanmean)
    plot(best, ax, np.nanmean, {
        'color': colors[alg],
        'label': alg,
        'hist': False,
    })

    data = np.nanmean(best.load(), axis=1)
    mean = np.mean(data)
    median = np.percentile(data, 50, interpolation='nearest')

    ylo, yhi = ax.get_ylim()
    ax.vlines(mean, 0, yhi, color=colors[alg])
    ax.vlines(median, 0, yhi, linestyles='dotted', color=colors[alg])



def iterateDomains(exps, path, should_save, save_type, section, domain):
    np.random.seed(0)
    print(domain)
    for alg in ALG_ORDER:
        for result_file in ['step_return', 'episodic_return', 'big_episodic_return', 'big_step_return']:
            for cutoff in [-1, 500, 1000]:
                for target in [1, 8, 32]:
                    f, axes = plt.subplots(1)
                    generatePlot(axes, exps[domain], result_file, alg, cutoff, target)

                    name = f'{alg}_{cutoff}_{target}'

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
