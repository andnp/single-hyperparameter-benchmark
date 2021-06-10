import os
import sys
from typing import List
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')

import numpy as np
from PyExpPlotting.tools import findExperiments
from PyExpPlotting.matplot import save, setDefaultConference

from PyExpUtils.utils.path import up, fileName
from PyExpUtils.results.results import getBest, whereParametersEqual
from analysis.results import getCurveReducer, filterCutoff, loadAllResults
from analysis.colors import colors
from experiment.tools import parseCmdLineArgs

from PyExpUtils.results.voting import scoreMetaparameters, confidenceRanking, buildBallot, small

from multiprocessing.pool import Pool
from functools import partial
from itertools import product

np.random.seed(0)

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
# ALG_ORDER = ['DQN']
ALG_ORDER = ['DQN', 'DeepQ', 'QLearning', 'ESARSA']
RUNS = [3, 10, 50, 100]

BOOTSTRAPS = 100

def scoreResults(all_results, domains, algs, cutoff: float, reducer):
    all_scores = {}
    for domain in domains:
        domain_scores = {}
        for alg in algs:
            _, results = all_results[domain][alg]
            results = filterCutoff(results, domain, cutoff)

            scores = []
            for r in results:
                scores += [reducer(line) for line in r.load()]

            domain_scores[alg] = np.array(scores)

        all_scores[domain] = domain_scores

    return all_scores

def probProfile(scored_results, algs):
    def score(x, domain):
        cdfs = [np.sum(scored_results[domain][alg] < x) / len(scored_results[domain][alg]) for alg in algs]
        avg_cdf = np.mean(cdfs)

        return avg_cdf

    return score

def getVoteParam(all_results, domains: List[str], alg: str, cutoff: float, runs: int, stderrs: int, reducer):
    ballots = []
    for domain in domains:
        r_idx = np.random.choice(range(250), replace=True, size=runs)
        exp, results = all_results[domain][alg]
        results = filterCutoff(results, domain, cutoff)
        results = (r.reducer(lambda x: x[r_idx]) for r in results)

        scores = scoreMetaparameters(results, exp, reducer)
        scores = list(filter(lambda s: not np.isnan(s.score), scores))
        ranks = confidenceRanking(scores, stderrs=stderrs, prefer='big')
        ballot = buildBallot(ranks)

        ballots.append(ballot)

    best = small(ballots)

    return best

def evaluate(all_results, vote_param, domains, alg, cutoff, reducer, scoreFunc, bs_scores):
    d_scores = []
    for domain in domains:
        # ----------------------------------
        # now look at this particular domain
        # ----------------------------------

        # then we need to plot this alg with those meta-parameters
        exp, results = all_results[domain][alg]
        params = exp.getPermutation(int(vote_param))['metaParameters']

        # load the results and clean
        results = filterCutoff(results, domain, cutoff)
        results = list(results)

        # get the best according to vote
        vote_results = whereParametersEqual(results, params)
        best = getBest(vote_results, prefer='big', reducer=reducer)
        vote_scores = (reducer(line) for line in best.load())
        vote_scores = [scoreFunc(x, domain) for x in vote_scores]

        score = np.mean(vote_scores)

        scores = bs_scores.get(domain, [])
        scores.append(score)
        bs_scores[domain] = scores
        d_scores.append(score)

    return d_scores

def generatePlot(ax, all_results, domains, cutoff, stderrs, runs):
    reducer = getCurveReducer('auc')
    labels = []
    ticks = []

    scored_results = scoreResults(all_results, domains, ALG_ORDER, cutoff, reducer)
    scoreFunc = probProfile(scored_results, ALG_ORDER)

    for a_idx, alg in enumerate(ALG_ORDER):
        # ----------------------------------------------------------
        # first we need to vote on meta-paramters across all domains
        # ----------------------------------------------------------
        bs_scores = {}
        avg_scores = []
        for bs in range(BOOTSTRAPS):
            print(alg, cutoff, stderrs, runs, bs)
            vote_param = getVoteParam(all_results, domains, alg, cutoff, runs, stderrs, reducer)

            scores = evaluate(all_results, vote_param, domains, alg, cutoff, reducer, scoreFunc, bs_scores)
            avg_scores.append(np.mean(scores))

        for i, domain in enumerate(domains):
            scores = bs_scores[domain]
            score = np.mean(scores)
            ci = np.std(scores)

            offset = a_idx + i * (len(ALG_ORDER) + 1)

            if i == 0:
                bar_label = alg
            else:
                bar_label = None

            ax.bar(offset, score, yerr=ci, color=colors[alg], width=1, alpha=0.7, label=bar_label)
            scores.append(score)

            ticks.append(offset)
            if a_idx == 0:
                labels.append(domain)
            else:
                labels.append(None)

        avg_scores = np.array(avg_scores)
        score = np.mean(avg_scores)
        ax.hlines(score, -0.5, (len(ALG_ORDER) + 1) * len(domains) - 1.5, color=colors[alg])
        ci = np.std(avg_scores)
        ax.axhspan(score - ci, ci + score, alpha=0.3, color=colors[alg])

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35)

def iterateCutoffsRuns(exps, path, should_save, save_type, section, d):
    result_file = 'step_return'
    cutoff, stderrs, runs = d
    f, axes = plt.subplots(1)

    all_results = loadAllResults(exps, f'{result_file}.h5', ALG_ORDER, parallelize=False)

    generatePlot(axes, all_results, exps.keys(), cutoff, stderrs, runs)

    plt.legend()

    name = f'{runs}_{stderrs}_{cutoff}_no-rm'

    if should_save:
        save(
            save_path=f'{up(path)}/plots/{section}/{result_file}',
            plot_name=name,
            save_type=save_type,
            width=1,
            height_ratio=0.5,
            f=f,
        )

        plt.close()

    else:
        plt.show()
        exit()

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    section = fileName(path)
    exps = findExperiments(key='{domain}', path=up(path))

    # del exps['JitterAcrobot']

    it = product([500, 1000], [0, 1, 2], RUNS)
    if should_save:
        pool = Pool(4)
        pool.map(partial(iterateCutoffsRuns, exps, path, should_save, save_type, section), it)

    else:
        list(map(partial(iterateCutoffsRuns, exps, path, should_save, save_type, section), it))
