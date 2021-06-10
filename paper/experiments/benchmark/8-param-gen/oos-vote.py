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
from analysis.results import getCurveReducer, filterCutoff, mase, loadAllResults
from analysis.colors import colors
from experiment.tools import parseCmdLineArgs

from PyExpUtils.results.voting import scoreMetaparameters, confidenceRanking, buildBallot, small

np.random.seed(0)

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
ALG_ORDER = ['DQN', 'DeepQ']

def getVoteParam(all_results, domains: List[str], alg: str, cutoff: float, reducer):
    ballots = []
    for domain in domains:
        exp, results = all_results[domain][alg]
        results = filterCutoff(results, domain, cutoff)

        scores = scoreMetaparameters(results, exp, reducer)
        scores = list(filter(lambda s: not np.isnan(s.score), scores))
        ranks = confidenceRanking(scores, stderrs=1, prefer='big')
        ballot = buildBallot(ranks)

        ballots.append(ballot)

    best = small(ballots)

    return best

def generatePlot(ax, all_results, domains, cutoff):
    reducer = getCurveReducer('auc')

    labels = []
    ticks = []
    for a_idx, alg in enumerate(ALG_ORDER):
        scores = []

        all_vote_param = getVoteParam(all_results, domains, alg, cutoff, reducer)

        for i, domain in enumerate(domains):
            print(domain)

            # ----------------------------------------------------------
            # first we need to vote on meta-paramters across all domains
            # ----------------------------------------------------------

            vote_domains = [d for d in domains if d != domain]
            vote_param = getVoteParam(all_results, vote_domains, alg, cutoff, reducer)

            # ----------------------------------
            # now look at this particular domain
            # ----------------------------------

            exp, results = all_results[domain][alg]
            params = exp.getPermutation(int(vote_param))['metaParameters']

            # load the results and clean
            results = filterCutoff(results, domain, cutoff)
            results = list(results)

            # get the best according to oos vote
            vote_results = whereParametersEqual(results, params)
            best = getBest(vote_results, prefer='big', reducer=reducer)
            vote_scores = np.array([reducer(line) for line in best.load()])

            # get the best according to is vote
            all_vote_params = exp.getPermutation(int(all_vote_param))['metaParameters']
            vote_results = whereParametersEqual(results, all_vote_params)
            best = getBest(vote_results, prefer='big', reducer=reducer)
            best_scores = reducer(best.mean())

            score = mase(vote_scores, best_scores)

            offset = a_idx + i * (len(ALG_ORDER) + 1)

            if i == 0:
                bar_label = alg
            else:
                bar_label = None

            ax.bar(offset, score, color=colors[alg], width=1, alpha=0.7, label=bar_label)
            scores.append(score)

            ticks.append(offset)
            if a_idx == 0:
                labels.append(domain)
            else:
                labels.append(None)

        ax.hlines(np.mean(scores), -0.5, (len(ALG_ORDER) + 1) * len(domains) - 1.5, color=colors[alg])

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35)

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    section = fileName(path)
    exps = findExperiments(key='{domain}', path=up(path))

    del exps['JitterAcrobot']

    np.random.seed(0)
    for result_file in ['step_return']:
        all_results = loadAllResults(exps, f'{result_file}.h5', ALG_ORDER)
        for cutoff in [500, 1000, -1]:
            f, axes = plt.subplots(1)
            generatePlot(axes, all_results, exps.keys(), cutoff)

            plt.legend()

            name = f'{cutoff}'

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
