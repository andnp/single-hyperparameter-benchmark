import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')

import numpy as np
from PyExpPlotting.tools import findExperiments
from PyExpPlotting.matplot import save, setDefaultConference

from PyExpUtils.utils.path import up, fileName
from PyExpUtils.results.results import getBest, whereParametersEqual
from analysis.results import getCurveReducer, loadAllResults, filterCutoff
from analysis.colors import colors
from experiment.tools import parseCmdLineArgs

from PyExpUtils.results.voting import scoreMetaparameters, confidenceRanking, buildBallot, small

np.random.seed(0)

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
ALG_ORDER = ['DQN', 'DeepQ', 'ESARSA', 'QLearning']

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

def getVoteParam(all_results, domains, alg, cutoff: float, reducer):
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

    scored_results = scoreResults(all_results, domains, ALG_ORDER, cutoff, reducer)
    scoreFunc = probProfile(scored_results, ALG_ORDER)

    for a_idx, alg in enumerate(ALG_ORDER):
        print(alg)
        # ----------------------------------------------------------
        # first we need to vote on meta-paramters across all domains
        # ----------------------------------------------------------

        # grab the paths for every domain for a given algorithm
        vote_param = getVoteParam(all_results, domains, alg, cutoff, reducer)

        scores = []
        for i, domain in enumerate(domains):
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
