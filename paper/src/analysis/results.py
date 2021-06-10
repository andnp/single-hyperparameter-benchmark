from functools import partial
import os
import sys
from typing import List, Sequence
import numpy as np
sys.path.append(os.getcwd())

import PyExpUtils.utils.path as Path
from PyExpUtils.utils.dict import get
from experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, BaseResult
from PyExpUtils.results.backends.h5 import H5Result
from multiprocessing.pool import Pool

def getCurveReducer(bestBy: str):
    if bestBy == 'auc':
        return np.nanmean

    if bestBy == 'end':
        return lambda m: np.nanmean(m[-int(m.shape[0] * .25):])

    raise NotImplementedError('Only now how to plot by "auc" or "end"')

def rename(alg: str, exp_path: str):
    return alg

def findExpPath(arr: Sequence[str], alg: str):
    for exp_path in arr:
        if f'{alg.lower()}.json' == Path.fileName(exp_path.lower()):
            return exp_path

    raise Exception(f'Expected to find exp_path for {alg}')

def whereParametersEqual(results, vals):
    def check(r):
        for key in vals:
            val = vals[key]

            if get(r.params, key, val) != val:
                return False

        return True

    return filter(check, results)

def loadDomain(all_exp_paths, result_file, algs, parallelize, domain):
    alg_results = {}
    for alg in algs:
        exp_path = findExpPath(all_exp_paths[domain], alg)

        exp = ExperimentModel.load(exp_path)
        results = loadResults(exp, result_file, cache=False)
        results = list(results)

        if parallelize:
            for r in results:
                r.load()

        alg_results[alg] = (exp, results)

    if parallelize:
        print("Loaded", domain)

    return alg_results

def loadAllResults(all_exp_paths, result_file, algs, parallelize=True):
    mapper = map
    if parallelize:
        pool = Pool()
        mapper = pool.map

    domains = list(all_exp_paths.keys())
    all_results = list(mapper(partial(loadDomain, all_exp_paths, result_file, algs, parallelize), domains))

    results = {}
    for i, domain in enumerate(domains):
        results[domain] = all_results[i]

    return results

def filterCutoff(results, path, cutoff):
    co = cutoff
    if 'CliffWorld' in path and cutoff > 0:
        co = int(cutoff / 10)

    return whereParametersEqual(results, { 'experiment': { 'cutoff': co} })

def smape(a, b):
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b)

    res = []
    for i in range(len(num)):
        if np.isclose(den[i], 0):
            res.append(0)

        else:
            res.append(num / den)

    return np.mean(res)

def mase(a, b):
    num = np.abs(a - b).mean()
    den = np.abs(b).mean()

    return num / den

def absoluteBest(all_results, domains, algs, cutoff, reducer):
    best_per_domain = {}
    for domain in domains:
        best_per_domain[domain] = -np.inf
        for alg in algs:
            _, results = all_results[domain][alg]
            results = filterCutoff(results, domain, cutoff)

            scores = [reducer(r.mean()) for r in results]

            m = np.max(scores)

            if m > best_per_domain[domain]:
                best_per_domain[domain] = m

    return best_per_domain

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

def probProfile(all_results, domains, algs, cutoff: float, reducer):
    scored_results = scoreResults(all_results, domains, algs, cutoff, reducer)

    def score(x, domain):
        cdfs = [np.sum(scored_results[domain][alg] < x) / len(scored_results[domain][alg]) for alg in algs]
        avg_cdf = np.mean(cdfs)

        return avg_cdf

    return score

class LimitedRunsResult(BaseResult):
    def __init__(self, r: H5Result, runs: List[int]):
        super().__init__(r.path, r.exp, r.idx)

        self._r = r
        self.run_indices = runs

    def load(self):
        lines = self._r.load()

        return np.array([lines[r] for r in self.run_indices])

    def mean(self):
        return self.load().mean(axis=0)

    def stderr(self):
        return self.load().std(axis=0) / np.sqrt(len(self.run_indices))

    def runs(self):
        return len(self.run_indices)

def limitRuns(results, runs: int):
    if runs == -1:
        return results

    indices = np.random.choice(range(250), size=runs, replace=True)
    return (LimitedRunsResult(r, indices) for r in results)
