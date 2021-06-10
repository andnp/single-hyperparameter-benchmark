import pickle
from experiment.SelectionModel import SelectionModel
from PyExpUtils.results.results import BaseResult
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.Collector import Collector

class AnalysisResult(BaseResult):
    def __init__(self, path: str, exp: SelectionModel, i: int) -> None:
        self.path = path
        self.exp = exp
        self.idx = i
        self.params = exp.getPermutation(i)['metaParameters']

        with open(self.path, 'rb') as f:
            self.collector: Collector = pickle.load(f)

    def getAlgs(self, domain: str, metric: str):
        out = []
        for key in self.collector.run_data:
            if domain in key and metric in key:
                out.append(self.collector.run_data[key])

        return out

    def getDomains(self, alg: str, metric: str):
        out = []
        for key in self.collector.run_data:
            if alg in key and metric in key:
                out.append(self.collector.run_data[key])

        return out

    def getScore(self, alg: str, domain: str, metric: str):
        return self.collector.run_data[f'{domain}-{alg}-{metric}']

def loadResults(exp: SelectionModel, fn: str = 'collected.pkl'):
    for i, path in enumerate(listResultsPaths(exp)):
        data_path = f'{path}/{fn}'
        yield AnalysisResult(data_path, exp, i)
