import sys
import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

class SelectionModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path, save_key='analysis_results/{name}/{dataset}/{procedure}/{params}')

        self.dataset = d['dataset']
        self.procedure = d['procedure']
        self.bootstraps = d.get('bootstraps', 1)
        self.parallelize = d.get('parallelize', False)

        d['name'] = self.getExperimentName().replace('analysis/', '')

def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = SelectionModel(d, path)
    return exp
