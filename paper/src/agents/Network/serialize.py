from typing import Dict, Iterable
import torch
import torch.optim as optim

def deserializeOptimizer(learnables: Iterable[torch.Tensor], params: Dict):
    name = params['name']
    alpha = params['alpha']

    if name == 'ADAM':
        b1 = params['beta1']
        b2 = params['beta2']

        return optim.Adam(learnables, lr=alpha, betas=(b1, b2))

    if name == 'RMSProp':
        b = params['beta']

        return optim.RMSprop(learnables, lr=alpha, alpha=b)

    if name == 'SGD':
        return optim.SGD(learnables, lr=alpha)

    raise NotImplementedError()
