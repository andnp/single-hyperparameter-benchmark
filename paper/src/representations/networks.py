from typing import Any, Dict
from utils.torch import OneLayerRelu, TwoLayerRelu, TwoLayerReluBN

def getNetwork(inputs: int, outputs: int, params: Dict[str, Any], seed: int):
    name = params['type']

    if name == 'TwoLayerReluBN':
        hidden = params['hidden']
        return TwoLayerReluBN(inputs, outputs, hidden, seed)

    if name == 'TwoLayerRelu':
        hidden = params['hidden']
        return TwoLayerRelu(inputs, outputs, hidden, seed)

    if name == 'OneLayerRelu':
        hidden = params['hidden']
        return OneLayerRelu(inputs, outputs, hidden, seed)

    raise NotImplementedError()
