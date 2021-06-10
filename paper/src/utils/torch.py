from typing import List, Optional
import torch
import torch.nn as nn
from collections import namedtuple

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

Batch = namedtuple(
    'batch',
    'states, nterm_next_states, actions, rewards, is_terminals, is_non_terminals, size'
)

def getBatchColumns(samples):
    s, a, sp, r, gamma = list(zip(*samples))
    states = torch.tensor(s, dtype=torch.float32, device=device)
    actions = torch.tensor(a, device=device).unsqueeze(1)
    rewards = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
    gamma = torch.tensor(gamma, device=device)

    is_terminal = gamma == 0

    sps = [x for x in sp if x is not None]
    if len(sps) > 0:

        non_final_next_states = torch.tensor(sps, dtype=torch.float32, device=device)
    else:
        non_final_next_states = torch.zeros((0, states.shape[1]))

    non_term = torch.logical_not(is_terminal).to(device)

    return Batch(states, non_final_next_states, actions, rewards, is_terminal, non_term, len(samples))

def toNP(maybeTensor):
    if type(maybeTensor) == torch.Tensor:
        return maybeTensor.cpu()

    return maybeTensor

def addGradients_(net1, net2):
    for (param1, param2) in zip(net1.parameters(), net2.parameters()):
        if param1.grad is not None and param2.grad is not None:
            param1.grad.add_(param2.grad)

def excludeParameters(param_list, exclude):
    exclude = [id(p) for layer in exclude for p in list(layer.parameters())]

    for param in param_list:
        if id(param) not in exclude:
            yield param

def getAllGrads(net):
    return [p.grad for p in net.parameters()]

# assumes we always use relu
def initializeParameters(m: nn.Module):
    if type(m) == nn.Linear:
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(m.weight, gain)

# ----------------------
# Typical Network Models
# ----------------------

# A simple base class that allows modifying the network with additional heads
# and controlling whether those heads will pass back gradients to higher layers
class BaseNetwork(nn.Module):
    features: int
    output: nn.Linear

    def __init__(self, inputs: int, outputs: int, seed: int):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.seed = seed

        self.model = nn.Sequential()
        self.output_layers: List[nn.Module] = []
        self.output_grads: List[bool] = []

    def addOutput(self, outputs: int, grad: bool = True, bias: bool = True, initial_value: Optional[float] = None):
        layer = nn.Linear(self.features, outputs, bias=bias)

        if initial_value is None:
            nn.init.xavier_uniform_(layer.weight)

        else:
            nn.init.constant_(layer.weight, initial_value)

        if bias:
            nn.init.zeros_(layer.bias)

        self.output_layers.append(layer)
        self.output_grads.append(grad)

        num = len(self.output_layers)
        self.add_module(f'output-{num}', layer)

        return layer

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        outs = []
        for layer, grad in zip(self.output_layers, self.output_grads):
            if grad:
                outs.append(layer(x))
            else:
                outs.append(layer(x.detach()))

        return outs

# The typical 2-layer network that I use for many of the small control domains
class TwoLayerRelu(BaseNetwork):
    def __init__(self, inputs: int, outputs: int, layer_size: int, seed: int):
        super().__init__(inputs, outputs, seed)

        # first layer
        self.model.add_module('layer-0-weights', nn.Linear(inputs, layer_size))
        self.model.add_module('layer-0-activation', nn.ReLU())

        # second layer
        self.model.add_module('layer-1-weights', nn.Linear(layer_size, layer_size))
        self.model.add_module('layer-1-activation', nn.ReLU())

        # output
        self.output = nn.Linear(layer_size, outputs)

        # register the output and mark it as impacting higher layer gradients
        self.output_layers.append(self.output)
        self.output_grads.append(True)

        self.model.apply(initializeParameters)

class OneLayerRelu(BaseNetwork):
    def __init__(self, inputs: int, outputs: int, layer_size: int, seed: int):
        super().__init__(inputs, outputs, seed)

        # first layer
        self.model.add_module('layer-0-weights', nn.Linear(inputs, layer_size))
        self.model.add_module('layer-0-activation', nn.ReLU())

        # output
        self.output = nn.Linear(layer_size, outputs)

        # register the output and mark it as impacting higher layer gradients
        self.output_layers.append(self.output)
        self.output_grads.append(True)

        self.model.apply(initializeParameters)

class TwoLayerReluBN(BaseNetwork):
    def __init__(self, inputs: int, outputs: int, layer_size: int, seed: int):
        super().__init__(inputs, outputs, seed)

        # first layer
        self.model.add_module('layer-0-weights', nn.Linear(inputs, layer_size))
        self.model.add_module('layer-0-bn', nn.BatchNorm1d(layer_size))
        self.model.add_module('layer-0-activation', nn.ReLU())

        # second layer
        self.model.add_module('layer-1-weights', nn.Linear(layer_size, layer_size))
        self.model.add_module('layer-1-bn', nn.BatchNorm1d(layer_size))
        self.model.add_module('layer-1-activation', nn.ReLU())

        # output
        self.output = nn.Linear(layer_size, outputs)

        # register the output and mark it as impacting higher layer gradients
        self.output_layers.append(self.output)
        self.output_grads.append(True)

        self.model.apply(initializeParameters)

def cloneNetworkWeights(fromNet: BaseNetwork, toNet: BaseNetwork):
    toNet.load_state_dict(fromNet.state_dict())
