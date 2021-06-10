from typing import Dict
import numpy as np
import torch
import torch.nn.functional as f
import copy

from PyExpUtils.utils.Collector import Collector
from agents.BaseAgent import BaseAgent
from utils.random import getControlledSeed
from utils.torch import device, getBatchColumns, Batch, cloneNetworkWeights
from agents.Network.serialize import deserializeOptimizer
from utils.ReplayBuffer import ReplayBuffer
from representations.networks import getNetwork

class DQN(BaseAgent):
    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        super().__init__(features, actions, params, seed, collector)

        # set up initialization of the value function network
        # and target network
        init_controlled = params.get('init_controlled', True)
        init_seed = getControlledSeed(init_controlled, seed)
        self.value_net = getNetwork(features, actions, params['representation'], init_seed)
        self.target_net = copy.deepcopy(self.value_net)
        self.target_refresh = params.get('target_refresh', 1)

        # set up the optimizer
        self.optimizer_params = params['optimizer']
        self.optimizer_params['alpha'] = self.alpha
        self.optimizer = deserializeOptimizer(self.value_net.parameters(), self.optimizer_params)

        # set up the experience replay buffer
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']
        buffer_controlled = params.get('buffer_controlled', True)
        buffer_seed = getControlledSeed(buffer_controlled, seed)
        self.buffer = ReplayBuffer(self.buffer_size, buffer_seed)

        # figure out which loss function to use
        if params.get('loss', 'huber') == 'huber':
            self.loss_func = f.smooth_l1_loss

        else:
            self.loss_func = f.mse_loss

        self.steps = 0

        self.initializeTargetNet()

    def initializeTargetNet(self):
        # if we aren't using target nets, then save some compute
        if self.target_refresh > 1:
            self.target_net = copy.deepcopy(self.value_net)
            cloneNetworkWeights(self.value_net, self.target_net)
        else:
            self.target_net = self.value_net

    def values(self, x):
        self.value_net.eval()
        x = torch.tensor([x], dtype=torch.float32, device=device)
        qs = self.value_net(x)[0].detach().cpu().numpy()
        self.value_net.train()
        return qs[0]

    def forward(self, batch):
        q, = self.value_net(batch.states)
        qp, = self.target_net(batch.nterm_next_states)
        return {
            "value": q,
            "next_value": qp,
        }

    def bootstrap(self, batch: Batch, gamma: float, next_values: torch.Tensor):
        q_sp_ap = torch.zeros(batch.size, 1, device=device)
        if batch.nterm_next_states.shape[0] > 0:
            q_sp_ap[batch.is_non_terminals] = next_values.max(1).values.unsqueeze(1)

        target = batch.rewards + gamma * q_sp_ap

        return target, {
            "q_sp_ap": q_sp_ap
        }

    def updateNetwork(self, batch: Batch, gamma: float, predictions: Dict[str, torch.Tensor]):
        q_s = predictions['value']
        q_s_a = q_s.gather(1, batch.actions)

        target, _ = self.bootstrap(batch, gamma, predictions['next_value'])
        loss = self.loss_func(target.detach(), q_s_a)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        delta = loss.mean().detach().cpu().numpy()
        return np.sqrt(delta)

    def update(self, s, a, sp, r, gamma):
        if gamma == 0:
            sp = None

        self.buffer.add((s, a, sp, r, gamma))
        self.steps += 1

        if self.steps % self.target_refresh == 0 and self.target_refresh > 1:
            cloneNetworkWeights(self.value_net, self.target_net)

        if len(self.buffer) > self.batch_size + 1:
            samples, idcs = self.buffer.sample(self.batch_size)
            batch = getBatchColumns(samples)
            predictions = self.forward(batch)
            tde = self.updateNetwork(batch, gamma, predictions)

            self.buffer.update_priorities(idcs, tde)
