from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Policy(ABC):
    @abstractmethod
    def act(self, obs, hidden_states=None):
        pass

class SimpleActorPolicy(nn.Module, Policy):
    def __init__(self, layers: List[int], optim_lr:float) -> None:
        '''
        The first element of layers is the dimension of (flattened) observation.
        The last element of layers is the number of (discrete) actions. It will be the output shape.
        '''
        super(SimpleActorPolicy, self).__init__()
        self.layers = [nn.Linear(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.optimizer = optim.Adam(self.parameters(), optim_lr)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            obs = F.relu(layer(obs))
        logits = self.layers[-1](obs)
        return logits

    def act(self, obs, hidden_states=None):
        logits = self.forward(obs.flatten(1))
        actions = torch.argmax(logits, dim=-1)
        return actions, logits
