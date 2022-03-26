from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(ABC):
    @abstractmethod
    def act(self, obs, hidden_states=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Outputs: actions, logits, hidden_states
        '''
        pass


class SimpleActorPolicy(nn.Module, Policy):
    def __init__(self, layers: List[int], optim_lr: float, device: torch.device = None) -> None:
        '''
        The first element of layers is the dimension of (flattened) observation.
        The last element of layers is the number of (discrete) actions. It will be the output shape.
        '''
        super(SimpleActorPolicy, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.layers = nn.ModuleList([nn.Linear(x, y)
                                     for x, y in zip(layers[:-1], layers[1:])])
        self.to(self.device)
        # self.optimizer = optim.Adam(self.parameters(), optim_lr)
        self.optimizer = optim.SGD(self.parameters(), optim_lr)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - 3.5) / 4.5  # normalize observations
        obs = obs.to(self.device)
        for layer in self.layers[:-1]:
            obs = F.relu(layer(obs))
        logits = self.layers[-1](obs)
        return logits

    def act(self, obs, hidden_states=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        params:
            obs: batch of tensor observations of shape B x obs.shape
        outputs:
            actions: selected actions of shape B
            logits: batch of action logits of shape B x num_actions
        '''
        logits = self.forward(obs.flatten(1))
        actions = torch.argmax(logits, dim=-1)
        return actions, logits


class ConvActorPolicy(nn.Module, Policy):
    def __init__(self, first_expansion_channels: int, mid_channels: int, kernel_sizes: List[int], optim_lr: float, one_hotify: bool = False, device: torch.device = None) -> None:
        super(ConvActorPolicy, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.first_conv = nn.Conv2d(
            in_channels=16 if one_hotify else 1, out_channels=first_expansion_channels, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(first_expansion_channels)
        self.kernel_sizes = kernel_sizes
        self.mid_convs = nn.ModuleList([nn.Conv2d(in_channels=first_expansion_channels,
                                       out_channels=mid_channels, kernel_size=ks,
                                        padding='same'
                                                  ) for ks in kernel_sizes])
        self.bn2 = nn.BatchNorm2d(len(kernel_sizes)*mid_channels)
        self.last_conv = nn.Conv2d(in_channels=len(
            kernel_sizes)*mid_channels, out_channels=1, kernel_size=3, padding='same')
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), optim_lr)
        self.one_hotify = one_hotify

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.one_hotify:
            obs = F.one_hot((obs+3).long(), 16).permute(0, -1, *list(range(1, len(obs.shape)))).float()
        else:
            obs = (obs-3.5)/4.5
            obs = obs.unsqueeze(1)  # add channel dimension

        obs = obs.to(self.device)

        obs = F.relu(self.bn1(self.first_conv(obs)))
        obs = F.relu(self.bn2(torch.cat([conv(obs)
                     for conv in self.mid_convs], dim=1)))
        logits = self.last_conv(obs).flatten(1)
        return logits

    def act(self, obs, hidden_states=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        params:
            obs: batch of tensor observations of shape B x obs.shape
        outputs:
            actions: selected actions of shape B
            logits: batch of action logits of shape B x num_actions
        '''
        logits = self.forward(obs)
        actions = torch.argmax(logits, dim=-1).detach()
        return actions, logits


# model = SimpleActorPolicy([36, 32, 32, 36], 1e-3)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# model = ConvActorPolicy(16, 4, [1, 2, 3, 4, 5], 1e-3)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# obs = torch.randn(64, 6, 6)
# actions, logits = p.act(obs)
# print(actions.shape, logits.shape)
# print(actions)
