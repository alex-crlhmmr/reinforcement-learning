from abc import ABC, abstractmethod
from torch.distributions import Distribution, Categorical, Normal
from network_utils import build_mlp
from typing import Tuple
import torch.nn as nn
import torch

class BaseActor(nn.Module, ABC):

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    @abstractmethod
    def act(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Given an observation, return an action and the log probability of that action."""
        pass

    @abstractmethod
    def forward(self, obs) -> Distribution:
        """ Given an observation, return a distribution over actions."""
        pass

    @abstractmethod
    def evaluate(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given stored obs and actions, return log_probs and entropy under current policy."""
        pass


class GaussianActor(BaseActor):

    def __init__(self, device: torch.device, obs_dim: int, action_dim: int, hidden_sizes: list):
        super().__init__(device)
        self.actor_net = build_mlp(obs_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device))

    def act(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def forward(self, obs) -> Distribution:
        mean = self.actor_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    def evaluate(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1) 
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy
    

class CategoricalActor(BaseActor):

    def __init__(self, device: torch.device, obs_dim: int, action_dim: int, hidden_sizes: list):
        super().__init__(device)
        self.actor_net = build_mlp(obs_dim, hidden_sizes, action_dim)

    def act(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def forward(self, obs) -> Distribution:
        logits = self.actor_net(obs)
        return Categorical(logits=logits)
    
    def evaluate(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

   