from abc import ABC, abstractmethod
import torch.nn as nn
import torch



class BaseAgent(nn.Module, ABC):

    def __init__(self, device: torch.device, num_actions: int, gamma: float = 0.9):
        super().__init__()
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.step = 0

    @abstractmethod
    def act(self, obs, explore: bool = True):
        pass

    @abstractmethod
    def update(self, batch):
        pass

    def sync_target(self):
        pass

    def as_tensor(self, x, dtype=None):
        t = torch.as_tensor(x)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(self.device)
