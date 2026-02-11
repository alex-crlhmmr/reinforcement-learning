from network_utils import build_mlp
import torch.nn as nn
import torch



class BaseCritic(nn.Module):

    def __init__(self, device: torch.device, obs_dim: int, hidden_sizes: list):
        super().__init__()
        self.critic_net = build_mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs) -> torch.Tensor:
        return self.critic_net(obs).squeeze(-1)     