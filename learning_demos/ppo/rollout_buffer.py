
import torch


class RolloutBuffer:
    def __init__(self, rollout_steps, obs_dim, action_dim, device):
        self.rollout_steps = rollout_steps                                                                                                      
        self.device = device
        self.ptr = 0                                                                                                                            
                    
        self.states = torch.zeros(rollout_steps, obs_dim, device=device)
        self.actions = torch.zeros(rollout_steps, action_dim, device=device)
        self.rewards = torch.zeros(rollout_steps, device=device)
        self.dones = torch.zeros(rollout_steps, device=device)
        self.log_probs = torch.zeros(rollout_steps, device=device)
        self.values = torch.zeros(rollout_steps, device=device)

        self.advantages = torch.zeros(rollout_steps, device=device)
        self.returns = torch.zeros(rollout_steps, device=device)

    def push(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_GAE(self, )