
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

    def compute_gae(self, last_value, gamma, gae_lambda):                                                                                       
        gae = 0                                                                                                                                 
        for t in reversed(range(self.rollout_steps)): 
            if t == self.rollout_steps - 1:                                                                                                
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
        
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


    def get_batches(self, mini_batch_size):                                                                                                     
        indices = torch.randperm(self.rollout_steps)
        for start in range(0, self.rollout_steps, mini_batch_size):
            batch_idx = indices[start:start + mini_batch_size]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
            )

    def reset(self):
        self.ptr = 0
