
from network_utils import build_mlp
from base_agent import BaseAgent
from typing import List
import numpy as np
import random
import torch



class DQNAgent(BaseAgent):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_sizes: List[int] = (128,128), 
                 activation="relu",
                 gamma=0.99, 
                 lr=1e-3,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=1000,
                 loss_fn = torch.nn.MSELoss(),
                 device=torch.device('cpu')):
        
        super().__init__(device, output_dim, gamma)
        self.online_net = build_mlp(input_dim, hidden_sizes, output_dim, activation)
        self.target_net = build_mlp(input_dim, hidden_sizes, output_dim, activation)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.loss_fn = loss_fn
        

    def act(self, obs, explore: bool = True):
        
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        obs_tensor = self.as_tensor(obs, dtype=torch.float32).unsqueeze(0) 
        
        with torch.no_grad():
            q_values = self.online_net(obs_tensor)
        
        return q_values.argmax(dim =1).item()
        

    def update(self, batch: tuple):
        states, actions, rewards, next_states, dones = batch
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0] 
        
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.step * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay) 
        return loss.item()
        


    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())