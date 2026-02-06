from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from config import Config 
from trainer import Trainer
import gymnasium as gym
import torch
import random




cfg = Config(
    env_name="CartPole-v1",
    hidden_size=(128, 128, 128),
    seed=42,
    gamma=0.90,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=1000,
    lr=1e-3,
    loss_fn="mse",
    batch_size=100,
    memory_capacity=5000,
    target_update_freq=1000,
    max_steps=100000,
    eval_freq=1000
)

loss_map = {
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss()
}

def main():

    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(cfg.env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                                                                      
    agent = DQNAgent(input_dim=env.observation_space.shape[0],
                     output_dim=env.action_space.n,
                     hidden_sizes=cfg.hidden_size, 
                     gamma=cfg.gamma, 
                     lr=cfg.lr, 
                     epsilon_start=cfg.epsilon_start, 
                     epsilon_end=cfg.epsilon_end, 
                     epsilon_decay=cfg.epsilon_decay,
                     loss_fn=loss_map[cfg.loss_fn],
                     device=device)
    
    buffer = ReplayBuffer(capacity=cfg.memory_capacity)
    trainer = Trainer(env, agent, buffer, cfg)
    rewards = trainer.train()
    plt.plot(rewards)


    

if __name__ == "__main__":
    main()