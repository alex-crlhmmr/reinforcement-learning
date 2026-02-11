from torch.utils.tensorboard import SummaryWriter
from rollout_buffer import RolloutBuffer
import matplotlib.pyplot as plt
from actor import GaussianActor
from trainer import PPOTrainer
from ppo_agent import PPOAgent
from critic import BaseCritic
from config import Config
import gymnasium as gym
import random
import torch
import time
import os


cfg = Config(
    env_name="Humanoid-v4",
    seed=42,
    critic_hidden_sizes=(256, 256, 256),
    actor_hidden_sizes=(256, 256, 256),
    lr=3e-4,
    clip_epsilon=0.2,
    num_epochs=10,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    rollout_steps=2048,
    mini_batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    max_training_steps=20_000_000,
    eval_freq=10000,
    save_path="./outputs/ppo/vanilla"
)                                                                                     


def main():

    seed = cfg.seed
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(cfg.env_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = GaussianActor(
        device=device,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_sizes=cfg.actor_hidden_sizes
    ).to(device)

    critic = BaseCritic(
        device=device,
        obs_dim=env.observation_space.shape[0],
        hidden_sizes=cfg.critic_hidden_sizes
    ).to(device)

    agent = PPOAgent(
        actor=actor,
        critic=critic,
        lr=cfg.lr,
        clip_epsilon=cfg.clip_epsilon,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        num_epochs=cfg.num_epochs,
        mini_batch_size=cfg.mini_batch_size,
        max_grad_norm=cfg.max_grad_norm
    )

    buffer = RolloutBuffer(
        rollout_steps=cfg.rollout_steps,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )

    writer = SummaryWriter(f"runs/ppo_{cfg.env_name}")

    trainer = PPOTrainer(
        env=env,
        agent=agent,
        buffer=buffer,
        config=cfg,
        writer=writer
    )

    rewards = trainer.train()

    os.makedirs(cfg.save_path, exist_ok=True)                                                                                                                     
    torch.save(agent.actor.state_dict(), f"{cfg.save_path}/{cfg.env_name}_actor_{cfg.seed}_{int(time.time())}.pth")
    torch.save(agent.critic.state_dict(), f"{cfg.save_path}/{cfg.env_name}_critic_{cfg.seed}_{int(time.time())}.pth")
    

    plt.plot(rewards)                                                                                                                                          
    plt.xlabel("Episode")                                                                                                                                      
    plt.ylabel("Reward")
    plt.title(f"PPO Training Rewards - {cfg.env_name}")
    plt.show() 


if __name__ == "__main__":
    # tensorboard --logdir runs
    main()