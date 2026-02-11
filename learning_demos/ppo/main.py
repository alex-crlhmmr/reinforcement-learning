from ppo_agent import PPOAgent
from critic import BaseCritic
from actor import GaussianActor
from rollout_buffer import RolloutBuffer
import matplotlib.pyplot as plt
from config import Config
from trainer import PPOTrainer
import gymnasium as gym
import random
import torch
import time
import os


cfg = Config(
    env_name="LunarLanderContinuous-v2",
    seed=42,
    critic_hidden_sizes=(128, 128, 128),
    actor_hidden_sizes=(128, 128, 128),
    lr=3e-4,
    clip_epsilon=0.2,
    num_epochs=10,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    rollout_steps=2048,
    mini_batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    max_training_steps=120_000,
    eval_freq=5000,
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

    trainer = PPOTrainer(
        env=env,
        agent=agent,
        buffer=buffer,
        config=cfg
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
    main()