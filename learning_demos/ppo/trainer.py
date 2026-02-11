from actor import GaussianActor, CategoricalActor
from torch.utils.tensorboard import SummaryWriter
from rollout_buffer import RolloutBuffer
from config import Config as cfg
from gymnasium import Env as env
from ppo_agent import PPOAgent
import numpy as np
import torch


class PPOTrainer:
    def __init__(self, env: env, agent: PPOAgent, buffer: RolloutBuffer, config: cfg, writer: SummaryWriter = None):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config
        self.writer = writer

    def train(self):
        episode_rewards = []
        episode_reward = 0
        obs, _ = self.env.reset()                          
        total_steps = 0
        
        while total_steps < self.config.max_training_steps:
            
            # Collect
            with torch.no_grad():
                for _ in range(self.config.rollout_steps):
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    action_t, log_prob = self.agent.actor.act(obs_t)
                    value = self.agent.critic(obs_t)
                    if isinstance(self.agent.actor, GaussianActor):
                        action = action_t.cpu().numpy()
                    elif isinstance(self.agent.actor, CategoricalActor):
                        action = action_t.item()
                    next_obs, reward, term, trunc, _ = self.env.step(action)
                    done = term or trunc
                    self.buffer.push(obs_t, action_t, reward, done, log_prob, value)
                    episode_reward += reward
                    total_steps += 1

                    if done:
                        episode_rewards.append(episode_reward)
                        if self.writer:
                            self.writer.add_scalar("Reward/Episode", episode_reward, total_steps)
                        else:
                            print(f"Step: {total_steps}, Episode Reward: {episode_reward}")
                        episode_reward = 0
                        obs, _ = self.env.reset()
                    else:
                        obs = next_obs

                    if total_steps % self.config.eval_freq == 0:
                        eval_reward = self.evaluate()
                        if self.writer:
                            self.writer.add_scalar("Reward/Eval", eval_reward, total_steps)
                        else:
                            print(f"Step: {total_steps}, Eval Reward: {eval_reward}")

            # GAE + Update + Reset
            with torch.no_grad():
                last_value = self.agent.critic(torch.tensor(obs, dtype=torch.float32))
            self.buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)
            self.agent.update(self.buffer)
            self.buffer.reset()

        return episode_rewards

    def evaluate(self, episodes=5):
      total_reward = 0
      for _ in range(episodes):
          obs, _ = self.env.reset()
          done = False
          while not done:
              obs_t = torch.tensor(obs, dtype=torch.float32)
              with torch.no_grad():
                  action_t, _ = self.agent.actor.act(obs_t)
              if isinstance(self.agent.actor, GaussianActor):
                  action = action_t.cpu().numpy()
              elif isinstance(self.agent.actor, CategoricalActor):
                  action = action_t.item()
              obs, reward, terminated, truncated, _ = self.env.step(action)
              done = terminated or truncated
              total_reward += reward
      return total_reward / episodes


