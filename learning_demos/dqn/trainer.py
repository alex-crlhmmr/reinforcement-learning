from config import Config as cfg
from gymnasium import Env as env
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import numpy as np


class Trainer:
    def __init__(self, env: env, agent: DQNAgent, buffer: ReplayBuffer, config: cfg):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config

    def train(self):
        episode_rewards = []
        while self.agent.step < self.config.max_steps:
            obs, _ = self.env.reset(seed=cfg.seed)
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                self.agent.step += 1

                if len(self.buffer) >= self.config.batch_size:
                    batch = self.buffer.sample(self.config.batch_size)
                    self.agent.update(batch)

                if self.agent.step % self.config.target_update_freq == 0:
                    self.agent.sync_target()

                if self.agent.step % self.config.eval_freq == 0:
                    eval_reward = self.evaluate()
                    print(f"Step: {self.agent.step}, Eval Reward: {eval_reward}")

                episode_reward += reward
                if done:
                    episode_rewards.append(episode_reward)                                                                                                                                                   
                    print(f"Step: {self.agent.step}, Episode Reward: {episode_reward}, Epsilon: {self.agent.epsilon}")  
        
        return episode_rewards                                                   

    def evaluate(self, episodes=5):
        total_reward = 0
        for _ in range(episodes):
            obs, _ = self.env.reset(seed=cfg.seed)
            done = False
            while not done:
                action = self.agent.act(obs, explore=False)
                obs, reward, terminted, truncated, _ = self.env.step(action)
                done = terminted or truncated
                total_reward += reward
        return total_reward / episodes
