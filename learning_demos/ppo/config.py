from dataclasses import dataclass


@dataclass
class Config:
    # env
    env_name: str = "CartPole-v1"
    seed: int = 42

    # Critic
    critic_hidden_sizes: tuple = (128, 128)

    # Actor
    actor_hidden_sizes: tuple = (128, 128)

    # PPO Agent
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    num_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Rollout Buffer
    rollout_steps: int = 2048
    mini_batch_size: int = 64

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Training
    max_training_steps: int = 100000
    eval_freq: int = 5000
    save_path: str = "./output/ppo/vanilla"

    
