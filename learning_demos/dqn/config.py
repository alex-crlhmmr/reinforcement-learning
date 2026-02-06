from dataclasses import dataclass


@dataclass
class Config:
    # env
    env_name: str = "gridworld"
    seed: int = 42

    # agent
    hidden_size: list = (128, 128, 128)
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000
    lr: float = 1e-3
    loss_fn: str = "mse"

    # buffer
    batch_size: int = 64
    memory_capacity: int = 10000

    # training
    target_update_freq: int = 100
    max_steps: int = 10000
    eval_freq: int = 1000
