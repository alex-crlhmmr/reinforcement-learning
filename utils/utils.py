from envs.gridworld import GridWorld
from typing import Tuple, List
import numpy as np
import os

def rollout_episode(env: GridWorld, pi: np.ndarray) -> Tuple[List[int], List[float], List[int]]:
    """
    Roll out a single episode in the environment following a fixed policy π.

    The episode starts from env.reset() and proceeds until termination or truncation.
    Actions are sampled from the provided policy distribution pi[s].

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance (provides reset(), step(), rng, etc.)
    pi : np.ndarray
        Stochastic policy, shape (n_states, n_actions).
        pi[s, a] = probability of taking action a in state s.

    Returns
    -------
    Tuple[List[int], List[float], List[int]]
        - states : List[int]
            Sequence of visited states S_0, S_1, ..., S_{T-1}
        - rewards : List[float]
            Sequence of rewards R_1, R_2, ..., R_T (aligned with transitions from states[t])
            rewards[t] is the reward received after taking an action in states[t]
        - actions : List[int]
            Sequence of actions A_0, A_1, ..., A_{T-1}

    Notes
    -----
    - The lengths of states, rewards, and actions are all equal to the episode length T.
    - states[t] corresponds to S_t, actions[t] corresponds to A_t,
      and rewards[t] corresponds to R_{t+1}.
    - The policy must define a valid probability distribution for each state visited.
    """
    states: List[int] = []
    rewards: List[float] = []
    actions: List[int] = []

    s = env.reset()
    done = False

    while not done:
        states.append(s)

        a = int(env.rng.choice(env.n_actions, p=pi[s]))
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        actions.append(a)
        rewards.append(float(r))

    return states, rewards, actions


def save_policy(path: str, env: GridWorld, pi_opt: np.ndarray, V_opt: np.ndarray, gamma: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        pi_opt=pi_opt.astype(np.float32),
        V_opt=V_opt.astype(np.float32),
        gamma=np.float32(gamma),
        height=np.int32(env.height),
        width=np.int32(env.width),
        wall_density=np.float32(env.wall_density),
        num_traps=np.int32(env.num_traps),
        slip_prob=np.float32(env.slip_prob),
        seed=np.int32(-1 if env.seed is None else env.seed),
        start_state=np.int32(env.start_state),
        goal_state=np.int32(env.goal_state),
        trap_states=np.array(sorted(list(env.trap_states)), dtype=np.int32),
    )

def load_policy(path: str):
    d = np.load(path, allow_pickle=False)
    return d["pi_opt"], d["V_opt"], float(d["gamma"]), dict(d)
