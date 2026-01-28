from envs.tabular.gridworld import GridWorld
from envs.base import TabularEnvironment
from typing import Tuple, List
import numpy as np
import os

def rollout_episode_tabular(env: TabularEnvironment, pi: np.ndarray) -> Tuple[List[int], List[float], List[int]]:
    """
    Roll out a single episode in the environment following a fixed policy π.

    The episode starts from env.reset() and proceeds until termination or truncation.
    Actions are sampled from the provided policy distribution pi[s].

    Parameters
    ----------
    env : TabularEnvironment
        The TabularEnvironment environment instance (provides reset(), step(), rng, etc.)
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

def evaluate_policy_tabular(
    env: TabularEnvironment,
    pi: np.ndarray,
    num_episodes: int = 1000,
    gamma: float = 0.9
) -> dict:
    """
    Evaluate a policy by running episodes and aggregating generic metrics.

    Works for any environment that follows the BaseEnvironment/TabularEnvironment
    step() API. Does NOT assume what "success" means.

    Parameters
    ----------
    env : TabularEnvironment
        Environment to evaluate in.
    pi : np.ndarray
        Stochastic policy, shape (n_states, n_actions). pi[s, a] is prob of action a in state s.
    n_episodes : int
        Number of episodes.

    Returns
    -------
    dict
        avg_return, std_return, avg_length, terminated_rate, truncated_rate
    """
    returns = np.zeros(num_episodes, dtype=np.float64)
    lengths = np.zeros(num_episodes, dtype=np.int64)
    terminated_count = 0
    truncated_count = 0

    for ep in range(num_episodes):
        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1}/{num_episodes} ...")
        s = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        discount = 1.0

        while not done:
            a = int(env.rng.choice(env.n_actions, p=pi[s]))
            s, r, terminated, truncated, _ = env.step(a)

            ep_return += discount * float(r)
            discount *= gamma

            ep_len += 1
            done = terminated or truncated

        returns[ep] = ep_return
        lengths[ep] = ep_len
        terminated_count += int(terminated)
        truncated_count += int(truncated)

    return {
        "avg_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)),
        "avg_length": float(lengths.mean()),
        "terminated_rate": float(terminated_count / num_episodes),
        "truncated_rate": float(truncated_count / num_episodes),
    }



def save_policy(
        path: str, 
        env: GridWorld, 
        pi_opt: np.ndarray, 
        V_opt: np.ndarray, 
        gamma: float):
    
    """
    Save an evaluated or optimal policy and value function to disk.

    This function stores the policy π, value function V, discount factor γ,
    and the minimal GridWorld configuration needed to reconstruct the
    environment in a compressed NumPy archive (.npz).

    Parameters
    ----------
    path : str
        File path where the policy will be saved (directories are created if needed).
    env : GridWorld
        GridWorld environment instance the policy was computed on.
    pi_opt : np.ndarray
        Policy array of shape (nS, nA), where pi_opt[s, a] is the probability
        of taking action a in state s.
    V_opt : np.ndarray
        Value function array of shape (nS,), indexed by state id.
    gamma : float
        Discount factor used when computing the policy/value function.
    """
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
    """
    Load a saved policy, value function, and environment metadata from disk.

    This function reads a policy archive created by `save_policy` and returns
    the stored policy π, value function V, discount factor γ, and a dictionary
    of environment parameters suitable for reconstructing the original
    GridWorld environment.

    Parameters
    ----------
    path : str
        Path to a saved policy file produced by `save_policy`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, dict]
        - pi_opt : np.ndarray
            Loaded policy array of shape (nS, nA).
        - V_opt : np.ndarray
            Loaded value function array of shape (nS,).
        - gamma : float
            Discount factor used during learning.
        - meta : dict
            Dictionary of environment metadata (grid size, wall density,
            number of traps, slip probability, seed, and terminal states).
    """
    d = np.load(path, allow_pickle=False)
    pi_opt = d["pi_opt"]
    V_opt = d["V_opt"]
    gamma = float(d["gamma"])

    # meta excludes big arrays; keep only config
    meta_keys = ["height", "width", "wall_density", "num_traps", "slip_prob", "seed",
                 "start_state", "goal_state", "trap_states"]
    meta = {k: d[k] for k in meta_keys if k in d}

    return pi_opt, V_opt, gamma, meta
