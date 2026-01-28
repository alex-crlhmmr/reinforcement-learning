from envs.tabular.gridworld import GridWorld
from utils.utils import rollout_episode, load_policy
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np



def first_visit_mc_prediction(
        env: GridWorld, 
        pi: np.ndarray, 
        num_episodes: int = 5000, 
        gamma: float = 0.9) -> np.ndarray:
    """
    Estimate the value function V^π using First-Visit Monte Carlo prediction.

    For each episode, for each state s, only the *first occurrence* of s in that
    episode is used to update V(s). The estimate is the sample mean of returns.

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance.
    pi : np.ndarray
        Policy to evaluate, shape (n_states, n_actions).
        pi[s, a] = probability of taking action a in state s.
    num_episodes : int, optional
        Number of episodes to sample, by default 5000.
    gamma : float, optional
        Discount factor, by default 0.9.

    Returns
    -------
    np.ndarray
        Estimated value function V^π(s), shape (n_states,).

    Notes
    -----
    - Uses an incremental sample-average estimator:
        V(s) = (1/N(s)) * sum_i G_i(s)
      where G_i(s) is the return observed on the i-th first-visit to s.
    - Returns are computed efficiently in O(T) per episode via a backward pass:
        G_t = R_{t+1} + gamma * G_{t+1}.
    - First-visit MC can have high variance and may converge slowly, especially
      in large or stochastic environments.
    """
    nS = max(env.states) + 1
    N = np.zeros(nS, dtype=np.int64)
    G_sum = np.zeros(nS, dtype=np.float64)
    V_pi = np.zeros(nS, dtype=np.float64)

    for ep in range(1, num_episodes+1):
        
        if ep % 500 == 0:
            print(f"Episode {ep}/{num_episodes} ...")
        
        states, rewards, _ = rollout_episode(env, pi)

        G_episode = np.zeros(len(rewards), dtype=np.float64)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            G_episode[t] = G

        visited = set()
        for t, s in enumerate(states):
            if s not in visited:
                visited.add(s)
                N[s] += 1
                G_sum[s] += G_episode[t]
                V_pi[s] = G_sum[s] / N[s]

    return V_pi


def every_visit_mc_prediction(
        env: GridWorld, pi: 
        np.ndarray, num_episodes: int = 5000, 
        gamma: float = 0.9) -> np.ndarray:
    """
    Estimate the value function V^π using Every-Visit Monte Carlo prediction.

    For each episode, for each state s, *every occurrence* of s in that episode
    is used to update V(s). The estimate is the sample mean of returns over
    all visits.

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance.
    pi : np.ndarray
        Policy to evaluate, shape (n_states, n_actions).
        pi[s, a] = probability of taking action a in state s.
    num_episodes : int, optional
        Number of episodes to sample, by default 5000.
    gamma : float, optional
        Discount factor, by default 0.9.

    Returns
    -------
    np.ndarray
        Estimated value function V^π(s), shape (n_states,).

    Notes
    -----
    - Uses an incremental sample-average estimator across *all* visits:
        V(s) = (1/N(s)) * sum_i G_i(s)
      where G_i(s) is the return observed on the i-th visit to s (not just first).
    - Returns are computed efficiently in O(T) per episode via a backward pass.
    - Every-visit MC often has lower variance than first-visit because it uses
      more samples per episode, but it can overweight states that appear many
      times within long episodes.
    """
    nS = max(env.states) + 1
    N = np.zeros(nS, dtype=np.int64)
    G_sum = np.zeros(nS, dtype=np.float64)
    V_pi = np.zeros(nS, dtype=np.float64)


    for ep in range(1, num_episodes+1):
        
        if ep % 500 == 0:
            print(f"Episode {ep}/{num_episodes} ...")
        
        states, rewards, _ = rollout_episode(env, pi)

        G_episode = np.zeros(len(rewards), dtype=np.float64)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            G_episode[t] = G

        for t, s in enumerate(states):
            N[s] += 1
            G_sum[s] += G_episode[t]
            V_pi[s] = G_sum[s] / N[s]

    return V_pi


def first_visit_mc_prediction_with_history(
        env: GridWorld,
        pi: np.ndarray,
        V_ref: np.ndarray,
        num_episodes: int = 5000,
        gamma: float = 0.9,
        eval_every: int = 100,
        min_visits: int = 10,) -> Tuple[np.ndarray, dict]:
    """
    First-Visit Monte Carlo prediction with convergence diagnostics over time.

    This function estimates V^π using first-visit MC, and periodically evaluates
    how close the current estimate is to a reference value function V_ref.
    It records RMSE/MAE (on sufficiently visited states) and coverage over episodes.

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance.
    pi : np.ndarray
        Policy to evaluate, shape (n_states, n_actions).
        pi[s, a] = probability of taking action a in state s.
    V_ref : np.ndarray
        Reference value function to compare against (e.g., from DP),
        shape (n_states,).
    num_episodes : int, optional
        Number of episodes to sample, by default 5000.
    gamma : float, optional
        Discount factor, by default 0.9.
    eval_every : int, optional
        Evaluate diagnostics every eval_every episodes, by default 100.
    min_visits : int, optional
        Only include states with at least min_visits (first-visits) in the
        RMSE/MAE computation, by default 10.

    Returns
    -------
    Tuple[np.ndarray, dict]
        - V_pi : np.ndarray
            Estimated value function V^π(s), shape (n_states,).
        - history : dict
            Diagnostic history containing:
              * "episodes": np.ndarray of evaluation episode indices
              * "rmse": np.ndarray of RMSE values
              * "mae": np.ndarray of MAE values
              * "coverage": np.ndarray of coverage values
              * "N": np.ndarray visit counts (first-visit counts per state)

    Notes
    -----
    - RMSE/MAE are computed on a *moving subset* of states defined by:
        non-wall states (env.states),
        excluding terminal states,
        and requiring N[s] >= min_visits.
      Because this subset can expand over time, RMSE/MAE are not guaranteed
      to decrease monotonically.
    - Coverage is defined as:
        (# of eligible states in mask) / (# of non-wall states).
    - Returns are computed in O(T) per episode via a backward pass.
    """
    nS = max(env.states) + 1
    N = np.zeros(nS, dtype=np.int64)
    G_sum = np.zeros(nS, dtype=np.float64)
    V_pi = np.zeros(nS, dtype=np.float64)

    episodes = []
    rmse_hist = []
    mae_hist = []
    coverage_hist = []

    for ep in range(1, num_episodes + 1):
        states, rewards, _ = rollout_episode(env, pi)

        G_episode = np.zeros(len(rewards), dtype=np.float64)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            G_episode[t] = G


        visited = set()
        for t, s in enumerate(states):
            if s not in visited:
                visited.add(s)
                N[s] += 1
                G_sum[s] += G_episode[t]
                V_pi[s] = G_sum[s] / N[s]

        if ep % eval_every == 0 or ep == 1:
            # compare only on sufficiently visited, non-terminal, non-wall states
            mask = np.zeros(nS, dtype=bool)
            for s in env.states:
                if env.is_terminal(s):
                    continue
                if N[s] >= min_visits:
                    mask[s] = True

            coverage = mask.sum() / max(1, len(env.states))
            if mask.sum() > 0:
                diff = V_pi[mask] - V_ref[mask]
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
            else:
                rmse, mae = float("nan"), float("nan")

            episodes.append(ep)
            rmse_hist.append(rmse)
            mae_hist.append(mae)
            coverage_hist.append(coverage)

            if ep % (5 * eval_every) == 0:
                print(f"ep={ep:5d} RMSE={rmse:.3f} MAE={mae:.3f} "
                    f"mask={mask.sum()} coverage={coverage*100:.1f}%")


    history = {
        "episodes": np.array(episodes),
        "rmse": np.array(rmse_hist),
        "mae": np.array(mae_hist),
        "coverage": np.array(coverage_hist),
        "N": N,
    }
    return V_pi, history


def every_visit_mc_prediction_with_history(
        env: GridWorld,
        pi: np.ndarray,
        V_ref: np.ndarray,
        num_episodes: int = 5000,
        gamma: float = 0.9,
        eval_every: int = 100,
        min_visits: int = 10,) -> Tuple[np.ndarray, dict]:
    """
    Every-Visit Monte Carlo prediction with convergence diagnostics over time.

    This function estimates V^π using every-visit MC, and periodically evaluates
    how close the current estimate is to a reference value function V_ref.
    It records RMSE/MAE (on sufficiently visited states) and coverage over episodes.

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance.
    pi : np.ndarray
        Policy to evaluate, shape (n_states, n_actions).
        pi[s, a] = probability of taking action a in state s.
    V_ref : np.ndarray
        Reference value function to compare against (e.g., from DP),
        shape (n_states,).
    num_episodes : int, optional
        Number of episodes to sample, by default 5000.
    gamma : float, optional
        Discount factor, by default 0.9.
    eval_every : int, optional
        Evaluate diagnostics every eval_every episodes, by default 100.
    min_visits : int, optional
        Only include states with at least min_visits (total visits) in the
        RMSE/MAE computation, by default 10.

    Returns
    -------
    Tuple[np.ndarray, dict]
        - V_pi : np.ndarray
            Estimated value function V^π(s), shape (n_states,).
        - history : dict
            Diagnostic history containing:
              * "episodes": np.ndarray of evaluation episode indices
              * "rmse": np.ndarray of RMSE values
              * "mae": np.ndarray of MAE values
              * "coverage": np.ndarray of coverage values
              * "N": np.ndarray visit counts (every-visit counts per state)

    Notes
    -----
    - RMSE/MAE are computed on a *moving subset* of states defined by:
        non-wall states (env.states),
        excluding terminal states,
        and requiring N[s] >= min_visits.
      Because this subset can expand over time, RMSE/MAE are not guaranteed
      to decrease monotonically.
    - Coverage is defined as:
        (# of eligible states in mask) / (# of non-wall states).
    - Returns are computed in O(T) per episode via a backward pass.
    - Compared to first-visit MC, every-visit MC usually reaches the
      min_visits threshold sooner (higher coverage earlier) because
      repeated visits within an episode count toward N[s].
    """
    nS = max(env.states) + 1
    N = np.zeros(nS, dtype=np.int64)
    G_sum = np.zeros(nS, dtype=np.float64)
    V_pi = np.zeros(nS, dtype=np.float64)

    episodes = []
    rmse_hist = []
    mae_hist = []
    coverage_hist = []

    for ep in range(1, num_episodes + 1):
        states, rewards, _ = rollout_episode(env, pi)

        G_episode = np.zeros(len(rewards), dtype=np.float64)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            G_episode[t] = G


        for t, s in enumerate(states):
            N[s] += 1
            G_sum[s] += G_episode[t]
            V_pi[s] = G_sum[s] / N[s]

        if ep % eval_every == 0 or ep == 1:
            # compare only on sufficiently visited, non-terminal, non-wall states
            mask = np.zeros(nS, dtype=bool)
            for s in env.states:
                if env.is_terminal(s):
                    continue
                if N[s] >= min_visits:
                    mask[s] = True

            coverage = mask.sum() / max(1, len(env.states))
            if mask.sum() > 0:
                diff = V_pi[mask] - V_ref[mask]
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
            else:
                rmse, mae = float("nan"), float("nan")

            episodes.append(ep)
            rmse_hist.append(rmse)
            mae_hist.append(mae)
            coverage_hist.append(coverage)

            if ep % (5 * eval_every) == 0:
                print(f"ep={ep:5d} RMSE={rmse:.3f} MAE={mae:.3f} "
                    f"mask={mask.sum()} coverage={coverage*100:.1f}%")


    history = {
        "episodes": np.array(episodes),
        "rmse": np.array(rmse_hist),
        "mae": np.array(mae_hist),
        "coverage": np.array(coverage_hist),
        "N": N,
    }
    return V_pi, history


def plot_mc_stabilization(history: dict):
    """
    Plot Monte Carlo convergence diagnostics recorded by *_with_history functions.

    Produces two plots:
      1) RMSE and MAE vs episode number
      2) Coverage vs episode number

    Parameters
    ----------
    history : dict
        Dictionary returned by first_visit_mc_prediction_with_history or
        every_visit_mc_prediction_with_history. Expected keys:
          - "episodes": np.ndarray of evaluation episode indices
          - "rmse": np.ndarray of RMSE values at those episodes
          - "mae": np.ndarray of MAE values at those episodes
          - "coverage": np.ndarray of coverage values at those episodes

    Returns
    -------
    None
    """
    eps = history["episodes"]
    rmse = history["rmse"]
    mae = history["mae"]
    cov = history["coverage"]

    plt.figure(figsize=(9, 4))
    plt.plot(eps, rmse, label="RMSE vs V_opt")
    plt.plot(eps, mae, label="MAE vs V_opt")
    plt.xlabel("Episodes")
    plt.ylabel("Error")
    plt.title("First-Visit MC: Error vs Episodes (stabilization)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.plot(eps, cov)
    plt.xlabel("Episodes")
    plt.ylabel("Coverage (fraction of states with N>=min_visits)")
    plt.title("How much of the state space is well-sampled")
    plt.tight_layout()
    plt.show()




def main():
    policy_path = "./outputs/gridworld/policy_iteration/optimal_policy.npz"
    pi_opt, V_opt, gamma, meta = load_policy(policy_path)

    env = GridWorld(
        height=int(meta["height"]),
        width=int(meta["width"]),
        wall_density=float(meta["wall_density"]),
        num_traps=int(meta["num_traps"]),
        slip_prob=float(meta["slip_prob"]),
        seed=None if int(meta["seed"]) == -1 else int(meta["seed"]),
    )

    nS = max(env.states) + 1
    assert pi_opt.shape[0] >= nS, f"Policy has {pi_opt.shape[0]} states, env needs {nS}"
    assert V_opt.shape[0] >= nS, f"V_opt has {V_opt.shape[0]} states, env needs {nS}"


    print(
        "[WARNING] Monte Carlo policy evaluation is being run on-policy with\n"
        "          a fixed start-state reset.\n"
        "          Estimated values V_pi will converge only for states that\n"
        "          are sufficiently visited under the policy.\n"
        "          States outside the policy's typical trajectories may have\n"
        "          high variance or unreliable estimates.\n"
        "          DP values (V_ref) are unconditioned and serve only as a\n"
        "          global reference.\n"
    )


    V_pi_fv, hist_fv = first_visit_mc_prediction_with_history(
        env, pi_opt, V_ref=V_opt,
        num_episodes=50000, gamma=gamma,
        eval_every=100, min_visits=10
    )

    V_pi_ev, hist_ev = every_visit_mc_prediction_with_history(
        env, pi_opt, V_ref=V_opt,
        num_episodes=50000, gamma=gamma,
        eval_every=100, min_visits=10
    )

    plot_mc_stabilization(hist_fv)
    plot_mc_stabilization(hist_ev)


if __name__ == "__main__":
    main()
