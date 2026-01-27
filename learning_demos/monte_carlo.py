from envs.gridworld import GridWorld
from utils.utils import load_policy
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np



def rollout_episode(env: GridWorld, pi: np.ndarray) -> Tuple[List[int], List[float], List[int]]:
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



def first_visit_mc_prediction(env: GridWorld, pi: np.ndarray, num_episodes: int = 5000, gamma: float = 0.9) -> np.ndarray:
    N = np.zeros(env.n_states, dtype=np.int64)   # visit counts
    G_sum = np.zeros(env.n_states, dtype=np.float64)  # sum of returns
    V_pi = np.zeros(env.n_states, dtype=np.float64)

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


def every_visit_mc_prediction(env: GridWorld, pi: np.ndarray, num_episodes: int = 5000, gamma: float = 0.9) -> np.ndarray:
    N = np.zeros(env.n_states, dtype=np.int64)   # visit counts
    G_sum = np.zeros(env.n_states, dtype=np.float64)  # sum of returns
    V_pi = np.zeros(env.n_states, dtype=np.float64)

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
    min_visits: int = 10,
):
    N = np.zeros(env.n_states, dtype=np.int64)
    G_sum = np.zeros(env.n_states, dtype=np.float64)
    V_pi = np.zeros(env.n_states, dtype=np.float64)

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
            mask = np.zeros(env.n_states, dtype=bool)
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
    min_visits: int = 10,
):
    N = np.zeros(env.n_states, dtype=np.int64)
    G_sum = np.zeros(env.n_states, dtype=np.float64)
    V_pi = np.zeros(env.n_states, dtype=np.float64)

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
            mask = np.zeros(env.n_states, dtype=bool)
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
    policy_path = "./outputs/policy_iteration/optimal_policy.npz"
    pi_opt, V_opt, gamma, meta = load_policy(policy_path)

    env = GridWorld(
        height=int(meta["height"]),
        width=int(meta["width"]),
        wall_density=float(meta["wall_density"]),
        num_traps=int(meta["num_traps"]),
        slip_prob=float(meta["slip_prob"]),
        seed=None if int(meta["seed"]) == -1 else int(meta["seed"]),
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
