from envs.gridworld import GridWorld
from utils.utils import load_policy
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np





def td0_learning(
    env: GridWorld,
    pi: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 10_000,
    gamma: float = 0.9,) -> np.ndarray:
    """
    TD(0) policy evaluation: learn V^pi from sampled transitions.

    Parameters
    ----------
    env : GridWorld
        Environment.
    pi : np.ndarray
        Policy (n_states, n_actions).
    alpha : float
        Step size.
    max_iter : int
        Number of transitions (updates) to perform.
    gamma : float
        Discount factor.

    Returns
    -------
    np.ndarray
        Learned value function V_pi (n_states,).
    """
    V_pi = np.zeros(env.n_states, dtype=np.float64)
    s = env.reset()

    for _ in range(int(max_iter)):
        a = int(env.rng.choice(env.n_actions, p=pi[s]))
        next_state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        target = reward + (0.0 if done else gamma * V_pi[next_state])

        V_pi[s] += alpha * (target - V_pi[s])

        s = env.reset() if done else next_state

    return V_pi


def td0_learning_with_history(
    env: GridWorld,
    pi: np.ndarray,
    V_ref: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 100_000,
    gamma: float = 0.9,
    eval_every: int = 1000,
    min_visits: int = 10,
):
    """
    TD(0) policy evaluation with convergence diagnostics over time.

    This function learns V^π using TD(0) updates from sampled transitions generated
    by following policy π, and periodically evaluates how close the current estimate
    is to a reference value function V_ref.

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
    alpha : float, optional
        TD(0) step size, by default 0.1.
    max_iter : int, optional
        Number of TD updates (transitions) to perform, by default 100000.
    gamma : float, optional
        Discount factor, by default 0.9.
    eval_every : int, optional
        Evaluate diagnostics every eval_every TD updates, by default 1000.
    min_visits : int, optional
        Only include states with at least min_visits updates in the RMSE/MAE
        computation, by default 10.

    Returns
    -------
    Tuple[np.ndarray, dict]
        - V_pi : np.ndarray
            Learned value function V^π(s), shape (n_states,).
        - history : dict
            Diagnostic history containing:
              * "steps": np.ndarray of evaluation step indices
              * "rmse": np.ndarray of RMSE values
              * "mae": np.ndarray of MAE values
              * "coverage": np.ndarray of coverage values
              * "N": np.ndarray visit/update counts per state
    """
    V_pi = np.zeros(env.n_states, dtype=np.float64)

    # N[s] counts how many TD updates were applied to state s
    N = np.zeros(env.n_states, dtype=np.int64)

    steps = []
    rmse_hist = []
    mae_hist = []
    coverage_hist = []

    s = env.reset()

    for t in range(1, int(max_iter) + 1):
        a = int(env.rng.choice(env.n_actions, p=pi[s]))
        next_state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        target = reward + (0.0 if done else gamma * V_pi[next_state])
        V_pi[s] += alpha * (target - V_pi[s])
        N[s] += 1

        s = env.reset() if done else next_state

        if t % eval_every == 0 or t == 1:
            mask = np.zeros(env.n_states, dtype=bool)
            for st in env.states:
                if env.is_terminal(st):
                    continue
                if N[st] >= min_visits:
                    mask[st] = True

            coverage = mask.sum() / max(1, len(env.states))
            if mask.sum() > 0:
                diff = V_pi[mask] - V_ref[mask]
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
            else:
                rmse, mae = float("nan"), float("nan")

            steps.append(t)
            rmse_hist.append(rmse)
            mae_hist.append(mae)
            coverage_hist.append(coverage)

            if t % (5 * eval_every) == 0:
                print(f"step={t:6d} RMSE={rmse:.3f} MAE={mae:.3f} "
                      f"mask={mask.sum()} coverage={coverage*100:.1f}%")

    history = {
        "steps": np.array(steps),
        "rmse": np.array(rmse_hist),
        "mae": np.array(mae_hist),
        "coverage": np.array(coverage_hist),
        "N": N,
    }
    return V_pi, history




def td_lambda_learning(
    env: GridWorld,
    pi: np.ndarray,
    alpha: float = 0.1,
    lambda_: float = 0.6,
    max_iter: int = 10_000,
    gamma: float = 0.9,) -> np.ndarray:
    
    
    V_pi = np.zeros(env.n_states, dtype=np.float64)
    e = np.zeros(env.n_states, dtype=np.float64)
    s = env.reset()

    for _ in range(int(max_iter)):
        a = int(env.rng.choice(env.n_actions, p=pi[s]))
        next_state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        target = reward + (0.0 if done else gamma * V_pi[next_state])
        delta = target - V_pi[s]

        e *= gamma * lambda_
        e[s] += 1.0

        V_pi += alpha * delta * e

        if done:
            s = env.reset()
            e[:] = 0.0   
        else:
            s = next_state

    return V_pi



def td_lambda_learning_with_history(
    env: GridWorld,
    pi: np.ndarray,
    V_ref: np.ndarray,
    alpha: float = 0.1,
    lambda_: float = 0.6,
    max_iter: int = 100000,
    gamma: float = 0.9,
    eval_every: int = 1000,
    min_visits: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """
    TD(lambda) policy evaluation with eligibility traces + diagnostics,
    compatible with plot_td_stabilization(history).
    """
    V_pi = np.zeros(env.n_states, dtype=np.float64)

    # eligibility trace
    e = np.zeros(env.n_states, dtype=np.float64)

    # N[s] counts how many updates were "triggered" from state s being current
    N = np.zeros(env.n_states, dtype=np.int64)

    steps = []
    rmse_hist = []
    mae_hist = []
    coverage_hist = []

    s = env.reset()

    for t in range(1, int(max_iter) + 1):
        a = int(env.rng.choice(env.n_actions, p=pi[s]))
        next_state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        target = reward + (0.0 if done else gamma * V_pi[next_state])
        delta = target - V_pi[s]

        # accumulating traces (your version)
        e *= gamma * lambda_
        e[s] += 1.0

        # TD(lambda) update: updates all states weighted by eligibility
        V_pi += alpha * delta * e

        # bookkeeping: count visits/updates for coverage like TD(0) code
        N[s] += 1

        if done:
            s = env.reset()
            e[:] = 0.0
        else:
            s = next_state

        # diagnostics (same structure as your TD(0) with_history)
        if t % eval_every == 0 or t == 1:
            mask = np.zeros(env.n_states, dtype=bool)
            for st in env.states:
                if env.is_terminal(st):
                    continue
                if N[st] >= min_visits:
                    mask[st] = True

            coverage = mask.sum() / max(1, len(env.states))
            if mask.sum() > 0:
                diff = V_pi[mask] - V_ref[mask]
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
            else:
                rmse, mae = float("nan"), float("nan")

            steps.append(t)
            rmse_hist.append(rmse)
            mae_hist.append(mae)
            coverage_hist.append(coverage)

            if t % (5 * eval_every) == 0:
                print(f"step={t:6d} RMSE={rmse:.3f} MAE={mae:.3f} "
                      f"mask={mask.sum()} coverage={coverage*100:.1f}%")

    history = {
        "steps": np.array(steps),
        "rmse": np.array(rmse_hist),
        "mae": np.array(mae_hist),
        "coverage": np.array(coverage_hist),
        "N": N,
    }
    return V_pi, history



def plot_td_stabilization(history: dict):
    """
    """
    steps = history["steps"]
    rmse = history["rmse"]
    mae = history["mae"]
    cov = history["coverage"]

    plt.figure(figsize=(9, 4))
    plt.plot(steps, rmse, label="RMSE vs V_ref")
    plt.plot(steps, mae, label="MAE vs V_ref")
    plt.xlabel("TD updates (transitions)")
    plt.ylabel("Error")
    plt.title("TD: Error vs Updates (stabilization)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.plot(steps, cov)
    plt.xlabel("TD updates (transitions)")
    plt.ylabel("Coverage (fraction of states with N>=min_visits)")
    plt.title("TD: How much of the state space is well-sampled")
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

    print(
        "[WARNING] TD(0) and TD(λ) is being run with a fixed start-state reset.\n"
        "          The learned value function V_pi will be accurate primarily\n"
        "          for states that are frequently visited under the policy.\n"
        "          It is NOT expected to match the DP value function V_ref\n"
        "          on states that are rarely or never visited.\n"
        "          This comparison is therefore conditioned on the policy's\n"
        "          visitation distribution, not global over all states.\n"
    )


    V_td0, hist_td0 = td0_learning_with_history(
        env, pi_opt, V_ref=V_opt,
        alpha=0.1,
        max_iter=2000000,     
        gamma=gamma,
        eval_every=2000,
        min_visits=10
    )

    V_td_lambda, hist_td_lambda = td_lambda_learning_with_history(
        env, pi_opt, V_ref=V_opt,
        alpha=0.1,
        lambda_=0.8,
        max_iter=2000000,     
        gamma=gamma,
        eval_every=2000,
        min_visits=10
    )



    plot_td_stabilization(hist_td0)
    plot_td_stabilization(hist_td_lambda)


if __name__ == "__main__":
    main()
