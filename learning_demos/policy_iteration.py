from envs.gridworld import GridWorld
from utils.utils import save_policy
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

def iterative_policy_evaluation(env: GridWorld, pi: np.ndarray, tol: float = 1e-6, max_iter: int = 1000, gamma: float = 0.9) -> np.ndarray:
    """
    Perform iterative policy evaluation to compute the value function V^π
    for a given (fixed) policy under the environment's MDP.

    This is the policy evaluation step of policy iteration.
    It solves the Bellman expectation equation iteratively until convergence.

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance providing transition and reward model
    pi : np.ndarray
        Policy to evaluate, shape (n_states, n_actions).
        pi[s, a] = probability of taking action a in state s.
    tol : float, optional
        Convergence tolerance (maximum change in value function), by default 1e-6
    max_iter : int, optional
        Maximum number of iterations, by default 1000
    gamma : float, optional
        Discount factor, by default 0.9

    Returns
    -------
    np.ndarray
        Value function V^π(s), shape (n_states,)

    Notes
    -----
    - Uses synchronous updates (copies value array each iteration).
    - Terminal states are assigned value 0.
    - Convergence is checked via the maximum difference between successive value arrays.
    """
    V = np.zeros(env.n_states)
    
    for iteration in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in range(env.n_states):
            if env.is_terminal(s):
                V_new[s] = 0
                continue

            v = 0.0
            for a in range(env.n_actions):
                prob_a = pi[s,a]
                if prob_a == 0:
                    continue
                
                transitions = env.P(s, a)
                if not transitions:
                    continue

                for next_s, p in transitions.items():
                    reward = env.R(s, a, next_s)
                    v += prob_a * p * (reward + gamma * V[next_s] * (not env.is_terminal(next_s)))
            V_new[s] = v
            delta = max(delta, abs(v - V[s]))

        V = V_new
        if delta < tol:
            print(f"Policy evaluation converged after {iteration+1} iterations (δ={delta:.2e})")
            break
    else:
        print(f"Warning: Did not converge within {max_iter} iterations (δ={delta:.2e})")

    return V

def policy_iteration(env: GridWorld, tol: float = 1e-6, max_iter: int = 1000, gamma: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the optimal policy and value function using Policy Iteration.

    Alternates between:
      1. Policy Evaluation: compute V^π using iterative_policy_evaluation
      2. Policy Improvement: make policy greedy w.r.t. current V

    Stops when the policy is stable (no changes between iterations) or max_iter is reached.

    Parameters
    ----------
    env : GridWorld
        The GridWorld environment instance
    tol : float, optional
        Convergence tolerance for policy evaluation, by default 1e-6
    max_iter : int, optional
        Maximum number of policy iteration cycles, by default 1000
    gamma : float, optional
        Discount factor, by default 0.9

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - pi_opt : np.ndarray
            Optimal (deterministic) policy, shape (n_states, n_actions)
            pi_opt[s, a] = 1.0 for the best action, 0 otherwise
        - V_opt : np.ndarray
            Optimal value function V^*(s), shape (n_states,)

    Notes
    -----
    - Starts with a uniform random policy.
    - Uses deterministic greedy improvement (one-hot policy).
    - Policy convergence is detected when no state changes its action.
    - The returned policy can be used directly with env.render(policy=...)
      after converting to action indices via np.argmax(pi_opt, axis=1)
    """
    pi = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)

    for iterations in range(max_iter):
        V = iterative_policy_evaluation(env, pi, tol=tol, max_iter=max_iter, gamma=gamma)
        policy_stable = True
        pi_new = np.zeros((env.n_states, env.n_actions))

        for s in range(env.n_states):
            if env.is_terminal(s):
                pi_new[s] = pi[s]
                continue  

            action_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                q = 0.0
                for next_s, p in env.P(s, a).items():
                    reward = env.R(s, a, next_s)
                    q += p * (reward + gamma * V[next_s] * (not env.is_terminal(next_s)))
                action_values[a] = q

            best_a = np.argmax(action_values)
            pi_new[s, best_a] = 1.0

            if not np.array_equal(pi_new[s], pi[s]):
                policy_stable = False

        pi = pi_new

        if policy_stable:
            print(f"Policy iteration converged after {iterations+1} iterations.")
            break
    else:
        print(f"Warning: Policy iteration did not converge within {max_iter} iterations.")

    return pi, V



def main():
    env = GridWorld(
        height=30,
        width=30,
        wall_density=0.15,
        num_traps=30,
        slip_prob=0.2,
        seed=70
    )

    pi_opt, V_opt = policy_iteration(env, tol=1e-6, max_iter=1000, gamma=0.9)

    # Save optimal policy to file
    save_policy(
        path="./outputs/policy_iteration/optimal_policy.npz",
        env=env,
        pi_opt=pi_opt,
        V_opt=V_opt,
        gamma=0.9
    )
    print("Saved optimal policy to: ./outputs/policy_iteration/optimal_policy.npz")

    # Reset environment so current_state = start_state
    env.reset()

    # Get the action indices for the optimal policy
    policy_indices = np.argmax(pi_opt, axis=1)

    # Render with optimal policy (arrows) and values
    fig = env.render(
        values=V_opt,
        policy=policy_indices,              
        title="Optimal Policy + Value Function (after Policy Iteration)",
        show_state_indices=False,
        figsize=(12, 10)
    )

    # Save to file
    fig.savefig("./outputs/policy_iteration/optimal_policy_value.png", dpi=150, bbox_inches='tight')
    print("Saved optimal policy visualization to: ./outputs/policy_iteration/optimal_policy_value.png")

    plt.close(fig)


if __name__ == "__main__":
    main()