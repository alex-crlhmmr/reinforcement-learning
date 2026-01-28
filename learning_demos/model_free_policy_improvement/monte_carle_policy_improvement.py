from envs.base import TabularEnvironment
from envs.tabular.gridworld import GridWorld
from utils.utils import rollout_episode_tabular, evaluate_policy_tabular, load_policy
from typing import Tuple
import numpy as np


def monte_carlo_policy_improvement(env: TabularEnvironment, num_episodes: int = 5000, gamma: float = 0.9, eps_min: float = 0.05, first_visit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions))
    eps0 = 1.0

    pi = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions, dtype=np.float64)
    for ep in range(1, num_episodes + 1):
        if ep % 500 == 0:
            print(f"Episode {ep}/{num_episodes} ...")
        
        states, rewards, actions = rollout_episode_tabular(env, pi)

        G_episode = np.zeros(len(rewards), dtype=np.float64)
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            G_episode[t] = G

        if first_visit:
            visited = set()
            for t, (s, a) in enumerate(zip(states, actions)):
                if (s, a) in visited:
                    continue
                visited.add((s, a))
                N[s, a] += 1
                Q[s, a] += (G_episode[t] - Q[s, a]) / N[s, a]

        else:
            for t, (s, a) in enumerate(zip(states, actions)):
                N[s, a] += 1
                Q[s, a] += (G_episode[t] - Q[s, a]) / N[s, a]
        
        # Epsilon-greedy policy improvement
        eps = max(eps_min, eps0 / np.sqrt(ep))
        visited_states = set(states)  
        for s in visited_states:
            best_a = int(np.argmax(Q[s]))
            pi[s] = eps / env.n_actions
            pi[s, best_a] += 1.0 - eps

        
    return pi, Q


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

    print(
        "[WARNING] MC control here collects episodes starting from env.reset(),\n"
        "          which (in this GridWorld) always resets to the SAME start_state.\n"
        "          So the learned policy/value estimates will be strongest for states\n"
        "          that are actually visited from that start under exploration.\n"
        "          DP (pi_opt/V_opt) is computed for the full MDP (all states).\n"
    )

    pi_mc, Q_mc = monte_carlo_policy_improvement(
        env,
        num_episodes=80000,
        gamma=gamma,
        first_visit=True,
    )

    pi_greedy = np.zeros_like(pi_mc)
    best_actions = np.argmax(Q_mc, axis=1)
    pi_greedy[np.arange(env.n_states), best_actions] = 1.0


    # Evaluate both policies (generic evaluator; should work for any TabularEnvironment)
    metrics_dp = evaluate_policy_tabular(env, pi_opt, num_episodes=1000, gamma=gamma)
    metrics_mc = evaluate_policy_tabular(env, pi_mc, num_episodes=1000, gamma=gamma)
    metrics_mc_greedy = evaluate_policy_tabular(env, pi_greedy, num_episodes=1000, gamma=gamma)

    print("\n=== DP policy (pi_opt) ===")
    for k, v in metrics_dp.items():
        print(f"{k}: {v}")

    print("\n=== MC learned policy (pi_mc) ===")
    for k, v in metrics_mc.items():
        print(f"{k}: {v}")

    print("\n=== MC learned greedy policy (pi_greedy) ===")
    for k, v in metrics_mc_greedy.items():
        print(f"{k}: {v}")

    

if __name__ == "__main__":
    main()