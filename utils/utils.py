from envs.gridworld import GridWorld
import numpy as np
import os


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
