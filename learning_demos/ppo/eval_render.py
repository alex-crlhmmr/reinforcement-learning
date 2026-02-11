from gymnasium.wrappers import RecordVideo
from actor import GaussianActor 
import gymnasium as gym       
import torch    
import glob                                                                                                              

                
env_name = "Humanoid-v4"
env = gym.make(env_name, render_mode="rgb_array")
env = RecordVideo(env, video_folder="outputs/ppo/vanilla/videos")                                                                           

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = GaussianActor(
    device=torch.device("cpu"),
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_sizes=(256, 256, 256)
)
path = glob.glob("outputs/ppo/vanilla/Humanoid-v4_actor_42_*.pth")[-1]
actor.load_state_dict(torch.load(path))
actor.eval()

for _ in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, _ = actor.act(obs_t)
        obs, reward, term, trunc, _ = env.step(action.cpu().numpy())
        done = term or trunc
        total_reward += reward
    print(f"Episode Reward: {total_reward}")

env.close()