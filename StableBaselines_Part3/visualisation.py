import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def visualisation (model_path, video_folder):
    env = gym.make("racetrack-v0", render_mode = 'rgb_array')
    model = PPO.load(model_path, env=env)
    env = RecordVideo(env, video_folder = video_folder, episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()