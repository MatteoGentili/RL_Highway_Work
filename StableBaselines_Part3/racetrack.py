from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from visualisation import visualisation

TRAIN = True

config = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'on_road'],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True,
        "lateral_min": -1.0, 
        "lateral_max": 1.0 
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 10,
    "collision_reward": -10, # The reward received when colliding with a vehicle.
    "lane_centering_cost": 40, #The cost associated with keeping the vehicle in the center of the lane. 
    # This can be used to encourage the vehicle to stay centered in its lane.
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

test_folder = "StableBaselines_Part3/test_2"
model_folder = "StableBaselines_Part3/test_2/model"
video_folder = "StableBaselines_Part3/test_2/videos"

if __name__ == "__main__":
    n_cpu = 6
    batch_size = 64
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={"config": config})
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log=test_folder,
    )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save(model_folder)
        del model

    # Visualisation par video
    visualisation(model_folder, video_folder)