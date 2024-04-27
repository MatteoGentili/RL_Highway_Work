from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import os

# Configurations
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

# Define the hyperparameters to test
learning_rates = [1e-3, 5e-4, 1e-4]
gammas = [0.9, 0.8]
batch_size = [32, 64, 128]

# Total number of episodes
total_episodes = int(1e5)

if __name__ == '__main__':
    # Loop through all hyperparameter combinations
    for lr in learning_rates:
        for gamma in gammas:
            for bs in batch_size:
                # Create a folder name based on hyperparameters
                folder_name = f"lr_{lr}_gamma_{gamma}_bs_{bs}"
                
                # Set the random seed
                set_random_seed(42)
                
                # Create the environment
                env = make_vec_env(
                    "racetrack-v0", 
                    n_envs=6, 
                    vec_env_cls=SubprocVecEnv, 
                    env_kwargs={"config": config})
                
                # Create the model
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    batch_size = bs,
                    n_epochs=10,
                    learning_rate=lr, 
                    gamma=gamma, 
                    verbose=2, 
                    tensorboard_log=folder_name)
                
                # Train the model
                print(f"Training model for lr={lr}, gamma={gamma}, bs={bs}")
                model.learn(total_timesteps=total_episodes)
                print(f"Training completed for lr={lr}, gamma={gamma}, bs={bs}")
                
                # Save the best model
                rewards = [info['r'] for info in model.ep_info_buffer]  # Extraction des récompenses
                best_reward = max(rewards) if rewards else None  # Trouver la récompense maximale

                # Construire le chemin absolu pour enregistrer le modèle
                save_path = os.path.join(folder_name, f"best_model_{folder_name}_reward_{best_reward}")

                # Enregistrer le modèle
                model.save(save_path)
               

                print(f"------ Finished training for flr={lr}, gamma={gamma}, bs={bs} ------")
