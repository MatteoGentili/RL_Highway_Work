import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(DQN, self).__init__()
        layers = [nn.Linear(state_dim, hidden_layers[0])]
        layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
        layers.append(nn.Linear(hidden_layers[-1], action_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
    
class ParameterConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class HighwayEnvManager:
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.config_environment()

    def config_environment(self):
        config = {
            "observation": {
                "type": "OccupancyGrid",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "grid_size": [[-20, 20], [-20, 20]],
                "grid_step": [5, 5],
                "absolute": False,
            },
            "action": {
                "type": "DiscreteAction",
            },
            "lanes_count": 3,
            "vehicles_count": 10,
            "duration": 20,  # [s]
            "initial_spacing": 0,
            "collision_reward": -100,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 5,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.1,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": -1,
            "reward_speed_range": [
                20,
                30,
            ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
            "simulation_frequency": 5,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": True,
            "render_agent": True,
            "offscreen_rendering": False,
            "disable_collision_checks": True,
        }
        self.env.unwrapped.configure(config)
        
    def reset_environment(self):
        return self.env.reset()[0].flatten()

    def step(self, action):
        return self.env.step(action)

    def get_heading(self):
        return self.env.unwrapped.vehicle.heading

def select_action(state, policy_net, epsilon, action_dim, device):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], dtype=torch.long, device=device)

def train_dqn(episodes):
    writer = SummaryWriter(f"runs_H/TEST")
    env_manager = HighwayEnvManager("highway-fast-v0")
    state_dim = env_manager.reset_environment().shape[0]
    action_dim = env_manager.env.action_space.n
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer(10000)

    for episode in range(episodes):
        state = env_manager.reset_environment()
        total_reward = 0
        while True:
            action = select_action(torch.tensor([state], dtype=torch.float32, device=device), policy_net, 1.0, action_dim, device)
            next_state, reward, done, truncated, info= env_manager.step(action.item())
            next_state = next_state.flatten()
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        writer.add_scalar("Total Reward", total_reward, episode)
        if episode % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    writer.close()

if __name__ == "__main__":
    paramConf = ParameterConfig(
        env_name="highway-fast-v0",
        learning_rate=0.0005,
        epsilon=0.1,
        hidden_layers=[64, 64],
        batch_size=128,
        num_episodes=50000
    )
    train_dqn(30000)
