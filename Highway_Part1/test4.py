import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment Setup
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
    "duration": 5,  # [s]
    "initial_spacing": 0,
    "collision_reward": -100,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 5,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.5,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0.2,
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




env = gym.make("highway-v0", render_mode="rgb_array")
env.unwrapped.configure(config)
env.reset()
initial_state = env.reset()[0].flatten()  # Assuming env.reset() returns state and additional info


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

state_dim = initial_state.shape[0] 
action_dim = env.action_space.n

# Initialize DQN
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)


# Replay Buffer
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
    
# Action Selection Function
def select_action(state, policy_net, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)
    
def calculate_reward(reward):
    head = env.unwrapped.vehicle.heading
    if (head > np.pi/2 and head < 3*np.pi/2) or (head < -np.pi/2 and head > -3*np.pi/2):
        reward = -0.2

    on_road_reward = info['rewards']['on_road_reward']
    speed_reward = info['speed']

    if on_road_reward > 0:
        reward += on_road_reward
    else :
        reward = -50

    if  float(speed_reward) <= 20 :
        reward = -50
    elif float(speed_reward) > 30 :
        reward = -50

    return reward, on_road_reward, speed_reward
    


lrList = [5e-4]
gammaList = [0.99]
best_reward = 0
totalRewardList = []
offRoad = 0 # Nombre de fois où la voiture est sortie de la route

speedAverage = 0

# LOAD_MODEL = None
LOAD_MODEL = r"saveReward\best_model_reward_890.1227488151658.pth"
TRAIN = True
if TRAIN:
    # Configuration pour l'entraînement
    train_config = {
        "simulation_frequency": 5,  # Plus de mises à jour de simulation par seconde
        "policy_frequency": 5,  # L'agent prend une décision chaque seconde
        # Autres paramètres de configuration
    }
    env.unwrapped.configure(train_config)
    env.reset()
else :
    # Configuration pour l'évaluation
    eval_config = {
        "lanes_count": 4,
        "vehicles_count": 100,
        "simulation_frequency": 5,  # Moins de mises à jour pour une simulation plus "lente"
        "policy_frequency": 5,  # L'agent prend une décision toutes les 2 secondes
        # Autres paramètres de configuration
    }
    env.unwrapped.configure(eval_config)
    env.reset()




def writeSummary(epsilon, epsilon_decay, epsilon_min, batch_size, gamma, lr):
    return  (f"runs_H/TRY.01_{epsilon}_epsilon_decay{epsilon_decay}_epsilon_min{epsilon_min}_batch_size{batch_size}_gamma{gamma}_lr{lr}_16layers")



if TRAIN:

    policy_net.train()
    target_net.train()
else :
    if LOAD_MODEL is not None:
        print("Model Loading ...")
        policy_net.load_state_dict(torch.load(LOAD_MODEL))
        policy_net.eval()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    else:
        policy_net.eval()
        target_net.eval()

for gamma in gammaList:
    for lr in lrList:
        # Setup other components (optimizer, replay buffer, etc.)
        if TRAIN:
            optimizer = optim.Adam(policy_net.parameters(), lr=lr)
            memory = ReplayBuffer(15000)
            epsilon = 1
            epsilon_decay = 0.99975 # A tester : 0.99975
            epsilon_min = 0.001
            batch_size = 64
            gamma = gamma
            num_episodes = 200_000
            writer = SummaryWriter(writeSummary(epsilon, epsilon_decay, epsilon_min, batch_size, gamma, lr))
        else:
            optimizer = optim.Adam(policy_net.parameters(), lr=lr)
            memory = ReplayBuffer(15000)
            epsilon = 0
            epsilon_decay = 0.0 
            epsilon_min = 0.00
            batch_size = 64
            gamma = gamma
            num_episodes = 200
            writer = SummaryWriter(writeSummary(epsilon, epsilon_decay, epsilon_min, batch_size, gamma, lr))
        
       
        
        offRoadList = []

        # Training Loop
        for episode in range(num_episodes):

            state = env.reset()[0].flatten()

            time_step_reward = 0.01  # Récompense ajoutée pour chaque pas de temps
            speedAverage = 0
            total_reward = 0
            time_average = 0
            last_action = None

            time_step = 0
            speedCarInfo = 0

            writer.add_scalar("Epsilon", epsilon, episode)

            while True:

                if TRAIN:
                    if episode % 100 == 0:
                        env.render()

                if not TRAIN:
                    env.render()
                

                state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
                action = select_action(state_tensor, policy_net, epsilon).item()
                next_state, reward, done, truncated, info = env.step(action)

                next_state = next_state.flatten()

                reward, on_road_reward, speed_reward = calculate_reward(reward)
                reward += time_step * time_step_reward

                total_reward += reward
                writer.add_scalar("TotReward", total_reward, episode)

                if info['rewards']['on_road_reward'] <= 0:
                    offRoad += 1

                memory.push(state, action, reward, next_state, done)
                state = next_state
                
                if on_road_reward > 0:
                    
                    # Update policy if memory is sufficient
                    if len(memory) > batch_size:
                        states, actions, rewards, next_states, dones = memory.sample(batch_size)
                        states = torch.tensor(states, dtype=torch.float32).to(device)
                        actions = torch.tensor(actions, dtype=torch.long).to(device)
                        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                        dones = torch.tensor(dones, dtype=torch.uint8).to(device)

                        # Double DQN update
                        next_state_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                        next_q_values = target_net(next_states).gather(1, next_state_actions).squeeze(1)
                        expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                        # Loss calculation and backpropagation
                        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        loss = F.mse_loss(current_q_values, expected_q_values)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if on_road_reward <= 0:
                    offRoadList.append(episode)
                    break
                if speed_reward <= 10:
                    break
                

                time_step += 1

                if time_step > 0:
                    speedCarInfo += float(info['speed'])/time_step
                    
                else :
                    speedCarInfo += float(info['speed'])
                writer.add_scalar("Speed", speedCarInfo, episode)

                if done:
                    break

            speedAverage += speedCarInfo
            time_average += time_step
            writer.add_scalar("TimeStep", time_step, episode)


            if episode % 2000 == 0:  # Update the target network
                target_net.load_state_dict(policy_net.state_dict())
                print(f"{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} OffRoad per 2000 : ", offRoad)
                offRoad = 0
            
            if episode % 100 == 0:
                print(f"{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} Moyenne Reward, episode {episode}", np.mean(totalRewardList[-99:]))
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            totalRewardList.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                "save the model"
                torch.save(policy_net.state_dict(), f"saveReward/best_model_reward_{total_reward}.pth")
                print(f"{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} Episode {episode}: Total reward = {total_reward}")
            
        print("Training completed.")
        
writer.close()