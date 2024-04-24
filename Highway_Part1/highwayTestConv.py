########################################################
################        IMPORT          ################
########################################################

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

########################################################
################        CONFIG          ################
########################################################

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
    "duration": 10,  # [s]
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
    "simulation_frequency": 3,  # [Hz]
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

########################################################

env = gym.make("highway-v0", render_mode="rgb_array")
env.unwrapped.configure(config)
env.reset()
initial_state = env.reset()[0].flatten()

########################################################

########################################################
################        DQN          ###################
########################################################

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

########################################################

########################################################
########        REPLAY BUFFER + ACTIONS         ########
########################################################

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
    
########################################################

########################################################
################        REWARD          ################
########################################################

def calculate_reward(reward, info, time_step):
    # PARAMETERS INITALISATION : 
    time_step_reward = 1.1
    head = env.unwrapped.vehicle.heading
    collisionFlag = False


    ### In Order to always have the car in the right position
    if (head > np.pi/2 and head < 3*np.pi/2) or (head < -np.pi/2 and head > -3*np.pi/2):
        reward -= 50

    on_road_reward = info['rewards']['on_road_reward']
    speed_reward = info['speed']
    collision_reward = info['rewards']['collision_reward']

    ### Car always on the Road, if not break and reward = -100 
    if on_road_reward > 0:
        reward += on_road_reward*3
    else :
        reward -= 200

    ### In order to have a car that changes lane if necessary and not create a collision with other cars.
    if collision_reward != 0:
        collisionFlag = True
        reward -= 200

    ### In order to have a car with a good speed
    """
    During the tests I saw that the car tried to but at low speed in order to avoid collision. 
    """
    if  float(speed_reward) <= 20 :
        reward -= 30
    elif float(speed_reward) > 50 :
        reward -= 5

    reward += time_step ** time_step_reward
    
    # print('time step reward : ', time_step ** time_step_reward)


    return reward, on_road_reward, speed_reward, collisionFlag



########################################################
################         CURVES         ################
########################################################

def writeSummary(epsilon, epsilon_decay, epsilon_min, batch_size, gamma, lr):
    return  (f"runs_HighwayPart1/test2")

########################################################
##############        PARAMETERS          ##############
########################################################

## DQN : 
lrList = [5e-4]
gammaList = [0.99]

lr = lrList[0]
gamma = gammaList[0]


## Rewards + Curves 
best_reward = 0
offRoad = 0 

totalRewardList = []
offRoadList = []


########################################################
#############        TRAIN + EVAL          #############
########################################################

#######################
TRAIN = True #########
#######################

if TRAIN :
    # TRAIN CONFIG
    policy_net.train()
    target_net.train()

    ########################################################
    ##############        DQN PARAMS.          #############
    ########################################################
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(15000)
    epsilon = 1
    epsilon_decay = 0.99975 # A tester : 0.99975
    epsilon_min = 0.001
    batch_size = 128
    gamma = gamma
    num_episodes = 100_000
    writer = SummaryWriter(writeSummary(epsilon, epsilon_decay, epsilon_min, batch_size, gamma, lr))

else :
    LOAD_MODEL = r"saveReward/Test1_best_model_reward_643627.3935895181.pth"
    # EVAL CONFIG
    eval_config = {
        "simulation_frequency": 50, 
    }
    env.unwrapped.configure(eval_config)
    env.reset()

    if LOAD_MODEL is not None :
        print("Model Loading ...")
        policy_net.load_state_dict(torch.load(LOAD_MODEL))
        policy_net.eval()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    
    else : 
        policy_net.eval()
        target_net.eval()

    ########################################################
    ##############        DQN PARAMS.          #############
    ########################################################


    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(15000)
    epsilon = 0
    epsilon_decay = 0
    epsilon_min = 0
    batch_size = 64
    gamma = gamma
    num_episodes = 200
    writer = SummaryWriter(writeSummary(epsilon, epsilon_decay, epsilon_min, batch_size, gamma, lr))


########################################################
###############        SIMULATION         ##############
########################################################
print("Training started.")
# print(dir(env.unwrapped.vehicle))

for episode in range(num_episodes):

    state = env.reset()[0].flatten()
    start_time = time.time()

    #### PARAMETERS : 
    total_reward = 0
    totalTimePerEpisode = 0
    time_step = 0
    speedCarInfo = 0
    speedValuePerEpisode = 0 

    while True:

        if TRAIN:
            if episode % 500 == 0 and episode > 8000 : env.render()
            env.render()
            # pass

        if not TRAIN:
            env.render()
        print("head of the car : ", env.unwrapped.vehicle.heading)
        
        #### NEXT STATE : 
        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
        action = select_action(state_tensor, policy_net, epsilon).item()
        next_state, reward, done, truncated, info = env.step(action)
        print("off road reward : ", info['rewards']['on_road_reward'])
        next_state = next_state.flatten()

        #### REWARD CALCULATED : 
        reward, on_road_reward, speed_reward, collisionFlag = calculate_reward(reward, info, time_step)

        if on_road_reward <= 0:
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


        total_reward += reward
        totalRewardList.append(total_reward)
        time_step += 1

        writer.add_scalar("TotReward", total_reward, episode)
        speedCarInfo += float(info['speed'])     
       
        if done or on_road_reward <= 0:
            break

    totalTimePerEpisode += time_step
    speedValuePerEpisode = speedCarInfo/time_step
    writer.add_scalar("TimeStep", totalTimePerEpisode, episode)
    writer.add_scalar("SpeedValue",speedValuePerEpisode,  episode)

    if episode % 2000 == 0:  # Update the target network
        target_net.load_state_dict(policy_net.state_dict())
        print(f"{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} OffRoad per 2000 : ", offRoad)
        offRoad = 0
    
    if episode % 100 == 0:
        print(f"{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} Moyenne Reward, episode {episode}", np.mean(totalRewardList[-99:]))
        
    if total_reward > best_reward:
        best_reward = total_reward
        end_time = time.time()
        torch.save(policy_net.state_dict(), f"saveReward/test2_best_model_reward_{total_reward}.pth")
        print("--------------------------------------------------------------")
        print(f"\n{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} Episode n° {episode}: Best Reward = {total_reward}")
        print(f"Informations on this Episode : ")
        print(f"+ Number of steps : {totalTimePerEpisode}")
        print(f"+ Time of the episode : {end_time - start_time} s")
        print(f"+ Speed During the episode : {speedValuePerEpisode}")
        if on_road_reward <= 0 :
            print(f"+ Does the crash (off Road ): Yes")
        else : 
            print(f"+ Does the crash (off Road ): No")
        if collisionFlag :
            print("+ Collision Flag : True")
        else :
            print("+ Collision Flag : False")
        print("\n--------------------------------------------------------------")


    writer.add_scalar("Epsilon", epsilon, episode)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    
    
print("Training completed.")

writer.close()