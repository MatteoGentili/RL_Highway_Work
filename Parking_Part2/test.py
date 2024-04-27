from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import datetime

from torch.utils.tensorboard import SummaryWriter

""""

A LIRE TOUS LES COMMENTAIRES JE T'AI MIS DES TRUCS À CHANGER ET DES TRUCS À AJOUTER, JE T'AI MIS DES EXEMPLES DE CE QUE TU PEUX METTRE DANS TES TENSORBOARD,ETC....
Y a deja pas mal de taffe il fuat faire les différents tests, en gros il faut faire une liste avec les param que tu veux tester, puis faire une boucle for pour les tester,
et à chaque fois changer le nom du fichier pour que les courbes soient de couleurs différentes, et que tu puisses voir les résultats de chaque test.


"""



"""
hidden_space1 = 16  # Nothing special with 16, feel free to change 
Faut changer ça pas laisser ces coms avant de push, je sens qu'il regardera bien en détail le code. 

############################################################################################################
 # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

Pourquoi Tanh voire si pas mieux LeakyReLU ou juste ReLU ! 

############################################################################################################

self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

ici je pense que le learning rate est good, sur tous les tests que j'ai pu faire c'était ça que je trouvais le plus sympa, à voir pour des tests.
Gamma je mettrais 0.995, plus c'est  grand plus c'est long à apprendre, mais plus c'est précis, à voir pour des tests.
Eps je mettrais [1e-8, 1e-7, 5e-6, 1e-6, 5e-5] à tester, je pense que 1e-6 est bon, mais à voir. AU moins ça add du contenu ! 

############################################################################################################

    def update(self):
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0

        loss à 0 dans le update, weird non ? 

############################################################################################################
        

Je t'ai mis le tensorBoard pour l'appeler ici c'est dans le terminal : tensorboard --logdir=runs_H
Ajout du break pour éviter de crasher, et ajout de la sauvegarde du meilleur modèle.
Le truc ou j'ai pas le temps de tester faire unnreward positif plus il s'approche de la case bleu ça doit etre possible, puis ensuite si il est pas parallèle enlever des points aussi 



"""


plt.rcParams["figure.figsize"] = (10, 5)

import highway_env
highway_env.register_highway_envs()

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
        
        
class REINFORCE:
    """REINFORCE algorithm."""

    """
    A CHANGER :     def __init__(self, obs_space_dims: int, action_space_dims: int):

    il faut que tu add les param pour pouvoir faire ta boucle de test ex :

    learningRateList = [X,Y,Z,....]

    for lr in learningRateList:
        le début de la boucle 
                agent = REINFORCE(obs_space_dims, action_space_dims, AJOUT DES PARAM A CHANGER)



    """


    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm .

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        

    

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state_array = np.concatenate([state[key] for key in state.keys()])
        state = torch.tensor(np.array([state_array]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        
# Create and wrap the environment

#config: 
config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False
    },
    "action": {
        "type": "ContinuousAction"
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False
}

# Write the summary
""" 
A toi de ce que tu veux mettre le mieux c'est à chaque test que le nom du fichier changer car sinon ça fait des courbes de mêmes couleurs et c'est pas ce que l'on veut 
et le mieux c'est de mettre le nom du fichier en fonction des paramètres que tu as changé pour le test. 
Exemple: test_1_lr_1e-4_gamma_0.99_eps_1e-6 

"""
def writeSummary(learning_rate, gamma, eps):
    return  (f"runs_H/test{learning_rate}_{gamma}_{eps}")


env = gym.make("parking-v0", render_mode="rgb_array")
env.unwrapped.configure(config)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = 1000  # Total number of episodes
# Observation-space of parking-v0 
obs_space_dims = sum(space.shape[0] for space in env.observation_space.spaces.values())
# Action-space of parking-v0 
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []



for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)
        
        total_reward_per_episode = 0
        best_reward = 0

        done = False
        while not done:
            "render the environment if you want to see the agent in action"
            wrapped_env.render()
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            total_reward_per_episode += reward

            """ Print info tu verras y a des trucs intéressants dedans"""
            #print(info)

            """
            Pour ma part je préfère break, et surtout d'avbord mettre reward négatif comme ça il prend encore plus conscience qu'il faut pas crash
            """
            if info['crashed']:
                break
            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        # Ajout de la condition pour sauvegarder le meilleur modèle
        if total_reward_per_episode > best_reward:
            best_reward = total_reward_per_episode
            print("Seed:", seed, "Episode:", episode, "Best Reward:", best_reward)
            "save the model"
            torch.save(agent.net.state_dict(), f"parking_model{best_reward}.pth")

        if episode % 100 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            # ajout du temps pour que tu  puisses voir l'évolution de l'apprentissage
            print(f"{datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} Episode:", episode, "Average Reward:", avg_reward)
        """
        Voilà comment faire un writer je te laisse faire de même pour les autres variables que tu veux voir dans tensorboard
        
        """
        writer.add_scalar("Reward", total_reward_per_episode, episode)


    rewards_over_seeds.append(reward_over_episodes)