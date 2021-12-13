'''
RL Project
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
from REINFORCEwBaseline import *

# Create the Mountain Car Environment
env = gym.make('MountainCar-v0')

# Show some information about states and actions
print('Environment Details')
print('-' * 50)

state_space = env.observation_space
print('state_space.low: {}'.format(state_space.low))
print('state_space.high: {}'.format(state_space.high))
n_features = state_space.shape[0]

action_space = env.action_space
print('action_space.n: {}'.format(action_space.n))
n_actions = action_space.n

# Run REINFORCE on MountainCar
num_runs = 2
iterations = 500
episode_lengths_history = np.zeros((num_runs, iterations))
for run_no in range(num_runs):
    print('Run No.: {}'.format(run_no))
    # theta, episode_lengths = reinforce(env, sigma=0.1, gamma=1., order=10, iterations=iterations, hidden_features=64, alpha=0.005)
    theta, W, episode_lengths = os_actor_critic(env, sigma=1., gamma=1., order=10, iterations=iterations, hidden_features_theta=6, hidden_features_w=3, \
                               alpha_theta=0.005, alpha_w=0.005, render=False)
    
    # Remember episode lengths
    episode_lengths_history[run_no] = episode_lengths.copy()
    
# Calculate the average
episode_lengths_avg = np.mean(episode_lengths_history, axis=0)
episode_lengths_std = np.std(episode_lengths_history, axis=0)

# Plot the graph
plt.plot(np.arange(iterations), episode_lengths_avg)
plt.fill_between(np.arange(iterations), episode_lengths_avg - episode_lengths_std / 2, \
                 episode_lengths_avg + episode_lengths_std / 2, color='blue', alpha=0.2)
plt.title('Avg. Episode Length v/s No. Episodes')
plt.xlabel('No. Episodes')
plt.ylabel('Avg. Episode Length')
plt.show()

# Display the policy
print('Theta:')
print(theta)
# print('\nW:')
# print(W)

# Run the learnt policy a few times
to_dipslay = 3
for i in range(to_dipslay):
    generate_episode(env, theta, sigma=1., order=10, k=2, render=True)

# Terminate the environment
env.close()