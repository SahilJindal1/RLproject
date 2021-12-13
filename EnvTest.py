# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:55:04 2021

@author: chira
"""
import gym

# Create the Environment
env = gym.make('CartPole-v0')

# Print some stats
s_space = env.observation_space
a_space = env.action_space

print('s_space.low: {}'.format(s_space.low))
print('s_space.high: {}'.format(s_space.high))
print('s_space.high.shape: {}'.format(s_space.high.shape))
print('a_space.n: {}'.format(a_space.n))

# Run some episodes with random actions
num_episodes = 10
for e in range(num_episodes):    
    # Reset the environment
    S_t = env.reset()
    done = False
    
    # Loop till episode is not done
    G = 0
    while not done:
        env.render()
        
        a_t = env.action_space.sample()
        
        S_tp1, R_t, done, _ = env.step(a_t)
        
        S_t = S_tp1
        
        G += R_t
        
    print('Episode: {}\tReturn: {}'.format(e, G))
    
        
# Cleanup
env.close()