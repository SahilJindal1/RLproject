'''
Rl Project
'''
import numpy as np
from nn_utils import *
from itertools import product

# Reuse combinations for fourier basis to save memory
combinations = None
combinations_generated = False

def policy_forward(features, theta, sigma=0.01):
    # Unpack parameters
    theta_w1 = theta['w1']
    theta_b1 = theta['b1']
    # theta_w2 = theta['w2']
    # theta_b2 = theta['b2']
    
    # Generate scores for each action
    scores, cache_1 = affine_forward(features, theta_w1, theta_b1)
    # out_2, cache_2 = relu_forward(out_1)
    # scores, cache_3 = affine_forward(out_2, theta_w2, theta_b2)    
    
    # cache = (cache_1, cache_2, cache_3, sigma)
    cache = (cache_1, sigma)
    
    # Generate probabilities for each action
    probs = softmax_forward(sigma * scores)
    
    return probs, cache

def policy_backward(action, probs, cache):
    # Unpack cache
    # cache_1, cache_2, cache_3, sigma = cache
    cache_1, sigma = cache
    
    # Begin backpropagation
    dscores = softmax_backward(action, probs) * sigma
    # dout_2, dtheta_w2, dtheta_b2 = affine_backward(dscores, cache_3)
    # dout_1 = relu_backward(dout_2, cache_2)
    dfeatures, dtheta_w1, dtheta_b1 = affine_backward(dscores, cache_1)
    
    dtheta = {}
    dtheta['w1'] = dtheta_w1
    dtheta['b1'] = dtheta_b1
    # dtheta['w2'] = dtheta_w2
    # dtheta['b2'] = dtheta_b2
    
    return dfeatures, dtheta   

def get_action(features, theta, sigma):
    # Calculate probabilities
    probs, _ = policy_forward(features, theta, sigma)
    
    # Sample an action based on probabities
    n_actions = probs.shape[0]
    action = np.random.choice(n_actions, 1, p=probs)[0]
    
    return action

def get_features(env, state, order=5, k=4):
    global combinations, combinations_generated
    
    # Clipped State
    # low = np.array([-2.4, -5.0, -0.21, -2.5])         # CartPole: Uncomment these is using Cart Pole
    # high = np.array([2.4, 5.0, 0.21, 2.5])
    
    low = env.observation_space.low                   # Comment these is using Cart Pole
    high = env.observation_space.high
    clipped = np.clip(state, low, high)
    
    # Normalize the state
    normalized = (clipped - low) / (high - low)
    
    # Compute Fourier-Basis features
    if combinations_generated == False:
        combinations = np.array(list(product(range(order + 1), repeat=k)))
        combinations_generated = True
        
    features = np.cos(np.pi * np.matmul(combinations, np.expand_dims(normalized, axis=1)))
    
    
    return np.squeeze(features)

def generate_episode(env, theta, sigma, order, k, render=False):
    # Reset the position
    S_t = env.reset()
    done = False
    
    # Initialize the lists store observations from the episode
    states = [S_t]
    actions = []
    rewards = []
    
    # Loop till episode ends
    while not done:
        # Render if requested
        if render == True:
            env.render()
        
        # Get an action depending on the policy
        features = get_features(env, S_t, order, k)
        a_t = get_action(features, theta, sigma)
        
        # Perform the action
        S_tp1, R_t, done, _ = env.step(a_t)
        
        # Store the observations
        states.append(S_tp1)
        actions.append(a_t)
        rewards.append(R_t)
        
        # Update state
        S_t = S_tp1
        
    return states, actions, rewards
    

# REINFORCE for continuous state, discrete action environments
def reinforce(env, sigma=1., order=3, gamma=0.999, hidden_features=10, iterations=2000, alpha=0.0008, print_every=20):
    # State space length
    s_space = env.observation_space

    k = s_space.shape[0] 
    n_features = (order + 1) ** k # Fourier Basis Features
    
    # Get the number of actions
    a_space = env.action_space
    n_actions = a_space.n
    
    # Initialize the policy parameters
    theta_w1 = np.random.randn(n_features, n_actions)
    theta_b1 = np.zeros((n_actions,))
    # theta_w2 = np.random.randn(hidden_features, n_actions)
    # theta_b2 = np.zeros((n_actions,))
    # theta = {'w1': theta_w1, 'b1': theta_b1, 'w2': theta_w2, 'b2': theta_b2}
    theta = {'w1': theta_w1, 'b1': theta_b1}
    
    # Store episode lengths for plotting
    episode_lengths = []
    
    # Loop over episodes
    for iteration in range(iterations):
        # Generate episodes
        states, actions, rewards = generate_episode(env, theta, sigma, order, k)
    
        # Loop for each step of the episode
        T = len(actions)
        episode_lengths.append(T) # Store this for plotting
        for t in range(T):
            # Calculate the discounted return
            G_t = 0
            for k in range(t, T):
                G_t += gamma ** (k - t) * rewards[k]
                
            # Calculate the gradient
            features = get_features(env, states[t], order, k)
            probs, cache = policy_forward(features, theta, sigma)
            dfeatures, dtheta = policy_backward(actions[t], probs, cache)
            
            # Update policy parameters
            theta['w1'] += alpha * G_t * dtheta['w1']
            theta['b1'] += alpha * G_t * dtheta['b1']
            # theta['w2'] += alpha * G_t * dtheta['w2']
            # theta['b2'] += alpha * G_t * dtheta['b2']
            
        # Display episode length every few episodes
        if iteration % print_every == 0:
            print('Iteration: {}\tEpisode Length: {}'.format(iteration, T))
            
    return theta, np.array(episode_lengths)
            

def value_forward(features, W, bias=True):
    # Unpack parameters
    W_w1 = W['w1']
    W_b1 = W['b1']
    # W_w2 = W['w2']
    # W_b2 = W['b2']
    
    # Generate scores for each action
    value, cache_1 = affine_forward(features, W_w1, W_b1)
    # out_2, cache_2 = relu_forward(out_1)
    # value, cache_3 = affine_forward(out_2, W_w2, W_b2)    
    
    # cache = (cache_1, cache_2, cache_3)
    cache = cache_1
    
    return value, cache

def value_backward(cache):
    # Unpack cache
    # cache_1, cache_2, cache_3 = cache
    cache_1 = cache
    
    # Begin backpropagation
    dvalue = np.ones((1,))
    # dout_2, dW_w2, dW_b2 = affine_backward(dvalue, cache_3)
    # dout_1 = relu_backward(dout_2, cache_2)
    dfeatures, dW_w1, dW_b1 = affine_backward(dvalue, cache_1)
    
    dW = {}
    dW['w1'] = dW_w1
    dW['b1'] = dW_b1
    # dW['w2'] = dW_w2
    # dW['b2'] = dW_b2
    
    return dfeatures, dW  



# One-Step Actor Critic
def os_actor_critic(env, sigma=1., order=5, gamma=0.99, iterations=2000, \
                    hidden_features_theta=6, hidden_features_w=10, \
                    alpha_w=0.05, alpha_theta=0.0008, print_every=20, \
                    render=True):
    # State space length
    s_space = env.observation_space
    k = s_space.shape[0] 
    n_features = (order + 1) ** k # Fourier Basis Features
    
    # Get the number of actions
    a_space = env.action_space
    n_actions = a_space.n
    
    # Initialize the policy parameters
    theta_w1 = np.random.randn(n_features, n_actions)
    theta_b1 = np.zeros((n_actions,))
    # theta_w2 = np.random.randn(hidden_features_theta, n_actions) / 1000.
    # theta_b2 = np.zeros((n_actions,))
    # theta = {'w1': theta_w1, 'b1': theta_b1, 'w2': theta_w2, 'b2': theta_b2}
    theta = {'w1': theta_w1, 'b1': theta_b1}
    
    # Initialize the value function parameters
    W_w1 = np.random.randn(n_features, 1)
    W_b1 = np.zeros((1,))
    # W_w2 = np.random.randn(hidden_features_w, 1)
    # W_b2 = np.zeros((1,))
    # W = {'w1': W_w1, 'b1': W_b1, 'w2': W_w2, 'b2': W_b2}
    W = {'w1': W_w1, 'b1': W_b1}
    
    # Store episode lengths for plotting
    episode_lengths = []
    
    # Loop over episodes
    for iteration in range(iterations):
        # Generate episodes
        S_t = env.reset()
        done = False
        
        T = 0
        while not done:
            if render == True:
                env.render()
            
            # Get an action depending on the policy
            f_t = get_features(env, S_t, order, k)
            a_t = get_action(f_t, theta, sigma)
            
            # Perform the action
            S_tp1, R_t, done, _ = env.step(a_t)
        
            # Calculate the TD-error
            v_t, cache_W = value_forward(f_t, W)
            
            f_tp1 = get_features(env, S_tp1, order, k)
            v_tp1, _ = value_forward(f_tp1, W)
            
            delta = R_t + gamma * v_tp1 - v_t
            
            # Calculate gradient wrt W
            _, dW = value_backward(cache_W)
            
            # Update critic  
            W['w1'] += alpha_w * delta * dW['w1']
            W['b1'] += alpha_w * delta * dW['b1']
            # W['w2'] += alpha_w * delta * dW['w2']
            # W['b2'] += alpha_w * delta * dW['b2']
            
            # Calculate the gradient wrt theta
            probs, cache = policy_forward(f_t, theta, sigma)
            _, dtheta = policy_backward(a_t, probs, cache)
            
            # print(delta, W['w1'].max())
            
            # Update policy parameters
            theta['w1'] += alpha_theta * delta * dtheta['w1']
            theta['b1'] += alpha_theta * delta * dtheta['b1']
            # theta['w2'] += alpha_theta * delta * dtheta['w2']
            # theta['b2'] += alpha_theta * delta * dtheta['b2']
            
            # Update the state
            S_t = S_tp1
        
            # Increment episode length
            T += 1
            
        # Store episode lengths
        episode_lengths.append(T) # Store this for plotting
        
        # if (iteration + 1) % 100 == 0:
        #     sigma += 0.01
        # Decay alpha every few iterations
        # if (iteration + 1) % decay_step == 0:
        #     alpha_w *= alpha_decay
        #     alpha_theta *= alpha_decay
            
        # Display episode length every few episodes
        if iteration % print_every == 0:
            print('Iteration: {}\tEpisode Length: {}'.format(iteration, T))
            
    return theta, W, np.array(episode_lengths)          
            
            
            
            
            
            
            
            
            
            
            