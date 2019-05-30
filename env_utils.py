
import numpy as np
import gym
import my_envs
from warnings import warn

def init_env(config):
    env_name = config['environment']

    if env_name == 'Pendulum-v0':
        env = gym.make('Pendulum-v0')
    elif env_name == 'my_Pendulum-v0':
        env = gym.make('my_Pendulum-v0')
    elif env_name == 'my_Cartpole-v0':
        env = gym.make('my_Cartpole-v0')
    elif env_name == 'my_Acrobot-v0':
        env = gym.make('my_Acrobot-v0')
    else:
        env = None
        print("ERROR: Invalid environment {:s} specified in config file".format(env_name))

    return env

def normalize_pendulum_state(state):
    # normalize theta and thetadot to values between -1 and 1
    #theta_norm     = -1.0 + (state[0] + np.pi)/np.pi
    theta_norm     = -1.0 + (state[0] + 2*np.pi)/(2*np.pi)
    theta_dot_norm = -1.0 + (state[1] + 6.0)/6.0
    
    return np.array([theta_norm, theta_dot_norm])

def normalize_pendulum_action(action):
    norm_action = action / 3.0
    
    return norm_action
    
def normalize_pendulum_state2(state):
    norm_state = np.zeros(3)
    norm_state[0] = (state[0] + 1.0) / 2.0
    norm_state[1] = (state[1] + 1.0) / 2.0
    norm_state[2] = (state[2] + 8.0) / 16.0
    
    return norm_state

def normalize_pendulum_action2(action):
    norm_action = action / 2.0
    
    return norm_action

def normalize_pendulum_reward2(reward):
    norm_reward = (reward + 16.2736044) / 16.2736044
    
    return norm_reward
    
def normalize_cartpole_state(state):
    pass
    
def normalize_cartpole_action(action):
    pass
    
def normalize_acrobot_state(state):
    pass
    
def normalize_acrobot_action(action):
    pass
        
def get_normalizers(env_name):
    if env_name == 'Pendulum-v0':
        return normalize_pendulum_state2, normalize_pendulum_action2, normalize_pendulum_reward2
    elif env_name == 'my_Pendulum-v0':
        return normalize_pendulum_state, normalize_pendulum_action
    elif env_name == 'my_Cartpole-v0':
        return normalize_cartpole_state, normalize_cartpole_action
    elif env_name == 'my_Acrobot-v0':
        return normalize_acrobot_state, normalize_acrobot_action

        