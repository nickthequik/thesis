
import numpy as np
import gym
import my_envs

def init_env(config):
    env_name = config['environment']

    if env_name == 'my_Pendulum-v0':
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
    norm_state = np.zeros(state.size)
    
    # 2 state - theta, thetadot
    # wrapped_theta = (((state[0]+np.pi) % (2*np.pi)) - np.pi)
    # norm_state[0] = -1.0 + (wrapped_theta + np.pi)/np.pi
    # norm_state[1] = -1.0 + (state[1] + 8.0) / 8.0
    
    # 3 state - cos(theta), sin(theta), thetadot
    norm_state[0] = state[0]
    norm_state[1] = state[1]
    norm_state[2] = -1.0 + (state[2] + 8.0) / 8.0
    
    return norm_state

def normalize_pendulum_action(action):
    norm_action = action / 2.0
    
    return norm_action

def normalize_pendulum_return(returns, gamma):
    if gamma == 1:
        norm_factor = 16.2736044 * returns.size
        # norm_factor = 26.8 * returns.size
    else:
        norm_factor = 16.2736044 * (1-gamma**returns.size)/(1-gamma)
        # norm_factor = 18.2696044 * (1-gamma**returns.size)/(1-gamma)
        # norm_factor = (1-gamma**returns.size)/(1-gamma)
        # norm_factor = 26.8 * (1-gamma**returns.size)/(1-gamma)
        
    return returns / norm_factor
    
def normalize_cartpole_state(state):
    pass
    
def normalize_cartpole_action(action):
    pass
    
def normalize_acrobot_state(state):
    pass
    
def normalize_acrobot_action(action):
    pass
        
def get_normalizers(env_name):
    if env_name == 'my_Pendulum-v0':
        return normalize_pendulum_state, normalize_pendulum_action, normalize_pendulum_return
    elif env_name == 'my_Cartpole-v0':
        return normalize_cartpole_state, normalize_cartpole_action
    elif env_name == 'my_Acrobot-v0':
        return normalize_acrobot_state, normalize_acrobot_action

        