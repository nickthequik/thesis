
import numpy as np

def get_discounted_returns(rewards, gamma):
    disc_returns = np.zeros(rewards.size)
    disc_returns[-1] = rewards[-1]
    for i in reversed(range(rewards.size-1)):
        disc_returns[i] = rewards[i] + gamma * disc_returns[i+1]
    
    return disc_returns

def moving_average(data, window):
    if window == 0 or window >= data.size:
        # average for max possible samples
        means = np.zeros(data.size)
        sum = 0
        for i in range(data.size):
            sum += data[i]
            means[i] = sum / (i+1)
    else:
        # central moving average
        means = np.zeros(data.size)
        for i in range(data.size):
            begin = max(0,i-(window-1)//2)
            end   = min(data.size, i+(window-1)//2 + 1)
            means[i] = np.mean(data[begin:end])

    return means

def standard_dev(data, window):
    # corrected std dev estimator
    if window == 0 or window >= data.size:
        stddevs = np.zeros(data.size)
        for i in range(data.size):
            stddevs[i] = np.std(data[0:i+1])
    else:
        stddevs = np.zeros(data.size)
        x = (window - 1) // 2
        for i in range(data.size):
            begin = max(0,i-x)
            end   = min(data.size, i+x)
            stddevs[i] = np.std(data[begin:end+1])

    return stddevs

def concat_episodes(episodes_data):
    num_eps    = episodes_data.num_eps
    timesteps  = episodes_data.timesteps
    state_dim  = episodes_data.state_dim
    action_dim = episodes_data.action_dim

    concat_states   = np.zeros((state_dim, timesteps * num_eps))
    concat_actions  = np.zeros((action_dim, timesteps * num_eps))
    concat_rewards  = np.zeros(timesteps * num_eps)

    for i in range(num_eps):
        episode = episodes_data.episode_list[i]
        states  = episode.states
        actions = episode.actions
        rewards = episode.rewards

        concat_states[:, i*timesteps:(i+1)*timesteps]  = states
        concat_actions[:, i*timesteps:(i+1)*timesteps] = actions
        concat_rewards[i*timesteps:(i+1)*timesteps]    = rewards

    return concat_states, concat_actions, concat_rewards

# This would have to be updated to work for action space with dimension > 1
def discretize_actions(low, high, num_actions):
    actions = np.linspace(low[0], high[0], num_actions)
    print("Actions:")
    print(actions)
    return actions.reshape(actions.size, 1)
