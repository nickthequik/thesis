
import numpy as np

from misc_utils import moving_average, standard_dev, get_discounted_returns

class EpisodeList:
    def __init__(self, state_dim, action_dim, timesteps, window):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_list = []
        self.num_eps = 0
        self.timesteps = timesteps
        self.window = window

    def append(self, episode):
        self.num_eps += 1
        self.episode_list.append(episode)

    def return_MA(self):
        if self.window == 0:
            N = self.num_eps
        else:
            begin = max(0, self.num_eps - self.window)
            N = self.num_eps - begin

        returns = [self.episode_list[self.num_eps-i-1].retrn for i in range(N)]
        return np.mean(returns)

class Episode:
    def __init__(self, state_dim, action_dim, timesteps):
        self.states   = np.zeros((state_dim, timesteps))
        self.actions  = np.zeros((action_dim, timesteps))
        self.rewards  = np.zeros(timesteps)
        
        # normalized states and action for training neural networks
        self.norm_states  = np.zeros((state_dim, timesteps))
        self.norm_actions = np.zeros((action_dim, timesteps))
        self.norm_rewards = np.zeros(timesteps)

    def calculate_stats(self, ep_length, gamma):
        self.length = ep_length
        self.retrn = np.sum(self.rewards)
        #self.disc_retrn = get_discounted_returns(self.rewards, gamma)
        self.disc_retrn = get_discounted_returns(self.norm_rewards, gamma)
        
        return self.retrn

def store_episodes_data(data_dir, train_data, loss=None, all=False):
    num_eps    = train_data.num_eps
    timesteps  = train_data.timesteps

    # store all episodes' data including states, actions, and rewards
    if all:
        state_dim  = train_data.state_dim
        action_dim = train_data.action_dim

        rpe = state_dim + action_dim + 1
        row_dim = rpe * num_eps
        stacked_data  = np.zeros((row_dim, timesteps))
        episode_length = np.zeros(num_eps)

        for i in range(num_eps):
            episode = train_data.episode_list[i]
            states  = episode.states
            actions = episode.actions
            rewards = episode.rewards

            stacked_data[i*rpe:(i+1)*rpe, :] = np.vstack((states, actions, rewards))
            episode_length[i] = episode.length

    # store only episodes' reward data
    else:
        stacked_data  = np.zeros((num_eps, timesteps))
        episode_length = np.zeros(num_eps)

        for i in range(num_eps):
            episode = train_data.episode_list[i]

            stacked_data[i,:] = episode.rewards
            episode_length[i] = episode.length


    if loss is None:
        np.savez_compressed(data_dir + '/train_data.npz',
                            stacked_data=stacked_data,
                            episode_length=episode_length)
    else:
        np.savez_compressed(data_dir + '/train_data.npz',
                            stacked_data=stacked_data,
                            episode_length=episode_length,
                            loss=loss)

def get_episodes_stats(config, episodes_data, elapsed_time):
    window = config['window']
    num_eps = episodes_data.num_eps
    returns = np.array([episodes_data.episode_list[i].retrn for i in range(num_eps)])

    returns_MA = moving_average(returns, window)
    returns_SD = standard_dev(returns, window)

    episode_stats = {'returns': returns,
                     'returns_MA': returns_MA,
                     'returns_SD': returns_SD,
                     'elapsed_time': elapsed_time,
                     'final_avg_rew': returns_MA[-1],
                     'final_sd_rew': returns_SD[-1]}

    return episode_stats

def store_episodes_stats(data_dir, train_data, episode_stats):
    elapsed_time  = episode_stats['elapsed_time']
    final_avg_rew = episode_stats['final_avg_rew']
    final_sd_rew  = episode_stats['final_sd_rew']
    with open(data_dir + '/stats.txt', mode='w') as f:
        f.write('Training Time = {:g} seconds\n'.format(elapsed_time))
        f.write('Final Average Reward = {:g}\n'.format(final_avg_rew))
        f.write('Final SD Reward = {:g}\n'.format(final_sd_rew))

def print_episode_stats(retrn, returns_MA, num_eps, ep_index, print_freq):
    if (ep_index+1) % print_freq == 0:
        print("\nEPISODE {:d}/{:d}".format(ep_index+1, num_eps))
        print("\tReturn:  {:g}".format(retrn))
        print("\tAvg Return: {:g}".format(returns_MA))
