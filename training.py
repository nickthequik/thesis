
import numpy as np
from ep_utils   import Episode, EpisodeList, print_episode_stats
from plot_utils import RewardPlotter, TrainingTracker
from env_utils  import get_normalizers

def train_agent(agent, env, config):
    num_eps   = config['episodes']
    timesteps = config['timesteps']
    gamma     = config['gamma']

    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]

    normalize_state, normalize_action, normalize_returns = get_normalizers(config['environment'])

    window = config['window']
    episode_list = EpisodeList(state_dim, action_dim, timesteps, window)
    reward_plot  = RewardPlotter(num_eps)
    #train_track  = TrainingTracker(config, num_eps)

    loss = np.zeros(num_eps)
    for i in range(num_eps):
        episode = Episode(state_dim, action_dim, timesteps)
        state = env.reset()

        # sample episode from environment
        for t in range(timesteps):
            #env.render()
            
            norm_state = normalize_state(state)
            episode.states[:,t] = state
            episode.norm_states[:,t] = norm_state

            norm_state = np.append(norm_state, t/499)

            action = agent.act(norm_state)
            
            norm_action = normalize_action(action)
            episode.actions[:,t] = action
            episode.norm_actions[:,t] = norm_action
            
            state, reward, _, _ = env.step(action)

            episode.rewards[t] = reward

        # compute and store stats about episode
        total_reward = episode.calculate_returns(t, gamma, normalize_returns)

        # store episode in episode list
        episode_list.append(episode)

        # calculate moving average of past episodes
        returns_MA = episode_list.return_MA()

        # print episode stats and update reward plot
        update_freq = 1000
        print_episode_stats(total_reward, returns_MA, num_eps, i, update_freq)
        reward_plot.update(i, total_reward, returns_MA, update_freq)

        # train agent on episode
        loss[i] = agent.train(episode_list)
        #train_track.evaluate(i, agent)
        
        #input("Press Return to continue...")

    #train_track.plot()
    reward_plot.close()
        
    return episode_list, loss
