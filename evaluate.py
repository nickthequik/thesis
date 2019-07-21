
import sys
import time
import numpy as np
from file_utils import get_exp_cfg, make_data_dir
from env_utils  import init_env, get_normalizers
from agents     import init_agent
from ep_utils   import *
from plot_utils import plot_episodes_data, RewardPlotter, get_episode_plotter

def main():
    exp_dir   = sys.argv[1] # get directory with argv
    iter_dir = sys.argv[2] # get agent iteration to evaluate
    num_eps   = int(sys.argv[3]) # get number of episodes to evaluate for

    # read config file
    config = get_exp_cfg(exp_dir)
    config['train'] = False
    config['window'] = 0
    print('Evaluation Configuration:')
    print(config)
    
    # initilize environment
    env = init_env(config)

    # create directory to store results
    agent_dir = exp_dir + '/' + iter_dir
    eval_dir = make_data_dir(agent_dir, 'eval')

    # initilize agent
    agent = init_agent(config, env)
    # agent.load(agent_dir)
    agent.load(agent_dir, True)
    # agent.load(agent_dir, 0)

    start_time = time.time()

    # evaluate agent
    episodes_data = evaluate_agent(agent, env, config, num_eps)

    elapsed_time = time.time() - start_time

    # save episode data
    store_episodes_data(eval_dir, episodes_data)
    episodes_stats = get_episodes_stats(config, episodes_data, elapsed_time)
    store_episodes_stats(eval_dir, episodes_data, episodes_stats)
    plot_episodes_data(eval_dir, episodes_data, episodes_stats, config)

    env.close()

def evaluate_agent(agent, env, config, num_eps):
    timesteps = config['timesteps']
    gamma     = config['gamma']

    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]

    normalize_state, normalize_action, normalize_returns = get_normalizers(config['environment'])
    plot_episode = get_episode_plotter(config)

    window = config['window']
    episode_list = EpisodeList(state_dim, action_dim, timesteps, window)
    reward_plot  = RewardPlotter(num_eps)

    for i in range(num_eps):
        episode = Episode(state_dim, action_dim, timesteps)
        # state = env.reset()
        state = env.reset(state=np.array([-1, 0, 0]))
        
        # print("x0:")
        # print(state)

        # sample episode from environment
        for t in range(timesteps):
            env.render()
            
            norm_state = normalize_state(state)
            episode.states[:,t] = state
            episode.norm_states[:,t] = norm_state

            # norm_state = np.append(norm_state, t/(timesteps-1))

            action = agent.act(norm_state, greedy=True)
            # action = agent.act(state, greedy=True)
            
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
        update_freq = 100
        print_episode_stats(total_reward, returns_MA, num_eps, i, update_freq)
        reward_plot.update(i, total_reward, returns_MA, update_freq)
        
        # plot_episode(episode)
        # input("Press enter to continue")

    reward_plot.close()
    # print(repr(episode.actions))
    return episode_list

if __name__ == '__main__':
    main()
