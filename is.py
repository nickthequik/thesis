
import sys
import time
import numpy as np
import gym
import my_envs

from scipy.stats import norm
from ep_utils    import Episode, EpisodeList, store_episodes_data, store_episodes_stats, get_episodes_stats, print_episode_stats
from env_utils   import get_normalizers, init_env
from plot_utils  import get_episode_plotter, plot_episodes_data, plot_loss_data, RewardPlotter
from agents      import init_agent, get_loss
from file_utils  import get_exp_cfg, make_data_dir

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

def main():
    # get directory with argv
    exp_dir = sys.argv[1]
    
    # read config file
    config = get_exp_cfg(exp_dir)
    config['train'] = True
    print('Experiment Configuration:')
    print(config)
    
    # initilize environment
    env = init_env(config)
    if not env: sys.exit()

    # repeat experiment for iter iterations
    iter = config['iterations']
    for i in range(iter):
        # create directory to store results
        data_dir = make_data_dir(exp_dir, i+1)

        # initilize agent
        agent = init_agent(config, env)
        if not agent: sys.exit()

        # keep track of start time
        start_time = time.time()

        # start training agent
        episodes_data, loss = GPS_train(data_dir, agent, env, config)

        elapsed_time = time.time() - start_time
        print("Elapsed time: {:g}".format(elapsed_time))

        # save episodes data and stats
        store_episodes_data(data_dir, episodes_data, loss=loss)
        episodes_stats = get_episodes_stats(config, episodes_data, elapsed_time)
        store_episodes_stats(data_dir, episodes_data, episodes_stats)
        plot_episodes_data(data_dir, episodes_data, episodes_stats, config)
        plot_loss_data(data_dir, loss, config)

    env.close()
    
def GPS_train(data_dir, agent, env, config):
    num_eps   = config['episodes']
    timesteps = config['timesteps']
    
    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # save initial weights
    agent.save(data_dir, True)
    
    normalize_state, normalize_action, normalize_returns = get_normalizers(config['environment'])
    normalize = [normalize_state, normalize_action, normalize_returns]
    
    window = config['window']
    policy_samples = EpisodeList(state_dim, action_dim, timesteps, window)
    reward_plot  = RewardPlotter(num_eps*10)
    update_freq = 1000

    for i in range(10):
        episode, total_reward = generate_policy_sample(config, env, agent, normalize)
        policy_samples.append(episode)
        
        returns_MA = policy_samples.return_MA()
        print_episode_stats(total_reward, returns_MA, num_eps, i, update_freq)
        reward_plot.update(i, total_reward, returns_MA, update_freq)
     
    # sample from experienced episodes
    x, u, r, p = concat_episodes(policy_samples, config) 
    
    loss = get_loss(config)
    for i in range(num_eps): 
        # train the best agent
        loss[i] = agent.train([x, u, r, p])
        # print('Loss: {:g}'.format(loss[i,0]))
        agent.save(data_dir, False)
            
        # get samples from new agent
        for j in range(10):
            episode, total_reward = generate_policy_sample(config, env, agent, normalize)
            policy_samples.append(episode)
            
            returns_MA = policy_samples.return_MA()
            ep_index = 10*i + j
            print_episode_stats(total_reward, returns_MA, num_eps, ep_index, update_freq)
            reward_plot.update(ep_index, total_reward, returns_MA, update_freq)
          
        # sample from experienced episodes
        x, u, r, p = concat_episodes(policy_samples, config)
        
        # predict performance of new and best agent
        cur_policy_action_probs = np.squeeze(agent.policy.action_prob(x, u))
        agent.load(data_dir, True)
        best_policy_action_probs = np.squeeze(agent.policy.action_prob(x, u))
        
        cur_policy_estimated_reward = estimate_expected_reward(r, p, cur_policy_action_probs)
        best_policy_estimated_reward = estimate_expected_reward(r, p, best_policy_action_probs)
        
        # print('New Policy Estimated Reward')
        # print(cur_policy_estimated_reward)
        # print('Best Policy Estimated Reward')
        # print(best_policy_estimated_reward)
        
        if cur_policy_estimated_reward > best_policy_estimated_reward:
            # print("Updating Best Agent")
            agent.load(data_dir, False)
            agent.save(data_dir, True)
            # decrease regularization
            agent.policy.policy_model.decrease_regularization_weight()
        else:
            # print("Keeping Old Best Agent")
            # increase regularization
            agent.policy.policy_model.increase_regularization_weight()
            
    return policy_samples, loss

# change to allow variable timesteps if needed
def estimate_expected_reward(r, guiding_action_probs, policy_action_probs):
    r = r.reshape((-1, 100))

    guiding_action_probs = guiding_action_probs.reshape((-1,100))
    guiding_trajectory_prob = np.cumprod(guiding_action_probs, axis=1, dtype='float64')

    policy_action_probs = policy_action_probs.reshape((-1, 100))
    policy_trajectory_probs = np.cumprod(policy_action_probs, axis=1, dtype='float64')

    Zt = np.sum(policy_trajectory_probs / guiding_trajectory_prob, axis=0)
    unnorm_reward = np.sum((policy_trajectory_probs / guiding_trajectory_prob) * r, axis=0)
    expected_return = np.sum(unnorm_reward / Zt)

    importance_weights = policy_trajectory_probs / guiding_trajectory_prob
    
    #######################################################################
    # print("Importance Weights")
    # print(policy_trajectory_probs[:,0])
    # print(guiding_trajectory_prob[:,0])
    # print(importance_weights[:, 0])
    # print(Zt[0])
    # print(np.sum(importance_weights[:, 0]/Zt[0]))
    # input("...")
    #######################################################################

    return expected_return
    
def generate_policy_sample(config, env, gps_agent, normalize):
    timesteps = config['timesteps']
    gamma     = config['gamma']
    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    normalize_state = normalize[0]
    normalize_action = normalize[1]
    normalize_returns = normalize[2]
    plot_episode = get_episode_plotter(config)
    episode = Episode(state_dim, action_dim, timesteps)
    
    state = env.reset()
    for t in range(timesteps):
        #env.render()
        
        norm_state = normalize_state(state)
        episode.states[:,t] = state
        episode.norm_states[:,t] = norm_state

        # action, prob = gps_agent.act(state)
        action, prob = gps_agent.act(norm_state)
        
        norm_action = normalize_action(action)
        episode.actions[:,t] = action
        episode.probs[t] = prob
        episode.norm_actions[:,t] = norm_action
        
        state, reward, _, _ = env.step(action)

        episode.rewards[t] = reward
        
    # plot_episode(episode)
    
    total_reward = episode.calculate_returns(t, gamma, normalize_returns)
        
    return episode, total_reward

def concat_episodes(episode_list, config):
    batch_size = config['batch_size']
    buff_size  = config['buff_size']
    timesteps  = episode_list.timesteps
    state_dim  = episode_list.state_dim
    action_dim = episode_list.action_dim

    num_eps = episode_list.num_eps    
    num_samples = min(num_eps, batch_size)
    begin = max(0, num_eps - buff_size) 
    train_buff = np.arange(begin, num_eps)
    samples = np.random.choice(train_buff, num_samples, replace=False)

    concat_states   = np.zeros((state_dim, timesteps * num_samples))
    concat_actions  = np.zeros((action_dim, timesteps * num_samples))
    concat_rewards  = np.zeros(timesteps * num_samples)
    concat_probs    = np.zeros(timesteps * num_samples)

    # print("Samples:")
    # print(samples)

    for i in range(num_samples):
        ep_index = samples[i]
        episode = episode_list.episode_list[ep_index]
        # states  = episode.states
        states  = episode.norm_states
        actions = episode.actions
        rewards = episode.rewards
        probs   = episode.probs

        concat_states[:, i*timesteps:(i+1)*timesteps]  = states
        concat_actions[:, i*timesteps:(i+1)*timesteps] = actions
        concat_rewards[i*timesteps:(i+1)*timesteps]    = rewards
        concat_probs[i*timesteps:(i+1)*timesteps]      = probs

    return concat_states.T, concat_actions.T, concat_rewards, concat_probs
        
if __name__ == '__main__':
    main()
    