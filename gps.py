
import sys
import time
import numpy as np
import gym
import my_envs

from scipy.stats import norm
from ep_utils    import Episode, EpisodeList, store_episodes_data, store_episodes_stats, get_episodes_stats
from env_utils   import get_normalizers, init_env
from plot_utils  import get_episode_plotter, plot_episodes_data, plot_loss_data
from agents      import init_agent, get_loss
from file_utils  import get_exp_cfg, make_data_dir
from misc_utils  import concat_episodes

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
    window = config['window']
    
    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # keeps track of how many agent policies have been saved
    num_agents = 0
    
    # keeps track of best agent
    best_agent = 0
    
    # load guiding ilqr policies
    guiding_policies = load_guiding_policies('experiments/pendulum/ilqr/low_variance')
    guiding_samples = EpisodeList(state_dim, action_dim, timesteps, window)
    
    for i in range(len(guiding_policies)):
        # generate 5 guiding samples per guiding policy
        for j in range(5):
            episode = generate_guiding_sample(config, env, guiding_policies[i])
            guiding_samples.append(episode)
        
    guide_total_rewards = np.zeros(guiding_samples.num_eps)
    for i in range(guiding_samples.num_eps):
        guide_total_rewards[i] = guiding_samples.episode_list[i].total_reward
    
    print("Average Guided Reward")
    print(np.mean(guide_total_rewards))
            
            
    # try training with only guiding samples to see if it works...
            
            
            
    loss = agent.trainer.pretraining(guiding_samples)
    agent.save(data_dir, num_agents)
    num_agents += 1
    
    plt.figure()
    plt.plot(loss)
    plt.show()
    plt.close()
    
    policy_samples = EpisodeList(state_dim, action_dim, timesteps, window)
    
    # generate intial samples from pretrained policy
    for i in range(5):
        episode = generate_policy_sample(config, env, agent)
        policy_samples.append(episode)
        
    sample_total_rewards = np.zeros(policy_samples.num_eps)
    for i in range(policy_samples.num_eps):
        sample_total_rewards[i] = policy_samples.episode_list[i].total_reward
    
    print("Average Sample Reward")
    print(np.mean(sample_total_rewards))
    
    loss = get_loss(config)
    # each 'episode' generates 5 policy samples and 
    for ii in range(num_eps): 
        # Generate the guiding probabilities
        guiding_states, guiding_actions, guiding_rewards = concat_episodes(guiding_samples)    
        sample_states, sample_actions, sample_rewards = concat_episodes(policy_samples) 

        x = np.transpose(np.hstack((guiding_states, sample_states)))
        u = np.transpose(np.hstack((guiding_actions, sample_actions)))
        r = np.hstack((guiding_rewards, sample_rewards))
        
        # get action probabilities from guiding policies
        guiding_policies_probs = np.zeros((len(guiding_policies), x.shape[0]))
        for i in range(len(guiding_policies)):
            guiding_policy = guiding_policies[i]
            for j in range(x.shape[0]):
                t = j%timesteps
                guiding_policies_probs[i, j] = guiding_policy.get_action_prob(x[j], u[j], t)
        
        # get action probabilities from all agents
        other_agent_probs = np.zeros((num_agents, x.shape[0]))
        for i in range(num_agents):
            agent.load(data_dir, i)
            other_agent_probs[i, :] = np.squeeze(agent.policy.action_prob(x, u))
        
        guiding_action_probs = np.mean(np.vstack((guiding_policies_probs, other_agent_probs)), axis=0)
        
        # train the best agent
        print("Loading Best Agent for Training")
        agent.load(data_dir, best_agent)
        loss[ii] = agent.train([x, u, r, guiding_action_probs])
        print('Loss')
        print(np.squeeze(loss))

        agent.save(data_dir, num_agents)
        num_agents += 1
            
        # get samples from new agent
        for i in range(5):
            episode = generate_policy_sample(config, env, agent)
            policy_samples.append(episode)
        
        #######################################################################
        # sample_total_rewards = np.zeros(policy_samples.num_eps)
        # for i in range(policy_samples.num_eps):
        #     sample_total_rewards[i] = policy_samples.episode_list[i].total_reward
        # 
        # print("Average Sample Reward")
        # print(np.mean(sample_total_rewards))
        #######################################################################
            
        # predict performance of new and best agent
        guiding_states, guiding_actions, guiding_rewards = concat_episodes(guiding_samples)    
        sample_states, sample_actions, sample_rewards = concat_episodes(policy_samples) 

        x = np.transpose(np.hstack((guiding_states, sample_states)))
        u = np.transpose(np.hstack((guiding_actions, sample_actions)))
        r = np.hstack((guiding_rewards, sample_rewards))
        
        cur_policy_action_probs = np.squeeze(agent.policy.action_prob(x, u))
        agent.load(data_dir, best_agent)
        best_policy_action_probs = np.squeeze(agent.policy.action_prob(x, u))
        
        # get action probabilities from guiding policies
        guiding_policies_probs = np.zeros((len(guiding_policies), x.shape[0]))
        for i in range(len(guiding_policies)):
            guiding_policy = guiding_policies[i]
            for j in range(x.shape[0]):
                t = j%timesteps
                guiding_policies_probs[i, j] = guiding_policy.get_action_prob(x[j], u[j], t)
        
        # get action probabilities from all agents
        other_agent_probs = np.zeros((num_agents, x.shape[0]))
        for i in range(num_agents):
            agent.load(data_dir, i)
            other_agent_probs[i, :] = np.squeeze(agent.policy.action_prob(x, u))
        
        guiding_action_probs = np.mean(np.vstack((guiding_policies_probs, other_agent_probs)), axis=0)
        
        cur_policy_estimated_reward = estimate_expected_reward(r, guiding_action_probs, cur_policy_action_probs)
        best_policy_estimated_reward = estimate_expected_reward(r, guiding_action_probs, best_policy_action_probs)
        
        print('cur_policy_estimated_reward')
        print(cur_policy_estimated_reward)
        print('best_policy_estimated_reward')
        print(best_policy_estimated_reward)
        
        if cur_policy_estimated_reward > best_policy_estimated_reward:
            best_agent = num_agents - 1
            # load agent
            print("Updating Best Agent")
            # agent.load(data_dir, best_agent)
            # decrease regularization
            #agent.policy.policy_model.decrease_regularization_weight()
        else:
            # load best agent
            print("Keeping Old Best Agent")
            # agent.load(data_dir, best_agent)
            #increase regularization
            #agent.policy.policy_model.increase_regularization_weight()
            
    print('Best Agent')
    print(best_agent)
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
    
    # print("Importance Weights")
    # print(importance_weights)
    # 
    # print("Expected Reward")
    # print(expected_return)

    return expected_return
    
def generate_policy_sample(config, env, gps_agent):
    timesteps = config['timesteps']
    gamma     = config['gamma']
    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    normalize_state, normalize_action, normalize_returns = get_normalizers(config['environment'])
    plot_episode = get_episode_plotter(config)
    episode = Episode(state_dim, action_dim, timesteps)
    
    state = env.reset()
    for t in range(timesteps):
        #env.render()
        
        norm_state = normalize_state(state)
        episode.states[:,t] = state
        episode.norm_states[:,t] = norm_state

        action = gps_agent.act(state)
        # action = gps_agent.act(norm_state)
        
        norm_action = normalize_action(action)
        episode.actions[:,t] = action
        episode.norm_actions[:,t] = norm_action
        
        state, reward, _, _ = env.step(action)

        episode.rewards[t] = reward
        
    # plot_episode(episode)
    
    episode.calculate_returns(t, gamma, normalize_returns)
        
    return episode

def generate_guiding_sample(config, env, guiding_policy):
    timesteps = config['timesteps']
    gamma     = config['gamma']
    state_dim  = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    normalize_state, normalize_action, normalize_returns = get_normalizers(config['environment'])
    plot_episode = get_episode_plotter(config)
    episode = Episode(state_dim, action_dim, timesteps)
    
    state = env.reset(state=guiding_policy.x0)
    for t in range(timesteps):
        #env.render()
        
        norm_state = normalize_state(state)
        episode.states[:,t] = state
        episode.norm_states[:,t] = norm_state

        action = guiding_policy.act(state, t)
        
        norm_action = normalize_action(action)
        episode.actions[:,t] = action
        episode.norm_actions[:,t] = norm_action
        
        state, reward, _, _ = env.step(action)

        episode.rewards[t] = reward
        
    # plot_episode(episode)
    
    episode.calculate_returns(t, gamma, normalize_returns)
        
    return episode
    
def load_guiding_policies(dir):
    guiding_policies = []
    for i in range(7):
        path = dir + '/traj{:d}.npz'.format(i)
        guiding_policies.append(GuidingPolicy(path))

    return guiding_policies

class GuidingPolicy:
    def __init__(self, traj_path):
        ilqr_policy = np.load(traj_path)
        self.x0   = ilqr_policy['x0']
        self.x    = ilqr_policy['x']
        self.u    = ilqr_policy['u']
        self.Kfb = ilqr_policy['Kfb']
        self.Quu  = ilqr_policy['Quu']
        self.u_lim = 2

    def act(self, x, t):
        dx = x - self.x[:,t]
        u_det = self.u[:,t] + self.Kfb[:,:,t] @ dx
        u_det = np.min([self.u_lim, np.max([-self.u_lim, u_det])])

        # stochastic policy from deterministic iLQR
        std_dev = np.sqrt(1/self.Quu[:,:,t])
        u_stoch = norm.rvs(loc=u_det, scale=std_dev, size=1)

        return u_stoch

    def get_action_prob(self, x, u, t):
        dx = x - self.x[:,t]
        u_det = self.u[:,t] + self.Kfb[:,:,t] @ dx
        u_det = np.min([self.u_lim, np.max([-self.u_lim, u_det])])

        std_dev = np.sqrt(1/self.Quu[:,:,t])
        action_prob = norm.pdf(u, loc=u_det, scale=std_dev)

        return np.squeeze(action_prob)
        
if __name__ == '__main__':
    main()
    
    
# sample_total_rewards = np.zeros(policy_samples.num_eps)
# for i in range(policy_samples.num_eps):
#     sample_total_rewards[i] = policy_samples.episode_list[i].total_reward
# 
# guide_total_rewards = np.zeros(policy_samples.num_eps)
# for i in range(policy_samples.num_eps):
#     guide_total_rewards[i] = guiding_samples.episode_list[i].total_reward
# 
# print("Average Total Reward")
# print(np.mean(np.hstack((sample_total_rewards, guide_total_rewards))))
# print("Average Sample Reward")
# print(np.mean(sample_total_rewards))
# print("Average Guided Reward")
# print(np.mean(guide_total_rewards))

# generate a bunch of policy samples to see how accurate estimate was
# for i in range(1000):
#     episode = generate_policy_sample(config, env, agent)
#     policy_samples.append(episode)
# 
# total_rewards = np.zeros(policy_samples.num_eps)
# for i in range(policy_samples.num_eps):
#     total_rewards[i] = policy_samples.episode_list[i].total_reward
# 
# print("Average Total Reward")
# print(np.mean(total_rewards))