
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from misc_utils import concat_episodes

def make_histogram(data, opt, save_dir):
    plt.figure()
    plt.hist(data, bins=100, density=True)
    plt.title(opt['title'])
    plt.xlabel(opt['x_label'])
    plt.ylabel(opt['y_label'])
    #plt.tight_layout()
    plt.savefig(save_dir + '/' + opt['title'] + '.png')
    plt.close()

def make_plot(data, opt, save_dir):
    plt.figure()
    plt.plot(data)
    plt.title(opt['title'])
    plt.xlabel(opt['x_label'])
    plt.ylabel(opt['y_label'])
    plt.xlim(0,data.size)
    if 'y_lims' in opt:
        y_lims = opt['y_lims']
        plt.ylim(y_lims[0],y_lims[1])
    plt.grid(b=True, which='both')
    #plt.tight_layout()
    plt.savefig(save_dir + '/' + opt['title'] + '.png')
    plt.close()

def plot_loss_data(data_dir, loss, config):
    agent_name = config['agent']

    if agent_name == 'Random':
        return
    elif agent_name =='MonteCarloControl' or agent_name =='GPS':
        plot_mcc_loss(data_dir, loss,)
    elif agent_name == 'REINFORCE':
        plot_mcpg_loss(data_dir, loss,)
    
def plot_mcc_loss(data_dir, loss,):
    loss_mean = np.mean(loss[:,0])
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'y_lims':   [0, 4*loss_mean],
                'title':   'Training Loss'}
    make_plot(loss[:,0], loss_opt, data_dir)
    
def plot_mcpg_loss(data_dir, loss,):
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'title':   'Policy Gradient Training Loss'}
    make_plot(loss[:,0], loss_opt, data_dir)
    
    vf_loss_mean = np.mean(loss[:,1])
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'y_lims':   [0, 4*vf_loss_mean],
                'title':   'Value Function Training Loss'}
    make_plot(loss[:,1], loss_opt, data_dir)

def plot_reward_stats(data_dir, episodes_data, episode_stats):
    returns    = episode_stats['returns']  # undiscounted
    returns_MA = episode_stats['returns_MA']
    returns_SD = episode_stats['returns_SD']

    plt.figure()
    plt.plot(returns, label="Episode Reward")
    plt.plot(returns_MA, label="Average Reward")
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.xlim(0,returns_MA.size)
    plt.legend()
    plt.grid(b=True, which='both')
    #plt.tight_layout()
    plt.savefig(data_dir + '/Reward per Episode.png' )
    plt.close()

    sd_opt = {'x_label': 'Episode',
              'y_label': 'Standard Deviation',
              'title':   'Reward Standard Deviation'}
    make_plot(returns_SD , sd_opt, data_dir)

def plot_episodes_data(data_dir, episodes_data, episode_stats, config):
    env_name = config['environment']

    if env_name == 'my_Pendulum-v0':
        plot_pendulum_data(data_dir, episodes_data, episode_stats)
    elif env_name == 'my_Cartpole-v0':
        plot_cattpole_data(data_dir, episodes_data)
    elif env_name == 'my_Acrobot-v0':
        plot_acrobot_data(data_dir, episodes_data)

def plot_pendulum_data(data_dir, episodes_data, episode_stats):
    # stack all states, action, and rewards into long arrays
    concat_states,concat_actions,concat_rewards = concat_episodes(episodes_data)

    # call histogram function on states, actions, and rewards
    x1_opt = {'x_label': 'Angle [rad/pi]',
              'y_label': 'Frequency',
              'title':   'Angle Histogram'}
    #concat_states[0,:] = (((concat_states[0,:]+np.pi) % (2*np.pi)) - np.pi)
    # make_histogram(concat_states[0,:]/np.pi, x1_opt, data_dir)

    thetas = np.arctan2(concat_states[1,:], concat_states[0,:])
    make_histogram(thetas/np.pi, x1_opt, data_dir)

    x2_opt = {'x_label': 'Angular Velocity [rad/s]',
              'y_label': 'Frequency',
              'title':   'Angular Velocity Histogram'}
    # make_histogram(concat_states[1,:], x2_opt, data_dir)
    make_histogram(concat_states[2,:], x2_opt, data_dir)

    act_opt = {'x_label': 'Torque [Nm]',
               'y_label': 'Frequency',
               'title':   'Action Histogram'}
    make_histogram(concat_actions[0,:], act_opt, data_dir)

    rew_opt = {'x_label': 'Reward',
               'y_label': 'Frequency',
               'title':   'Reward Histogram'}
    make_histogram(concat_rewards, rew_opt, data_dir)

    plot_reward_stats(data_dir, episodes_data, episode_stats)

def plot_cattpole_data(data_dir, train_data):
    pass

def plot_acrobot_data(data_dir, train_data):
    pass

def get_episode_plotter(config):
    env_name = config['environment']

    if env_name == 'my_Pendulum-v0':
        return plot_pendulum_episode
    elif env_name == 'my_Cartpole-v0':
        pass
    elif env_name == 'my_Acrobot-v0':
        pass

def unwrap(x):
    for i in range(x.size - 1):
        if x[i+1] - x[i] > np.pi:
            x[i+1:] -= 2*np.pi
        elif x[i] - x[i+1] > np.pi:
            x[i+1:] += 2*np.pi

def plot_pendulum_episode(episode):
    fig = plt.figure()
    fig.set_size_inches(10,5)
    
    states = episode.states
    actions = episode.actions
    rewards = episode.rewards
    
    thetas = np.arctan2(states[1,:],states[0,:])
    unwrap(thetas)
    
    # Phase Plane plot
    ax0 = fig.add_subplot(1,2,1)
    ax0.set_title("Phase Plane Trajectory")
    ax0.set_xlabel("Theta [rad/pi]")
    ax0.set_ylabel("Theta Dot [rad/s]")
    ax0.grid(b=True, which='both')
    # trajectory, = ax0.plot(states[0, :]/np.pi, states[1, :])
    # target,     = ax0.plot(0, 0, marker='x', color='red')
    # start,      = ax0.plot(states[0, 0]/np.pi, states[1, 0], marker='o', color='red')
    # end,        = ax0.plot(states[0, -1]/np.pi, states[1, -1], marker='s', color='red')
    trajectory, = ax0.plot(thetas/np.pi, states[2, :])
    target,     = ax0.plot(0, 0, marker='x', color='red')
    start,      = ax0.plot(thetas[0]/np.pi, states[2, 0], marker='o', color='red')
    end,        = ax0.plot(thetas[-1]/np.pi, states[2, -1], marker='s', color='red')

    # States and actions
    ax1 = fig.add_subplot(1,2,2)
    ax1.set_title("Time Domain Plot")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.grid(b=True, which='both')
    # theta,    = ax1.plot(states[0, :]/np.pi, label="Theta [rad/pi]", color='tab:blue')
    # thetadot, = ax1.plot(states[1, :], label="Theta Dot [rad/s]", color='tab:orange')
    # action,   = ax1.plot(actions[0, :], label="Torque [Nm]", color='tab:red')
    theta,    = ax1.plot(thetas/np.pi, label="Theta [rad/pi]", color='tab:blue')
    thetadot, = ax1.plot(states[2, :], label="Theta Dot [rad/s]", color='tab:orange')
    action,   = ax1.plot(actions[0, :], label="Torque [Nm]", color='tab:red')

    # Reward
    ax2 = ax1.twinx()
    reward, = ax2.plot(rewards, label="Reward", color='tab:green')

    # plt.pause(0.01)
    plt.show()
    plt.close()

class RewardPlotter:
    def __init__(self, num_eps):
        self.returns = np.zeros(num_eps)
        self.returns_MA = np.zeros(num_eps)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.line1, = self.ax.plot(0, label="Episode Reward")
        self.line2, = self.ax.plot(0, label="Average Reward")

        self.ax.set_title("Total Reward per Episode")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.ax.legend()
        self.ax.grid(b=True, which='both')

    def update(self, ep_index, retrn, return_MA, draw_freq):
        self.returns[ep_index] = retrn
        self.returns_MA[ep_index] = return_MA

        if (ep_index+1) % draw_freq == 0:
            x = np.arange(ep_index+1)
            self.line1.set_ydata(self.returns[0:ep_index+1])
            self.line1.set_xdata(x)
            self.line2.set_ydata(self.returns_MA[0:ep_index+1])
            self.line2.set_xdata(x)
            self.ax.relim()
            self.ax.autoscale()
            plt.pause(0.01)
            
    def close(self):
        plt.close(self.fig)

# keep track of specific states' Q value throughout training to monitor progress
class TrainingTracker:
    def __init__(self, config, num_eps):
        env_name = config['environment']
        
        if env_name == 'Pendulum-v0':
            # normalized states
            self.states  = np.array([[-1, -1/2, 0, 1/2, 1],
                                     [ 0,    0, 0,   0, 0],
                                     [ 0,    0, 0,   0, 0]])
                                     
            #self.actions = np.array([-1, 0, 1]).reshape((3,1))
        
        self.Qvalues = np.zeros((self.states.shape[1], num_eps))
        
    def evaluate(self, iter, agent):
        for i in range(self.states.shape[1]):
            self.Qvalues[i,iter] = agent.policy.Qfunction(self.states[:,i], np.array([[0]]))
        
    def plot(self):
        plt.figure()
        
        for i in range(self.Qvalues.shape[0]):
            plt.plot(self.Qvalues[i,:], label="State {:d}".format(i))
            
        plt.title('Q Values')
        plt.xlabel('Episode')
        plt.ylabel('Q Value')
        plt.xlim(0,self.Qvalues.shape[1])
        plt.grid(b=True, which='both')
        plt.legend()
        plt.show()
        plt.close()
        
        
        
        
        
        
