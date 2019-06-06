
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from misc_utils import concat_episodes

def plot_episodes_data(data_dir, episodes_data, episode_stats, config):
    env_name = config['environment']

    if env_name == 'my_Pendulum-v0':
        plot_pendulum_data(data_dir, episodes_data, episode_stats)
    elif env_name == 'my_Cartpole-v0':
        plot_cattpole_data(data_dir, episodes_data)
    elif env_name == 'my_Acrobot-v0':
        plot_acrobot_data(data_dir, episodes_data)

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

def plot_loss_data(data_dir, loss):
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'y_lims':   [0,0.01],
                'title':   'Training Loss'}
    make_plot(loss, loss_opt, data_dir)

def plot_reward_stats(data_dir, episodes_data, episode_stats):
    returns    = episode_stats['returns']
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

def plot_pendulum_data(data_dir, episodes_data, episode_stats):
    # stack all states, action, and rewards into long arrays
    concat_states,concat_actions,concat_rewards = concat_episodes(episodes_data)

    # call histogram function on states, actions, and rewards
    x1_opt = {'x_label': 'Angle [rad/pi]',
              'y_label': 'Frequency',
              'title':   'Angle Histogram'}
    make_histogram(concat_states[0,:]/np.pi, x1_opt, data_dir)

    x2_opt = {'x_label': 'Angular Velocity [rad/s]',
              'y_label': 'Frequency',
              'title':   'Angular Velocity Histogram'}
    make_histogram(concat_states[1,:], x2_opt, data_dir)

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
            self.Qvalues[i,iter] = agent.policy.get_Qvalue(self.states[:,i], np.array([[0]]))
        
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
        
        
        
        
        
        
