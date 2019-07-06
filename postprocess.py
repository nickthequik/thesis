
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import make_plot
from misc_utils import moving_average

def main():
    # dir = 'experiments/pendulum/official/mcc/gamma'
    # subdirs = ['0.89', '0.94', '1.0']
    # var_name = 'Gamma'
    # df = load_experiment_data(dir, subdirs, var_name)
    # average_performance_plot(df, var_name, subdirs)
    
    # dir = 'experiments/pendulum/official/mcc/buffer'
    # subdirs = ['50', '100', '400', '700', '1200', '2000']
    # # subdirs = ['50', '400', '2000']
    # var_name = 'Training Buffer Size'
    # df = load_experiment_data(dir, subdirs, var_name)
    # average_performance_plot(df, var_name, subdirs)
    
    # dir = 'experiments/pendulum/official/mcc/num_actions'
    # subdirs = ['11', '21', '41', '81', '401']
    # var_name = 'Number of Actions'
    # df = load_experiment_data(dir, subdirs, var_name)
    # average_performance_plot(df, var_name, subdirs)
    
    # dir = 'experiments/pendulum/official/mcc/time_state'
    # # subdirs = ['with_time', 'without_time','with_time_100steps', 'without_time_100steps']
    # subdirs = ['with_time_100steps', 'without_time_100steps']
    # var_name = 'Time in State'
    # df = load_experiment_data(dir, subdirs, var_name)
    # average_performance_plot(df, var_name, subdirs)
    
    # dir = 'experiments/pendulum/official/mcc/trig_state'
    # subdirs = ['trig_state_no_time', 'theta_state_no_time']
    # var_name = 'State Space Formulation'
    # df = load_experiment_data(dir, subdirs, var_name)
    # average_performance_plot(df, var_name, subdirs)
    
    replot_loss('experiments/pendulum/mcc/official/different_reward/trig_reward/1', [0, 0.02])

def replot_loss(data_dir, y_lims):
    data = np.load(data_dir +'/train_data.npz')
    loss = data['loss']
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'title':   'Training Loss',
                'y_lims':   y_lims}
    make_plot(loss[0,:], loss_opt, data_dir)

# create dataframe for the different runs of an experiment
def load_experiment_data(dir, subdirs, var_name):
    total_rewards = np.zeros(0)
    eps           = np.zeros(0)
    iters         = np.zeros(0)
    vars          = np.zeros(0)
    
    for i in range(len(subdirs)):
        exp_dir = dir + '/' + subdirs[i]
        subsubdirs = [f.path for f in os.scandir(exp_dir) if f.is_dir()]
        # loop through experimnet iterations
        for j in subsubdirs:
            # data = np.load(j + '/train_data.npz')
            data = np.load(j + '/eval/train_data.npz')
            rewards = data['stacked_data']
            
            # calculate total reward
            total_reward = np.sum(rewards, axis=1)
            
            # apply moving average filter to rewards
            total_reward = moving_average(total_reward, 0)
            
            total_rewards = np.concatenate((total_rewards, total_reward))
            eps = np.concatenate((eps, np.arange(0,total_reward.size)))
            iter = np.repeat(j.split('/')[-1], total_reward.size)
            iters = np.concatenate((iters, iter))
            
            var = np.repeat(subdirs[i], total_reward.size)
            vars = np.concatenate((vars, var))
        
    # make pandas frame from experiment's rewards
    d = {'Episode': eps, var_name: vars, 'Iter': iters, 'Total Reward': total_rewards}
    df = pd.DataFrame(data=d)

    return df
    
def average_performance_plot(df, var_name, subdirs):
    sns.set(style='darkgrid')
    # sns.lineplot(x='Episode', y='Total Reward', hue='Gamma' ,ci='sd', data=df, 
    #              palette=sns.color_palette('deep', 12))
    sns.lineplot(x='Episode', y='Total Reward', hue=var_name, hue_order=subdirs,
                 ci='sd', data=df, palette=sns.color_palette('deep', len(subdirs)))             
    
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    
    fig = plt.gcf()
    fig.set_size_inches(9,6)
    plt.tight_layout()
    
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    