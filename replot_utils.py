
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import make_plot
from misc_utils import moving_average

def replot_loss(data_dir, y_lims):
    data = np.load(data_dir +'/train_data.npz')
    loss = data['loss']
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'title':   'Training Loss',
                'y_lims':   y_lims}
    make_plot(loss, loss_opt, data_dir)

# create dataframe for the different runs of an experiment
def load_experiment_data(dir):
    total_rewards = np.zeros(0)
    eps           = np.zeros(0)
    iters         = np.zeros(0)
    gammas        = np.zeros(0)
    
    # loop through experiment directories
    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    for i in subdirs:
        subsubdirs = [f.path for f in os.scandir(i) if f.is_dir()]
        # loop through experimnet iterations
        for j in subsubdirs:
            data = np.load(j + '/train_data.npz')
            rewards = data['stacked_data']
            
            # calculate total reward
            total_reward = np.sum(rewards, axis=1)
            
            # apply moving average filter to rewards
            total_reward = moving_average(total_reward, 501)
            
            total_rewards = np.concatenate((total_rewards, total_reward))
            eps = np.concatenate((eps, np.arange(0,total_reward.size)))
            iter = np.repeat(j.split('/')[-1], total_reward.size)
            iters = np.concatenate((iters, iter))
            gamma = np.repeat(i.split('/')[-1], total_reward.size)
            gammas = np.concatenate((gammas, gamma))
        
    # make pandas frame from experiment's rewards
    d = {'Episode': eps, 'Gamma': gammas, 'Iter': iters, 'Total Reward': total_rewards}
    df = pd.DataFrame(data=d)

    return df
    
def average_performance_plot(df):
    sns.set(style='darkgrid')
    # sns.lineplot(x='Episode', y='Total Reward', hue='Gamma' ,ci='sd', data=df, 
    #              palette=sns.color_palette('deep', 12))
    sns.lineplot(x='Episode', y='Total Reward', hue='Gamma' ,ci=None, data=df, 
                 palette=sns.color_palette('deep', 12))             
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    