
import random
import numpy as np
from tensorflow.keras.models     import Sequential 
from tensorflow.keras.layers     import Dense
from tensorflow.keras.optimizers import Adam

hidden_layer_nodes = 24

def make_sequential_mlp(input_size, output_size, config):
    global hidden_layer_nodes

    mlp = Sequential([Dense(units              = hidden_layer_nodes,
                            input_dim          = input_size,
                            activation         = 'relu'),
                            #kernel_initializer = 'he_uniform',
                            #bias_initializer   = 'zeros'),
                            
                      Dense(units              = hidden_layer_nodes,
                            activation         = 'relu'),
                            #kernel_initializer = 'he_uniform',
                            #bias_initializer   = 'zeros'),

                      Dense(units              = output_size,
                            activation         = 'linear')])
                            #kernel_initializer = 'he_uniform',
                            #bias_initializer   = 'zeros')])

    adam = Adam(lr=config['lr'])
    print("Learning Rate:")
    print(config['lr'])
    mlp.compile(loss='mean_squared_error', optimizer=adam)
    return mlp

class QFunction:
    def __init__(self, input_size, config):
        self.mlp = make_sequential_mlp(input_size, 1, config)

    # get action-value function of current state for each action
    # assumes that given state and actions have already been normalized
    def __call__(self, state, actions):
        input = np.repeat(state.reshape(1,state.size), actions.size, axis=0)
        input = np.hstack((input, actions))
        # print('Input:')
        # print(input)
        Qvalues = self.mlp.predict(input, batch_size=actions.size)
        # print('Qvalues:')
        # print(Qvalues)
        return Qvalues

class QFunctionUpdater:
    def __init__(self, policy, config):
        self.Qfunction = policy.Qfunction
        self.batch_size = config['batch_size']
        self.buff_size  = config['buff_size']

    # num_eps is not 0 indexed because it is incremented before this fcn is cld
    def __call__(self, episode_list):
        # samples from entire episode history
        # num_eps = episode_list.num_eps    
        # num_samples = min(num_eps, self.batch_size)
        # samples = np.random.choice(num_eps, num_samples, replace=False)
        
        # samples from training buffer
        num_eps = episode_list.num_eps    
        num_samples = min(num_eps, self.batch_size)
        begin = max(0, num_eps - self.buff_size) 
        train_buff = np.arange(begin, num_eps)
        samples = np.random.choice(train_buff, num_samples, replace=False)
        
        # this puts the most recent sampled episode in the training batch
        if num_eps-1 not in samples:
            samples[-1] = num_eps-1
        
        # print("Train Buffer:")
        # print(train_buff)
        # print("Training Samples")
        # print(samples)
        
        # similar to old code
        # num_eps = episode_list.num_eps    
        # num_samples = min(num_eps, self.batch_size)
        # train_buff = range(0, num_eps)
        # samples = random.sample(train_buff, num_samples - 1)
        # samples.append(num_eps-1)
        
        loss = np.zeros(num_samples)
        for i in range(num_samples):
            ep_index = samples[i]
            episode = episode_list.episode_list[ep_index]
            states = episode.norm_states
            actions = episode.norm_actions
            disc_retrn = episode.disc_retrn
            
            x = np.transpose(np.vstack((states, actions)))
            #print("Episode")
            #print("Training Input:")
            #print(x)
            # print("Discounted Returns:")
            # print(disc_retrn)

            loss[i] = self.Qfunction.mlp.train_on_batch(x, disc_retrn)
        
        return np.mean(loss)










        
