
import random
import numpy as np
from tensorflow.keras.models     import Sequential 
from tensorflow.keras.layers     import Dense
from tensorflow.keras.optimizers import Adam

hidden_layer_nodes = 50

def make_sequential_mlp(input_size, output_size, config):
    global hidden_layer_nodes

    mlp = Sequential([Dense(units              = hidden_layer_nodes,
                            input_dim          = input_size,
                            activation         = 'relu'),   
                      Dense(units              = hidden_layer_nodes,
                            activation         = 'relu'),
                      Dense(units              = output_size,
                            activation         = 'linear')])

    if config['train']:
        adam = Adam(lr=config['lr'])
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
        #print('Input:')
        #print(input)
        Qvalues = self.mlp.predict(input, batch_size=actions.size)
        #print('Qvalues:')
        #print(Qvalues)
        return Qvalues
        
    def save(self, name):
        self.mlp.save_weights(name)
        
    def load(self, name):
        self.mlp.load_weights(name)

class QFunctionUpdater:
    def __init__(self, policy, config):
        self.Qfunction  = policy.Qfunction
        self.batch_size = config['batch_size']
        self.buff_size  = config['buff_size']

    # num_eps is not 0 indexed because it is incremented before this fcn is cld
    def __call__(self, episode_list):
        # samples from training buffer
        num_eps = episode_list.num_eps    
        num_samples = min(num_eps, self.batch_size)
        begin = max(0, num_eps - self.buff_size) 
        train_buff = np.arange(begin, num_eps)
        samples = np.random.choice(train_buff, num_samples, replace=False)
        
        # this puts the most recent sampled episode in the training batch
        #if num_eps-1 not in samples:
        #    samples[-1] = num_eps-1
        
        # print("Train Buffer:")
        # print(train_buff)
        # print("Training Samples")
        # print(samples)
        
        loss = np.zeros(num_samples)
        for i in range(num_samples):
            ep_index = samples[i]
            episode = episode_list.episode_list[ep_index]
            states = episode.norm_states
            actions = episode.norm_actions
            disc_retrn = episode.norm_returns
        
            t = np.arange(500).reshape((1,500)) / 499
        
            x = np.transpose(np.vstack((states, t, actions)))
            #print("Training Input:")
            #print(x)
            # print("Discounted Returns:")
            # print(disc_retrn)
        
            loss[i] = self.Qfunction.mlp.train_on_batch(x, disc_retrn)
            #print("Loss: {:g}".format(loss[i]))
        
        return np.mean(loss)
        
        # input = np.zeros((500*num_samples, 4))
        # returns = np.zeros(500*num_samples)
        # for i in range(num_samples):
        #     ep_index = samples[i]
        #     episode = episode_list.episode_list[ep_index]
        #     states = episode.norm_states
        #     actions = episode.norm_actions
        #     disc_retrn = episode.norm_returns
        # 
        #     t = np.arange(500).reshape((1,500)) / 499
        # 
        #     x = np.transpose(np.vstack((states, t, actions)))
        # 
        #     input[i*500:(i+1)*500, :] = x
        #     returns[i*500:(i+1)*500] = disc_retrn
        # 
        # loss = self.Qfunction.mlp.fit(input, returns, batch_size=32, epochs=1, verbose=0)
        # 
        # return loss.history['loss'][0]









        
