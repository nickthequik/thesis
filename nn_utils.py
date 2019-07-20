
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses     import binary_crossentropy
from tensorflow.keras.models     import Sequential, Model
from tensorflow.keras.layers     import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.distributions    import Normal
from misc_utils                  import concat_episodes

# hidden_layer_nodes = 24
hidden_layer_nodes = 50

######################### MCC ###########################

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
        # print('Qvalues:')
        # print(Qvalues)
        return Qvalues
        
    def save(self, name):
        self.mlp.save_weights(name)
        
    def load(self, name):
        print('loading weights from {:s}'.format(name))
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
            
            # time in state
            # t = np.arange(100).reshape((1,100)) / 99
            # x = np.transpose(np.vstack((states, t, actions)))
            
            # time not in state
            x = np.transpose(np.vstack((states, actions)))
            
            #print("Training Input:")
            #print(x)
            # print("Discounted Returns:")
            # print(disc_retrn)
        
            # 1 batch == 1 episode
            loss[i] = self.Qfunction.mlp.train_on_batch(x, disc_retrn)
        
        return np.mean(loss)

######################### REINFORCE ###########################

def reinforce_loss(y_true, y_pred):
    advantage = y_true
    log_prob  = y_pred
    return -K.mean(log_prob*advantage, axis=0)

def action(args):
    # mean, std = args
    mean = args
    std = 0.75
    action = Normal(loc=mean[0], scale=std).sample(1)
    return [action[0], mean[0]]

def log_prob(args):
    # mean, std, act_taken = args
    mean, act_taken = args
    std = 0.75
    log_prob = Normal(loc=mean, scale=std).log_prob(act_taken)
    return log_prob

def make_policy_mlp(input_size, output_size, config):
    global hidden_layer_nodes
    
    input = Input(shape=(input_size,))
    x = Dense(units=hidden_layer_nodes, activation='relu')(input)
    x = Dense(units=hidden_layer_nodes, activation='relu')(x)
    mean = Dense(units=output_size, activation='linear')(x)
    # mean = Dense(units=output_size, activation='tanh')(x)
    # std  = Dense(units=output_size, activation='softplus')(x)
    
    # action_out = Lambda(action)([mean, std])
    action_out = Lambda(action)(mean)
    model_act  = Model(inputs=input, outputs=action_out)
    
    if config['train']:
        act_taken    = Input(shape=(1,))
        # log_prob_out = Lambda(log_prob)([mean, std, act_taken])
        log_prob_out = Lambda(log_prob)([mean, act_taken])
        model_train  = Model(inputs=[input, act_taken], outputs=log_prob_out)
        
        adam = Adam(lr=config['pglr'])
        model_train.compile(loss=reinforce_loss, optimizer=adam)
    else:
        model_train = None
        
    return model_act, model_train 

class NNPolicy:
    def __init__(self, input_size, output_size, config):
        self.mlp_act, self.mlp_train = make_policy_mlp(input_size, output_size, config)

    def __call__(self, state):
        norm_action = self.mlp_act.predict(state.reshape(1,state.size))
        return norm_action
        
    def save(self, name):
        self.mlp_act.save_weights(name)
        
    def load(self, name):
        print('loading weights from {:s}'.format(name))
        self.mlp_act.load_weights(name)
        
class PolicyGradientUpdater:
    def __init__(self, policy, config):
        self.policy_mlp = policy.policy_model.mlp_train
        self.batch_size = config['batch_size']
        self.buff_size  = config['buff_size']
        self.value_function = make_sequential_mlp(policy.state_dim, 1, config)

    def __call__(self, episode_list):
        # get most recent episode
        episode = episode_list.episode_list[-1]
        states = episode.norm_states
        # actions = episode.norm_actions
        actions = episode.actions
        disc_retrn = episode.norm_returns
        
        states = np.transpose(states)
        actions = np.transpose(actions)
        
        if self.value_function is not None:
            vf_loss = self.update_value_function(episode_list)
            baseline = self.value_function.predict(states).reshape(disc_retrn.shape)
        else:
            baseline = np.mean(disc_retrn)
                
        advantage = disc_retrn - baseline
        
        pg_loss = self.policy_mlp.train_on_batch(x=[states, actions], y=advantage)
        
        return np.array([pg_loss, vf_loss])
        
    def update_value_function(self, episode_list):
        # samples from training buffer
        num_eps = episode_list.num_eps    
        num_samples = min(num_eps, self.batch_size)
        begin = max(0, num_eps - self.buff_size) 
        train_buff = np.arange(begin, num_eps)
        samples = np.random.choice(train_buff, num_samples, replace=False)
        
        loss = np.zeros(num_samples)
        for i in range(num_samples):
            ep_index = samples[i]
            episode = episode_list.episode_list[ep_index]
            states = episode.norm_states
            disc_retrn = episode.norm_returns
            
            # time in state
            # t = np.arange(100).reshape((1,100)) / 99
            # x = np.transpose(np.vstack((states, t)))
            
            # time not in state
            x = np.transpose(states)
        
            # 1 batch == 1 episode
            loss[i] = self.value_function.train_on_batch(x, disc_retrn)
        
        return np.mean(loss)

######################### GPS ###########################

def gps_loss(regularization_weight, guiding_action_prob):
    
    def loss(y_true, y_pred):
        reward = K.reshape(y_true, (-1, 100))
        #reward = K.print_tensor(reward, message = "reward:")
        
        guiding_action_probs = K.reshape(guiding_action_prob, (-1, 100))
        #guiding_action_prob = K.print_tensor(guiding_action_prob, message = "guiding_action_probs:")
        guiding_trajectory_probs = tf.cumprod(guiding_action_probs, axis=1)
        # guiding_trajectory_probs = K.print_tensor(guiding_trajectory_probs, message = "guiding_trajectory_probs:")

        policy_action_probs = K.reshape(y_pred, (-1, 100))
        #policy_action_probs = K.print_tensor(policy_action_probs, message = "policy_action_probs:")
        policy_trajectory_probs = tf.cumprod(policy_action_probs, axis=1)
        #policy_trajectory_probs = K.print_tensor(policy_trajectory_probs, message = "policy_trajectory_probs:")

        Zt = K.sum(policy_trajectory_probs / guiding_trajectory_probs, axis=0)
        #Zt = K.print_tensor(Zt, message = "Zt:")

        unnorm_reward = K.sum((policy_trajectory_probs / guiding_trajectory_probs) * reward, axis=0)
        #unnorm_reward = K.print_tensor(unnorm_reward, message = "unnorm_reward:")

        regularization = regularization_weight * K.log(Zt)
        # regularization = K.print_tensor(regularization, message = "regularization:")
        
        expected_reward = K.sum((unnorm_reward / Zt) + regularization)
        # expected_reward = K.sum((unnorm_reward / Zt))

        # negative bc keras minimizes loss function instead of maximize
        return -expected_reward

    return loss

def gps_action(args):
    # mean, std = args
    mean = args
    std = np.float64(0.75)
    dist = Normal(loc=mean[0], scale=std)
    action = dist.sample(1)
    prob = dist.prob(action)
    return [action[0], mean[0], prob[0]]

def prob(args):
    # mean, std, act_taken = args
    mean, act_taken = args
    std = np.float64(0.75)
    prob = Normal(loc=mean, scale=std).prob(act_taken)
    return prob

def make_gps_mlp(input_size, output_size, config):
    global hidden_layer_nodes
    K.set_floatx('float64')
    
    input = Input(shape=(input_size,))
    x = Dense(units=hidden_layer_nodes, activation='relu')(input)
    x = Dense(units=hidden_layer_nodes, activation='relu')(x)
    mean = Dense(units=output_size, activation='linear')(x)
    # mean = Dense(units=output_size, activation='tanh')(x)
    # std  = Dense(units=output_size, activation='softplus')(x)
    
    # action_out = Lambda(action)([mean, std])
    action_out = Lambda(gps_action)(mean)
    model_act  = Model(inputs=input, outputs=action_out)
    
    if config['train']:
        act_taken    = Input(shape=(output_size,))
        guiding_prob = Input(shape=(1,))
        # log_prob_out = Lambda(log_prob)([mean, std, act_taken])
        prob_out = Lambda(prob)([mean, act_taken])
        
        # Model for full training
        reg_weight = K.variable(1e-2, name='regularization_weight')
        model_train  = Model(inputs=[input, act_taken, guiding_prob], outputs=prob_out)
        adam1 = Adam(lr=config['lr'])
        model_train.compile(loss=gps_loss(reg_weight, guiding_prob), optimizer=adam1)
        
        # Model for pretraining
        model_pretrain = Model(inputs=[input, act_taken], outputs=prob_out)
        # adam2 = Adam(lr=config['lr'])
        adam2 = Adam(lr=0.001)
        model_pretrain.compile(loss=binary_crossentropy, optimizer=adam2)
    else:
        model_train = None
        model_pretrain = None
        reg_weight = None
        
    return model_act, model_train, model_pretrain, reg_weight
        
class GPSNN:
    def __init__(self, input_size, output_size, config):
        self.mlp_act, self.mlp_train, self.mlp_pretrain, self.reg_weight = make_gps_mlp(input_size, output_size, config)

    def __call__(self, state):
        norm_action = self.mlp_act.predict(state.reshape(1,state.size))
        return norm_action
        
    def action_prob(self, state, action):
        prob = self.mlp_pretrain.predict(x=[state, action])
        return prob
        
    def decrease_regularization_weight(self):
        w = K.get_value(self.reg_weight)
        if w > 1e-4:
            w /= 10.
            # print('decreasing regularization to {:g}'.format(w))
            K.set_value(self.reg_weight, w)

    def increase_regularization_weight(self):
        w = K.get_value(self.reg_weight)
        if w < 1e-2:
            w *= 10.
            # print('increasing regularization to {:g}'.format(w))
            K.set_value(self.reg_weight, w)
        
    def save(self, name):
        self.mlp_act.save_weights(name)
        
    def load(self, name):
        self.mlp_act.load_weights(name)

class GPSUpdater:
    def __init__(self, policy, config):
        self.mlp_train = policy.policy_model.mlp_train
        self.mlp_pretrain = policy.policy_model.mlp_pretrain

    def __call__(self, episode_list):
        x = episode_list[0]
        u = episode_list[1]
        r = episode_list[2]
        guiding_action_probs = episode_list[3]
        
        loss = self.mlp_train.train_on_batch(x=[x, u, guiding_action_probs], y=r)
        return loss
        
    def pretraining(self, guiding_samples):
        concat_states, concat_actions, concat_rewards = concat_episodes(guiding_samples)

        x = np.transpose(concat_states)
        u = np.transpose(concat_actions)
        y = np.ones(x.shape[0])

        # Binary Crossentropy with y all ones maximizes log probability
        num_train_loops = 2000
        loss = np.zeros(num_train_loops)
        for i in range(num_train_loops):
            loss[i] = self.mlp_pretrain.train_on_batch(x=[x, u], y=y)

        return loss
        
        
        
        
        
        
