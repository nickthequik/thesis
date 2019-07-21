
import numpy as np
from abc        import ABC, abstractmethod
from misc_utils import discretize_actions
from nn_utils   import QFunction, QFunctionUpdater, \
                       NNPolicy, PolicyGradientUpdater, \
                       GPSNN, GPSUpdater

def init_agent(config, env):
    agent_name = config['agent']

    if agent_name == 'Random':
        agent = RandomAgent(env)
    elif agent_name =='MonteCarloControl':
        agent = MCCAgent(env, config)
    elif agent_name == 'REINFORCE':
        agent = REINFORCEAgent(env, config)
    elif agent_name == "GPS":
        agent = GPSAgent(env, config)
    else:
        agent = None
        print("ERROR: Invalid agent specified in config file")

    return agent

def get_loss(config):
    agent_name = config['agent']
    num_eps   = config['episodes']
    
    if agent_name == 'Random' or agent_name =='MonteCarloControl' or agent_name =='GPS':
        loss = np.zeros((num_eps, 1))
    elif agent_name == 'REINFORCE':
        loss = np.zeros((num_eps, 2))
    else:
        agent = None
        print("ERROR: Invalid agent specified in config file")

    return loss

class Policy(ABC):
    @abstractmethod
    def __call__(self, state):
        ...

class EpsilonGreedyPolicy(Policy):
    def __init__(self, env, config):
        self.state_dim   = env.state_space.shape[0]
        self.action_dim  = env.action_space.shape[0]
        self.high        = env.action_space.high
        self.low         = env.action_space.low
        self.num_actions = config['num_actions']
        self.epsilon_dec = config['epsilon_dec']
        self.epsilon     = 1.0

        # time not in state
        self.Qfunction = QFunction(self.state_dim+self.action_dim, config)
        
        # time in state
        # self.Qfunction = QFunction(self.state_dim+self.action_dim+1, config) 
        
        self.actions = discretize_actions(self.low, self.high, self.num_actions)
        self.norm_actions = self.actions/self.high

    def __call__(self, norm_state):
        # choose action based on epsilon greedy policy
        if np.random.uniform() > self.epsilon:
            # state is assumed to be normalized already
            Qvalues = self.Qfunction(norm_state, self.norm_actions)
            return self.actions[np.argmax(Qvalues)]
        else:
            return self.actions[np.random.randint(0, self.actions.size)]

    def greedy_action(self, norm_state):
        Qvalues = self.Qfunction(norm_state, self.norm_actions)
        # print(Qvalues)
        return self.actions[np.argmax(Qvalues)]

    def decay_epsilon(self):
        if self.epsilon > 0.05:
            self.epsilon *= self.epsilon_dec
        #print("Epsilon Value: {:g}".format(self.epsilon))

class GaussianPolicy(Policy):
    def __init__(self, env, config):
        self.state_dim   = env.state_space.shape[0]
        self.action_dim  = env.action_space.shape[0]
        self.high        = env.action_space.high
        self.low         = env.action_space.low
        
        # time not in state
        self.policy_model = NNPolicy(self.state_dim, self.action_dim, config)
        
        # time in state
        # self.policy = NNPolicy(self.state_dim+1, self.action_dim, config)

    def __call__(self, norm_state):
        output = self.policy_model(norm_state)
        action = output[0]
        return action

    def greedy_action(self, norm_state):
        output = self.policy_model(norm_state)
        greedy_action = output[1]
        return greedy_action

class GPSPolicy(Policy):
    def __init__(self, env, config):
        self.state_dim   = env.state_space.shape[0]
        self.action_dim  = env.action_space.shape[0]
        self.policy_model = GPSNN(self.state_dim, self.action_dim, config)

    def __call__(self, state):
        output = self.policy_model(state)
        action = output[0]
        prob = output[2]
        return action, prob

    def greedy_action(self, state):
        output = self.policy_model(state)
        greedy_action = output[1]
        return greedy_action
        
    def action_prob(self, state, action):
        return self.policy_model.action_prob(state, action)

class RandomPolicy(Policy):
    def __init__(self, env):
        self.action_dim = env.action_space.shape[0]
        self.high       = env.action_space.high
        self.low        = env.action_space.low

    def __call__(self, state):
        return np.random.uniform(self.low, self.high)

class Agent(ABC):
    def __init__(self, policy, trainer):
        self.policy = policy
        self.trainer = trainer

    @abstractmethod
    def act(self, state):
        ...

    @abstractmethod
    def train(self, episode_list):
        ...

class MCCAgent(Agent):
    def __init__(self, env, config):
        policy  = EpsilonGreedyPolicy(env, config)
        
        if config['train']:
            trainer = QFunctionUpdater(policy, config)
        else:
            trainer = None
            
        super().__init__(policy, trainer)

    def act(self, state, greedy=False):
        if not greedy:
            return self.policy(state)
        else:
            return self.policy.greedy_action(state)

    def train(self, episode_list):
        self.policy.decay_epsilon()
        loss = self.trainer(episode_list)
        return loss
        
    def save(self, name):
        self.policy.Qfunction.save(name + '/MCCAgent_weights.h5')
        
    def load(self, name):
        self.policy.Qfunction.load(name + '/MCCAgent_weights.h5')

class REINFORCEAgent(Agent):
    def __init__(self, env, config):
        policy  = GaussianPolicy(env, config)
        
        if config['train']:
            trainer = PolicyGradientUpdater(policy, config)
        else:
            trainer = None
            
        super().__init__(policy, trainer)

    def act(self, state, greedy=False):
        if not greedy:
            return self.policy(state)
        else:
            return self.policy.greedy_action(state)

    def train(self, episode_list):
        loss = self.trainer(episode_list)
        return loss
        
    def save(self, name):
        self.policy.policy_model.save(name + '/REINFORCEAgent_weights.h5')
        self.trainer.value_function.save_weights(name + '/ValueFunction_weights.h5')
        
    def load(self, name):
        self.policy.policy_model.load(name + '/REINFORCEAgent_weights.h5')

class GPSAgent(Agent):
    def __init__(self, env, config):
        policy  = GPSPolicy(env, config)
        
        if config['train']:
            trainer = GPSUpdater(policy, config)
        else:
            trainer = None    
        
        super().__init__(policy, trainer)

    def act(self, state, greedy=False):
        if not greedy:
            return self.policy(state)
        else:
            return self.policy.greedy_action(state)

    def pretrain(self, episode_list):
        loss = self.trainer.pretraining(episode_list)
        return loss

    def train(self, episode_list):
        loss = self.trainer(episode_list)
        return loss
        
    def save(self, dir, best):
        if best:
            # print("Saving Best Agent")
            fn = dir + '/ISAgent_best_weights.h5'
            self.policy.policy_model.save(fn)
        else:
            # print("Saving Temp Agent")
            fn = dir + '/ISAgent_temp_weights.h5'
            self.policy.policy_model.save(fn)
    
    def load(self, dir, best):
        if best:
            # print("Loading Best Agent")
            fn = dir + '/ISAgent_best_weights.h5'
            self.policy.policy_model.load(fn)
        else:
            # print("Loading Temp Agent")
            fn = dir + '/ISAgent_temp_weights.h5'
            self.policy.policy_model.load(fn)
    
    # def save(self, dir, agent_num):
    #     print("Saving Agent {:d}".format(agent_num))
    #     fn = dir + '/GPSAgent{:d}_weights.h5'.format(agent_num)
    #     self.policy.policy_model.save(fn)
    # 
    # def load(self, dir, agent_num):
    #     print("Loading Agent {:d}".format(agent_num))
    #     fn = dir + '/GPSAgent{:d}_weights.h5'.format(agent_num)
    #     self.policy.policy_model.load(fn)

class RandomAgent(Agent):
    def __init__(self, env):
        policy = RandomPolicy(env)
        super().__init__(policy, None)

    def act(self, state):
        return self.policy(state)

    def train(self, episode_list):
        pass
        
    def save(self, name):
        pass
