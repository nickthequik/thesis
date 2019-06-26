
import numpy as np
from abc        import ABC, abstractmethod
from misc_utils import discretize_actions
from nn_utils   import QFunction, QFunctionUpdater

def init_agent(config, env):
    agent_name = config['agent']

    if agent_name == 'Random':
        agent = RandomAgent(env)
    elif agent_name =='MonteCarloControl':
        agent = MCCAgent(env, config)
    else:
        agent = None
        print("ERROR: Invalid agent specified in config file")

    return agent

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
        #self.norm_actions = np.linspace(0, 1, self.num_actions).reshape(self.num_actions, 1)
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
        return self.actions[np.argmax(Qvalues)]

    def decay_epsilon(self):
        # if self.epsilon > 0.05:
        self.epsilon *= self.epsilon_dec
        #print("Epsilon Value: {:g}".format(self.epsilon))

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
        trainer = QFunctionUpdater(policy, config)
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

class RandomAgent(Agent):
    def __init__(self, env):
        policy = RandomPolicy(env)
        super().__init__(policy, None)

    def act(self, state):
        return self.policy(state)

    def train(self, episode_list):
        pass
