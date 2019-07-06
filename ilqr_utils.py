
import numpy as np
import gym
import my_envs

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

def get_ilqr_env(env_name):
    if env_name == 'my_Pendulum-v0':
        env = PendulumWrapper()
    elif env_name == 'my_Cartpole-v0':
        env = CartpoleWrapper('my_Cartpole-v0')
    elif env_name == 'my_Acrobot-v0':
        env = AcrobotWrapper('my_Acrobot-v0')
    else:
        env = None
        print("ERROR: Invalid environment {:s} specified".format(env_name))

    return env

# wrap theta to be between pi and -pi
def normalize_theta(x):
    xnorm = np.zeros(2)
    xnorm[0] = (((x[0]+np.pi) % (2*np.pi)) - np.pi)
    xnorm[1] = x[1]
    return xnorm

class EnvWrapper(ABC):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        
    @abstractmethod
    def linearize_system(self, x, l):
        ...
        
    @abstractmethod
    def cost(self, x, u, final=False):
        ...
        
    @abstractmethod
    def step(self, u):
        ...
        
    @abstractmethod
    def reset(self):
        ...

class PendulumWrapper(EnvWrapper):
    def __init__(self):
        super().__init__('my_Pendulum-v0')
        self.state_dim   = self.env.state_space.shape[0]
        self.action_dim  = self.env.action_space.shape[0]
        self.u_lim       = self.env.max_torque
        self.dt          = self.env.dt
        
        # self.x0          = np.array([-0.94113802,  0.33802252, -0.78166668])
        self.x0          = np.array([(2 * np.random.rand() - 1), 
                                     (2 * np.random.rand() - 1), 
                                     (2 * np.random.rand() - 1)])
        
        self.x_target    = np.array([1., 0, 0])
        
        # Quadratic Cost Terms
        self.Q  = 10*np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]])
        self.Qf = 10*np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]])
        self.R  = 10*np.array([[0.5]])
        # self.Q  = np.array([[10, 0, 0], [0, 10, 0], [0, 0, .1]])
        # self.Qf = np.array([[10, 0, 0], [0, 10, 0], [0, 0, .1]])
        # self.R  = np.array([[.05]])
        # self.Q  = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 1]])
        # self.Qf = np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]])
        # self.R  = np.array([[5]])
        
    def linearize_system(self, x, u, l, N):
        fx  = np.zeros((self.state_dim, self.state_dim, N))
        fu  = np.zeros((self.state_dim, self.action_dim, N))
        lx  = np.zeros((self.state_dim, N+1))
        lu  = np.zeros((self.action_dim, N))
        lxx = np.zeros((self.state_dim, self.state_dim, N+1))
        luu = np.zeros((self.action_dim, self.action_dim, N))
        
        for t in range(N):
            # linearized dynamics
            # fx[:,:,t] = np.eye(2) + np.array([[0, 1],[15.*np.cos(x[0,t]), 0]]) * self.dt
            # fu[:,:,t] = np.array([[0],[3.]]) * self.dt
            
            # dynamics with trig state state
            fx[:,:,t] = np.eye(3) + np.array([[0, -x[2,t], -x[1,t]],[x[2,t], 0, x[0,t]],[0, 15., 0]]) * self.dt
            fu[:,:,t] = np.array([[0],[0],[3.]]) * self.dt
            
            # quadratized cost
            lx[:,t] = self.Q @ (x[:,t] - self.x_target)
            lu[:,t] = self.R @ u[:,t]
            lxx[:,:,t] = self.Q
            luu[:,:,t] = self.R
            
        # final state cost terms
        lx[:,N]    = self.Qf @ (x[:,N] - self.x_target)
        lxx[:,:,N] = self.Qf

        lin_sys = [fx, fu, lx, lu, lxx, luu]
        return lin_sys
        
    def cost(self, x, u, final=False):
        # x = normalize_theta(x)
        if not final:
            cost = (x - self.x_target).T @ self.Q @ (x - self.x_target) + u.T @ self.R @ u
            # cost = 0.5*(1-np.cos(x[0]))**2 + 0.5 * 0.1 * x[1]**2 + 0.5 * 0.05 * u**2
        else:
            cost = (x - self.x_target).T @ self.Qf @ (x - self.x_target)
            # cost = 0.5*(1-np.cos(x[0]))**2 + 0.5 * 0.1 * x[1]**2
            
        return cost
        
    def step(self, u):
        x, reward, _, _ = self.env.step(u)
        return x, reward
    
    def final_cost(self, x):
        _, reward, _, _ = self.env.step([0])
        cost = self.cost(x, 0, final=True)
        
        return cost, reward
    
    def reset(self):
        self.env.reset(state=self.x0)
        return self.x0
    
class CartpoleWrapper(EnvWrapper):
    pass
    
class AcrobotWrappe(EnvWrapper):
    pass