
import numpy as np
import gym
import my_envs

if __name__ == '__main__':
    env = gym.make('my_Pendulum-v0')
    #env = gym.make('my_Cartpole-v0')
    #env = gym.make('my_Acrobot-v0')

    x0 = np.array([np.pi/2, 0])
    states
    rewards = np.zeros(100)

    state = env.reset(state=x0)
    for t in range(100):
        #env.render()
        states[:,t] = state
        state, reward, _, _ = env.step1([0])
        rewards[t] = reward    

    env.close()
