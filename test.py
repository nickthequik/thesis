
import numpy as np
import gym
import my_envs

if __name__ == '__main__':
    env = gym.make('my_Pendulum-v0')
    #env = gym.make('my_Cartpole-v0')
    #env = gym.make('my_Acrobot-v0')

    for ep in range(10):
        state = env.reset_to_state([2,-5])
        #state = env.reset()

        for t in range(500):
            env.render()
            #action = env.action_space.sample()
            action = [0]
            state, reward, _, _ = env.step(action)
        
            #print("State: [{:g}, {:g}] \
            #       Action: {:g} \
            #       Reward: {:g}".format(state[0], state[1], action[0], reward))

    env.close()
