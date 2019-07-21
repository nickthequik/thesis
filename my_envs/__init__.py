from gym.envs.registration import register

register(id='my_Pendulum-v0',
         entry_point='my_envs.my_pendulum:MyPendulumEnv'
)

register(id='my_Cartpole-v0',
         entry_point='my_envs.my_cartpole:MyCartPoleEnv'
)

register(id='my_Acrobot-v0',
         entry_point='my_envs.my_acrobot:MyAcrobotEnv'
)
