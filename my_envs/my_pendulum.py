
import numpy as np
import gym

from gym       import spaces
from gym.utils import seeding
from os        import path

class MyPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed  = 8
        self.max_torque = 2.
        self.dt         = 0.05
        self.viewer     = None
        self.t          = 0

        self.action_space = spaces.Box(low=-self.max_torque, 
                                       high=self.max_torque, 
                                       shape=(1,), 
                                       dtype=np.float32)
        
        # high = np.array([np.pi, self.max_speed])
        high = np.array([1, 1, self.max_speed])
        self.state_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 3/4
        l = 2/3
        dt = self.dt

        self.t += 1

        u = np.clip(u[0], -self.max_torque, self.max_torque)
        self.last_u = u # for rendering
        
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        # costs = -np.cos(th)
        # costs = 3*(1 - np.cos(th))**2 + 3*np.sin(th)**2 + .1*thdot**2 + .001*(u**2)
        # reward = get_reward(th, self.t)

        newthdot = thdot + (-g/l * np.sin(th + np.pi) + 1./(m*l**2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        
        return self._get_obs(), -costs, False, {}

    def reset(self, state=None):
        if state is None:
            high = np.array([np.pi, 1])
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            # self.state = state
            theta = np.arctan2(state[1], state[0])
            thetadot = state[2]
            self.state = np.array([theta, thetadot])
        
        self.t = 0
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        # return np.array([theta, thetadot])
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
    
def get_reward(th, t):
    th = angle_normalize(th)
    #if (th > -0.523599 and th < 0.523599): # 30 degrees
    if (th > -0.261799 and th < 0.261799): # 15 degrees
        return 1
    else:
        return -1
    