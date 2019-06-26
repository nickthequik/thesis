
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def get_plotter(env_name, env):
    if env_name == 'my_Pendulum-v0':
        return PendulumPlotter(env)
    elif env_name == 'my_Cartpole-v0':
        ...
    elif env_name == 'my_Acrobot-v0':
        ...
    else:
        print("ERROR: Invalid environment {:s} specified".format(env_name))

class PendulumPlotter:
    def __init__(self, env):
        self.fig = plt.figure()
        self.fig.set_size_inches(14.5,5)
        
        # Phase Plane plot
        self.ax0 = self.fig.add_subplot(1,3,1)
        self.ax0.set_title("Phase Plane Trajectory")
        self.ax0.set_xlabel("Theta [rad/pi]")
        self.ax0.set_ylabel("Theta Dot [rad/s]")
        self.ax0.grid(b=True, which='both')
        self.trajectory, = self.ax0.plot(0, 0)
        self.target,     = self.ax0.plot(np.arctan2(env.x_target[1], env.x_target[0])/np.pi, env.x_target[2], marker='x', color='red')
        self.start,      = self.ax0.plot(np.arctan2(env.x0[1],env.x0[0])/np.pi, env.x0[2], marker='o', color='red')
        # self.target,     = self.ax0.plot(env.x_target[0]/np.pi, env.x_target[1], marker='x', color='red')
        # self.start,      = self.ax0.plot(env.x0[0]/np.pi, env.x0[1], marker='o', color='red')
        self.end,        = self.ax0.plot(0, 0, marker='s', color='red')

        # States and actions
        self.ax1 = self.fig.add_subplot(1,3,2)
        self.ax1.set_title("Time Domain Plot")
        self.ax1.set_xlabel("Time Step")
        self.ax1.set_ylabel("Value")
        self.ax1.grid(b=True, which='both')
        self.theta,    = self.ax1.plot(0, label="Theta [rad/pi]", color='tab:blue')
        self.thetadot, = self.ax1.plot(0, label="Theta Dot [rad/s]", color='tab:orange')
        self.action,   = self.ax1.plot(0, label="Torque [Nm]", color='tab:red')

        # Cost and reward
        self.ax3 = self.ax1.twinx()
        self.ax3.set_ylabel("Cost/Reward")
        self.cost,   = self.ax3.plot(0, label="Cost", color='black')
        self.reward, = self.ax3.plot(0, label="Reward", color='tab:green')
        
        # Total Cost
        self.ax2 = self.fig.add_subplot(1,3,3)
        self.ax2.set_title("Total Cost")
        self.ax2.set_xlabel("Time Step")
        self.ax2.set_ylabel("Total Cost")
        self.ax2.grid(b=True, which='both')
        self.J, = self.ax2.plot(0, label="Total Cost")

    def plot(self, x, u, l, J, r, Kff, Kfb, a, i):
        # update phase plane plot
        # self.trajectory.set_xdata(x[0,:]/np.pi)
        # self.trajectory.set_ydata(x[1,:])
        # self.end.set_xdata(x[0,-1]/np.pi)
        # self.end.set_ydata(x[1,-1])
        # self.ax0.relim()
        # self.ax0.autoscale()
        
        theta = np.arctan2(x[1,:], x[0,:])
        self.trajectory.set_xdata(theta/np.pi)
        self.trajectory.set_ydata(x[2,:])
        self.end.set_xdata(theta[-1]/np.pi)
        self.end.set_ydata(x[2,-1])
        self.ax0.relim()
        self.ax0.autoscale()
        
        # update states, actions, cost, rewards plot
        # self.ax1.set_title("Control Actions, Iteration = {:d}, Alpha={:g}".format(i, a))
        # t1 = np.arange(x.shape[1])
        # t2 = np.arange(u.shape[1])
        # self.theta.set_xdata(t1)
        # self.theta.set_ydata(x[0,:]/np.pi)
        # self.thetadot.set_xdata(t1)
        # self.thetadot.set_ydata(x[1,:])
        # self.action.set_xdata(t2)
        # self.action.set_ydata(u[0,:])
        # self.cost.set_xdata(t1)
        # self.cost.set_ydata(l)
        # self.reward.set_xdata(t1)
        # self.reward.set_ydata(r)
        # self.ax1.relim()
        # self.ax1.autoscale()
        # self.ax3.relim()
        # self.ax3.autoscale()
        
        self.ax1.set_title("Control Actions, Iteration = {:d}, Alpha={:g}".format(i, a))
        t1 = np.arange(x.shape[1])
        t2 = np.arange(u.shape[1])
        self.theta.set_xdata(t1)
        # self.theta.set_ydata(x[0,:]/np.pi)
        self.theta.set_ydata(theta/np.pi)
        self.thetadot.set_xdata(t1)
        self.thetadot.set_ydata(x[2,:])
        self.action.set_xdata(t2)
        self.action.set_ydata(u[0,:])
        self.cost.set_xdata(t1)
        self.cost.set_ydata(l)
        self.reward.set_xdata(t1)
        self.reward.set_ydata(r)
        self.ax1.relim()
        self.ax1.autoscale()
        self.ax3.relim()
        self.ax3.autoscale()
        
        # update total cost plot
        t3 = np.arange(J.size)
        self.J.set_xdata(t3)
        self.J.set_ydata(J)
        self.ax2.relim()
        self.ax2.autoscale()
        
        self.fig.tight_layout()
        plt.pause(0.01)
        
    def close(self):
        plt.close(self.fig)
        
