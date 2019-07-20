
import sys
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from numpy.linalg    import inv
from ilqr_utils      import get_ilqr_env
from ilqr_plot_utils import get_plotter

def main():
    num_args = len(sys.argv)
    
    if num_args == 5:
        env_name  = sys.argv[1]
        iter      = int(sys.argv[2])
        N         = int(sys.argv[3]) # control horizon
        num_iters = int(sys.argv[4])
        characterize_performance(num_iters, env_name, iter, N)
    elif num_args == 4:
        env_name = sys.argv[1]
        iter     = int(sys.argv[2])
        N        = int(sys.argv[3]) # control horizon
        get_ilqr_controller(env_name, iter, N)
    

def characterize_performance(num_iters, env_name, iter, N):
    total_reward = np.zeros(num_iters)
    
    for i in range(num_iters):
        print(i)
        x, u, Kfb, Quu, l, r = ilqr(env_name, iter, N)
        total_reward[i] = np.sum(r)
        print(np.mean(total_reward[0:i+1]))
        
    print(np.mean(total_reward))

def get_ilqr_controller(env_name, iter, N):
    x0 = np.array([-1., 0., 0.])
    # x0 = np.array([0., 1., 0.])
    # x0 = np.array([0., -1., 0.])
    # x0 = np.array([0.5, 0.5, 0.])
    # x0 = np.array([0.5, -0.5, 0.])
    # x0 = np.array([-0.5, 0.5, 0.])
    # x0 = np.array([-0.5, -0.5, 0.])
    x, u, Kfb, Quu, l, r = ilqr(env_name, iter, N, x0=x0)    
    dir = './experiments/pendulum/ilqr/high_variance'
    fn  = '/traj0.npz'
    np.savez_compressed(dir + fn, x0=x0, x=x, u=u, Kfb=Kfb, Quu=Quu)

def ilqr(env_name, iter, N, x0=None):
    alpha = [1, 0.5]
    # alpha = [1, 0.5, 0.05, 0.01]
    # alpha = [10**x for x in np.linspace(0, -3, 11)]
    
    env = get_ilqr_env(env_name)
    if not env: sys.exit()
    
    if x0 is not None:
        env.x0 = x0
    
    ilqr_plot = get_plotter(env_name, env)
    
    x = np.zeros((env.state_dim, N+1))  # nominal trajectory
    u = np.zeros((env.action_dim, N))   # nominal controls
                    
    J = np.zeros(iter+1)                # total cost
    
    Kff = np.zeros((env.action_dim, N))
    Kfb = np.zeros((env.action_dim, env.state_dim, N))
    
    # initial rollout with no control
    x, u, l, J[0], r, a = forward_pass(x, u, Kff, Kfb, env, [1], N)
    
    ilqr_plot.plot(x, u, l, J, r, Kff, Kfb, a, 0)
    
    for i in range(iter):
        # Linearize system dynamics and quadratize costs
        lin_sys = env.linearize_system(x, u, l, N)
        
        # update the control based on the sampled trajectory
        Kff, Kfb, Quu = backward_pass(x, u, l, N, lin_sys, env)
        
        # generate rollout with the new controls
        x, u, l, J[i+1], r, a = forward_pass(x, u, Kff, Kfb, env, alpha, N)
        
        ilqr_plot.plot(x, u, l, J, r, Kff, Kfb, a, i)
        #input("Press Enter to continue...")
        
    print("Final Total Cost: {:g}".format(J[-1]))
    print("Final Total Reward: {:g}".format(np.sum(r)))
    input("Press Enter to continue...")
    
    plt.close()
    
    plt.figure()
    plt.plot(np.sqrt(1/Quu[0,0,:]))
    plt.show()
    plt.close()
    
    return x, u, Kfb, Quu, l, r

def forward_pass(x, u, Kff, Kfb, env, alpha, N):
    len_alpha = len(alpha)
    xnew = np.zeros((env.state_dim, N+1, len_alpha))
    unew = np.zeros((env.action_dim, N, len_alpha))
    lnew = np.zeros((len_alpha, N+1))
    rnew = np.zeros((len_alpha, N+1))
    Jnew = np.zeros(len_alpha)
    
    for i in range(len_alpha):
        xnew[:, 0, i] = env.reset()
        for t in range(N):
            # env.env.render()
            
            dx = xnew[:,t,i] - x[:,t]
            
            unew[:,t,i] = u[:,t] + alpha[i] * Kff[:,t] + Kfb[:,:,t] @ dx 
            
            # clip action to be within limits
            unew[:,t,i] = np.min([env.u_lim, np.max([-env.u_lim, unew[:,t,i]])])
            
            xnew[:,t+1,i], rnew[i,t] = env.step(unew[:,t,i])
            lnew[i, t] = env.cost(xnew[:,t,i], unew[:,t,i])
            
        # get final time step cost
        lnew[i,N], rnew[i,N] = env.final_cost(xnew[:,N,i])
        
        #  caculate total cost of trajectory
        Jnew[i] = np.sum(lnew[i,:])
    
    # find which alpha resulted in lowest cost
    min_index = np.argmin(Jnew)
    return (xnew[:,:,min_index], unew[:,:,min_index], lnew[min_index,:], 
            Jnew[min_index], rnew[min_index,:], alpha[min_index])
    
def backward_pass(x, u, l, N, lin_sys, env, debug=False):
    fx  = lin_sys[0]    # System Jacobian (A)
    fu  = lin_sys[1]    # System input matrix (B)
    lx  = lin_sys[2]    # Cost functuon gradient wrt x
    lu  = lin_sys[3]    # Cost function gradient wrt u
    lxx = lin_sys[4]    # Cost function Hessian  wrt x
    luu = lin_sys[5]    # Cost function Hessian  wrt u
    
    Kff = np.zeros((env.action_dim, N))                 # feedforward gain
    Kfb = np.zeros((env.action_dim, env.state_dim, N))  # feedback gain 
    
    Qu  = np.zeros((env.action_dim, N))                 # Q function gradient wrt u
    Qx  = np.zeros((env.state_dim, N))                  # Q function gradient wrt x
    Quu = np.zeros((env.action_dim, env.action_dim, N)) # Q function Hessian wrt u
    Qxx = np.zeros((env.state_dim, env.state_dim, N))   # Q function Hessian wrt x
    Qux = np.zeros((env.action_dim, env.state_dim, N))  # Q function Hessian wrt ux
    
    Vx  = np.zeros((env.state_dim, N+1))                # Value function gradient
    Vxx = np.zeros((env.state_dim, env.state_dim, N+1)) # Value function Hessian
    v   = np.zeros(N+1)                                 # Value function cost term
    
    # initilize value function boundary conditions
    v[N]       = l[N]
    Vx[:,N]    = lx[:,N]
    Vxx[:,:,N] = lxx[:,:,N]
    
    for t in reversed(range(N)):
        Qu[:,t] = lu[:,t] + fu[:,:,t].T @ Vx[:,t+1]
        Qx[:,t] = lx[:,t] + fx[:,:,t].T @ Vx[:,t+1]
        
        Quu[:,:,t] = luu[:,:,t] + fu[:,:,t].T @ Vxx[:,:,t+1] @ fu[:,:,t]
        Qxx[:,:,t] = lxx[:,:,t] + fx[:,:,t].T @ Vxx[:,:,t+1] @ fx[:,:,t]
        Qux[:,:,t] =              fu[:,:,t].T @ Vxx[:,:,t+1] @ fx[:,:,t]
    
        # this needs to be changed for higher order systems to check the eigs
        if np.any(Quu[:,:,t] < 0):
            print("WARNING: NEGATIVE HESSIAN")
    
        # inv_Quu = inv(Quu[:,:,t])
        # Kff[:,t]   = - inv_Quu @ Qu[:,t]
        # Kfb[:,:,t] = - inv_Quu @ Qux[:,:,t]
        Kff[:,t]   = - Qu[:,t] / Quu[:,:,t]
        Kfb[:,:,t] = - Qux[:,:,t] / Quu[:,:,t]
        
        # clamp control action to limits and adjust feedback gain accordingly
        if u[:,t] + Kff[:,t] > env.u_lim:
            Kff[:,t] = env.u_lim - u[:,t]
            Kfb[:,:,t] = np.zeros((env.action_dim,env.state_dim))
        elif u[:,t] + Kff[:,t] < -env.u_lim:
            Kff[:,t] = -env.u_lim - u[:,t]
            Kfb[:,:,t] = np.zeros((env.action_dim,env.state_dim))
    
        # update Value function gradient and Hessian
        v[t] = v[t+1] + l[t] + 1/2 * Kff[:,t].T @ Quu[:,:,t] @ Kff[:,t] + \
                                     Kff[:,t].T @ Qu[:,t]
        
        Vx[:,t] = Qx[:,t] + Kfb[:,:,t].T @ Quu[:,:,t] @ Kff[:,t] + \
                            Kfb[:,:,t].T @ Qu[:,t] + Qux[:,:,t].T @ Kff[:,t]
                            
        Vxx[:,:,t] = Qxx[:,:,t] + Kfb[:,:,t].T @ Quu[:,:,t] @ Kfb[:,:,t] + \
                                  Kfb[:,:,t].T @ Qux[:,:,t] + \
                                  Qux[:,:,t].T @ Kfb[:,:,t]
            
    # to improve this, make a class similar to the plotter, or add to plotter
    # plot_debug_info(Vxx, Vx, v, fx, fu, Qu, Qux, Quu, Kff, Kfb)
                          
    return Kff, Kfb, Quu

fig = None
axes = None
ax6 = None
ax7 = None

def plot_debug_info(Vxx, Vx, v, fx, fu, Qu, Qux, Quu, Kff, Kfb):
    global fig
    global axes
    global ax6
    global ax7
    
    if fig is None:
        fig, axes = plt.subplots(2,3)
        fig.set_size_inches(15,6.5)
        ax6 = axes[0,0].twinx()
        ax7 = axes[0,2].twinx()
    else:
        plt.sca(axes[0,0])
        plt.cla()
        plt.sca(axes[0,1])
        plt.cla()
        plt.sca(axes[1,0])
        plt.cla()
        plt.sca(axes[1,1])
        plt.cla()
        plt.sca(axes[0,2])
        plt.cla()
        plt.sca(axes[1,2])
        plt.cla()
        plt.sca(ax6)
        plt.cla()
        plt.sca(ax7)
        plt.cla()

    # Gains
    axes[0,0].plot(Kfb[0,0,:], color='red')
    axes[0,0].plot(Kfb[0,1,:], color='green')
    ax6.plot(Kff[0,:], color='blue')
    axes[0,0].set_title("Gains")
    axes[0,0].set_xlabel("Index")
    axes[0,0].set_ylabel("Value")

    # Costate Matrix Values
    axes[0,1].plot(Vxx[0,0,:], color='red')
    axes[0,1].plot(Vxx[0,1,:], color='green')
    axes[0,1].plot(Vxx[1,0,:], color='blue', linestyle='--')
    axes[0,1].plot(Vxx[1,1,:], color='magenta')
    axes[0,1].set_title("Vxx")
    axes[0,1].set_xlabel("Index")
    axes[0,1].set_ylabel("Value")

    # Linearized System Dynamics
    axes[1,0].plot(fx[0,0,:], color='red')
    axes[1,0].plot(fx[0,1,:], color='red')
    axes[1,0].plot(fx[1,0,:], color='red')
    axes[1,0].plot(fx[1,1,:], color='red')
    axes[1,0].plot(fu[0,0,:], color='green')
    axes[1,0].plot(fu[1,0,:], color='green')
    axes[1,0].set_title("Linearized Values")
    axes[1,0].set_xlabel("Index")
    axes[1,0].set_ylabel("Value")

    # Costate Vector
    axes[1,1].plot(Vx[0,:], color='red')
    axes[1,1].plot(Vx[1,:], color='green')
    axes[1,1].set_title("Vx")
    axes[1,1].set_xlabel("Index")
    axes[1,1].set_ylabel("Value")

    # Value function stuff
    axes[0,2].plot(v, color='red')
    axes[0,2].set_title("Value Function")
    axes[0,2].set_xlabel("Index")
    axes[0,2].set_ylabel("Value")
    ax7.plot(Qu[0,:], color='green')
    ax7.plot(Quu[0,0,:], color='blue')

    axes[1,2].plot(Qux[0,0,:], color='red')
    axes[1,2].plot(Qux[0,1,:], color='green')
    axes[1,2].set_title("G=Qux")
    axes[1,2].set_xlabel("Index")
    axes[1,2].set_ylabel("Value")

    # ax2.plot(x[0,0,:])
    # ax2.plot(x[1,0,:])
    # ax2.set_xlabel("Time Step")
    # ax2.set_ylabel("Value")
    # ax3 = ax2.twinx()
    # ax3.plot(l, color='black')
    # ax3.set_ylabel("Cost")
    # ax3.set_title("Trajectory Info")
    # plt.grid(b=True, which='both')

    fig.tight_layout()
    # plt.show()
    plt.pause(0.01)
    # plt.close()

if __name__ == "__main__":
    main()
