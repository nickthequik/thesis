
import numpy as np
import gym
import my_envs

from ilqr_plot import plot_trajectory_info, plot_debug_info

def main():
    env = gym.make('my_Pendulum-v0')

    n = 2
    m = 1
    dt = 0.05
    N = 150
    max_iter = 300
    u_lim = 2
    #alpha = [10**x for x in np.linspace(0, -3, 11)]
    alpha = [1, 0.5, 0.05, 0.01]
    #alpha = [1]

    # Quadratic Cost Terms
    Q  = dt * np.array([[100, 0], [0, 70]])
    Qf = dt * np.array([[10000, 0], [0, 10000]])
    R  = dt * 150

    # Optimal Control Parameters
    K_fb = np.zeros((m,n,N))
    K_ff = np.zeros((m,1,N))

    # Hessian Quu used in GPS
    H = np.zeros((m,m,N))

    # Total Trajectory Costs
    J = np.zeros(max_iter)

    # Initial Control Sequence
    u = np.zeros((m,N))

    # Initial State
    x = np.zeros((n,1,N+1))
    #x0 = np.array([[np.pi], [0]])
    #x0 = np.array([[.05*np.pi], [0]])
    x0 = np.array([[(2 * np.random.rand() - 1) * np.pi], [2 * np.random.rand() - 1]])
    x[:,:,0] = x0

    x_target = np.array([[0], [0]])

    J_alpha = np.zeros(len(alpha))
    l_alpha = np.zeros((N+1,len(alpha)))
    r_alpha = np.zeros((N+1,len(alpha)))
    u_alpha = np.zeros((N,len(alpha)))
    x_alpha = np.zeros((2,1,N+1,len(alpha)))

    for i in range(max_iter):
        print("Iteration {:d}".format(i+1))
        for j in range(len(alpha)):
            x_alpha[:,:,:,j], u_alpha[:,j], l_alpha[:,j], J_alpha[j], r_alpha[:,j] = forward_pass(x, u, K_ff, K_fb, u_lim, alpha[j], env, Q, Qf, R, 0)

        min_index = np.argmin(J_alpha)
        a = alpha[min_index]
        l = l_alpha[:,min_index]
        r = r_alpha[:,min_index]
        J[i] = J_alpha[min_index]
        xnew = x_alpha[:,:,:,min_index]
        unew = u_alpha[:,min_index].reshape((m,N))

        #print("Total Reward: {:g}".format(np.sum(r)))

        #plot_trajectory_info(xnew, unew, l, r, K_ff, K_fb, H, x0, x_target, a, i, 0)

        #forward_pass(xnew, unew, K_ff, K_fb, u_lim, a, env, Q, Qf, R, 1)

        K_ff, K_fb, H = backward_pass(xnew, unew, u_lim, Q, Qf, R, x_target, 0)

        x = xnew.copy()
        u = unew.copy()

    plot_trajectory_info(xnew, unew, l, r, K_ff, K_fb, H, x0, x_target, a, i, 0)

    print("Total Reward for deterministic policy is {:g}".format(np.sum(r)))

    demo_policy(x, u, K_ff, K_fb, u_lim, a, env, Q, Qf, R)

    env.close()

    #save_policy(x, u, a * K_ff, K_fb, H, multiplier)

def forward_pass(x, u, K_ff, K_fb, u_lim, alpha, env, Q, Qf, R, render):
    N = u.shape[1] # number of time steps
    m = u.shape[0] # dim of action space
    n = x.shape[0] # dim of state space

    xnew = np.zeros((n,1,N+1))
    xnew[:,:,0] = x[:,:,0] # initial state
    unew = np.zeros((m,N))
    l = np.zeros(N+1)
    r = np.zeros(N+1)
    dx = np.zeros((n,1,N+1))

    env.reset_to_state(x[:,:,0].reshape((2,)))

    for i in range(N):
        if render:
            env.render()

        dx[:,:,i] = xnew[:,:,i] - x[:,:,i]
        dx[:,:,i] = wrapToPi(dx[:,:,i])

        unew[:,i] = u[:,i] + K_ff[:,:,i] * alpha + K_fb[:,:,i] @ dx[:,:,i]

        unew[:,i] = np.min([u_lim, np.max([-u_lim, unew[:,i]])])

        # u is being passed in as a 1D array which makes the returned state a 2D array
        st8, r[i], _, _ = env.step(unew[:,i])
        xnew[:,:,i+1] = st8.reshape(-1,1)
        l[i] = calc_cost(xnew[:,:,i], unew[:,i], Q, R)

    _, r[N], _, _ = env.step([0])
    l[N] = calc_cost(xnew[:,:,N], 0, Qf, 0)

    return xnew, unew, l, np.sum(l), r

def backward_pass(x, u, u_lim, Q, Qf, R, x_target, debug):
    N = u.shape[1] # number of time steps
    m = u.shape[0] # dim of action space
    n = x.shape[0] # dim of state space

    K_fb = np.zeros((m,n,N))
    K_ff = np.zeros((m,1,N))

    g = np.zeros((m,1,N))
    G = np.zeros((m,n,N))
    H = np.zeros((m,m,N))

    S = np.zeros((n,n,N+1))
    v = np.zeros((n,1,N+1))
    s = np.zeros(N+1)

    A = np.zeros((n,n,N))
    B = np.zeros((n,m,N))

    
    S[:,:,N] = Qf
    v[:,:,N] = Qf @ x[:,:,-1]
    s[N] = (x[:,:,-1] - x_target).T @ Qf @ (x[:,:,-1] - x_target)
    # x_wrap = np.array([wrapToPi(x[0,:,-1]), x[1,:,-1]])
    # S[:,:,N] = Qf
    # v[:,:,N] = Qf @ (x_wrap - x_target)
    # s[N] = (x_wrap - x_target).T @ Qf @ (x_wrap - x_target)

    for i in reversed(range(N)):
        A_k, B_k = linearize_pendulum(x[:,:,i])
        A[:,:,i] = A_k
        B[:,:,i] = B_k

        g[:,:,i] = R * u[:,i] + B_k.T @ v[:,:,i+1]
        G[:,:,i] = B_k.T @ S[:,:,i+1] @ A_k
        H[:,:,i] = R + B_k.T @ S[:,:,i+1] @ B_k

        if H[:,:,i] < 0:
            print("WARNING: NEGATIVE HESSIAN")

        K_fb[:,:,i] = - G[:,:,i] / H[:,:,i]
        K_ff[:,:,i] = - g[:,:,i] / H[:,:,i]

        if u[:,i] + K_ff[:,:,i] > u_lim:
            K_ff[:,:,i] = u_lim - u[:,i]
            K_fb[:,:,i] = np.zeros((m,n))
        elif u[:,i] + K_ff[:,:,i] < -u_lim:
            K_ff[:,:,i] = -u_lim - u[:,i]
            K_fb[:,:,i] = np.zeros((m,n))

        S[:,:,i] = Q + A_k.T @ S[:,:,i+1] @ A_k + \
                   K_fb[:,:,i].T @ H[:,:,i] @ K_fb[:,:,i] + \
                   K_fb[:,:,i].T @ G[:,:,i] + G[:,:,i].T @ K_fb[:,:,i]

        # v[:,:,i] = Q @ (x[:,:,i] - x_target) + A_k.T @ v[:,:,i+1] + \
        #            K_fb[:,:,i].T @ H[:,:,i] @ K_ff[:,:,i] + \
        #            K_fb[:,:,i].T @ g[:,:,i] + G[:,:,i].T @ K_ff[:,:,i]
        x_wrap = np.array([wrapToPi(x[0,:,i]), x[1,:,i]])
        v[:,:,i] = Q @ (x_wrap - x_target) + A_k.T @ v[:,:,i+1] + \
                   K_fb[:,:,i].T @ H[:,:,i] @ K_ff[:,:,i] + \
                   K_fb[:,:,i].T @ g[:,:,i] + G[:,:,i].T @ K_ff[:,:,i]

        # s[i] = s[i+1] + (x[:,:,i] - x_target).T @ Q @ (x[:,:,i] - x_target) + \
        #        1/2 * K_ff[:,:,i].T @ H[:,:,i] @ K_ff[:,:,i] + \
        #        K_ff[:,:,i].T @ g[:,:,i]
        s[i] = s[i+1] + (x_wrap - x_target).T @ Q @ (x_wrap - x_target) + \
               1/2 * K_ff[:,:,i].T @ H[:,:,i] @ K_ff[:,:,i] + \
               K_ff[:,:,i].T @ g[:,:,i]

    # debug plotting goes here
    if debug:
        plot_debug_info(S, v, s, A, B, g, G, H, K_ff, K_fb)

    return K_ff, K_fb, H

def linearize_pendulum(x):
    #print("Linearizing around theta={:g}, theta_dot={:g}".format(x[0,0], x[1,0]))
    A = np.eye(2) + 0.05 * np.array([[0, 1],[15. * np.cos(x[0,0]), 0]])
    B = 0.05 * np.array([[0],[3.]])

    return A, B

def calc_cost(x, u, Q, R):
    state_cost = x.T @ Q @ x
    action_cost = u * R * u
    return state_cost + action_cost

def demo_policy(x, u, K_ff, K_fb, u_lim, a, env, Q, Qf, R):
    done = False
    while not done:
        forward_pass(x, u, K_ff, K_fb, u_lim, a, env, Q, Qf, R, 1)
        again = input("Watch Policy Again? y/n\n")
        if again == 'n':
            done = True

def save_policy(x, u, K_ff, K_fb, H, mult):
    #np.savez('./trajectories/guiding_policy_{:d}'.format(mult), x, u, K_ff, K_fb, H)
    np.savez('./trajectories/guiding_policy_001', x, u, K_ff, K_fb, H)

def wrapToPi(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

if __name__ == "__main__":
    main()
