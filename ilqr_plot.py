
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np

def plot_trajectory_info(x, u, l, r, K_ff, K_fb, H, x0, x_target, a, i, debug):
    fig, (ax0, ax1, ax2) = plt.subplots(1,3)
    fig.set_size_inches(13,5)

    # Phase Plane
    ax0.plot(x[0,0,:], x[1,0,:])
    ax0.plot(x_target[0,0], x_target[1,0], marker='x', color='red')
    ax0.plot(x[0,0,0], x[1,0,0], marker='o', color='red')
    ax0.plot(x[0,0,-1], x[1,0,-1], marker='s', color='red')
    ax0.set_title("Phase Plane Trajectory")
    ax0.set_xlabel("Theta")
    ax0.set_ylabel("Theta Dot")
    ax0.grid(b=True, which='both')

    # Control Effort
    ax1.plot(u[0,:])
    ax1.set_title("Control Actions, Iteration = {:d}, Alpha={:g}".format(i, a))
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("u")
    ax1.grid(b=True, which='both')

    # States and Cost
    ax2.plot(x[0,0,:])
    ax2.plot(x[1,0,:])
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Value")
    ax2.grid(b=True, which='both')
    ax3 = ax2.twinx()
    #ax3.plot(l, color='black')
    ax3.plot(r, color='green')
    ax3.set_ylabel("Cost")
    ax3.set_title("Trajectory Info")

    fig.tight_layout()
    plt.show()
    plt.close()

    if debug:
        fig, (ax0, ax1, ax2) = plt.subplots(1,3)
        fig.set_size_inches(13,5)

        ax0.plot(K_fb[0,0,:], color='red')
        ax0.plot(K_fb[0,1,:], color='green')
        ax0.set_xlabel("Time Step")
        ax0.grid(b=True, which='both')
        ax3 = ax0.twinx()
        ax3.plot(K_ff[0,0,:], color='blue')

        ax1.plot(u[0,:], color='blue')
        ax1.set_xlabel("Time Step")
        ax1.grid(b=True, which='both')
        ax4 = ax1.twinx()
        ax4.plot(np.sqrt(1/H[0,0,:]), color='black')
        ax4.set_ylabel("Std Dev")

        ax2.plot(r, color='red')
        ax2.set_ylabel("Reward")
        ax2.set_xlabel("Time Step")
        ax2.grid(b=True, which='both')
        ax5 = ax2.twinx()
        ax5.plot(l, color='green')
        ax5.set_ylabel("Cost")

        fig.tight_layout()
        plt.show()
        plt.close()

def plot_debug_info(S, v, s, A, B, g, G, H, K_ff, K_fb):
    fig, axes = plt.subplots(2,3)
    fig.set_size_inches(15,6.5)

    # Gains
    axes[0,0].plot(K_fb[0,0,:], color='red')
    axes[0,0].plot(K_fb[0,1,:], color='green')
    ax6 = axes[0,0].twinx()
    ax6.plot(K_ff[0,0,:], color='blue')
    axes[0,0].set_title("Gains")
    axes[0,0].set_xlabel("Index")
    axes[0,0].set_ylabel("Value")

    # Costate Matrix Values
    axes[0,1].plot(S[0,0,:], color='red')
    axes[0,1].plot(S[0,1,:], color='green')
    axes[0,1].plot(S[1,0,:], color='blue', linestyle='--')
    axes[0,1].plot(S[1,1,:], color='magenta')
    axes[0,1].set_title("Costate Matrix Values")
    axes[0,1].set_xlabel("Index")
    axes[0,1].set_ylabel("Value")

    # Linearized System Dynamics
    axes[1,0].plot(A[0,0,:], color='red')
    axes[1,0].plot(A[0,1,:], color='red')
    axes[1,0].plot(A[1,0,:], color='red')
    axes[1,0].plot(A[1,1,:], color='red')
    axes[1,0].plot(B[0,0,:], color='green')
    axes[1,0].plot(B[1,0,:], color='green')
    axes[1,0].set_title("Linearized Values")
    axes[1,0].set_xlabel("Index")
    axes[1,0].set_ylabel("Value")

    # Costate Vector
    axes[1,1].plot(v[0,0,:], color='red')
    axes[1,1].plot(v[1,0,:], color='green')
    axes[1,1].set_title("Costate Vector")
    axes[1,1].set_xlabel("Index")
    axes[1,1].set_ylabel("Value")

    axes[0,2].plot(s, color='red')
    axes[0,2].set_title("Value Function")
    axes[0,2].set_xlabel("Index")
    axes[0,2].set_ylabel("Value")
    ax7 = axes[0,2].twinx()
    ax7.plot(g[0,0,:], color='green')
    ax7.plot(H[0,0,:], color='blue')

    axes[1,2].plot(G[0,0,:], color='red')
    axes[1,2].plot(G[0,1,:], color='green')
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
    plt.show()
    plt.close()
