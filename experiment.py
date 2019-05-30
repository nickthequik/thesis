
import sys
import time
from file_utils import get_exp_cfg, make_data_dir
from env_utils  import init_env
from plot_utils import plot_episodes_data, plot_loss_data
from agents     import init_agent
from training   import train_agent
from ep_utils   import store_episodes_data, store_episodes_stats, get_episodes_stats


def main():
    # get directory with argv
    exp_dir = sys.argv[1]

    # read config file
    config = get_exp_cfg(exp_dir)
    print('Experiment Configuration:')
    print(config)

    # initilize environment
    env = init_env(config)
    if not env: sys.exit()

    # repeat experiment for iter iterations
    iter = config['iterations']
    for i in range(iter):
        # create directory to store results
        data_dir = make_data_dir(exp_dir, i+1)

        # initilize agent
        agent = init_agent(config, env)
        if not agent: sys.exit()

        # keep track of start time
        start_time = time.time()

        # start training agent
        episodes_data, loss = train_agent(agent, env, config)

        elapsed_time = time.time() - start_time
        print("Elapsed time: {:g}".format(elapsed_time))

        # save episodes data and stats
        store_episodes_data(data_dir, episodes_data, loss=loss)
        episodes_stats = get_episodes_stats(config, episodes_data, elapsed_time)
        store_episodes_stats(data_dir, episodes_data, episodes_stats)
        plot_episodes_data(data_dir, episodes_data, episodes_stats, config)
        plot_loss_data(data_dir, loss)

    env.close()


if __name__ == '__main__':
    main()
