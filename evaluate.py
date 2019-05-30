
import sys
import time
from file_utils import get_exp_cfg, make_data_dir
from env_utils  import init_env
from agents     import init_agent

def main():
    # get directory with argv
    exp_dir = sys.argv[1]

    # get agent iteration to evaluate
    agent_dir = sys.argv[2]

    # read config file
    config = get_exp_cfg(exp_dir)

    # initilize environment
    env = init_env(config['environement'])

    # create directory to store results
    eval_dir = exp_dir + '/' + agent_dir
    make_data_dir(eval_dir, 'eval')

    # initilize agent
    agent = init_agent(config, env)

    # keep track of start time
    start_time = time.time()

    # start evaluating agent

    # save agent and stats



    env.close()


if '__name__' == __main__:
    main()
