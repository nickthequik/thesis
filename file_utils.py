
import os
import re

def parse_line(line):
    key, value = list(filter(None, re.split('[ =\n]', line)))
    return key, value

def interpret_kv_pair(key, value):
    if (key == 'episodes'   or key == 'iterations' or key == 'timesteps'  or
        key == 'window'     or key == 'batch_size' or key == 'num_actions' or
        key == 'buff_size'):
       value = int(value)
    elif (key == 'gamma' or key == 'lr' or key == 'epsilon_dec'):
       value = float(value)

    return value

def get_exp_cfg(exp_dir):
    config = {}
    # open config file
    with open(exp_dir + '/config.txt', mode='r') as f:

        line = f.readline()
        while line:
            key, value = parse_line(line)
            value = interpret_kv_pair(key, value)
            config[key] = value
            line = f.readline()

    return config

def make_data_dir(exp_dir, name):
    data_dir = exp_dir + '/{:s}'.format(str(name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    return data_dir
