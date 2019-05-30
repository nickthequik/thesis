
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from plot_utils import make_plot

def replot_loss(data_dir, y_lims):
    data = np.load(data_dir +'/train_data.npz')
    loss = data['loss']
    loss_opt = {'x_label': 'Episode',
                'y_label': 'Loss',
                'title':   'Training Loss',
                'y_lims':   y_lims}
    make_plot(loss, loss_opt, data_dir)
    