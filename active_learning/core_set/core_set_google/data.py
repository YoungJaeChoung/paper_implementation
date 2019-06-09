from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.datasets import mnist
import keras.backend as K

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


class Dataset(object):
    def __init__(self, X, y):
        self.data = X
        self.target = y


def get_keras_data(dataname):
    """Get datasets using keras API and return as a Dataset object."""
    if dataname == 'cifar10_keras':
      train, test = cifar10.load_data()
    elif dataname == 'cifar100_coarse_keras':
      train, test = cifar100.load_data('coarse')
    elif dataname == 'cifar100_keras':
      train, test = cifar100.load_data()
    elif dataname == 'mnist_keras':
      train, test = mnist.load_data()
    else:
     raise NotImplementedError('dataset not supported')

    X = np.concatenate((train[0], test[0]))
    y = np.concatenate((train[1], test[1]))

    if dataname == 'mnist_keras':
        # Add extra dimension for channel
        num_rows = X.shape[1]
        num_cols = X.shape[2]
        X = X.reshape(X.shape[0], 1, num_rows, num_cols)
        if K.image_data_format() == 'channels_last':
            X = X.transpose(0, 2, 3, 1)

    y = y.flatten()
    data = Dataset(X, y)
    return data


def make_figure_folder():
    if not os.path.exists("./figures"):
        os.makedirs("./figures")


if __name__=="__main__":
    # path
    print("Current Dir:", os.getcwd())

    # data
    dataname = 'cifar10_keras'
    data_cifar10 = get_keras_data(dataname)
    print("Shape [X]:", data_cifar10.data.shape)      # (60,000, 32, 32, 3)
    print("Shape [y]:", data_cifar10.target.shape)    # (60,000, )

    # plot
    """
    make_figure_folder()
    for idx in range(10):
        plt.imshow(data_cifar10.data[idx]/255)
        plt.savefig("./figures/cifar10 ({})".format(idx))
        plt.close()
    """




