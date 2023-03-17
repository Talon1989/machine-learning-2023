import numpy as np
import matplotlib.pyplot as plt


def onehot_transformation(y: np.ndarray):
    """
    :param y: 1d numpy.ndarray numeric array
    :return: onehot transformation of given y array
    """
    Y = np.zeros([len(np.unique(y)), y.shape[0]])
    for idx, val in enumerate(y):
        Y[int(val), idx] = 1
    return Y.T


def print_graph(arr1, arr2, label1, label2, title: str, x_label='epoch'):
    """
    :param arr1: np.ndarray to be scattered
    :param arr2: np.ndarray to be plotted
    :param label1: label of arr1
    :param label2: label of arr2
    :param title: title of the graph
    :param x_label: label of x-axis
    :return:
    """
    plt.scatter(np.arange(len(arr1)), arr1, c='g', s=1, label=label1)
    plt.plot(arr2, c='b', linewidth=1, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(label1)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    plt.clf()
























































