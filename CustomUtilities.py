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


def print_graph(arr1, arr2, label1, label2, title: str, x_label='epoch', scatter=True):
    """
    :param arr1: np.ndarray to be scattered
    :param arr2: np.ndarray to be plotted
    :param label1: label of arr1
    :param label2: label of arr2
    :param title: title of the graph
    :param x_label: label of x-axis
    :param scatter: scatter of plot for the first array
    :return:
    """
    if scatter:
        plt.scatter(np.arange(len(arr1)), arr1, c='g', s=1, label=label1)
    else:
        plt.plot(arr1, c='g', linewidth=1, label=label1)
    plt.plot(arr2, c='b', linewidth=1, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(label1)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    plt.clf()


def print_simple_graph(arr):
    plt.plot(arr, c='b', linewidth=1)
    plt.show()
    plt.clf()


def create_sequential_dataset(dataset, look_back=1):
    """
    :param dataset:
    :param look_back:
    :return: 2 arrays, first is array of sequences of # look_back, second is array of values right after the end of sequences
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


#  hardcoded to work with pytorch
def create_sequential_dataset_2(dataset, look_back=1):
    """
    :param dataset:
    :param look_back:
    :return: 2 arrays, first is array of sequences of # look_back, second is array of values right after the end of sequences
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i: i+look_back])
        dataY.append(dataset[i+1: i+look_back+1])
    return dataX, dataY
# def create_sequential_dataset_2(dataset, look_back=1):
#     """
#     :param dataset:
#     :param look_back:
#     :return: 2 arrays, first is array of sequences of # look_back, second is array of values right after the end of sequences
#     """
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back):
#         dataX.append(dataset[i: i+look_back])
#         dataY.append(dataset[i+look_back])
#     return dataX, dataY


















































