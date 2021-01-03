import pickle
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical

########################################################
# read cifar-10 data using keras
########################################################
def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_test


def load_cifar10_keras():
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    # train_data = train_data / 255.0
    # test_data = test_data / 255.0

    train_data, test_data = normalize(train_data, test_data)

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_data, train_labels, test_data, test_labels


########################################################
# read cifar-10 data, batch 1-5 training data
########################################################
def load_cifar10_train(path):

    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []
    cifar_label_train_onehot = np.zeros((50000, 10))

    for i in range(1, 6):
        with open(path + '/data_batch_' + str(i), 'rb') as fo:
            data_batch = pickle.load(fo, encoding='bytes')
            if i == 1:
                cifar_train_data = data_batch[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, data_batch[b'data']))
            cifar_train_filenames += data_batch[b'filenames']
            cifar_train_labels += data_batch[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    cifar_train_labels = np.array(cifar_train_labels)

    for cl in range(50000):
        cifar_label_train_onehot[cl, cifar_train_labels[cl]] = 1

    return cifar_train_data, cifar_label_train_onehot


########################################################
# read cifar-10 data, testing data
########################################################
def load_cifar10_eval(path):
    with open(path + '/test_batch', 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
        test_data = test_batch[b'data']
        test_label = test_batch[b'labels']

    cifar_test_data = test_data.reshape((len(test_data), 3, 32, 32))
    cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)

    cifar_label_test_onehot = np.zeros((10000, 10))

    for cl in range(10000):
        cifar_label_test_onehot[cl, test_label[cl]] = 1

    return cifar_test_data, cifar_label_test_onehot

