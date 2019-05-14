import pickle
import os
import sys
import argparse
from keras.utils import to_categorical
from sklearn.datasets import load_boston, load_diabetes

from refactoring.models import *
from refactoring.query_methods import *

import numpy as np


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_index', type=int, help="index of current experiment")
    p.add_argument('data_type', type=str, choices={'mnist', 'cifar10', 'cifar100'}, help="data type (mnist/cifar10/cifar100)")
    p.add_argument('batch_size', type=int, help="active learning batch size")
    p.add_argument('initial_size', type=int, help="initial sample size for active learning")    # Todo: initial sample size 도 정해주는 건가 ... ?
    p.add_argument('iterations', type=int, help="number of active learning batches to sample")  # todo: batch_size 랑은 뭐가 다른걸까 ... ?
    p.add_argument('method', type=str,
                   defulat="Discriminative",
                   choices={"Random", "CoreSet", "CoreSetMIP", "Discriminative", "DiscriminativeLearned", "DiscriminativeAE",
                            "DiscriminativeStocastic", "Uncertainty", "Bayesian", "UncertaintyEntropy", "BayesianEntropy", "EGL", "Adversarial"},
                   help="sampling method ('Random', 'CoreSet', 'CoreSetMIP', 'Discriminative', 'DiscriminativeLearned', 'DiscriminativeAE',"
                        "'DiscriminativeStocastic', 'Uncertainty', 'Bayesian', 'UncertaintyEntropy', 'BayesianEntropy', 'EGL', 'Adversarial'")
    p.add_argument("experiment_folder", type=str,
                   default=r"D:\소프트팩토리\소프트팩토리_대전\Git\paper_implementation\active_learning\core_set",
                   help="folder where the experiment results will be saved")
    p.add_argument( choices={None,'Random','CoreSet','CoreSetMIP','Discriminative','DiscriminativeLearned','DiscriminativeAE','DiscriminativeStochastic','Uncertainty','Bayesian','UncertaintyEntropy','BayesianEntropy','EGL','Adversarial'},
                   default=None,
                   help="second sampling method ('Random','CoreSet','CoreSetMIP','Discriminative','DiscriminativeLearned','DiscriminativeAE','DiscriminativeStochastic','Uncertainty','Bayesian','UncertaintyEntropy','BayesianEntropy','EGL','Adversarial')")
    p.add_argument('--initial_idx_path', '-idx', type=str,
                   default=None,
                   help="path to a folder with pickle file with the initial indices of the labeled set")
    p.add_argument("--gpu", "-gpu", type=int, default=2)
    args = p.parse_args()
    return args


def load_batch(fpath, label_key="labels"):
    # Todo: 이거 사용되는 예시 보고 정리
    with open(fpath, 'rb') as f:    # todo: rb ... ?
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding="bytes")
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_mnist():
    """
    load nad pre-process the MNIST data
    """

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_last":
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    else:
        x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
        x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))

    # standardize the dataset:
    x_train = np.array(x_train).astype('float32') / 255
    x_test = np.array(x_test).astype('float32') / 255

    # shuffle the data:
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def evaluate_sample(training_function, X_train, Y_train, X_test, Y_test, checkpoint_path):
    # Todo: 이거 제대로 된 것인지 확인 필요
    """
    A function that accepts a labeled-unlabeled data split
    and trains the relevant model on the labeled data,
    returning the model and it`s accuracy on the test set
    """

    # shuffle the training set:
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # create the validation set:
    X_validation = X_train[:int(0.2*X_train.shape[0])]
    Y_validation = Y_train[:int(0.2*Y_train.shape[0])]
    X_train = X_train[int(0.2*X_train.shape[0]):]
    Y_train = Y_train[int(0.2*X_train.shape[0]):]

    # train and evaluate the model
    model = training_function(X_train, Y_train,
                              X_validation, Y_validation,
                              checkpoint_path, gpu=args.gpu)
    if args.data_type in ['imdb', "wiki"]:
        acc = model.evaluate(X_test, Y_test, verbose=0)
    else:
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    return acc, model


def get_data(data_name="mnist"):
    """
    get data

    Args
    ----
    data_name: (str) i.e. 'mnist'

    Returns
    -------
    (X_train, Y_train)
    (X_test, Y_test)
    nb_labels
    input_shape
    evaluation_function
    """
    if data_name == "mnist":
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        nb_labels = 10
        if K.image_data_format() == "channels_last":
            input_shape = (28, 28, 1)
        else:
            input_shape = (1, 28, 28)
        # todo: evaluation_function 이 뭘까 ... ?
        evaluation_function = train_mnist_model

    # make categorical:
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, Y_train, X_test, Y_test, nb_labels, input_shape, evaluation_function


if __name__ == "__main__":
    # parse the arguments
    args = parse_input()    # Todo: 이거 없애봄

    # load the dataset:
    X_train, Y_train, X_test, Y_test, nb_labels, input_shape, evaluation_function = \
        get_data(data_name="mnist")

    # load the indices:
    if args.initial_idx_path is not None:
        idx_path = os.path.join(
            args.initial_idx_path,
            "{exp}_{size}_{data}.pkl".format(exp=args.experiment_index,
                                             size=args.initial_size,
                                             data=args.data_type))
        with open(idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        print("No initial indices found - drawing random indices...")
        labeled_idx = np.random.choice(X_train.shape[0], args.initial_size,
                                       replace=False)

    method = CoreSetSampling
    query_method = method(None, input_shape, num_labels, args.gpu)
