import pickle
import os
import sys
import argparse
from keras.utils import to_categorical
from sklearn.datasets import load_boston, load_diabetes

from models import *
from query_methods import *

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


class QueryMethod:
    """
    A general class for query strategies,
    with a general method for querying examples to be labeled
    """
    def __init__(self, model, input_shape=(28, 28), num_labels=10, gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        """
        get the indices of labeled examples
        after the given amount have been queried by the query strategy.

        Args
        ----
        X_train:
            the training set
        Y_train:
            the training labels
        labeled_idx:
            the indices of the labeled examples
        amount:
            the amount of examples to query

        Returns
        -------
            the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class CombinedSampling(QueryMethod):
    # Todo: 왜 반반씩 sample 을 뽑을까 ... ? 그냥 실험해본건가 ... ?
    """
    An implementation of a query strategy
    which naively combines two given query strategies,
    sampling half of the batch
    from one strategy and the other half from the other strategy
    """
    def __init__(self, model, input_shape, num_labels, method1, method2, gpu):
        super().__init__(model, input_shape, num_labels, gpu)
        # todo: method 사용 방법에 대해서 알야아 할 듯
        #  init 부분 그냥 대충 이렇게 될 것 같긴 한데 정확히 이해 안됨
        self.method1 = method1(model, input_shape, num_labels, gpu)
        self.method2 = method2(model, input_shape, num_labels, gpu)

    # todo: query 부분 이해 안됨
    # todo: amount 가 왜 //2 해서 들어갈까 ... ?
    def query(self, X_train, Y_train, labeled_idx, amount):
        labeled_idx = self.method1.query(X_train, Y_train, labeled_idx, int(amount/2))
        return self.method2.query(X_train, Y_train, labeled_idx, int(amount/2))

    # todo: model update 할 때 ㅡ 왜 같은 model 을 넣어주지 ... ?
    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model
        self.method1.update_model(new_model)
        self.method2.update_model(new_model)


class CoreSetSampling(QueryMethod):
    """

    """
    # Todo: num_labels ... ?
    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def greedy_k_center(self, labeled, unlabeled, amount):
        greedy_indices = []

        # get the minimum distances
        # between the labeled and unlabeled examples
        # (iteratively, to avoid memory isseus):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])),
                                          unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        # todo: 100 개 씩 하는거 정리 필요 - parameter 로 받음
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:(j+100), :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            # todo: np.vstack() ... ?
            #  왜 np.vtack 을 사용할까 ... ?
            min_dist = np.vstack((min_dist,
                                  np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        # todo: 왜 amount - 1 개 만큼 하지 ... ?
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])),
                                   unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self, X_train, Y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        # use the learned representation for the k-greedy-center algorith:
        # Todo: Model 부분 익숙치 않음 ㅡ Model 이 어디서 나온걸까 ... ?
        representation_model = \
            Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        representation = representation_model.predict(X_train, verbose=0)
        new_indices = self.greedy_k_center(representation[labeled_idx, :],
                                           representation[unlabeled_idx, :],
                                           amount)
        # Todo: 왜 np.hstack 을 사용할까 ... ?
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))


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

    return (X_train, Y_train), (X_test, Y_test), nb_labels, input_shape, evaluation_function


if __name__ == "__main__":
    # parse the arguments
    args = parse_input()    # Todo: 이거 없애봄

    # load the dataset:
    (X_train, Y_train), (X_test, Y_test), nb_labels, input_shape, evaluation_function = \
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