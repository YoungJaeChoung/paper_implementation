# https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py

import gc
from scipy.spatial import distance_matrix

from keras.models import Model
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Lambda
from keras import optimizers

# todo cleverhans ... ?
# from cleverhans.attacks import FastGradientMethod, DeepFool
# from cleverhans.utils_keras import KerasModelWrapper

from models import *


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


def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]),
                                                              labeled_idx))]


class CoreSetSampling(QueryMethod):
    def __init__(self, model, input_shape, nb_labels, gpu):
        super().__init__(model, input_shape, nb_labels, gpu)

    def greedy_k_center(self, labeled, unlabeled, amount):
        greedy_indices = []

        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])),
                          unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        nb_obs = labeled.shape[0]

        for idx in range(1, nb_obs, 100):
            if idx + 100 < nb_obs:
                dist = distance_matrix(labeled[idx:idx+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[idx:, :], unlabeled)
            min_dist = np.vstack((min_dist,
                                  np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for _ in range(amount-1):
            dist = distance_matrix(
                unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])),
                unlabeled
            )
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self, X_train, Y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        representation_model = Model(inputs=self.model.input,
                                     outputs=self.model.get_layer('softmax').input)
        representation = representation_model.predict(X_train, verbose=0)

        new_indices = self.greedy_k_center(representation[labeled_idx, :],
                                           representation[unlabeled_idx, :],
                                           amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))























