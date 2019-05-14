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