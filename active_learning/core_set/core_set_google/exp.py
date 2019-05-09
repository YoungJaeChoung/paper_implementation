from sampling_methods.sampling_def import SamplingMethod
from sklearn.metrics import pairwise_distances

import numpy as np
import data
import os


class kCenterGreedy(SamplingMethod):
    def __init__(self, X, y, seed=2019, metric='euclidean'):
        self.X = X
        self.y = y
        self.seed = seed
        self.flat_X = self.flatten_X()  # i.e. shape: (60000, 32, 32, 3) -> (60000, 3072)
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []
        super(kCenterGreedy, self).__init__(self.X, self.y, self.seed)

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        # Todo: 이 함수가 왜 필요하지 ... ?
        """ Update min distances given cluster centers.

        Args
        ----
        cluster_centers:
            indices of cluster centers
        only_new:
            only calculate distance for newly selected points
            and update min_distances.
        reset_dist:
            whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:    # if cluster_centers exist
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, model, already_selected, N, **kwargs):
        # Todo: kwargs 에 뭐가 들어가지 ... ?
        """
        Diversity promoting active learning method
        that greedily forms a batch to minimize the maximum distance to a cluster center
        among all unlabeled datapoints.

        Args
        ----
        todo: model 에는 뭐가 들어가지 ... ?
        model:
            model with scikit-like API with decision_function implemented
        already_selected:
            index of datapoints already selected
        N:
            batch size

        Returns
        -------
        indices of points selected to minimize distance to cluster centers
        """
        try:
            # Assumes that the transform function
            # takes in original data and not flattened data.
            # todo: oridignal data 랑 flattened data 랑 뭐가 다르지 ... ?
            print('Getting transformed features...')
            self.features = model.transform(self.X)
            print('Calculating distances...')
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            print('Using flat_X as features.')
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
                # New examples should not be in already selected since those points
                # should have min_distance of zero to a cluster center.
        assert ind not in already_selected

        self.update_distances([ind], only_new=True, reset_dist=False)
        new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        self.already_selected = already_selected
        return new_batch


if __name__=="__main__":
    # path
    print("Current Dir:", os.getcwd())

    # data
    data_cifar10 = data.get_keras_data("cifar10_keras")
    X, y = data_cifar10.data, data_cifar10.target
    core_set = kCenterGreedy(X, y)

    # ------------------------------
    core_set.X.shape        # (60000, 32, 32, 3)
    core_set.y.shape
    core_set.flat_X.shape   # (60000, 3072)

    core_set.update_distances(cluster_centers=[])
    core_set.select_batch_()