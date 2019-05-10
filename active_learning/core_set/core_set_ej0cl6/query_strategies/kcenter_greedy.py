# https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/kcenter_greedy.py

import numpy as np
from query_strategies.strategy import Strategy
from datetime import datetime


class KCenterGreedy(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(KCenterGreedy, self).__init__(X, Y, idxs_lb, net, handler, args)  # Todo: 이거 확인 필요

    def query(self, n):
        lb_flag = self.idxs_lb.copy()       # Todo: 이거 뭐지 ... ?
        embedding = self.get_embedding(self.X, self.Y)   # Todo: 이것도 뭐지 ... ?
        embedding = embedding.numpy()

        # calculate dist matrix
        print('calculate distance matrix')
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())      # Todo: 이게 왜 dist 가 될까 ... ?
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        dist_mat *= -2      # Todo: 왜 -2 를 곱하지 ... ?
        dist_mat += sq      # Todo: 왜 sq 를 더하지 ... ?
        dist_mat += sq.transpose()  # Todo: 왜 sq.T 를 더하지 ... ?
        dist_mat = np.sqrt(dist_mat)    # Todo: 왜 root 를 취할까 ... ?
        print(datetime.now() - t_start)

        mat = dist_mat[~lb_flag, :][:, lb_flag]     # Todo: 이거 뭐지 ... ?

        # Todo: 여기 뭐 하는 걸까 ... ?
        for i in range(n):
            print('greedy solution {}/{}'.format(i, n))
            mat_min = mat.min(axis=1)    # Todo: 이거 뭐지 ... ?
            q_idx_ = mat_min.argmax()    # Todo: 이거 뭐지 ... ?
            q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]    # Todo: 이거 뭐지 ... ?
            lb_flag[q_idx] = True    # Todo: 이거 뭐지 ... ?
            mat = np.delete(mat, q_idx_, 0)    # Todo: 이거 뭐지 ... ?
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)    # Todo: 이거 뭐지 ... ?

        return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]    # Todo: 이거 뭐지 ... ?
