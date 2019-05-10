

import numpy as np

nb_obs = 5
embedding = np.zeros([nb_obs, 2])
embedding[1] = np.array([1, 2])
embedding[2] = np.array([2, 3])
embedding[3] = np.array([5, 1])
embedding[4] = np.array([4, 2])
print(embedding)

# distance matrix
dist_mat = np.matmul(embedding, embedding.transpose())
diag = np.array(dist_mat.diagonal()).reshape(nb_obs, 1)
dist_mat *= -2
dist_mat += diag
dist_mat += diag.transpose()
dist_mat = np.sqrt(dist_mat)
print(dist_mat)

# distance between row vectors (labeled rows, unlabeled rows)
lb_flag = np.array([True, True, True, False, False])
mat = dist_mat[~lb_flag, :][:, lb_flag]
print(mat)

# select unlabeled rows which satisfy max(i)min(j) dist
n = 2
for i in range(n):
    print('greedy solution {}/{}'.format(i, n))
    mat_min = mat.min(axis=1)
    q_idx_ = mat_min.argmax()
    q_idx = np.arange(nb_obs)[~lb_flag][q_idx_]
    lb_flag[q_idx] = True
    mat = np.delete(mat, q_idx_, axis=0)
    mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

self.idxs_lb ^ lb_flag

np.array([1,2,3]) ^ np.array([False, True, True])
np.array([1,2,3,4]) ^ np.array([False, True, True, True])
np.array([1,2,3,4]) ^ np.array([False, True, True, False])

np.array([False, True, True]) ^ np.array([False, True, True])
np.array([1]) ^ np.array([True])
np.array([1]) ^ np.array([False])
# todo: ^ 이거 뭐지 ... ? xor ... ?