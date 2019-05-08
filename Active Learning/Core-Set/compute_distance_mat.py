# https://github.com/ozansener/active_learning_coreset/blob/master/coreset/compute_distance_mat.py

import numpy.matlib
import time
import pickle


# Todo: 이 파일은 왜 필요하지 ... ?
dat = pickle.load(open('feature_vectors_pickled'))
data = numpy.concatenate((dat['gt_f'], dat['f']), axis=0)
budget = 5000

start = time.clock()
num_images = data.shape[0]

dist_mat = numpy.matmul(data,data.transpose())

sq = numpy.array(dist_mat.diagonal()).reshape(num_images,1)
dist_mat *= -2
dist_mat += sq
dist_mat += sq.transpose()

elapsed = time.clock() - start
print("Time spent in (distance computation) is: ", elapsed)
numpy.save('distances.npy', dist_mat)

