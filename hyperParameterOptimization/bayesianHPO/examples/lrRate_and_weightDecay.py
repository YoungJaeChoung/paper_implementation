# http://research.sualab.com/introduction/practice/2019/04/01/bayesian-optimization-overview-2.html

from learning.optimizers import MomentumOptimizer as Optimizer
from learning.evaluators import AccuracyEvaluator as Evaluator

from alexNet.models.nn import ConvNet
from bayes_opt import BayesianOptimization
from datasets import asirra as dataset

import tensorflow as tf
import numpy as np
import os


# todo: 이 부분 데이터 따로 저장안하고 - 인터넷에서 로드하게끔 할 수 없나 ... ?
def read_data():
    root_dir = os.path.join('/', 'mnt', 'sdb2', 'Datasets', 'asirra')  # FIXME
    trainval_dir = os.path.join(root_dir, 'train')

    # 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
    # todo: read_asirra_subset
    X_train, y_train = dataset.read_asirra_subset(trainval_dir, one_hot=True)
    nb_rows = X_train.shape[0]
    nb_rows_val = int(nb_rows * 0.2)
    # todo: dataset.DataSet
    val_set = dataset.DataSet(X_train[:nb_rows_val], y_train[:nb_rows_val])
    train_set = dataset.DataSet(X_train[nb_rows_val:], y_train[nb_rows_val:])

    return train_set, val_set


""" 3. 특정한 초기 학습률 및 L2 정규화 계수 하에서 학습을 수행한 후, 검증 성능을 출력하는 목적 함수 정의 """
def train_and_validate(init_learning_rate_log, weight_decay_log):
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    dict_hyperParam['init_learning_rate'] = 10 ** init_learning_rate_log
    dict_hyperParam['weight_decay'] = 10 ** weight_decay_log

    model = ConvNet([227, 227, 3], 2, **dict_hyperParam)
    evaluator = Evaluator()
    optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **dict_hyperParam)

    sess = tf.Session(graph=graph, config=config)
    train_results = optimizer.train(sess, details=True, verbose=True, **dict_hyperParam)

    # 검증 정확도의 최댓값을 목적 함수의 출력값으로 반환
    best_val_score = np.max(train_results['eval_scores'])

    return best_val_score


if __name__ == "__main__":
    #
    # Data
    #

    train_set, val_set = read_data()

    # 중간 점검
    print('Training set stats:')
    print(train_set.images.shape)
    print(train_set.images.min(), train_set.images.max())
    print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
    print('Validation set stats:')
    print(val_set.images.shape)
    print(val_set.images.min(), val_set.images.max())
    print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


    #
    # Set HyperParameters
    #

    dict_hyperParam = dict()
    # todo: 평균 이미지를 왜 저장하지 ... ?
    image_mean = train_set.images.mean(axis=(0, 1, 2))    # 평균 이미지
    np.save('/tmp/asirra_mean.npy', image_mean)    # 평균 이미지를 저장
    dict_hyperParam['image_mean'] = image_mean

    # 1) learning
    dict_hyperParam['batch_size'] = 256
    dict_hyperParam['num_epochs'] = 200

    dict_hyperParam['augment_train'] = True
    dict_hyperParam['augment_pred'] = True

    dict_hyperParam['init_learning_rate'] = 0.01
    dict_hyperParam['momentum'] = 0.9
    dict_hyperParam['learning_rate_patience'] = 30
    dict_hyperParam['learning_rate_decay'] = 0.1
    dict_hyperParam['eps'] = 1e-8

    # 2) weight decay & dropout
    dict_hyperParam['weight_decay'] = 0.0005
    dict_hyperParam['dropout_prob'] = 0.5

    # 3) score threshold
    # todo: score_threshold...?
    dict_hyperParam['score_threshold'] = 1e-4


    #
    # Optimize HyperParameters (learning rate, weight decay)
    #

    # todo: dropout rate 도 optimize 시킬 수 있나 ... ?
    bayes_optimizer = BayesianOptimization(
        f=train_and_validate,
        # todo: pbounds 범위가 왜 이렇지 ... ?
        pbounds={
            'init_learning_rate_log': (-5, -1),    # FIXME
            'weight_decay_log': (-5, -1)            # FIXME
        },
        random_state=0,
        verbose=2
    )
    bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei', xi=0.01)

    for i, res in enumerate(bayes_optimizer.res):
        print('Iteration {}: \n\t{}'.format(i, res))
    print('Final result: ', bayes_optimizer.max)