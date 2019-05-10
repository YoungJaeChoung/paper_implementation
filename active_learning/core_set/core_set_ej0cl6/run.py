import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
import torch
from query_strategies.kcenter_greedy import KCenterGreedy
from query_strategies.core_set import CoreSet

from datetime import datetime
import os

"""
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning
"""

# parameters
SEED = 1

NUM_INIT_LB = 10000     # LB: Labeled
NUM_QUERY = 10        # todo: Label 붙일 데이터 개수를 말하는건가 ... ?
NUM_ROUND = 10

DATA_NAME = 'MNIST'
# DATA_NAME = 'FashionMNIST'
# DATA_NAME = 'SVHN'
# DATA_NAME = 'CIFAR10'

args_pool = {'MNIST':
                # Todo: transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 이거 뭐지 ... ?
                #  mean, std 인 듯 ㅡ 왜 이렇게 정했지 ... ?
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 0},     # original: 'num_workers': 1
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 0},   # te: test
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'FashionMNIST':
                {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'SVHN':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'CIFAR10':
                {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 0},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 0},
                 'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
            }
args = args_pool[DATA_NAME]

# set seed
np.random.seed(SEED)
torch.manual_seed(SEED)     # Todo: torch.manual_seed ... ?
torch.backends.cudnn.enabled = False    # Todo: torch.backends.cudnn.enabled ... ? GPU 사용 여부 관련된 것인가 ... ?

# load dataset
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
print("X_tr.shape:", X_tr.shape)
print("Y_tr.shape:", Y_tr.shape)
print("X_te.shape:", X_te.shape)
print("Y_te.shape:", Y_te.shape)
X_tr = X_tr[:40000]     # Todo: 왜 40000 개를 선택했지 ... ?
Y_tr = Y_tr[:40000]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)  # 40,000
idxs_tmp = np.arange(n_pool)            # array([    0,     1,     2, ..., 39997, 39998, 39999])
np.random.shuffle(idxs_tmp)             # idxs_tmp - shuffle
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# load network
net = get_net(DATA_NAME)
handler = get_handler(DATA_NAME)

# strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = LeastConfidenceDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# strategy = MarginSamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# strategy = EntropySamplingDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# strategy = KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
strategy = KCenterGreedy(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = BALDDropout(X_tr, Y_tr, idxs_lb, net, handler, args, n_drop=10)
# strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
# strategy = AdversarialBIM(X_tr, Y_tr, idxs_lb, net, handler, args, eps=0.05)
# strategy = AdversarialDeepFool(X_tr, Y_tr, idxs_lb, net, handler, args, max_iter=50)
# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)

# print info
print(DATA_NAME)
print('SEED {}'.format(SEED))
print(type(strategy).__name__)


# round 0 accuracy
if __name__=="__main__":
    # path
    os.chdir("D:/소프트팩토리/소프트팩토리_대전/Git/paper_implementation/active_learning/core_set")
    print("Current Dir:", os.getcwd())

    strategy.train()    # iter: 10
    P = strategy.predict(X_te, Y_te)    # Todo: 이거 어떻게 돌아가는걸까 ... ?
    acc = np.zeros(NUM_ROUND+1)
    acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print('Round 0\ntesting accuracy {}'.format(acc[0]))

    # Todo: 시간 많이 걸리는데 ㅡ 왜 그렇지 ... ? gpu를 안쓰나 ... ?
    for rd in range(1, NUM_ROUND+1):    # todo: 이 부분 이해 필요
        print('Round {}'.format(rd))

        # query (55.039 sec)
        start_time = datetime.now()
        q_idxs = strategy.query(NUM_QUERY)
        idxs_lb[q_idxs] = True
        print(datetime.now() - start_time)

        # update
        # Todo: self.idxs_lb 의 역할 ... ? lb data 만 뽑아서 학습하나 ... ?
        strategy.update(idxs_lb)    # update: self.idxs_lb = idxs_lb
        strategy.train()    # Todo: train 속도가 왜이리 오래 걸릴까 ... ? 데이터가 커서 그런가 ... ?

        # round accuracy
        P = strategy.predict(X_te, Y_te)    # Todo: predict 함수
        acc[rd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
        print('testing accuracy {}'.format(acc[rd]))

    # print results
    print('SEED {}'.format(SEED))
    print(type(strategy).__name__)
    print(acc)